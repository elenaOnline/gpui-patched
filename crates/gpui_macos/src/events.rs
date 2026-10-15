use gpui::{
    Capslock, KeyDownEvent, KeyUpEvent, Keystroke, Modifiers, ModifiersChangedEvent, MouseButton,
    MouseDownEvent, MouseExitEvent, MouseMoveEvent, MousePressureEvent, MouseUpEvent,
    NavigationDirection, PenInfo, PinchEvent, Pixels, PlatformInput, PressureStage, ScrollDelta,
    ScrollWheelEvent, TouchPhase, point, px,
};

use crate::{
    LMGetKbdType, NSStringExt, TISCopyCurrentKeyboardLayoutInputSource, TISGetInputSourceProperty,
    UCKeyTranslate, kTISPropertyUnicodeKeyLayoutData,
};
use cocoa::{
    appkit::{NSEvent, NSEventModifierFlags, NSEventPhase, NSEventType},
    base::{BOOL, NO, YES, id},
};
use core_foundation::data::{CFDataGetBytePtr, CFDataRef};
use core_graphics::event::CGKeyCode;
use objc::{msg_send, sel, sel_impl};
use std::{borrow::Cow, ffi::c_void};

const BACKSPACE_KEY: u16 = 0x7f;
const SPACE_KEY: u16 = b' ' as u16;
const ENTER_KEY: u16 = 0x0d;
const NUMPAD_ENTER_KEY: u16 = 0x03;
pub(crate) const ESCAPE_KEY: u16 = 0x1b;
const TAB_KEY: u16 = 0x09;
const SHIFT_TAB_KEY: u16 = 0x19;

pub fn key_to_native(key: &str) -> Cow<'_, str> {
    use cocoa::appkit::*;
    let code = match key {
        "space" => SPACE_KEY,
        "backspace" => BACKSPACE_KEY,
        "escape" => ESCAPE_KEY,
        "up" => NSUpArrowFunctionKey,
        "down" => NSDownArrowFunctionKey,
        "left" => NSLeftArrowFunctionKey,
        "right" => NSRightArrowFunctionKey,
        "pageup" => NSPageUpFunctionKey,
        "pagedown" => NSPageDownFunctionKey,
        "home" => NSHomeFunctionKey,
        "end" => NSEndFunctionKey,
        "delete" => NSDeleteFunctionKey,
        "insert" => NSHelpFunctionKey,
        "f1" => NSF1FunctionKey,
        "f2" => NSF2FunctionKey,
        "f3" => NSF3FunctionKey,
        "f4" => NSF4FunctionKey,
        "f5" => NSF5FunctionKey,
        "f6" => NSF6FunctionKey,
        "f7" => NSF7FunctionKey,
        "f8" => NSF8FunctionKey,
        "f9" => NSF9FunctionKey,
        "f10" => NSF10FunctionKey,
        "f11" => NSF11FunctionKey,
        "f12" => NSF12FunctionKey,
        "f13" => NSF13FunctionKey,
        "f14" => NSF14FunctionKey,
        "f15" => NSF15FunctionKey,
        "f16" => NSF16FunctionKey,
        "f17" => NSF17FunctionKey,
        "f18" => NSF18FunctionKey,
        "f19" => NSF19FunctionKey,
        "f20" => NSF20FunctionKey,
        "f21" => NSF21FunctionKey,
        "f22" => NSF22FunctionKey,
        "f23" => NSF23FunctionKey,
        "f24" => NSF24FunctionKey,
        "f25" => NSF25FunctionKey,
        "f26" => NSF26FunctionKey,
        "f27" => NSF27FunctionKey,
        "f28" => NSF28FunctionKey,
        "f29" => NSF29FunctionKey,
        "f30" => NSF30FunctionKey,
        "f31" => NSF31FunctionKey,
        "f32" => NSF32FunctionKey,
        "f33" => NSF33FunctionKey,
        "f34" => NSF34FunctionKey,
        "f35" => NSF35FunctionKey,
        _ => return Cow::Borrowed(key),
    };
    Cow::Owned(String::from_utf16(&[code]).unwrap())
}

unsafe fn read_modifiers(native_event: id) -> Modifiers {
    unsafe {
        let modifiers = native_event.modifierFlags();
        let control = modifiers.contains(NSEventModifierFlags::NSControlKeyMask);
        let alt = modifiers.contains(NSEventModifierFlags::NSAlternateKeyMask);
        let shift = modifiers.contains(NSEventModifierFlags::NSShiftKeyMask);
        let command = modifiers.contains(NSEventModifierFlags::NSCommandKeyMask);
        let function = modifiers.contains(NSEventModifierFlags::NSFunctionKeyMask);

        Modifiers {
            control,
            alt,
            shift,
            platform: command,
            function,
        }
    }
}

/// The phase of a trackpad gesture (magnify, scroll). A cancelled gesture
/// (the system took it over: Mission Control, an app switch, a conflicting
/// gesture) reports `Cancelled` and never an end, so consumers must see it
/// to unwind whatever the gesture started.
fn touch_phase(phase: NSEventPhase) -> TouchPhase {
    match phase {
        NSEventPhase::NSEventPhaseMayBegin | NSEventPhase::NSEventPhaseBegan => {
            TouchPhase::Started
        }
        NSEventPhase::NSEventPhaseEnded => TouchPhase::Ended,
        NSEventPhase::NSEventPhaseCancelled => TouchPhase::Cancelled,
        _ => TouchPhase::Moved,
    }
}

/// `NSEventSubtype` of a mouse event a tablet's pen produced.
const NS_EVENT_SUBTYPE_TABLET_POINT: i16 = 1;
/// `NSEventSubtype` of a mouse event that also reports a pen entering or
/// leaving the tablet's proximity.
const NS_EVENT_SUBTYPE_TABLET_PROXIMITY: i16 = 2;
/// `NSPointingDeviceTypeEraser`: the eraser end of a pen.
const NS_POINTING_DEVICE_TYPE_ERASER: u64 = 3;
/// What a window remembers about the tablet between events: proximity
/// events say which end of the pen is in use, the mouse events that follow
/// don't. Chromium keeps the same state per view
/// (`RenderWidgetHostViewCocoa`'s `_pointerType`, set from proximity events
/// in `-tabletEvent:` and `-mouseEvent:`), and reports pen and eraser alike
/// as `pointerType: "pen"`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TabletState {
    /// The eraser end of the pen is in proximity.
    pub eraser: bool,
}

impl TabletState {
    /// Notes a proximity event: `NSEventTypeTabletProximity`, or a mouse
    /// event of subtype `NSEventSubtypeTabletProximity`.
    ///
    /// # Safety
    ///
    /// `native_event` must be a valid `NSEvent` that carries proximity
    /// information.
    pub(crate) unsafe fn note_proximity(&mut self, native_event: id) {
        let device: u64 = unsafe { msg_send![native_event, pointingDeviceType] };
        self.eraser = device == NS_POINTING_DEVICE_TYPE_ERASER;
    }
}

/// Whether a pen's mouse event reports it leaving the tablet's proximity.
///
/// # Safety
///
/// `native_event` must be a valid mouse `NSEvent` ([`is_mouse_event`]).
unsafe fn leaves_proximity(native_event: id) -> bool {
    unsafe {
        let subtype: i16 = msg_send![native_event, subtype];
        if subtype != NS_EVENT_SUBTYPE_TABLET_PROXIMITY {
            return false;
        }
        let entering: BOOL = msg_send![native_event, isEnteringProximity];
        entering == NO
    }
}

fn is_mouse_event(event_type: NSEventType) -> bool {
    matches!(
        event_type,
        NSEventType::NSLeftMouseDown
            | NSEventType::NSLeftMouseUp
            | NSEventType::NSRightMouseDown
            | NSEventType::NSRightMouseUp
            | NSEventType::NSOtherMouseDown
            | NSEventType::NSOtherMouseUp
            | NSEventType::NSMouseMoved
            | NSEventType::NSLeftMouseDragged
            | NSEventType::NSRightMouseDragged
            | NSEventType::NSOtherMouseDragged
    )
}

/// The pen details of a mouse event, `None` unless a tablet's pen produced
/// it. Chromium builds a pen `PointerEvent` from the same fields
/// (`WebMouseEventBuilder::Build` in
/// `components/input/web_input_event_builders_mac.mm`): pressure from
/// `-[NSEvent pressure]` for subtype `NSEventSubtypeTabletPoint` only, tilt
/// as `-[NSEvent tilt]` × 90°, with y negated for an unflipped view so that
/// it is positive towards the user.
///
/// # Safety
///
/// `native_event` must be a valid mouse `NSEvent` ([`is_mouse_event`]):
/// `-subtype` raises for other event types.
unsafe fn pen_info(native_event: id, tablet: &mut TabletState) -> Option<PenInfo> {
    unsafe {
        let subtype: i16 = msg_send![native_event, subtype];
        match subtype {
            NS_EVENT_SUBTYPE_TABLET_POINT => {
                let tilt = native_event.tilt();
                Some(PenInfo {
                    pressure: native_event.pressure().clamp(0.0, 1.0),
                    tilt: point(
                        (tilt.x * 90.0).clamp(-90.0, 90.0) as f32,
                        (-tilt.y * 90.0).clamp(-90.0, 90.0) as f32,
                    ),
                    eraser: tablet.eraser,
                })
            }
            NS_EVENT_SUBTYPE_TABLET_PROXIMITY => {
                tablet.note_proximity(native_event);
                Some(PenInfo {
                    eraser: tablet.eraser,
                    ..PenInfo::default()
                })
            }
            _ => None,
        }
    }
}

pub(crate) unsafe fn platform_input_from_native(
    native_event: id,
    window_height: Option<Pixels>,
    tablet: &mut TabletState,
) -> Option<PlatformInput> {
    unsafe {
        let event_type = native_event.eventType();

        // Filter out event types that aren't in the NSEventType enum.
        // See https://github.com/servo/cocoa-rs/issues/155#issuecomment-323482792 for details.
        match event_type as u64 {
            0 | 21 | 32 | 33 | 35 | 36 | 37 => {
                return None;
            }
            _ => {}
        }

        // A pen entering or leaving the tablet's proximity (sent on its own
        // by some tablet drivers, instead of as a mouse event's subtype).
        if matches!(event_type, NSEventType::NSTabletProximity) {
            tablet.note_proximity(native_event);
            return None;
        }
        let pen = if is_mouse_event(event_type) {
            pen_info(native_event, tablet)
        } else {
            None
        };
        // The pen left the tablet's proximity: it no longer points anywhere
        // (Chromium sends `pointerleave` for it).
        if pen.is_some() && leaves_proximity(native_event) {
            return window_height.map(|window_height| {
                PlatformInput::MouseExited(MouseExitEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),
                    pressed_button: None,
                    modifiers: read_modifiers(native_event),
                })
            });
        }

        match event_type {
            NSEventType::NSFlagsChanged => {
                Some(PlatformInput::ModifiersChanged(ModifiersChangedEvent {
                    modifiers: read_modifiers(native_event),
                    capslock: Capslock {
                        on: native_event
                            .modifierFlags()
                            .contains(NSEventModifierFlags::NSAlphaShiftKeyMask),
                    },
                }))
            }
            NSEventType::NSKeyDown => Some(PlatformInput::KeyDown(KeyDownEvent {
                keystroke: parse_keystroke(native_event),
                is_held: native_event.isARepeat() == YES,
                prefer_character_input: false,
            })),
            NSEventType::NSKeyUp => Some(PlatformInput::KeyUp(KeyUpEvent {
                keystroke: parse_keystroke(native_event),
            })),
            NSEventType::NSLeftMouseDown
            | NSEventType::NSRightMouseDown
            | NSEventType::NSOtherMouseDown => {
                let button = match native_event.buttonNumber() {
                    0 => MouseButton::Left,
                    1 => MouseButton::Right,
                    2 => MouseButton::Middle,
                    3 => MouseButton::Navigate(NavigationDirection::Back),
                    4 => MouseButton::Navigate(NavigationDirection::Forward),
                    // Other mouse buttons aren't tracked currently
                    _ => return None,
                };
                window_height.map(|window_height| {
                    PlatformInput::MouseDown(MouseDownEvent {
                        button,
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            // MacOS screen coordinates are relative to bottom left
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                        modifiers: read_modifiers(native_event),
                        click_count: native_event.clickCount() as usize,
                        first_mouse: false,
                        pen,
                    })
                })
            }
            NSEventType::NSLeftMouseUp
            | NSEventType::NSRightMouseUp
            | NSEventType::NSOtherMouseUp => {
                let button = match native_event.buttonNumber() {
                    0 => MouseButton::Left,
                    1 => MouseButton::Right,
                    2 => MouseButton::Middle,
                    3 => MouseButton::Navigate(NavigationDirection::Back),
                    4 => MouseButton::Navigate(NavigationDirection::Forward),
                    // Other mouse buttons aren't tracked currently
                    _ => return None,
                };

                window_height.map(|window_height| {
                    PlatformInput::MouseUp(MouseUpEvent {
                        button,
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                        modifiers: read_modifiers(native_event),
                        click_count: native_event.clickCount() as usize,
                        pen,
                    })
                })
            }
            NSEventType::NSEventTypePressure => {
                let stage = native_event.stage();
                let pressure = native_event.pressure();

                window_height.map(|window_height| {
                    PlatformInput::MousePressure(MousePressureEvent {
                        stage: match stage {
                            1 => PressureStage::Normal,
                            2 => PressureStage::Force,
                            _ => PressureStage::Zero,
                        },
                        pressure,
                        modifiers: read_modifiers(native_event),
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                    })
                })
            }
            // Some mice (like Logitech MX Master) send navigation buttons as swipe events
            NSEventType::NSEventTypeSwipe => {
                let navigation_direction = match native_event.phase() {
                    NSEventPhase::NSEventPhaseEnded => match native_event.deltaX() {
                        x if x > 0.0 => Some(NavigationDirection::Back),
                        x if x < 0.0 => Some(NavigationDirection::Forward),
                        _ => return None,
                    },
                    _ => return None,
                };

                match navigation_direction {
                    Some(direction) => window_height.map(|window_height| {
                        PlatformInput::MouseDown(MouseDownEvent {
                            button: MouseButton::Navigate(direction),
                            position: point(
                                px(native_event.locationInWindow().x as f32),
                                window_height - px(native_event.locationInWindow().y as f32),
                            ),
                            modifiers: read_modifiers(native_event),
                            click_count: 1,
                            first_mouse: false,
                            pen: None,
                        })
                    }),
                    _ => None,
                }
            }
            NSEventType::NSEventTypeMagnify => window_height.map(|window_height| {
                let phase = touch_phase(native_event.phase());

                let magnification = native_event.magnification() as f32;

                PlatformInput::Pinch(PinchEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),
                    delta: magnification,
                    modifiers: read_modifiers(native_event),
                    phase,
                })
            }),
            NSEventType::NSScrollWheel => window_height.map(|window_height| {
                let phase = touch_phase(native_event.phase());

                let raw_data = point(
                    native_event.scrollingDeltaX() as f32,
                    native_event.scrollingDeltaY() as f32,
                );

                let delta = if native_event.hasPreciseScrollingDeltas() == YES {
                    ScrollDelta::Pixels(raw_data.map(px))
                } else {
                    ScrollDelta::Lines(raw_data)
                };

                PlatformInput::ScrollWheel(ScrollWheelEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),
                    delta,
                    touch_phase: phase,
                    modifiers: read_modifiers(native_event),
                })
            }),
            NSEventType::NSLeftMouseDragged
            | NSEventType::NSRightMouseDragged
            | NSEventType::NSOtherMouseDragged => {
                let pressed_button = match native_event.buttonNumber() {
                    0 => MouseButton::Left,
                    1 => MouseButton::Right,
                    2 => MouseButton::Middle,
                    3 => MouseButton::Navigate(NavigationDirection::Back),
                    4 => MouseButton::Navigate(NavigationDirection::Forward),
                    // Other mouse buttons aren't tracked currently
                    _ => return None,
                };

                window_height.map(|window_height| {
                    PlatformInput::MouseMove(MouseMoveEvent {
                        pressed_button: Some(pressed_button),
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                        modifiers: read_modifiers(native_event),
                        pen,
                    })
                })
            }
            NSEventType::NSMouseMoved => window_height.map(|window_height| {
                PlatformInput::MouseMove(MouseMoveEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),
                    pressed_button: None,
                    modifiers: read_modifiers(native_event),
                    pen,
                })
            }),
            NSEventType::NSMouseExited => window_height.map(|window_height| {
                PlatformInput::MouseExited(MouseExitEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),

                    pressed_button: None,
                    modifiers: read_modifiers(native_event),
                })
            }),
            _ => None,
        }
    }
}

unsafe fn parse_keystroke(native_event: id) -> Keystroke {
    unsafe {
        use cocoa::appkit::*;

        let characters = native_event
            .charactersIgnoringModifiers()
            .to_str()
            .to_string();
        let mut key_char = None;
        let first_char = characters.chars().next().map(|ch| ch as u16);
        let modifiers = native_event.modifierFlags();

        let control = modifiers.contains(NSEventModifierFlags::NSControlKeyMask);
        let alt = modifiers.contains(NSEventModifierFlags::NSAlternateKeyMask);
        let mut shift = modifiers.contains(NSEventModifierFlags::NSShiftKeyMask);
        let command = modifiers.contains(NSEventModifierFlags::NSCommandKeyMask);
        let function = modifiers.contains(NSEventModifierFlags::NSFunctionKeyMask)
            && first_char
                .is_none_or(|ch| !(NSUpArrowFunctionKey..=NSModeSwitchFunctionKey).contains(&ch));

        #[allow(non_upper_case_globals)]
        let key = match first_char {
            Some(SPACE_KEY) => {
                key_char = Some(" ".to_string());
                "space".to_string()
            }
            Some(TAB_KEY) => {
                key_char = Some("\t".to_string());
                "tab".to_string()
            }
            Some(ENTER_KEY) | Some(NUMPAD_ENTER_KEY) => {
                key_char = Some("\n".to_string());
                "enter".to_string()
            }
            Some(BACKSPACE_KEY) => "backspace".to_string(),
            Some(ESCAPE_KEY) => "escape".to_string(),
            Some(SHIFT_TAB_KEY) => "tab".to_string(),
            Some(NSUpArrowFunctionKey) => "up".to_string(),
            Some(NSDownArrowFunctionKey) => "down".to_string(),
            Some(NSLeftArrowFunctionKey) => "left".to_string(),
            Some(NSRightArrowFunctionKey) => "right".to_string(),
            Some(NSPageUpFunctionKey) => "pageup".to_string(),
            Some(NSPageDownFunctionKey) => "pagedown".to_string(),
            Some(NSHomeFunctionKey) => "home".to_string(),
            Some(NSEndFunctionKey) => "end".to_string(),
            Some(NSDeleteFunctionKey) => "delete".to_string(),
            // Observed Insert==NSHelpFunctionKey not NSInsertFunctionKey.
            Some(NSHelpFunctionKey) => "insert".to_string(),
            Some(NSF1FunctionKey) => "f1".to_string(),
            Some(NSF2FunctionKey) => "f2".to_string(),
            Some(NSF3FunctionKey) => "f3".to_string(),
            Some(NSF4FunctionKey) => "f4".to_string(),
            Some(NSF5FunctionKey) => "f5".to_string(),
            Some(NSF6FunctionKey) => "f6".to_string(),
            Some(NSF7FunctionKey) => "f7".to_string(),
            Some(NSF8FunctionKey) => "f8".to_string(),
            Some(NSF9FunctionKey) => "f9".to_string(),
            Some(NSF10FunctionKey) => "f10".to_string(),
            Some(NSF11FunctionKey) => "f11".to_string(),
            Some(NSF12FunctionKey) => "f12".to_string(),
            Some(NSF13FunctionKey) => "f13".to_string(),
            Some(NSF14FunctionKey) => "f14".to_string(),
            Some(NSF15FunctionKey) => "f15".to_string(),
            Some(NSF16FunctionKey) => "f16".to_string(),
            Some(NSF17FunctionKey) => "f17".to_string(),
            Some(NSF18FunctionKey) => "f18".to_string(),
            Some(NSF19FunctionKey) => "f19".to_string(),
            Some(NSF20FunctionKey) => "f20".to_string(),
            Some(NSF21FunctionKey) => "f21".to_string(),
            Some(NSF22FunctionKey) => "f22".to_string(),
            Some(NSF23FunctionKey) => "f23".to_string(),
            Some(NSF24FunctionKey) => "f24".to_string(),
            Some(NSF25FunctionKey) => "f25".to_string(),
            Some(NSF26FunctionKey) => "f26".to_string(),
            Some(NSF27FunctionKey) => "f27".to_string(),
            Some(NSF28FunctionKey) => "f28".to_string(),
            Some(NSF29FunctionKey) => "f29".to_string(),
            Some(NSF30FunctionKey) => "f30".to_string(),
            Some(NSF31FunctionKey) => "f31".to_string(),
            Some(NSF32FunctionKey) => "f32".to_string(),
            Some(NSF33FunctionKey) => "f33".to_string(),
            Some(NSF34FunctionKey) => "f34".to_string(),
            Some(NSF35FunctionKey) => "f35".to_string(),
            _ => {
                // Cases to test when modifying this:
                //
                //           qwerty key | none | cmd   | cmd-shift
                // * Armenian         s | ս    | cmd-s | cmd-shift-s  (layout is non-ASCII, so we use cmd layout)
                // * Dvorak+QWERTY    s | o    | cmd-s | cmd-shift-s  (layout switches on cmd)
                // * Ukrainian+QWERTY s | с    | cmd-s | cmd-shift-s  (macOS reports cmd-s instead of cmd-S)
                // * Czech            7 | ý    | cmd-ý | cmd-7        (layout has shifted numbers)
                // * Norwegian        7 | 7    | cmd-7 | cmd-/        (macOS reports cmd-shift-7 instead of cmd-/)
                // * Russian          7 | 7    | cmd-7 | cmd-&        (shift-7 is . but when cmd is down, should use cmd layout)
                // * German QWERTZ    ; | ö    | cmd-ö | cmd-Ö        (Zed's shift special case only applies to a-z)
                //
                let mut chars_ignoring_modifiers =
                    chars_for_modified_key(native_event.keyCode(), NO_MOD);
                let mut chars_with_shift =
                    chars_for_modified_key(native_event.keyCode(), SHIFT_MOD);
                let always_use_cmd_layout = always_use_command_layout();

                // Handle Dvorak+QWERTY / Russian / Armenian
                if command || always_use_cmd_layout {
                    let chars_with_cmd = chars_for_modified_key(native_event.keyCode(), CMD_MOD);
                    let chars_with_both =
                        chars_for_modified_key(native_event.keyCode(), CMD_MOD | SHIFT_MOD);

                    // We don't do this in the case that the shifted command key generates
                    // the same character as the unshifted command key (Norwegian, e.g.)
                    if chars_with_both != chars_with_cmd {
                        chars_with_shift = chars_with_both;

                    // Handle edge-case where cmd-shift-s reports cmd-s instead of
                    // cmd-shift-s (Ukrainian, etc.)
                    } else if chars_with_cmd.to_ascii_uppercase() != chars_with_cmd {
                        chars_with_shift = chars_with_cmd.to_ascii_uppercase();
                    }
                    chars_ignoring_modifiers = chars_with_cmd;
                }

                if !control && !command && !function {
                    let mut mods = NO_MOD;
                    if shift {
                        mods |= SHIFT_MOD;
                    }
                    if alt {
                        mods |= OPTION_MOD;
                    }

                    key_char = Some(chars_for_modified_key(native_event.keyCode(), mods));
                }

                if shift
                    && chars_ignoring_modifiers
                        .chars()
                        .all(|c| c.is_ascii_lowercase())
                {
                    chars_ignoring_modifiers
                } else if shift {
                    shift = false;
                    chars_with_shift
                } else {
                    chars_ignoring_modifiers
                }
            }
        };

        Keystroke {
            modifiers: Modifiers {
                control,
                alt,
                shift,
                platform: command,
                function,
            },
            key,
            key_char,
        }
    }
}

fn always_use_command_layout() -> bool {
    if chars_for_modified_key(0, NO_MOD).is_ascii() {
        return false;
    }

    chars_for_modified_key(0, CMD_MOD).is_ascii()
}

const NO_MOD: u32 = 0;
const CMD_MOD: u32 = 1;
const SHIFT_MOD: u32 = 2;
const OPTION_MOD: u32 = 8;

fn chars_for_modified_key(code: CGKeyCode, modifiers: u32) -> String {
    // Values from: https://github.com/phracker/MacOSX-SDKs/blob/master/MacOSX10.6.sdk/System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/Headers/Events.h#L126
    // shifted >> 8 for UCKeyTranslate
    const CG_SPACE_KEY: u16 = 49;
    // https://github.com/phracker/MacOSX-SDKs/blob/master/MacOSX10.6.sdk/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/CarbonCore.framework/Versions/A/Headers/UnicodeUtilities.h#L278
    #[allow(non_upper_case_globals)]
    const kUCKeyActionDown: u16 = 0;
    #[allow(non_upper_case_globals)]
    const kUCKeyTranslateNoDeadKeysMask: u32 = 0;

    let keyboard_type = unsafe { LMGetKbdType() as u32 };
    const BUFFER_SIZE: usize = 4;
    let mut dead_key_state = 0;
    let mut buffer: [u16; BUFFER_SIZE] = [0; BUFFER_SIZE];
    let mut buffer_size: usize = 0;

    let keyboard = unsafe { TISCopyCurrentKeyboardLayoutInputSource() };
    if keyboard.is_null() {
        return "".to_string();
    }
    let layout_data = unsafe {
        TISGetInputSourceProperty(keyboard, kTISPropertyUnicodeKeyLayoutData as *const c_void)
            as CFDataRef
    };
    if layout_data.is_null() {
        unsafe {
            let _: () = msg_send![keyboard, release];
        }
        return "".to_string();
    }
    let keyboard_layout = unsafe { CFDataGetBytePtr(layout_data) };

    unsafe {
        UCKeyTranslate(
            keyboard_layout as *const c_void,
            code,
            kUCKeyActionDown,
            modifiers,
            keyboard_type,
            kUCKeyTranslateNoDeadKeysMask,
            &mut dead_key_state,
            BUFFER_SIZE,
            &mut buffer_size as *mut usize,
            &mut buffer as *mut u16,
        );
        if dead_key_state != 0 {
            UCKeyTranslate(
                keyboard_layout as *const c_void,
                CG_SPACE_KEY,
                kUCKeyActionDown,
                modifiers,
                keyboard_type,
                kUCKeyTranslateNoDeadKeysMask,
                &mut dead_key_state,
                BUFFER_SIZE,
                &mut buffer_size as *mut usize,
                &mut buffer as *mut u16,
            );
        }
        let _: () = msg_send![keyboard, release];
    }
    String::from_utf16(&buffer[..buffer_size]).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gesture_phases() {
        assert_eq!(touch_phase(NSEventPhase::NSEventPhaseMayBegin), TouchPhase::Started);
        assert_eq!(touch_phase(NSEventPhase::NSEventPhaseBegan), TouchPhase::Started);
        assert_eq!(touch_phase(NSEventPhase::NSEventPhaseChanged), TouchPhase::Moved);
        assert_eq!(touch_phase(NSEventPhase::NSEventPhaseStationary), TouchPhase::Moved);
        assert_eq!(touch_phase(NSEventPhase::NSEventPhaseEnded), TouchPhase::Ended);
        assert_eq!(touch_phase(NSEventPhase::NSEventPhaseCancelled), TouchPhase::Cancelled);
    }

    /// A tablet mouse event as a tablet driver sends it (built, never
    /// posted): subtype, pressure and tilt fields on a CGEvent.
    fn tablet_event(
        event_type: core_graphics::event::CGEventType,
        subtype: i64,
        fields: &[(core_graphics::event::CGEventField, f64)],
    ) -> id {
        use core_graphics::event::{CGEvent, CGMouseButton, EventField};
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
        use foreign_types::ForeignType;
        use objc::class;
        let source = CGEventSource::new(CGEventSourceStateID::Private).unwrap();
        let event = CGEvent::new_mouse_event(
            source,
            event_type,
            core_graphics::geometry::CGPoint::new(10.0, 20.0),
            CGMouseButton::Left,
        )
        .unwrap();
        event.set_integer_value_field(EventField::MOUSE_EVENT_SUB_TYPE, subtype);
        for (field, value) in fields {
            if *field == EventField::TABLET_PROXIMITY_EVENT_POINTER_TYPE
                || *field == EventField::TABLET_PROXIMITY_EVENT_ENTER_PROXIMITY
            {
                event.set_integer_value_field(*field, *value as i64);
            } else {
                event.set_double_value_field(*field, *value);
            }
        }
        unsafe { msg_send![class!(NSEvent), eventWithCGEvent: event.as_ptr()] }
    }

    #[test]
    fn tablet_pens_on_mouse_events() {
        use core_graphics::event::{CGEventType, EventField};
        let window_height = Some(px(100.0));
        let mut tablet = TabletState::default();
        let point = |subtype, event_type| {
            tablet_event(
                event_type,
                subtype,
                &[
                    (EventField::MOUSE_EVENT_PRESSURE, 0.625),
                    (EventField::TABLET_EVENT_TILT_X, 0.5),
                    (EventField::TABLET_EVENT_TILT_Y, -0.25),
                ],
            )
        };
        let down = point(1, CGEventType::LeftMouseDown);
        let Some(PlatformInput::MouseDown(e)) =
            (unsafe { platform_input_from_native(down, window_height, &mut tablet) })
        else {
            panic!("a mouse down");
        };
        let pen = e.pen.expect("a pen");
        // the event stores pressure in 1/255 steps
        assert!((pen.pressure - 0.625).abs() < 0.01);
        // tilt is in degrees, y positive towards the user (AppKit's y is up)
        assert!((pen.tilt.x - 45.0).abs() < 0.01 && (pen.tilt.y - 22.5).abs() < 0.01);
        assert!(!pen.eraser);

        let drag = point(1, CGEventType::LeftMouseDragged);
        let Some(PlatformInput::MouseMove(e)) =
            (unsafe { platform_input_from_native(drag, window_height, &mut tablet) })
        else {
            panic!("a mouse move");
        };
        assert!(e.pen.is_some_and(|p| (p.pressure - 0.625).abs() < 0.01));

        // the eraser end comes into proximity: later points are the eraser's
        let proximity = tablet_event(
            CGEventType::MouseMoved,
            2,
            &[
                (EventField::TABLET_PROXIMITY_EVENT_POINTER_TYPE, 3.0),
                (EventField::TABLET_PROXIMITY_EVENT_ENTER_PROXIMITY, 1.0),
            ],
        );
        let Some(PlatformInput::MouseMove(e)) =
            (unsafe { platform_input_from_native(proximity, window_height, &mut tablet) })
        else {
            panic!("a mouse move");
        };
        assert_eq!(e.pen.map(|p| (p.pressure, p.eraser)), Some((0.0, true)));
        let up = point(1, CGEventType::LeftMouseUp);
        let Some(PlatformInput::MouseUp(e)) =
            (unsafe { platform_input_from_native(up, window_height, &mut tablet) })
        else {
            panic!("a mouse up");
        };
        assert!(e.pen.is_some_and(|p| p.eraser));

        // the pen leaves the tablet: it points nowhere
        let away = tablet_event(
            CGEventType::MouseMoved,
            2,
            &[
                (EventField::TABLET_PROXIMITY_EVENT_POINTER_TYPE, 3.0),
                (EventField::TABLET_PROXIMITY_EVENT_ENTER_PROXIMITY, 0.0),
            ],
        );
        assert!(matches!(
            unsafe { platform_input_from_native(away, window_height, &mut tablet) },
            Some(PlatformInput::MouseExited(_))
        ));

        // a mouse (subtype 0) has no pen, whatever its pressure
        let mouse = tablet_event(CGEventType::LeftMouseDown, 0, &[]);
        let Some(PlatformInput::MouseDown(e)) =
            (unsafe { platform_input_from_native(mouse, window_height, &mut tablet) })
        else {
            panic!("a mouse down");
        };
        assert_eq!(e.pen, None);
    }
}
