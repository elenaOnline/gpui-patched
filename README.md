# gpui-patched

Zed's [GPUI](https://www.gpui.rs) crates with a small patch queue, packaged as
a standalone Cargo workspace so other projects can depend on them by git
revision. Used by [ntex](https://github.com/elenaOnline/ntex) (Ntex Studio and
Ntex Figures).

Copied from <https://github.com/zed-industries/zed> at
`2c4bc2d7b2c5b7832ad964f39840d823d961cb0e`, with the patches in
[`patches/`](patches) applied. The patches add, among other things, a headless
Metal renderer path, transformed image and glyph painting, retained meshes,
tablet pen events and a few macOS input fixes.

```toml
[dependencies]
gpui = { git = "https://github.com/elenaOnline/gpui-patched", rev = "<commit>", default-features = false }
gpui_platform = { git = "https://github.com/elenaOnline/gpui-patched", rev = "<commit>", default-features = false, features = ["font-kit", "runtime_shaders"] }
```

`runtime_shaders` is needed on machines without the offline Metal toolchain.

Licence: all 22 crates are Apache-2.0 (`LICENSE-APACHE`, zed's
copy; each crate has one, as upstream). Zed's other licence (GPL-3.0,
`LICENSE-GPL` there) covers crates that are not here; the fonts zed's tests
use (OFL) are left out.

Changes from upstream (Apache-2.0 §4(b)): the patches change the files
listed below, and every `Cargo.toml` is rewritten to build outside zed's
workspace (fields and dependencies inherited from it written out, zed crates
by relative path, async-task from the git revision zed's workspace patches
it to, zed's workspace lints). Tests, benches, examples and
dev-dependencies are left out.

Patches:

- `0001-gpui-Treat-file-drops-as-mouse-input.patch`
- `0002-gpui-Add-Window-paint_image_transformed.patch`
- `0003-gpui-Add-transformed-glyph-and-emoji-painting.patch`
- `0004-gpui-Add-Window-with_font_smoothing.patch`
- `0005-gpui-Add-hidden-and-image-cursors.patch`
- `0006-gpui-Report-cancelled-trackpad-gestures-on-macOS.patch`
- `0007-gpui-Add-transformed-drop-and-inset-shadows.patch`
- `0008-gpui-Add-a-hatch-pattern-background.patch`
- `0009-gpui-Report-tablet-pens-on-mouse-events-on-macOS.patch`
- `0010-gpui-Add-Window-covered_areas_above.patch`
- `0011-gpui-Add-retained-meshes-Window-paint_mesh.patch`
- `0012-gpui-Add-evictable-glyphs-to-the-sprite-atlas.patch`
- `0013-gpui-Add-live-regions-disabled-state-synthetic-subtr.patch`
- `0014-gpui-Ask-the-app-before-quitting-on-macOS-applicatio.patch`
- `0015-gpui-Tell-which-threads-run-dispatched-tasks.patch`
- `0016-gpui-Rasterize-glyphs-with-one-font-instance-per-siz.patch`
- `0017-gpui-Draw-the-shadows-between-a-run-s-meshes-inside-.patch`
- `0018-gpui-Add-action-listeners-that-are-available-on-dema.patch`
- `0019-gpui-Offer-the-input-handler-s-text-to-macOS-Service.patch`
- `0020-gpui-Pass-an-input-method-s-clause-underlines-with-i.patch`
- `0021-gpui-Prepare-glyph-rasters-ahead-of-painting-them-of.patch`
- `0022-gpui-Add-an-app-store-feature-that-builds-macOS-with.patch`
- `0023-gpui-Add-an-embedded-macOS-platform-for-an-applicati.patch`
- `0024-gpui-Add-gradient-paints-for-meshes-Window-paint_mes.patch`
- `0025-gpui-Add-multiply-compositing-for-retained-meshes.patch`

Files the patches change:

- `crates/gpui/Cargo.toml`
- `crates/gpui/src/app.rs`
- `crates/gpui/src/app/test_app.rs`
- `crates/gpui/src/app/test_context.rs`
- `crates/gpui/src/app/visual_test_context.rs`
- `crates/gpui/src/color.rs`
- `crates/gpui/src/elements/div.rs`
- `crates/gpui/src/executor.rs`
- `crates/gpui/src/gestures.rs`
- `crates/gpui/src/gpui.rs`
- `crates/gpui/src/input.rs`
- `crates/gpui/src/interactive.rs`
- `crates/gpui/src/key_dispatch.rs`
- `crates/gpui/src/mesh.rs`
- `crates/gpui/src/platform.rs`
- `crates/gpui/src/platform/test/platform.rs`
- `crates/gpui/src/platform/test/window.rs`
- `crates/gpui/src/scene.rs`
- `crates/gpui/src/style.rs`
- `crates/gpui/src/text_system.rs`
- `crates/gpui/src/window.rs`
- `crates/gpui/src/window/a11y.rs`
- `crates/gpui_apple/Cargo.toml`
- `crates/gpui_apple/build.rs`
- `crates/gpui_apple/src/gpui_apple.rs`
- `crates/gpui_apple/src/metal_atlas.rs`
- `crates/gpui_apple/src/metal_meshes.rs`
- `crates/gpui_apple/src/metal_renderer.rs`
- `crates/gpui_apple/src/shaders.metal`
- `crates/gpui_linux/src/linux/platform.rs`
- `crates/gpui_linux/src/linux/wayland.rs`
- `crates/gpui_linux/src/linux/wayland/client.rs`
- `crates/gpui_linux/src/linux/x11/client.rs`
- `crates/gpui_macos/Cargo.toml`
- `crates/gpui_macos/src/dispatcher.rs`
- `crates/gpui_macos/src/events.rs`
- `crates/gpui_macos/src/gpui_macos.rs`
- `crates/gpui_macos/src/platform.rs`
- `crates/gpui_macos/src/text_system.rs`
- `crates/gpui_macos/src/window.rs`
- `crates/gpui_platform/Cargo.toml`
- `crates/gpui_platform/src/gpui_platform.rs`
- `crates/gpui_web/src/events.rs`
- `crates/gpui_web/src/platform.rs`
- `crates/gpui_wgpu/src/shaders.wgsl`
- `crates/gpui_wgpu/src/shaders_storage.wgsl`
- `crates/gpui_wgpu/src/shaders_webgl.wgsl`
- `crates/gpui_wgpu/src/wgpu_atlas.rs`
- `crates/gpui_wgpu/src/wgpu_renderer.rs`
- `crates/gpui_windows/src/direct_write.rs`
- `crates/gpui_windows/src/directx_atlas.rs`
- `crates/gpui_windows/src/directx_renderer.rs`
- `crates/gpui_windows/src/events.rs`
- `crates/gpui_windows/src/shaders.hlsl`
