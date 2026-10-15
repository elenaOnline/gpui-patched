use crate::metal_atlas::MetalAtlas;
use crate::metal_meshes::{MAX_PAGES, MESH_BLOCK_VERTICES, MeshStore};
use anyhow::{Context as _, Result};
use block2::RcBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, MeshBlendMode, MeshDraw, MeshGradient,
    MeshGradientShape, MeshGradientSpread, PaintSurface, Path, Point, PrimitiveBatch,
    ScaledPixels, Scene, Shadow, Size, TransformationMatrix, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::runtime::AnyObject;

use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;

use std::{
    cell::Cell,
    ffi::c_void,
    mem,
    mem::MaybeUninit,
    ops::Range,
    ptr, slice,
    sync::{Arc, atomic::Ordering},
};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    meshes_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    multiply_path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    /// Sprites rasterized into the path texture with the meshes around them.
    intermediate_monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    intermediate_polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    intermediate_shadows_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    mesh_store: MeshStore,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

/// A mesh draw as the mesh shaders read it.
#[repr(C)]
pub struct MeshDrawInstance {
    pub transformation: TransformationMatrix,
    pub background: Background,
    pub bounds: Bounds<ScaledPixels>,
    pub clip_bounds: Bounds<ScaledPixels>,
    /// The draw's first chunk (its chunks are consecutive).
    pub first_chunk: u32,
    pub vertex_count: u32,
    /// 1 + the index of the draw's gradient in the batch's gradients, or 0 for none.
    pub gradient: u32,
}

/// A mesh draw's gradient as the mesh shaders read it.
#[repr(C)]
pub struct MeshGradientInstance {
    /// Mesh space → gradient space.
    pub to_gradient: TransformationMatrix,
    /// 0: linear, 1: radial (two-point conical), 2: degenerate (the last stop).
    pub kind: u32,
    /// 0: pad, 1: reflect, 2: repeat.
    pub spread: u32,
    /// The gradient's first stop in the batch's stops.
    pub first_stop: u32,
    pub stop_count: u32,
    /// Where offset 0 is: a point, or the start circle's centre.
    pub start: [f32; 2],
    /// Where offset 1 is: a point, or the end circle's centre.
    pub end: [f32; 2],
    /// The circles' radii (radial gradients).
    pub start_radius: f32,
    pub end_radius: f32,
}

/// A gradient stop as the mesh shaders read it: a straight colour and its offset.
#[repr(C)]
pub struct MeshGradientStopInstance {
    pub color: [f32; 4],
    pub offset: f32,
}

/// One instance of a mesh batch's draw call: a block of a draw's mesh.
#[repr(C)]
pub struct MeshChunk {
    /// Index into the batch's draws.
    pub draw: u32,
    /// The vertex buffer ("page") holding the block.
    pub page: u32,
    /// Where the block starts in its page, in vertices.
    pub first_vertex: u32,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests (captures screenshots without ScreenCaptureKit)
        #[cfg(any(test, feature = "test-support"))]
        layer.set_framebuffer_only(false);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }

        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        }
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        let is_unified_memory = device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let meshes_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "meshes_rasterization",
            "mesh_vertex",
            "mesh_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let multiply_path_sprites_pipeline_state = build_path_sprite_pipeline_state_with_blend(
            &device, &library, "multiply_path_sprites", "path_sprite_vertex",
            "path_sprite_fragment", MTLPixelFormat::BGRA8Unorm, MeshBlendMode::Multiply,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let intermediate_monochrome_sprites_pipeline_state =
            build_intermediate_sprite_pipeline_state(
                &device,
                &library,
                "intermediate_monochrome_sprites",
                "monochrome_sprite_vertex",
                "monochrome_sprite_fragment",
                PATH_SAMPLE_COUNT,
            );
        let intermediate_polychrome_sprites_pipeline_state =
            build_intermediate_sprite_pipeline_state(
                &device,
                &library,
                "intermediate_polychrome_sprites",
                "polychrome_sprite_vertex",
                "polychrome_sprite_fragment",
                PATH_SAMPLE_COUNT,
            );
        // (shadows painted between meshes are drawn with them: see
        // `draw_mesh_run_to_intermediate`)
        let intermediate_shadows_pipeline_state = build_intermediate_sprite_pipeline_state(
            &device,
            &library,
            "intermediate_shadows",
            "shadow_vertex",
            "shadow_fragment",
            PATH_SAMPLE_COUNT,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let mesh_store = MeshStore::new(device.clone(), is_unified_memory);
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            meshes_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            multiply_path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            intermediate_monochrome_sprites_pipeline_state,
            intermediate_polychrome_sprites_pipeline_state,
            intermediate_shadows_pipeline_state,
            surfaces_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            mesh_store,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            let ns_size = NSSize {
                width: size.width.0 as f64,
                height: size.height.0 as f64,
            };
            unsafe {
                let _: () = msg_send![
                    layer.as_ref(),
                    setDrawableSize: ns_size
                ];
            }
        }
        self.update_path_intermediate_textures(size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let command_buffer = match self.render_frame(scene, drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let frame = self
            .mesh_store
            .begin_frame(scene.meshes.iter().map(|draw| &draw.mesh));
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let instance_bindings = write_instances(scene, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} meshes, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.meshes.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
        )?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let completed_frame = self.mesh_store.completion();
        let block = RcBlock::new(move |_: ptr::NonNull<AnyObject>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
            completed_frame.fetch_max(frame, Ordering::Release);
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
        unsafe {
            command_buffer.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
        }

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, drawable.texture(), viewport_size)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(drawable.texture())
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            // Update path intermediate textures for this size
            self.update_path_intermediate_textures(size);

            // Create an offscreen texture as render target
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            texture_descriptor.set_storage_mode(metal::MTLStorageMode::Managed);
            let target_texture = self.device.new_texture(&texture_descriptor);

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // On discrete GPUs (non-unified memory), Managed textures require an
            // explicit blit synchronize before the CPU can read back the rendered
            // data. Without this, get_bytes returns stale zeros.
            if !self.is_unified_memory {
                let blit = command_buffer.new_blit_command_encoder();
                blit.synchronize_resource(&target_texture);
                blit.end_encoding();
            }

            // Commit and wait for completion
            command_buffer.commit();
            command_buffer.wait_until_completed();

            read_texture_to_image(&target_texture)
        })
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            // (made again only for a new size, as a window's are on resize: a benchmark's frames
            // must not allocate two full-window textures each)
            let intermediates_fit = self.path_intermediate_texture.as_ref().is_some_and(|texture| {
                texture.width() == size.width.0 as u64 && texture.height() == size.height.0 as u64
            });
            if !intermediates_fit {
                self.update_path_intermediate_textures(size);
            }

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
            });
            if needs_new_target {
                let texture_descriptor = metal::TextureDescriptor::new();
                texture_descriptor.set_width(size.width.0 as u64);
                texture_descriptor.set_height(size.height.0 as u64);
                texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor.set_usage(
                    metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
                );
                texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
                self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .expect("just ensured the render target exists");

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };

        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            viewport_size,
            Some(metal::MTLClearColor::new(0., 0., 0., alpha)),
        );

        let batches: Vec<PrimitiveBatch> = scene.batches().collect();
        let mut next = 0;
        while let Some(batch) = batches.get(next) {
            next += 1;
            match batch.clone() {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        writer,
                        viewport_size,
                        command_buffer,
                    )?;

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        None,
                    );

                    if did_draw {
                        if let Err(error) = self.draw_paths_from_intermediate(
                            paths,
                            writer,
                            viewport_size,
                            command_encoder,
                        ) {
                            command_encoder.end_encoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Meshes(_) => {
                    // Normal meshes absorb intervening sprites and shadows to save passes.
                    // Other blend modes stop before them, preserving their source-over paint.
                    let start = next - 1;
                    let end = gpui::mesh_run_end(scene, &batches, start);
                    next = end;
                    command_encoder.end_encoding();

                    let drawn_bounds = self.draw_mesh_run_to_intermediate(
                        scene,
                        &batches[start..end],
                        instance_bindings,
                        writer,
                        viewport_size,
                        command_buffer,
                    )?;

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        None,
                    );

                    if let Some(bounds) = drawn_bounds
                        && let Err(error) = self.composite_intermediate(
                            &[PathSprite { bounds }],
                            scene.meshes[match &batches[start] { PrimitiveBatch::Meshes(range) => range.start, _ => unreachable!() }].blend_mode,
                            writer,
                            viewport_size,
                            command_encoder,
                        )
                    {
                        command_encoder.end_encoding();
                        return Err(error);
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    range.start,
                    instance_bindings,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.end_encoding();

        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertex_instance_bindings = writer.write(&vertices)?;

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );

        command_encoder.end_encoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        self.draw_shadows_with(
            &self.shadows_pipeline_state,
            shadows,
            instance_bindings,
            viewport_size,
            command_encoder,
        );
    }

    /// Draws a range of shadows with `pipeline_state` (the target's, or the multisampled path
    /// texture's for shadows inside a mesh run).
    fn draw_shadows_with(
        &self,
        pipeline_state: &metal::RenderPipelineState,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
            quads.start as u64,
        );
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        let sprite_instance_bindings = writer.write(&sprites)?;
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    /// Rasterizes a run of batches that starts and ends with meshes (and may hold sprites
    /// painted between them) into the path intermediate texture, multisampled like paths, in
    /// order: each batch of meshes is one instanced draw call, an instance per mesh block.
    /// Returns the bounds to composite, if anything was drawn.
    fn draw_mesh_run_to_intermediate(
        &self,
        scene: &Scene,
        run: &[PrimitiveBatch],
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<Option<Bounds<ScaledPixels>>> {
        let pages: Vec<metal::Buffer> = self.mesh_store.pages().cloned().collect();
        if pages.is_empty() {
            return Ok(None);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .context("no color attachment")?;
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));
        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }
        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_viewport(metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: i32::from(viewport_size.width) as f64,
            height: i32::from(viewport_size.height) as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        let mut union: Option<Bounds<ScaledPixels>> = None;
        let mut add = |bounds: Bounds<ScaledPixels>| {
            union = Some(match union {
                Some(union) => union.union(&bounds),
                None => bounds,
            });
        };
        let result = (|| -> Result<()> {
            for batch in run {
                match batch {
                    PrimitiveBatch::Meshes(range) => {
                        if let Some(bounds) = self.encode_meshes(
                            &scene.meshes[range.clone()],
                            &pages,
                            writer,
                            viewport_size,
                            command_encoder,
                        )? {
                            add(bounds);
                        }
                    }
                    PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                        for sprite in &scene.monochrome_sprites[range.clone()] {
                            add(sprite_bounds(
                                sprite.bounds,
                                sprite.transformation,
                                &sprite.content_mask,
                            ));
                        }
                        self.draw_sprites_with(
                            &self.intermediate_monochrome_sprites_pipeline_state,
                            &instance_bindings.monochrome_sprites,
                            *texture_id,
                            range.clone(),
                            viewport_size,
                            command_encoder,
                        );
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                        for sprite in &scene.polychrome_sprites[range.clone()] {
                            add(sprite_bounds(
                                sprite.bounds,
                                sprite.transformation,
                                &sprite.content_mask,
                            ));
                        }
                        self.draw_sprites_with(
                            &self.intermediate_polychrome_sprites_pipeline_state,
                            &instance_bindings.polychrome_sprites,
                            *texture_id,
                            range.clone(),
                            viewport_size,
                            command_encoder,
                        );
                    }
                    PrimitiveBatch::Shadows(range) => {
                        for shadow in &scene.shadows[range.clone()] {
                            add(shadow_bounds(shadow));
                        }
                        self.draw_shadows_with(
                            &self.intermediate_shadows_pipeline_state,
                            range.clone(),
                            instance_bindings,
                            viewport_size,
                            command_encoder,
                        );
                    }
                    _ => {}
                }
            }
            Ok(())
        })();
        command_encoder.end_encoding();
        result?;
        Ok(union.filter(|bounds| !bounds.is_empty()))
    }

    /// Encodes a batch of meshes as one instanced draw call on `command_encoder` (a pass into
    /// the path intermediate texture). Returns the bounds it may paint.
    fn encode_meshes(
        &self,
        meshes: &[MeshDraw],
        pages: &[metal::Buffer],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<Option<Bounds<ScaledPixels>>> {
        let mut draws = Vec::with_capacity(meshes.len());
        let mut chunks = Vec::new();
        let mut gradients: Vec<MeshGradientInstance> = Vec::new();
        let mut stops: Vec<MeshGradientStopInstance> = Vec::new();
        let mut union: Option<Bounds<ScaledPixels>> = None;
        for draw in meshes {
            let clip_bounds = draw.clipped_bounds();
            if clip_bounds.is_empty() {
                continue;
            }
            let Some(blocks) = self.mesh_store.blocks(draw.mesh.id()) else {
                continue;
            };
            let index = draws.len() as u32;
            let gradient = match &draw.gradient {
                Some(gradient) => {
                    gradients.push(mesh_gradient_instance(gradient, &mut stops));
                    gradients.len() as u32
                }
                None => 0,
            };
            draws.push(MeshDrawInstance {
                transformation: draw.transformation,
                background: draw.background,
                // (a gradient spans the clipped bounds, as for paths on every
                // backend and for meshes drawn through the path pipelines)
                bounds: clip_bounds,
                clip_bounds,
                first_chunk: chunks.len() as u32,
                vertex_count: draw.mesh.len() as u32,
                gradient,
            });
            chunks.extend(blocks.iter().map(|block| MeshChunk {
                draw: index,
                page: block.page,
                first_vertex: block.first_vertex,
            }));
            union = Some(match union {
                Some(bounds) => bounds.union(&clip_bounds),
                None => clip_bounds,
            });
        }
        if chunks.is_empty() {
            return Ok(None);
        }
        let draw_bindings = writer.write(&draws)?;
        let chunk_bindings = writer.write(&chunks)?;
        // (a batch without gradients binds the draws in their place: the shaders read them
        // only for draws that have one)
        let (gradient_bindings, stop_bindings) = if gradients.is_empty() {
            (None, None)
        } else {
            (Some(writer.write(&gradients)?), Some(writer.write(&stops)?))
        };

        command_encoder.set_render_pipeline_state(&self.meshes_rasterization_pipeline_state);
        // (slots past the last page hold the first: the shader reads only real pages)
        let slots = [
            MeshInputIndex::Page0,
            MeshInputIndex::Page1,
            MeshInputIndex::Page2,
            MeshInputIndex::Page3,
            MeshInputIndex::Page4,
            MeshInputIndex::Page5,
            MeshInputIndex::Page6,
            MeshInputIndex::Page7,
        ];
        for (index, slot) in slots.into_iter().enumerate().take(MAX_PAGES) {
            let page = pages.get(index).unwrap_or(&pages[0]);
            command_encoder.set_vertex_buffer(slot as u64, Some(page), 0);
        }
        command_encoder.set_vertex_buffer(
            MeshInputIndex::Draws as u64,
            Some(&draw_bindings.buffer),
            draw_bindings.offset as u64,
        );
        command_encoder.set_vertex_buffer(
            MeshInputIndex::Chunks as u64,
            Some(&chunk_bindings.buffer),
            chunk_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            MeshInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        let block_vertices = MESH_BLOCK_VERTICES;
        command_encoder.set_vertex_bytes(
            MeshInputIndex::BlockVertices as u64,
            mem::size_of_val(&block_vertices) as u64,
            &block_vertices as *const u32 as *const _,
        );
        command_encoder.set_fragment_buffer(
            MeshInputIndex::Draws as u64,
            Some(&draw_bindings.buffer),
            draw_bindings.offset as u64,
        );
        for (index, binding) in [
            (MeshInputIndex::Gradients as u64, &gradient_bindings),
            (MeshInputIndex::GradientStops as u64, &stop_bindings),
        ] {
            let binding = binding.as_ref().unwrap_or(&draw_bindings);
            command_encoder.set_vertex_buffer(index, Some(&binding.buffer), binding.offset as u64);
            command_encoder.set_fragment_buffer(
                index,
                Some(&binding.buffer),
                binding.offset as u64,
            );
        }
        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            u64::from(MESH_BLOCK_VERTICES),
            chunks.len() as u64,
        );
        Ok(union)
    }

    /// Composites the path intermediate texture within `sprites` onto the target.
    fn composite_intermediate(
        &self,
        sprites: &[PathSprite],
        blend_mode: MeshBlendMode,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(match blend_mode {
            MeshBlendMode::Normal => &self.path_sprites_pipeline_state,
            MeshBlendMode::Multiply => &self.multiply_path_sprites_pipeline_state,
        });
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );
        let sprite_instance_bindings = writer.write(sprites)?;
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );
        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        self.draw_sprites_with(
            &self.monochrome_sprites_pipeline_state,
            &instance_bindings.monochrome_sprites,
            texture_id,
            sprites,
            viewport_size,
            command_encoder,
        );
    }

    /// Draws a range of sprites (monochrome or polychrome, as `pipeline_state` and `binding`
    /// say) from one atlas texture.
    fn draw_sprites_with(
        &self,
        pipeline_state: &metal::RenderPipelineState,
        binding: &InstanceBinding,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&binding.buffer),
            binding.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&binding.buffer),
            binding.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        self.draw_sprites_with(
            &self.polychrome_sprites_pipeline_state,
            &instance_bindings.polychrome_sprites,
            texture_id,
            sprites,
            viewport_size,
            command_encoder,
        );
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if surfaces.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    surface.image_buffer.get_width_of_plane(0),
                    surface.image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    surface.image_buffer.get_width_of_plane(1),
                    surface.image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            // let y_texture = y_texture.get_texture().unwrap().
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// A sprite pipeline that rasterizes into the multisampled path texture, which holds
/// premultiplied colours (the sprite shaders output straight ones).
fn build_intermediate_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// The area a shadow paints: its quad (the element for an inset shadow, else the shadow rect
/// with the gaussian's tail), transformed, within its clip.
fn shadow_bounds(shadow: &Shadow) -> Bounds<ScaledPixels> {
    let painted = if shadow.transformation == TransformationMatrix::unit() {
        shadow.painted_geometry()
    } else {
        shadow
            .transformation
            .transform_bounds(shadow.painted_geometry())
            .dilate(ScaledPixels(1.))
    };
    painted.intersect(&shadow.content_mask.bounds)
}

/// The area a sprite paints (its transformed bounds within its clip).
fn sprite_bounds(
    bounds: Bounds<ScaledPixels>,
    transformation: TransformationMatrix,
    content_mask: &ContentMask<ScaledPixels>,
) -> Bounds<ScaledPixels> {
    let painted = if transformation == TransformationMatrix::unit() {
        bounds
    } else {
        transformation
            .transform_bounds(bounds)
            .dilate(ScaledPixels(1.))
    };
    painted.intersect(&content_mask.bounds)
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    build_path_sprite_pipeline_state_with_blend(
        device, library, label, vertex_fn_name, fragment_fn_name, pixel_format,
        MeshBlendMode::Normal,
    )
}

fn build_path_sprite_pipeline_state_with_blend(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    blend_mode: MeshBlendMode,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(match blend_mode {
        MeshBlendMode::Normal => metal::MTLBlendFactor::One,
        MeshBlendMode::Multiply => metal::MTLBlendFactor::DestinationColor,
    });
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: metal::Buffer,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    surfaces: InstanceBinding,
}

fn write_instances(scene: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBindings> {
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
        }))?,
    })
}

struct InstanceBufferWriter {
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory);
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
            if offset > 0 {
                current.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: offset as NSUInteger,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[repr(C)]
enum MeshInputIndex {
    Draws = 0,
    Chunks = 1,
    ViewportSize = 2,
    BlockVertices = 3,
    /// The vertex pages (`MAX_PAGES` slots from here).
    Page0 = 4,
    Page1 = 5,
    Page2 = 6,
    Page3 = 7,
    Page4 = 8,
    Page5 = 9,
    Page6 = 10,
    Page7 = 11,
    /// The batch's gradients (`MeshGradientInstance`).
    Gradients = 12,
    /// The batch's gradient stops (`MeshGradientStopInstance`).
    GradientStops = 13,
}

/// A mesh gradient as the mesh shaders read it, its stops appended to `stops`.
fn mesh_gradient_instance(
    gradient: &MeshGradient,
    stops: &mut Vec<MeshGradientStopInstance>,
) -> MeshGradientInstance {
    let first_stop = stops.len() as u32;
    let mut previous = 0f32;
    for (index, stop) in gradient.stops.iter().enumerate() {
        // (offsets out of order are raised to the one before)
        let offset = if index == 0 {
            stop.offset
        } else {
            stop.offset.max(previous)
        };
        previous = offset;
        let c = stop.color;
        stops.push(MeshGradientStopInstance {
            color: [c.r, c.g, c.b, c.a],
            offset,
        });
    }
    let (kind, start, end, start_radius, end_radius) =
        if gradient.stops.len() == 1 || gradient.is_degenerate() {
            (2, [0.; 2], [0.; 2], 0., 0.)
        } else {
            match gradient.shape {
                MeshGradientShape::Linear { start, end } => {
                    (0, [start.x, start.y], [end.x, end.y], 0., 0.)
                }
                MeshGradientShape::Radial {
                    start,
                    start_radius,
                    end,
                    end_radius,
                } => (
                    1,
                    [start.x, start.y],
                    [end.x, end.y],
                    start_radius,
                    end_radius,
                ),
            }
        };
    MeshGradientInstance {
        to_gradient: gradient.to_gradient,
        kind,
        spread: match gradient.spread {
            MeshGradientSpread::Pad => 0,
            MeshGradientSpread::Reflect => 1,
            MeshGradientSpread::Repeat => 2,
        },
        first_stop,
        stop_count: gradient.stops.len() as u32,
        start,
        end,
        start_radius,
        end_radius,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod mesh_tests {
    use super::*;
    use gpui::{
        ContentMask, Hsla, Mesh, MeshGradientStop, MeshVertex, Pixels, Rgba, TransformationMatrix,
        hatch_pattern, px,
    };

    const SIZE: i32 = 160;

    fn renderer() -> Option<MetalRenderer> {
        metal::Device::system_default()?;
        Some(MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        ))))
    }

    fn render(renderer: &mut MetalRenderer, scene: &mut Scene) -> image::RgbaImage {
        scene.finish();
        renderer
            .render_scene_to_image(scene, size(DevicePixels(SIZE), DevicePixels(SIZE)))
            .expect("rendering should succeed")
    }

    fn rgba(r: f32, g: f32, b: f32, a: f32) -> Rgba {
        Rgba { r, g, b, a }
    }

    fn whole_viewport() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: Bounds {
                origin: point(ScaledPixels(0.), ScaledPixels(0.)),
                size: size(ScaledPixels(SIZE as f32), ScaledPixels(SIZE as f32)),
            },
        }
    }

    /// Triangles (mesh space) and their colours: a star over a translucent square, both
    /// crossing each other.
    fn shapes() -> Vec<([Point<f32>; 3], Rgba)> {
        let mut triangles = Vec::new();
        let square = rgba(0.2, 0.4, 0.9, 0.6);
        triangles.push((
            [point(-30., -30.), point(30., -30.), point(30., 30.)],
            square,
        ));
        triangles.push((
            [point(-30., -30.), point(30., 30.), point(-30., 30.)],
            square,
        ));
        let star = rgba(0.9, 0.3, 0.1, 0.8);
        for i in 0..5 {
            let angle = |k: f32| (k * 0.4 - 0.5) * std::f32::consts::PI;
            let tip = point(
                45. * angle(i as f32 * 2.).cos(),
                45. * angle(i as f32 * 2.).sin(),
            );
            let left = point(
                18. * angle(i as f32 * 2. - 1.).cos(),
                18. * angle(i as f32 * 2. - 1.).sin(),
            );
            let right = point(
                18. * angle(i as f32 * 2. + 1.).cos(),
                18. * angle(i as f32 * 2. + 1.).sin(),
            );
            triangles.push(([left, tip, right], star));
            triangles.push(([left, right, point(0., 0.)], star));
        }
        triangles
    }

    fn mesh() -> Mesh {
        Mesh::new(
            shapes()
                .into_iter()
                .flat_map(|(points, color)| points.map(|point| MeshVertex::new(point, color))),
        )
    }

    /// Mesh space → scaled pixels: turned, scaled and moved.
    fn transformation() -> TransformationMatrix {
        let (sin, cos) = 0.3f32.sin_cos();
        TransformationMatrix {
            rotation_scale: [[1.3 * cos, -1.3 * sin], [1.3 * sin, 1.3 * cos]],
            translation: [80.5, 77.25],
        }
    }

    /// The same triangles as paths, transformed on the CPU: one path per colour run, in order.
    fn insert_paths(scene: &mut Scene, paint: impl Fn(Rgba) -> Background) {
        let t = transformation();
        let mut current: Option<(Rgba, Path<Pixels>)> = None;
        let flush = |scene: &mut Scene, entry: Option<(Rgba, Path<Pixels>)>| {
            if let Some((color, mut path)) = entry {
                path.content_mask = ContentMask {
                    bounds: Bounds {
                        origin: point(px(0.), px(0.)),
                        size: size(px(SIZE as f32), px(SIZE as f32)),
                    },
                };
                path.color = paint(color);
                scene.insert_primitive(path.scale(1.));
            }
        };
        for (points, color) in shapes() {
            if current.as_ref().is_some_and(|(c, _)| *c != color) {
                flush(scene, current.take());
            }
            let entry = current.get_or_insert_with(|| (color, Path::new(point(px(0.), px(0.)))));
            let [a, b, c] = points.map(|p| {
                let q = t.apply(point(px(p.x), px(p.y)));
                point(q.x, q.y)
            });
            entry
                .1
                .push_triangle((a, b, c), (point(0., 1.), point(0., 1.), point(0., 1.)));
        }
        flush(scene, current.take());
    }

    fn insert_mesh(scene: &mut Scene, mesh: &Mesh, background: Background) {
        let t = transformation();
        let b = mesh.bounds();
        let bounds = t
            .transform_bounds(Bounds {
                origin: b.origin.map(ScaledPixels),
                size: b.size.map(ScaledPixels),
            })
            .dilate(ScaledPixels(1.));
        scene.insert_primitive(MeshDraw {
            blend_mode: gpui::MeshBlendMode::Normal,
            order: 0,
            bounds,
            content_mask: whole_viewport(),
            transformation: t,
            background,
            gradient: None,
            mesh: mesh.clone(),
        });
    }

    fn max_difference(a: &image::RgbaImage, b: &image::RgbaImage) -> (u8, usize) {
        let mut max = 0u8;
        let mut differing = 0;
        for (p, q) in a.pixels().zip(b.pixels()) {
            let d =
                p.0.iter()
                    .zip(q.0.iter())
                    .map(|(x, y)| x.abs_diff(*y))
                    .max()
                    .unwrap_or(0);
            max = max.max(d);
            differing += usize::from(d > 0);
        }
        (max, differing)
    }

    #[test]
    fn test_meshes_render_like_paths() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let mut paths = Scene::default();
        insert_paths(&mut paths, |color| color.into());
        let expected = render(&mut renderer, &mut paths);

        let white: Background = Hsla::white().into();
        let mut meshes = Scene::default();
        insert_mesh(&mut meshes, &mesh(), white);
        let actual = render(&mut renderer, &mut meshes);
        let (max, differing) = max_difference(&expected, &actual);
        // The GPU transforms the vertices: the last bits of a position may differ from the
        // CPU's, which may move a multisample in or out of an edge now and then.
        assert!(
            max <= 1 || differing <= 3,
            "{differing} pixels differ, by up to {max}"
        );
        assert!(actual.pixels().any(|p| p.0[3] > 0), "the mesh drew nothing");
    }

    #[test]
    fn test_mesh_paint_multiplies_vertex_colors() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        // A half transparent white paint fades the vertex colours.
        let mut paths = Scene::default();
        insert_paths(&mut paths, |color| {
            rgba(color.r, color.g, color.b, color.a * 0.5).into()
        });
        let expected = render(&mut renderer, &mut paths);
        let mut meshes = Scene::default();
        insert_mesh(&mut meshes, &mesh(), rgba(1., 1., 1., 0.5).into());
        let actual = render(&mut renderer, &mut meshes);
        let (max, differing) = max_difference(&expected, &actual);
        assert!(
            max <= 1 || differing <= 3,
            "{differing} pixels differ, by up to {max}"
        );

        // A pattern is drawn in white vertices as it is.
        let to_pattern = TransformationMatrix {
            rotation_scale: [[1. / 16., 0.], [0., 1. / 16.]],
            translation: [0., 0.],
        };
        let hatch = hatch_pattern(rgba(0.9, 0.9, 1., 1.), rgba(0.1, 0.2, 0.8, 1.), to_pattern);
        let white_mesh = Mesh::new(shapes().into_iter().flat_map(|(points, _)| {
            points.map(|point| MeshVertex::new(point, rgba(1., 1., 1., 1.)))
        }));
        let mut paths = Scene::default();
        insert_paths(&mut paths, |_| hatch);
        let expected = render(&mut renderer, &mut paths);
        let mut meshes = Scene::default();
        insert_mesh(&mut meshes, &white_mesh, hatch);
        let actual = render(&mut renderer, &mut meshes);
        let (max, differing) = max_difference(&expected, &actual);
        assert!(
            max <= 1 || differing <= 3,
            "{differing} pixels differ, by up to {max}"
        );
    }

    #[test]
    fn test_meshes_are_clipped_and_drawn_in_order() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let square = |color: Rgba| {
            Mesh::new(
                [
                    point(0., 0.),
                    point(40., 0.),
                    point(40., 40.),
                    point(0., 0.),
                    point(40., 40.),
                    point(0., 40.),
                ]
                .map(|p| MeshVertex::new(p, color)),
            )
        };
        let red = square(rgba(1., 0., 0., 1.));
        let blue = square(rgba(0., 0., 1., 0.5));
        let at = |x: f32, y: f32| TransformationMatrix {
            rotation_scale: [[1., 0.], [0., 1.]],
            translation: [x, y],
        };
        let draw = |mesh: &Mesh, transformation: TransformationMatrix, clip: f32| MeshDraw {
            blend_mode: gpui::MeshBlendMode::Normal,
            order: 0,
            bounds: transformation.transform_bounds(Bounds {
                origin: point(ScaledPixels(0.), ScaledPixels(0.)),
                size: size(ScaledPixels(40.), ScaledPixels(40.)),
            }),
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: point(ScaledPixels(0.), ScaledPixels(0.)),
                    size: size(ScaledPixels(clip), ScaledPixels(SIZE as f32)),
                },
            },
            transformation,
            background: Hsla::white().into(),
            gradient: None,
            mesh: mesh.clone(),
        };
        let mut scene = Scene::default();
        // One layer: every draw gets the same order, so they are one batch.
        scene.push_layer(whole_viewport().bounds);
        scene.insert_primitive(draw(&red, at(10., 10.), SIZE as f32));
        // translucent blue over the red, clipped at x = 40
        scene.insert_primitive(draw(&blue, at(30., 30.), 40.));
        // red again, drawn twice from the same mesh
        scene.insert_primitive(draw(&red, at(100., 100.), SIZE as f32));
        scene.pop_layer();
        let image = render(&mut renderer, &mut scene);
        let pixel = |x: u32, y: u32| image.get_pixel(x, y).0;
        assert_eq!(pixel(15, 15), [255, 0, 0, 255], "red");
        // the blue over the red: half blue, half red
        let over = pixel(35, 35);
        assert!(
            over[0].abs_diff(128) <= 1 && over[2].abs_diff(127) <= 1,
            "{over:?}"
        );
        // clipped at x = 40
        assert_eq!(pixel(45, 45)[2], 0, "clipped");
        assert_eq!(pixel(120, 120), [255, 0, 0, 255], "the second red");
        // (the headless target is opaque black)
        assert_eq!(pixel(80, 80), [0, 0, 0, 255], "nothing between");
    }

    /// Gradients evaluated on the GPU agree with `MeshGradient::color_at` pixel for pixel
    /// (away from the mesh's edges and the gradients' hard edges): linear and radial,
    /// padded, repeated and reflected, a focal point outside its circle (a cone), through a
    /// skewed gradient space and a turned mesh, faded by the paint.
    #[test]
    fn test_mesh_gradients_match_their_cpu_evaluation() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let square = Mesh::new(
            [
                point(0., 0.),
                point(100., 0.),
                point(100., 100.),
                point(0., 0.),
                point(100., 100.),
                point(0., 100.),
            ]
            .map(|p| MeshVertex::new(p, rgba(1., 1., 1., 1.))),
        );
        let (sin, cos) = 0.2f32.sin_cos();
        let t = TransformationMatrix {
            rotation_scale: [[1.2 * cos, -1.2 * sin], [1.2 * sin, 1.2 * cos]],
            translation: [30., 8.],
        };
        let stop = |offset: f32, color: Rgba| MeshGradientStop { offset, color };
        let skew = TransformationMatrix {
            rotation_scale: [[1., 0.3], [0., 1.]],
            translation: [-5., 3.],
        };
        let gradients = [
            MeshGradient {
                shape: MeshGradientShape::Linear {
                    start: point(10., 10.),
                    end: point(50., 30.),
                },
                stops: vec![
                    stop(0., rgba(1., 0., 0., 1.)),
                    stop(0.4, rgba(0., 1., 0.5, 0.3)),
                    stop(1., rgba(0., 0., 1., 1.)),
                ],
                spread: MeshGradientSpread::Reflect,
                to_gradient: skew,
            },
            MeshGradient {
                shape: MeshGradientShape::Radial {
                    start: point(40., 35.),
                    start_radius: 0.,
                    end: point(50., 50.),
                    end_radius: 30.,
                },
                stops: vec![
                    stop(0.1, rgba(1., 1., 0.8, 1.)),
                    stop(0.6, rgba(0.9, 0.4, 0.1, 1.)),
                    stop(1., rgba(0.2, 0.1, 0.3, 0.8)),
                ],
                spread: MeshGradientSpread::Repeat,
                to_gradient: TransformationMatrix::unit(),
            },
            MeshGradient {
                shape: MeshGradientShape::Radial {
                    start: point(5., 50.),
                    start_radius: 0.,
                    end: point(60., 50.),
                    end_radius: 20.,
                },
                stops: vec![
                    stop(0., rgba(0., 0.5, 1., 1.)),
                    stop(1., rgba(1., 1., 1., 1.)),
                ],
                spread: MeshGradientSpread::Pad,
                to_gradient: TransformationMatrix::unit(),
            },
        ];
        let det = t.rotation_scale[0][0] * t.rotation_scale[1][1]
            - t.rotation_scale[0][1] * t.rotation_scale[1][0];
        for (index, gradient) in gradients.into_iter().enumerate() {
            let gradient = Arc::new(gradient);
            let mut scene = Scene::default();
            let b = square.bounds();
            scene.insert_primitive(MeshDraw {
            blend_mode: gpui::MeshBlendMode::Normal,
                order: 0,
                bounds: t
                    .transform_bounds(Bounds {
                        origin: b.origin.map(ScaledPixels),
                        size: b.size.map(ScaledPixels),
                    })
                    .dilate(ScaledPixels(1.)),
                content_mask: whole_viewport(),
                transformation: t,
                background: rgba(1., 1., 1., 0.8).into(),
                gradient: Some(gradient.clone()),
                mesh: square.clone(),
            });
            let image = render(&mut renderer, &mut scene);
            let (mut compared, mut differing, mut transparent) = (0, 0, 0);
            for (x, y, pixel) in image.enumerate_pixels() {
                // the pixel centre in mesh space
                let (px, py) = (
                    x as f32 + 0.5 - t.translation[0],
                    y as f32 + 0.5 - t.translation[1],
                );
                let m = t.rotation_scale;
                let mx = (m[1][1] * px - m[0][1] * py) / det;
                let my = (-m[1][0] * px + m[0][0] * py) / det;
                if !(2. ..=98.).contains(&mx) || !(2. ..=98.).contains(&my) {
                    continue;
                }
                let expected = gradient.color_at(point(mx, my)).unwrap_or_default();
                let a = expected.a * 0.8;
                let expected = [expected.r * a, expected.g * a, expected.b * a]
                    .map(|v| (v * 255.).round() as i32);
                compared += 1;
                transparent += usize::from(a == 0.);
                let d = (0..3)
                    .map(|c| (i32::from(pixel.0[c]) - expected[c]).abs())
                    .max()
                    .unwrap_or(0);
                differing += usize::from(d > 2);
            }
            assert!(compared > 10_000, "gradient {index}: {compared} pixels compared");
            // (a pixel on a hard edge, the reflection's turns or the repeat's seams may
            // land on either side)
            assert!(
                differing * 200 < compared,
                "gradient {index}: {differing} of {compared} pixels differ by more than 2"
            );
            if index == 2 {
                assert!(
                    transparent > 1_000,
                    "outside the cone nothing is painted: {transparent} pixels"
                );
            }
        }
    }

    /// `MeshGradient::color_at`, the CPU evaluation the shaders follow: straight
    /// interpolation, spreads, hard edges, degenerate gradients, focal points and cones.
    #[test]
    fn test_mesh_gradient_colors() {
        let red = Rgba {
            r: 1.,
            g: 0.,
            b: 0.,
            a: 1.,
        };
        let clear_blue = Rgba {
            r: 0.,
            g: 0.,
            b: 1.,
            a: 0.,
        };
        let linear = |spread| MeshGradient {
            shape: MeshGradientShape::Linear {
                start: point(0., 0.),
                end: point(10., 0.),
            },
            stops: vec![
                MeshGradientStop {
                    offset: 0.,
                    color: red,
                },
                MeshGradientStop {
                    offset: 1.,
                    color: clear_blue,
                },
            ],
            spread,
            to_gradient: TransformationMatrix::unit(),
        };
        // straight interpolation: halfway is half red, half blue, half transparent
        let half = linear(MeshGradientSpread::Pad).color_at(point(5., 3.)).unwrap();
        assert!((half.r - 0.5).abs() < 1e-6 && (half.b - 0.5).abs() < 1e-6);
        assert!((half.a - 0.5).abs() < 1e-6);
        let at = |spread, x: f32| linear(spread).color_at(point(x, 0.)).unwrap().r;
        assert_eq!(at(MeshGradientSpread::Pad, -4.), 1.);
        assert_eq!(at(MeshGradientSpread::Pad, 14.), 0.);
        assert!((at(MeshGradientSpread::Repeat, 12.5) - 0.75).abs() < 1e-5);
        assert!((at(MeshGradientSpread::Reflect, 12.5) - 0.25).abs() < 1e-5);
        assert!((at(MeshGradientSpread::Reflect, -2.5) - 0.75).abs() < 1e-5);

        // a hard edge where two stops share an offset
        let mut hard = linear(MeshGradientSpread::Pad);
        hard.stops.insert(
            1,
            MeshGradientStop {
                offset: 0.5,
                color: red,
            },
        );
        hard.stops.insert(
            2,
            MeshGradientStop {
                offset: 0.5,
                color: clear_blue,
            },
        );
        assert_eq!(hard.color_at(point(4.9, 0.)).unwrap().r, 1.);
        assert_eq!(hard.color_at(point(5.1, 0.)).unwrap().b, 1.);
        // (on the edge itself, the later colour, as Skia and resvg draw it)
        assert_eq!(hard.color_at(point(5., 0.)).unwrap().b, 1.);

        // degenerate: the last stop; no stops: nothing
        let mut flat = linear(MeshGradientSpread::Pad);
        flat.shape = MeshGradientShape::Linear {
            start: point(3., 3.),
            end: point(3., 3.),
        };
        assert_eq!(flat.color_at(point(0., 0.)), Some(clear_blue));
        flat.stops.clear();
        assert_eq!(flat.color_at(point(0., 0.)), None);

        // radial: offset 0 at the focal point, 1 on the circle; outside a cone, nothing
        let radial = |focal: Point<f32>| MeshGradient {
            shape: MeshGradientShape::Radial {
                start: focal,
                start_radius: 0.,
                end: point(0., 0.),
                end_radius: 10.,
            },
            ..linear(MeshGradientSpread::Pad)
        };
        let centred = radial(point(0., 0.));
        assert!((centred.offset_at(point(3., 4.)).unwrap() - 0.5).abs() < 1e-6);
        let focal = radial(point(5., 0.));
        assert!(focal.offset_at(point(5., 0.)).unwrap().abs() < 1e-6);
        assert!((focal.offset_at(point(-10., 0.)).unwrap() - 1.).abs() < 1e-5);
        assert!((focal.offset_at(point(0., 10.)).unwrap() - 1.).abs() < 1e-5);
        let outside = radial(point(20., 0.));
        assert!(outside.offset_at(point(20., 10.)).is_none());
        assert!(outside.offset_at(point(5., 0.)).is_some());
    }
}
