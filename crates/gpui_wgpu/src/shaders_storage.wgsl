// Storage-buffer instance transport. Concatenated after `shaders.wgsl` on backends
// with storage buffer support; `shaders_webgl.wgsl` is the WebGL2 counterpart that
// decodes the same records from a texture instead.
//
// All buffers share `@group(1) @binding(0)` because each pipeline binds only the
// buffer its own entry points read.

@group(1) @binding(0) var<storage, read> b_quads: array<Quad>;
@group(1) @binding(0) var<storage, read> b_shadows: array<Shadow>;
@group(1) @binding(0) var<storage, read> b_path_vertices: array<PathRasterizationVertex>;
@group(1) @binding(0) var<storage, read> b_path_sprites: array<PathSprite>;
@group(1) @binding(0) var<storage, read> b_underlines: array<Underline>;
@group(1) @binding(0) var<storage, read> b_mono_sprites: array<MonochromeSprite>;
@group(1) @binding(0) var<storage, read> b_poly_sprites: array<PolychromeSprite>;

fn load_quad(instance_id: u32) -> Quad {
    return b_quads[instance_id];
}

fn load_shadow(instance_id: u32) -> Shadow {
    return b_shadows[instance_id];
}

fn load_path_vertex(vertex_id: u32) -> PathRasterizationVertex {
    return b_path_vertices[vertex_id];
}

fn load_path_sprite(instance_id: u32) -> PathSprite {
    return b_path_sprites[instance_id];
}

fn load_underline(instance_id: u32) -> Underline {
    return b_underlines[instance_id];
}

fn load_mono_sprite(instance_id: u32) -> MonochromeSprite {
    return b_mono_sprites[instance_id];
}

fn load_poly_sprite(instance_id: u32) -> PolychromeSprite {
    return b_poly_sprites[instance_id];
}

// --- meshes --- //
//
// Retained meshes (`Window::paint_mesh`) for a renderer that keeps their vertices in a GPU
// buffer, as the Metal renderer does: a batch is one instanced draw with an instance per block
// of a draw's mesh (`MeshChunk`), `MESH_BLOCK_VERTICES` vertices each. (This renderer does not
// use these entry points yet: it rasterizes meshes through the path pipeline.)

const MESH_BLOCK_VERTICES: u32 = 48u;

struct MeshVertex {
    position: vec2<f32>,
    // red and green, blue and alpha, 16 bits each
    rg: u32,
    ba: u32,
}

struct MeshDrawInstance {
    transformation: TransformationMatrix,
    background: Background,
    // what a gradient spans: the clipped bounds, as for paths
    bounds: Bounds,
    clip_bounds: Bounds,
    first_chunk: u32,
    vertex_count: u32,
}

struct MeshChunk {
    draw: u32,
    // (one vertex buffer here: blocks are addressed by their first vertex in it)
    page: u32,
    first_vertex: u32,
}

@group(1) @binding(0) var<storage, read> b_mesh_draws: array<MeshDrawInstance>;
@group(1) @binding(1) var<storage, read> b_mesh_chunks: array<MeshChunk>;
@group(1) @binding(2) var<storage, read> b_mesh_vertices: array<MeshVertex>;

struct MeshVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) paint_solid: vec4<f32>,
    @location(2) @interpolate(flat) paint_color0: vec4<f32>,
    @location(3) @interpolate(flat) paint_color1: vec4<f32>,
    @location(4) @interpolate(flat) draw_id: u32,
    //TODO: use `clip_distance` once Naga supports it
    @location(5) clip_distances: vec4<f32>,
}

@vertex
fn vs_mesh(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) chunk_id: u32) -> MeshVarying {
    let chunk = b_mesh_chunks[chunk_id];
    let draw = b_mesh_draws[chunk.draw];
    var out = MeshVarying();
    out.draw_id = chunk.draw;
    let index = (chunk_id - draw.first_chunk) * MESH_BLOCK_VERTICES + vertex_id;
    if (index >= draw.vertex_count) {
        // past the end of the draw's last block: a degenerate triangle
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.clip_distances = vec4<f32>(-1.0);
        return out;
    }
    let v = b_mesh_vertices[chunk.first_vertex + vertex_id];
    let position = transpose(draw.transformation.rotation_scale) * v.position
        + draw.transformation.translation;
    out.position = to_device_position_impl(position);
    out.color = vec4<f32>(unpack2x16unorm(v.rg), unpack2x16unorm(v.ba));
    let paint = prepare_gradient_color(
        draw.background.tag,
        draw.background.color_space,
        draw.background.solid,
        draw.background.colors,
    );
    out.paint_solid = paint.solid;
    out.paint_color0 = paint.color0;
    out.paint_color1 = paint.color1;
    out.clip_distances = distance_from_clip_rect_impl(position, draw.clip_bounds);
    return out;
}

@fragment
fn fs_mesh(input: MeshVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let draw = b_mesh_draws[input.draw_id];
    let paint = gradient_color(draw.background, input.position.xy, draw.bounds,
        input.paint_solid, input.paint_color0, input.paint_color1);
    let color = input.color * paint;
    return vec4<f32>(color.rgb * color.a, color.a);
}
