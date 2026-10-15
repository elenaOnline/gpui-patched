//! Retained triangle meshes: geometry that is uploaded to the GPU once and then drawn in any
//! number of frames with a transformation and a paint (see [`crate::Window::paint_mesh`]).
//!
//! A [`Path`](crate::Path) is rebuilt and uploaded every frame it is painted, so its cost is
//! proportional to its vertices, every frame. A [`Mesh`] is uploaded the first time it is
//! painted and stays on the GPU while it is in use; painting it again only records its
//! transformation, paint and clip, whatever its size. That makes it the primitive for scenes of
//! many shapes that move, pan and zoom as a whole: canvases, diagrams, maps.
//!
//! Meshes are rasterized like paths (into the multisampled path texture, then composited), so
//! both are anti-aliased the same way.

use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
};

use crate::{Bounds, Point, Rgba, TransformationMatrix, point, size};

/// Identifies a [`Mesh`]'s geometry, which renderers keep on the GPU while the mesh is in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshId(pub u64);

/// How a mesh draw composites with the destination after rasterization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshBlendMode {
    #[default]
    /// Ordinary source-over compositing.
    Normal,
    /// Premultiplied source times destination, plus destination where source is absent.
    Multiply,
}

/// A vertex of a [`Mesh`], as renderers store it: a position in the mesh's own space and a
/// straight (not premultiplied) colour with 16 bits per channel, red and green in the low and
/// high halves of `rg`, blue and alpha in `ba`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct MeshVertex {
    /// The position (x, y).
    pub position: [f32; 2],
    /// Red (low 16 bits) and green (high 16 bits).
    pub rg: u32,
    /// Blue (low 16 bits) and alpha (high 16 bits).
    pub ba: u32,
}

impl MeshVertex {
    /// A vertex at `position` of `color`.
    pub fn new(position: Point<f32>, color: Rgba) -> Self {
        fn unorm16(v: f32) -> u32 {
            (v.clamp(0., 1.) * 65535. + 0.5) as u32
        }
        Self {
            position: [position.x, position.y],
            rg: unorm16(color.r) | unorm16(color.g) << 16,
            ba: unorm16(color.b) | unorm16(color.a) << 16,
        }
    }

    /// The vertex position in mesh space.
    pub fn position(&self) -> Point<f32> {
        point(self.position[0], self.position[1])
    }

    /// The vertex colour (as stored: each channel rounded to 16 bits).
    pub fn color(&self) -> Rgba {
        let channel = |v: u32| (v & 0xffff) as f32 / 65535.;
        Rgba {
            r: channel(self.rg),
            g: channel(self.rg >> 16),
            b: channel(self.ba),
            a: channel(self.ba >> 16),
        }
    }
}

/// An immutable triangle list with a colour per vertex, drawn with
/// [`Window::paint_mesh`](crate::Window::paint_mesh).
///
/// Every three vertices are a triangle. Build a mesh once and paint it every frame: renderers
/// upload it the first time it is painted and keep it on the GPU while it is in use (they drop
/// it once every `Mesh` handle is gone, and may drop meshes that have not been painted for a
/// while to stay within their memory budget, uploading them again if they are painted
/// again). Cloning a mesh is cheap and keeps its identity.
#[derive(Clone)]
pub struct Mesh(Arc<MeshData>);

struct MeshData {
    id: MeshId,
    vertices: Box<[MeshVertex]>,
    bounds: Bounds<f32>,
}

static NEXT_MESH_ID: AtomicU64 = AtomicU64::new(1);

impl Mesh {
    /// A mesh of the triangles `vertices` makes (a trailing partial triangle is dropped).
    pub fn new(vertices: impl IntoIterator<Item = MeshVertex>) -> Self {
        let mut vertices: Vec<MeshVertex> = vertices.into_iter().collect();
        vertices.truncate(vertices.len() / 3 * 3);
        let mut min = point(f32::INFINITY, f32::INFINITY);
        let mut max = point(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for v in &vertices {
            let p = v.position();
            min = point(min.x.min(p.x), min.y.min(p.y));
            max = point(max.x.max(p.x), max.y.max(p.y));
        }
        let bounds = if vertices.is_empty() {
            Bounds::default()
        } else {
            Bounds {
                origin: min,
                size: size(max.x - min.x, max.y - min.y),
            }
        };
        Self(Arc::new(MeshData {
            id: MeshId(NEXT_MESH_ID.fetch_add(1, Ordering::Relaxed)),
            vertices: vertices.into_boxed_slice(),
            bounds,
        }))
    }

    /// The mesh's identity: the same for every clone of it.
    pub fn id(&self) -> MeshId {
        self.0.id
    }

    /// The vertices; every three are a triangle.
    pub fn vertices(&self) -> &[MeshVertex] {
        &self.0.vertices
    }

    /// The number of vertices (a multiple of three).
    pub fn len(&self) -> usize {
        self.0.vertices.len()
    }

    /// Whether the mesh has no triangles.
    pub fn is_empty(&self) -> bool {
        self.0.vertices.is_empty()
    }

    /// The bounds of the vertices in mesh space.
    pub fn bounds(&self) -> Bounds<f32> {
        self.0.bounds
    }

    /// A handle that does not keep the mesh alive (renderers keep one per mesh they store, to
    /// drop its GPU copy once the mesh is gone).
    pub fn downgrade(&self) -> WeakMesh {
        WeakMesh(Arc::downgrade(&self.0))
    }
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("id", &self.0.id)
            .field("vertices", &self.0.vertices.len())
            .field("bounds", &self.0.bounds)
            .finish()
    }
}

impl PartialEq for Mesh {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// A [`Mesh`] handle that does not keep it alive.
#[derive(Clone, Debug)]
pub struct WeakMesh(Weak<MeshData>);

impl WeakMesh {
    /// Whether every [`Mesh`] handle of this mesh has been dropped.
    pub fn is_dropped(&self) -> bool {
        self.0.strong_count() == 0
    }

    /// The mesh, if it is still alive.
    pub fn upgrade(&self) -> Option<Mesh> {
        self.0.upgrade().map(Mesh)
    }
}

impl std::fmt::Debug for MeshData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshData").field("id", &self.id).finish()
    }
}

/// A colour of a [`MeshGradient`] at an offset along it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshGradientStop {
    /// Where along the gradient, from 0 to 1.
    pub offset: f32,
    /// The colour, straight (not premultiplied).
    pub color: Rgba,
}

/// What a [`MeshGradient`] paints beyond offsets 0 and 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshGradientSpread {
    /// The end colours continue.
    #[default]
    Pad,
    /// The gradient repeats, mirrored every other time.
    Reflect,
    /// The gradient repeats.
    Repeat,
}

/// The geometry of a [`MeshGradient`], in the gradient's own space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeshGradientShape {
    /// Offset 0 at `start`, 1 at `end`, constant across the line between them.
    Linear {
        /// Where offset 0 is.
        start: Point<f32>,
        /// Where offset 1 is.
        end: Point<f32>,
    },
    /// A two-point conical gradient, as the HTML canvas's `createRadialGradient` and SVG 2's
    /// `<radialGradient>` (with `fx`, `fy`, `fr`) draw it: offset `t` is on the circle whose
    /// centre and radius are `t` of the way from the start circle to the end circle (the
    /// largest `t` whose radius is not negative, where circles overlap). Where no circle
    /// passes (outside the cone between two circles), nothing is painted.
    Radial {
        /// The centre of the circle at offset 0.
        start: Point<f32>,
        /// The radius of the circle at offset 0.
        start_radius: f32,
        /// The centre of the circle at offset 1.
        end: Point<f32>,
        /// The radius of the circle at offset 1.
        end_radius: f32,
    },
}

/// A gradient a mesh can be painted with ([`Window::paint_mesh_gradient`](crate::Window::paint_mesh_gradient)):
/// colours interpolated between stops (straight, as SVG and the HTML canvas interpolate them)
/// along a line or between two circles.
///
/// A gradient without stops paints nothing; with one stop, or whose geometry is degenerate (a
/// line of no length, or two equal circles), it paints its last stop's colour.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshGradient {
    /// The line or circles, in the gradient's space.
    pub shape: MeshGradientShape,
    /// The colours, in order of offset. Offsets out of order are raised to the offset before
    /// them (equal offsets make a hard edge, whose later colour starts at its offset).
    pub stops: Vec<MeshGradientStop>,
    /// What is painted beyond offsets 0 and 1.
    pub spread: MeshGradientSpread,
    /// The mesh's space → the gradient's space.
    pub to_gradient: TransformationMatrix,
}

impl MeshGradient {
    /// The offset along the gradient at `point` (the gradient's space), before spreading, or
    /// `None` where the gradient paints nothing (outside a conical gradient's cone). A
    /// degenerate gradient gives 1.
    pub fn offset_at(&self, point: Point<f32>) -> Option<f32> {
        match self.shape {
            MeshGradientShape::Linear { start, end } => {
                let (dx, dy) = (end.x - start.x, end.y - start.y);
                let length_squared = dx * dx + dy * dy;
                if !(length_squared > 0.) {
                    return Some(1.);
                }
                Some(((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared)
            }
            MeshGradientShape::Radial {
                start,
                start_radius,
                end,
                end_radius,
            } => {
                // |p - c(t)| = r(t) for c(t) = c0 + t (c1 - c0), r(t) = r0 + t (r1 - r0):
                // a t² - 2 b t + c = 0
                let (cx, cy) = (end.x - start.x, end.y - start.y);
                let (px, py) = (point.x - start.x, point.y - start.y);
                let dr = end_radius - start_radius;
                let a = cx * cx + cy * cy - dr * dr;
                let b = px * cx + py * cy + start_radius * dr;
                let c = px * px + py * py - start_radius * start_radius;
                let radius = |t: f32| start_radius + t * dr;
                if cx == 0. && cy == 0. && dr == 0. {
                    return Some(1.);
                }
                if a.abs() <= 1e-6 * (cx * cx + cy * cy + dr * dr) {
                    let t = c / (2. * b);
                    return (t.is_finite() && radius(t) >= 0.).then_some(t);
                }
                let discriminant = b * b - a * c;
                if discriminant < 0. {
                    return None;
                }
                let root = discriminant.sqrt();
                let (t0, t1) = ((b - root) / a, (b + root) / a);
                let (high, low) = (t0.max(t1), t0.min(t1));
                if radius(high) >= 0. {
                    Some(high)
                } else if radius(low) >= 0. {
                    Some(low)
                } else {
                    None
                }
            }
        }
    }

    /// The straight colour the gradient paints at `point` (the mesh's space), or `None` where
    /// it paints nothing. Renderers evaluate gradients on the GPU the same way.
    pub fn color_at(&self, point: Point<f32>) -> Option<Rgba> {
        let (last, first) = (self.stops.last()?, self.stops.first()?);
        let m = self.to_gradient;
        let point = crate::point(
            m.rotation_scale[0][0] * point.x + m.rotation_scale[0][1] * point.y + m.translation[0],
            m.rotation_scale[1][0] * point.x + m.rotation_scale[1][1] * point.y + m.translation[1],
        );
        if self.stops.len() == 1 || self.is_degenerate() {
            return Some(last.color);
        }
        let t = self.offset_at(point)?;
        let t = match self.spread {
            MeshGradientSpread::Pad => t.clamp(0., 1.),
            MeshGradientSpread::Repeat => t - t.floor(),
            MeshGradientSpread::Reflect => {
                let u = t.abs() % 2.;
                if u > 1. { 2. - u } else { u }
            }
        };
        // (intervals are open at their end: on a hard edge, the later colour)
        let mut previous = *first;
        if t < first.offset {
            return Some(first.color);
        }
        for stop in &self.stops[1..] {
            let offset = stop.offset.max(previous.offset);
            if t < offset {
                let k = (t - previous.offset) / (offset - previous.offset);
                let mix = |a: f32, b: f32| a + (b - a) * k;
                let (a, b) = (previous.color, stop.color);
                return Some(Rgba {
                    r: mix(a.r, b.r),
                    g: mix(a.g, b.g),
                    b: mix(a.b, b.b),
                    a: mix(a.a, b.a),
                });
            }
            previous = MeshGradientStop { offset, ..*stop };
        }
        Some(last.color)
    }

    /// Whether the geometry is degenerate: a line of no length, or two equal circles (the
    /// gradient paints its last stop's colour).
    pub fn is_degenerate(&self) -> bool {
        match self.shape {
            MeshGradientShape::Linear { start, end } => start == end,
            MeshGradientShape::Radial {
                start,
                start_radius,
                end,
                end_radius,
            } => start == end && start_radius == end_radius,
        }
    }

    /// The mean of the stops' colours, weighted by the offsets each covers: what renderers
    /// that do not evaluate gradients paint instead.
    pub fn average_color(&self) -> Rgba {
        let Some(first) = self.stops.first() else {
            return Rgba::default();
        };
        if self.stops.len() == 1 || self.is_degenerate() {
            return self.stops[self.stops.len() - 1].color;
        }
        let mut sum = [0f32; 4];
        let mut add = |color: Rgba, weight: f32| {
            sum[0] += color.r * color.a * weight;
            sum[1] += color.g * color.a * weight;
            sum[2] += color.b * color.a * weight;
            sum[3] += color.a * weight;
        };
        let mut previous = *first;
        add(first.color, first.offset.clamp(0., 1.));
        for stop in &self.stops[1..] {
            let offset = stop.offset.clamp(previous.offset, 1.);
            let span = offset - previous.offset;
            add(previous.color, span / 2.);
            add(stop.color, span / 2.);
            previous = MeshGradientStop { offset, ..*stop };
        }
        add(previous.color, 1. - previous.offset);
        if sum[3] <= 0. {
            return Rgba::default();
        }
        Rgba {
            r: sum[0] / sum[3],
            g: sum[1] / sum[3],
            b: sum[2] / sum[3],
            a: sum[3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_vertices_round_trip() {
        let color = Rgba {
            r: 0.25,
            g: 1.0,
            b: 0.0,
            a: 0.5,
        };
        let v = MeshVertex::new(point(-3.5, 1e6), color);
        assert_eq!(v.position(), point(-3.5, 1e6));
        let c = v.color();
        for (a, b) in [(c.r, 0.25), (c.g, 1.0), (c.b, 0.0), (c.a, 0.5)] {
            assert!((a - b).abs() <= 0.5 / 65535., "{a} vs {b}");
        }
        // Out of range channels are clamped.
        let v = MeshVertex::new(
            point(0., 0.),
            Rgba {
                r: 2.0,
                g: -1.0,
                b: 0.5,
                a: 1.0,
            },
        );
        assert_eq!(v.color().r, 1.0);
        assert_eq!(v.color().g, 0.0);
    }

    #[test]
    fn test_mesh_bounds_identity_and_liveness() {
        let white = Rgba {
            r: 1.,
            g: 1.,
            b: 1.,
            a: 1.,
        };
        let mesh = Mesh::new(
            [
                point(1., 2.),
                point(5., -1.),
                point(3., 7.),
                point(100., 100.),
            ]
            .map(|p| MeshVertex::new(p, white)),
        );
        // The partial triangle is dropped.
        assert_eq!(mesh.len(), 3);
        assert_eq!(
            mesh.bounds(),
            Bounds {
                origin: point(1., -1.),
                size: size(4., 8.)
            }
        );
        let other = Mesh::new([]);
        assert!(other.is_empty());
        assert_ne!(mesh.id(), other.id());
        let clone = mesh.clone();
        assert_eq!(clone.id(), mesh.id());
        assert_eq!(clone, mesh);

        let weak = mesh.downgrade();
        drop(mesh);
        assert!(!weak.is_dropped());
        drop(clone);
        assert!(weak.is_dropped());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_mesh_draw_path_vertices() {
        use crate::{Background, ContentMask, MeshDraw, ScaledPixels, TransformationMatrix};
        let orange = Rgba {
            r: 1.,
            g: 0.5,
            b: 0.,
            a: 0.5,
        };
        let mesh = Mesh::new(
            [point(0., 0.), point(10., 0.), point(0., 10.)].map(|p| MeshVertex::new(p, orange)),
        );
        let draw = MeshDraw {
            blend_mode: crate::MeshBlendMode::Normal,
            order: 0,
            bounds: Bounds::default(),
            content_mask: ContentMask::default(),
            transformation: TransformationMatrix {
                rotation_scale: [[2., 0.], [0., 2.]],
                translation: [5., 7.],
            },
            background: Background::from(Rgba {
                r: 1.,
                g: 1.,
                b: 1.,
                a: 0.5,
            }),
            gradient: None,
            mesh,
        };
        let vertices: Vec<_> = draw.path_vertices().collect();
        assert_eq!(vertices.len(), 3);
        assert_eq!(vertices[1].0, point(ScaledPixels(25.), ScaledPixels(7.)));
        // The paint multiplies the vertex colour.
        let color = Rgba::from(vertices[0].1.solid);
        assert!((color.r - 1.).abs() < 1e-3 && (color.g - 0.5).abs() < 1e-3);
        assert!((color.a - 0.25).abs() < 1e-3, "{color:?}");
    }
}
