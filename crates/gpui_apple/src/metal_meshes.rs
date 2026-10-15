//! GPU storage for retained meshes (`Window::paint_mesh`).
//!
//! Every mesh painted in a frame is kept in a few shared vertex buffers ("pages"), in blocks of
//! [`MESH_BLOCK_VERTICES`] vertices that need not be contiguous: a draw is rasterized as one
//! instance per block (see `mesh_vertex` in the shaders), so a frame of meshes costs one
//! instanced draw call and a few bytes per block, whatever the vertices. Blocks are allocated
//! from a free list (no fragmentation), and a mesh's blocks are reused only once the frames
//! that drew it have completed on the GPU. When the pages are full a bigger one is added
//! (nothing is copied), up to [`MAX_PAGES`].
//!
//! A mesh stays stored while any `Mesh` handle to it is alive; beyond the memory budget the
//! meshes painted longest ago are dropped first (and uploaded again if they are painted again).

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Result;
use collections::{FxHashMap, VecDeque};
use gpui::{Mesh, MeshId, MeshVertex, WeakMesh};
use metal::{MTLResourceOptions, NSRange};

/// Vertices per block: whole triangles.
pub(crate) const MESH_BLOCK_VERTICES: u32 = 48;
const BLOCK_BYTES: u64 = MESH_BLOCK_VERTICES as u64 * size_of::<MeshVertex>() as u64;
/// The vertex buffers the mesh shaders can read.
pub(crate) const MAX_PAGES: usize = 8;
/// The first page's size in blocks (3 MB); each next one is twice as big, up to
/// [`MAX_PAGE_BLOCKS`].
const FIRST_PAGE_BLOCKS: u32 = 1 << 12;
const MAX_PAGE_BLOCKS: u32 = 1 << 17;
/// Above this many bytes of blocks in use, meshes that were not painted in the current frame
/// are dropped (least recently painted first) before another page is added.
const BUDGET_BYTES: u64 = 256 << 20;
/// How often (in frames) meshes whose handles were all dropped are released.
const SWEEP_FRAMES: u64 = 32;

/// A block of a page: where a mesh's vertices [`MESH_BLOCK_VERTICES`] at a time are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    pub page: u32,
    /// The block's first vertex in its page.
    pub first_vertex: u32,
}

struct Page {
    buffer: metal::Buffer,
    /// Capacity in blocks.
    blocks: u32,
    /// Blocks handed out at least once; the ones above are unused.
    high_water: u32,
}

struct StoredMesh {
    blocks: Vec<Block>,
    last_used: u64,
    mesh: WeakMesh,
}

pub(crate) struct MeshStore {
    device: metal::Device,
    unified_memory: bool,
    pages: Vec<Page>,
    free: Vec<Block>,
    /// Released blocks and the last frame that may still read them.
    retired: VecDeque<(u64, Vec<Block>)>,
    meshes: FxHashMap<MeshId, StoredMesh>,
    /// Blocks in use (stored or retired).
    used: u64,
    frame: u64,
    /// The last frame the GPU completed (set by command buffer completion handlers).
    completed: Arc<AtomicU64>,
    failed: bool,
}

impl MeshStore {
    pub(crate) fn new(device: metal::Device, unified_memory: bool) -> Self {
        Self {
            device,
            unified_memory,
            pages: Vec::new(),
            free: Vec::new(),
            retired: VecDeque::new(),
            meshes: FxHashMap::default(),
            used: 0,
            frame: 0,
            completed: Arc::new(AtomicU64::new(0)),
            failed: false,
        }
    }

    /// Starts a frame: stores every mesh `meshes` paints (uploading those it does not hold
    /// yet). Returns the frame's number, which the frame's command buffer reports to
    /// [`Self::completion`] once the GPU is done with it.
    pub(crate) fn begin_frame<'a>(&mut self, meshes: impl Iterator<Item = &'a Mesh>) -> u64 {
        self.frame += 1;
        self.reclaim();
        if self.frame.is_multiple_of(SWEEP_FRAMES) {
            self.release_dropped();
        }
        for mesh in meshes {
            if let Err(error) = self.ensure(mesh) {
                if !self.failed {
                    log::error!("mesh not drawn: {error:#}");
                    self.failed = true;
                }
            }
        }
        self.frame
    }

    /// The counter command buffer completion handlers advance to their frame's number.
    pub(crate) fn completion(&self) -> Arc<AtomicU64> {
        self.completed.clone()
    }

    /// The vertex buffers, in page order.
    pub(crate) fn pages(&self) -> impl Iterator<Item = &metal::Buffer> {
        self.pages.iter().map(|page| &page.buffer)
    }

    /// The blocks holding a stored mesh, in order.
    pub(crate) fn blocks(&self, id: MeshId) -> Option<&[Block]> {
        self.meshes.get(&id).map(|stored| stored.blocks.as_slice())
    }

    /// Stored meshes, blocks in use, and the pages' size in bytes.
    #[allow(dead_code)]
    pub(crate) fn usage(&self) -> (usize, u64, u64) {
        (
            self.meshes.len(),
            self.used,
            self.pages
                .iter()
                .map(|page| u64::from(page.blocks) * BLOCK_BYTES)
                .sum(),
        )
    }

    fn ensure(&mut self, mesh: &Mesh) -> Result<()> {
        if let Some(stored) = self.meshes.get_mut(&mesh.id()) {
            stored.last_used = self.frame;
            return Ok(());
        }
        let needed = mesh.len().div_ceil(MESH_BLOCK_VERTICES as usize) as u32;
        let blocks = self.allocate(needed)?;
        self.upload(mesh.vertices(), &blocks);
        self.meshes.insert(
            mesh.id(),
            StoredMesh {
                blocks,
                last_used: self.frame,
                mesh: mesh.downgrade(),
            },
        );
        Ok(())
    }

    fn available(&self) -> u64 {
        self.free.len() as u64
            + self
                .pages
                .iter()
                .map(|page| u64::from(page.blocks - page.high_water))
                .sum::<u64>()
    }

    fn allocate(&mut self, count: u32) -> Result<Vec<Block>> {
        if self.available() < u64::from(count) {
            if (self.used + u64::from(count)) * BLOCK_BYTES > BUDGET_BYTES {
                self.evict_least_recently_used(count);
            }
            while self.available() < u64::from(count) {
                self.add_page()?;
            }
        }
        let mut blocks = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let block = match self.free.pop() {
                Some(block) => block,
                None => {
                    let (index, page) = self
                        .pages
                        .iter_mut()
                        .enumerate()
                        .find(|(_, page)| page.high_water < page.blocks)
                        .ok_or_else(|| anyhow::anyhow!("mesh pages are full"))?;
                    page.high_water += 1;
                    Block {
                        page: index as u32,
                        first_vertex: (page.high_water - 1) * MESH_BLOCK_VERTICES,
                    }
                }
            };
            blocks.push(block);
        }
        self.used += u64::from(count);
        Ok(blocks)
    }

    /// Adds a page twice as big as the last one (the first is small).
    fn add_page(&mut self) -> Result<()> {
        anyhow::ensure!(
            self.pages.len() < MAX_PAGES,
            "meshes need more than {MAX_PAGES} pages"
        );
        let blocks = self.pages.last().map_or(FIRST_PAGE_BLOCKS, |page| {
            (page.blocks * 2).min(MAX_PAGE_BLOCKS)
        });
        let options = if self.unified_memory {
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            MTLResourceOptions::StorageModeManaged
        };
        let buffer = self
            .device
            .new_buffer(u64::from(blocks) * BLOCK_BYTES, options);
        log::debug!(
            "mesh page {}: {} MB",
            self.pages.len(),
            (u64::from(blocks) * BLOCK_BYTES) >> 20
        );
        self.pages.push(Page {
            buffer,
            blocks,
            high_water: 0,
        });
        Ok(())
    }

    /// Copies `vertices` into `blocks`.
    fn upload(&self, vertices: &[MeshVertex], blocks: &[Block]) {
        for (chunk, block) in vertices.chunks(MESH_BLOCK_VERTICES as usize).zip(blocks) {
            let buffer = &self.pages[block.page as usize].buffer;
            let base = buffer.contents().cast::<MeshVertex>();
            // SAFETY: `block` lies within its page, which holds `MESH_BLOCK_VERTICES` vertices
            // from `first_vertex`; no frame the GPU may still be running reads it (blocks are
            // reused only after the frames that drew them completed).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    base.add(block.first_vertex as usize),
                    chunk.len(),
                );
            }
            if !self.unified_memory {
                buffer.did_modify_range(NSRange {
                    location: u64::from(block.first_vertex) * size_of::<MeshVertex>() as u64,
                    length: (chunk.len() * size_of::<MeshVertex>()) as u64,
                });
            }
        }
    }

    /// Drops meshes not painted in this frame, least recently painted first, until `count`
    /// blocks are free or a quarter of the budget was released.
    fn evict_least_recently_used(&mut self, count: u32) {
        let frame = self.frame;
        let mut candidates: Vec<(u64, MeshId)> = self
            .meshes
            .iter()
            .filter(|(_, stored)| stored.last_used < frame)
            .map(|(id, stored)| (stored.last_used, *id))
            .collect();
        candidates.sort_unstable();
        let target = u64::from(count).max(BUDGET_BYTES / BLOCK_BYTES / 4);
        let mut released = 0u64;
        for (_, id) in candidates {
            if released >= target {
                break;
            }
            if let Some(stored) = self.meshes.remove(&id) {
                released += stored.blocks.len() as u64;
                self.retired.push_back((stored.last_used, stored.blocks));
            }
        }
        self.reclaim();
    }

    /// Releases the meshes whose handles were all dropped.
    fn release_dropped(&mut self) {
        let dropped: Vec<MeshId> = self
            .meshes
            .iter()
            .filter(|(_, stored)| stored.mesh.is_dropped())
            .map(|(id, _)| *id)
            .collect();
        for id in dropped {
            if let Some(stored) = self.meshes.remove(&id) {
                self.retired.push_back((stored.last_used, stored.blocks));
            }
        }
    }

    /// Frees retired blocks the GPU is done with.
    fn reclaim(&mut self) {
        let completed = self.completed.load(Ordering::Acquire);
        let mut kept = VecDeque::new();
        while let Some((last_used, blocks)) = self.retired.pop_front() {
            if last_used <= completed {
                self.used -= blocks.len() as u64;
                self.free.extend(blocks);
            } else {
                kept.push_back((last_used, blocks));
            }
        }
        self.retired = kept;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{Rgba, point};

    fn rgba(r: f32, g: f32, b: f32, a: f32) -> Rgba {
        Rgba { r, g, b, a }
    }

    #[test]
    fn test_mesh_blocks_are_reused_only_after_their_frames_completed() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let mut store = MeshStore::new(device, true);
        let color = rgba(1., 1., 1., 1.);
        // 102 vertices: three blocks
        let big = Mesh::new((0..102).map(|i| MeshVertex::new(point(i as f32, 0.), color)));
        let frame = store.begin_frame([&big].into_iter());
        let blocks = store.blocks(big.id()).expect("stored").to_vec();
        assert_eq!(blocks.len(), 3);
        let id = big.id();
        drop(big);
        // Released at the next sweep, but the GPU has not completed the frame yet.
        for _ in 0..SWEEP_FRAMES {
            store.begin_frame(std::iter::empty());
        }
        assert!(store.blocks(id).is_none());
        let other = Mesh::new((0..3).map(|i| MeshVertex::new(point(i as f32, 1.), color)));
        store.begin_frame([&other].into_iter());
        let reused = store.blocks(other.id()).expect("stored")[0];
        assert!(
            !blocks.contains(&reused),
            "a block the GPU may still read was reused"
        );
        // Once the frame completed its blocks are free again.
        store.completion().store(frame, Ordering::Release);
        let third = Mesh::new((0..3).map(|i| MeshVertex::new(point(i as f32, 2.), color)));
        store.begin_frame([&third].into_iter());
        assert!(blocks.contains(&store.blocks(third.id()).expect("stored")[0]));
    }

    #[test]
    fn test_mesh_pages_are_added_without_moving_stored_meshes() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let mut store = MeshStore::new(device, true);
        let color = rgba(0.25, 0.5, 0.75, 1.);
        // More than the first page holds.
        let vertices = (FIRST_PAGE_BLOCKS as usize + 10) * MESH_BLOCK_VERTICES as usize;
        let first = Mesh::new((0..48).map(|i| MeshVertex::new(point(i as f32, 0.), color)));
        store.begin_frame([&first].into_iter());
        let first_blocks = store.blocks(first.id()).expect("stored").to_vec();
        let big = Mesh::new((0..vertices).map(|i| MeshVertex::new(point(i as f32, 1.), color)));
        store.begin_frame([&first, &big].into_iter());
        assert_eq!(store.pages().count(), 2, "a second page");
        assert_eq!(
            store.blocks(first.id()).expect("stored"),
            first_blocks,
            "the first mesh did not move"
        );
        // The vertices are where the blocks say.
        let blocks = store.blocks(big.id()).expect("stored");
        let last = blocks.last().expect("blocks");
        let page = store.pages().nth(last.page as usize).expect("page");
        let stored = unsafe {
            *page
                .contents()
                .cast::<MeshVertex>()
                .add(last.first_vertex as usize)
        };
        let expected = big.vertices()[(blocks.len() - 1) * MESH_BLOCK_VERTICES as usize];
        assert_eq!(stored, expected);
    }
}
