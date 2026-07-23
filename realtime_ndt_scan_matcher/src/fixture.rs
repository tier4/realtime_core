//! Frozen WCET benchmark fixtures — capture-once, replay-everywhere.
//!
//! A fixture is one complete align input (target map tiles + source cloud + initial guess +
//! params) in a trivial little-endian binary layout, so the **same bytes** feed the Rust harness
//! (`examples/wcet_frame.rs`), the search (`examples/wcet_search.rs`), CI regression, and the C++
//! comparison (`bench/ndt_bench_replay.cpp` reads it with a few `fread`s). std-only (file IO);
//! not part of the `no_std` engine.
//!
//! The map is stored as **tiles** (one `add_target` / `setInputTarget(id)` call each) because the
//! multi-grid voxel map keeps one voxel grid *per tile*: overlapping tiles multiply the leaves a
//! radius search can collect, and driving the per-point neighbor count `K` toward
//! [`MAX_NEIGHBORS`](crate::ndt::MAX_NEIGHBORS) — a first-order WCET term — requires that overlap
//! (a single tile geometrically caps `K` at the ≤ 8 voxels sharing a corner when the search radius
//! equals the leaf size). This mirrors production dynamic map loading, which is inherently
//! multi-tile.
//!
//! Layout (all little-endian):
//! ```text
//! magic      8 × u8   = "NDTFIX01"
//! n_tiles    u64      target map tile count
//! n_source   u64      source cloud point count
//! guess      16 × f32 row-major 4×4 initial pose
//! trans_epsilon, step_size, resolution, outlier_ratio   4 × f64
//! max_iterations i32, reserved i32 (0)
//! tiles      n_tiles × [ n_pts u64, n_pts × 3 × f32 ]   (tile index == map id)
//! source     n_source × 3 × f32
//! ```
//! `regularization` is always `None` and `num_threads` 1 (the serial WCET baseline) — neither is
//! stored.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use nalgebra::Matrix4;

use crate::ndt::NdtParams;

const MAGIC: [u8; 8] = *b"NDTFIX01";
/// Sanity cap on stored point counts (guards a corrupt/hostile file from OOM-ing the reader).
const MAX_POINTS: u64 = 50_000_000;
/// Sanity cap on the tile count.
const MAX_TILES: u64 = 4096;

/// One frozen align input.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// Target map tiles; tile `i` is fed to `add_target(&tiles[i], i)` (C++: `setInputTarget`
    /// with map id `i`). Tiles may overlap — that is the `K`-adversarial case.
    pub tiles: Vec<Vec<[f32; 3]>>,
    /// Source cloud.
    pub source: Vec<[f32; 3]>,
    /// Initial pose guess (row-major in the file).
    pub guess: Matrix4<f32>,
    /// Align parameters (serial baseline: `regularization = None`, `num_threads = 1`).
    pub params: NdtParams,
}

pub(crate) fn write_points(w: &mut impl Write, pts: &[[f32; 3]]) -> io::Result<()> {
    for p in pts {
        for c in p {
            w.write_all(&c.to_le_bytes())?;
        }
    }
    Ok(())
}

pub(crate) fn read_exact_array<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut buf = [0_u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub(crate) fn read_points(r: &mut impl Read, n: u64) -> io::Result<Vec<[f32; 3]>> {
    if n > MAX_POINTS {
        return Err(io::ErrorKind::InvalidData.into());
    }
    let n = usize::try_from(n).map_err(|_| io::ErrorKind::InvalidData)?;
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        let mut p = [0.0_f32; 3];
        for c in &mut p {
            *c = f32::from_le_bytes(read_exact_array::<4>(r)?);
        }
        pts.push(p);
    }
    Ok(pts)
}

impl Fixture {
    /// Total map point count across all tiles.
    #[must_use]
    pub fn map_len(&self) -> usize {
        self.tiles
            .iter()
            .fold(0_usize, |acc, t| acc.saturating_add(t.len()))
    }

    /// Serialize to `path` (see the module doc for the layout).
    ///
    /// # Errors
    /// Propagates file-IO errors.
    #[expect(
        clippy::indexing_slicing,
        reason = "(r, c) ∈ 0..4 index a fixed-size nalgebra Matrix4"
    )]
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut f = io::BufWriter::new(File::create(path)?);
        f.write_all(&MAGIC)?;
        let n_tiles = u64::try_from(self.tiles.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
        let n_source = u64::try_from(self.source.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
        f.write_all(&n_tiles.to_le_bytes())?;
        f.write_all(&n_source.to_le_bytes())?;
        for r in 0..4 {
            for c in 0..4 {
                f.write_all(&self.guess[(r, c)].to_le_bytes())?;
            }
        }
        f.write_all(&self.params.trans_epsilon.to_le_bytes())?;
        f.write_all(&self.params.step_size.to_le_bytes())?;
        f.write_all(&self.params.resolution.to_le_bytes())?;
        f.write_all(&self.params.outlier_ratio.to_le_bytes())?;
        f.write_all(&self.params.max_iterations.to_le_bytes())?;
        f.write_all(&0_i32.to_le_bytes())?;
        for tile in &self.tiles {
            let n = u64::try_from(tile.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
            f.write_all(&n.to_le_bytes())?;
            write_points(&mut f, tile)?;
        }
        write_points(&mut f, &self.source)?;
        f.flush()
    }

    /// Deserialize from `path`.
    ///
    /// # Errors
    /// Propagates file-IO errors; rejects a bad magic or absurd tile/point counts as
    /// `InvalidData`.
    #[expect(
        clippy::indexing_slicing,
        reason = "(r, c) ∈ 0..4 index a fixed-size nalgebra Matrix4"
    )]
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut f = io::BufReader::new(File::open(path)?);
        if read_exact_array::<8>(&mut f)? != MAGIC {
            return Err(io::ErrorKind::InvalidData.into());
        }
        let n_tiles = u64::from_le_bytes(read_exact_array::<8>(&mut f)?);
        let n_source = u64::from_le_bytes(read_exact_array::<8>(&mut f)?);
        if n_tiles > MAX_TILES || n_source > MAX_POINTS {
            return Err(io::ErrorKind::InvalidData.into());
        }
        let mut guess = Matrix4::<f32>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                guess[(r, c)] = f32::from_le_bytes(read_exact_array::<4>(&mut f)?);
            }
        }
        let trans_epsilon = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
        let step_size = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
        let resolution = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
        let outlier_ratio = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
        let max_iterations = i32::from_le_bytes(read_exact_array::<4>(&mut f)?);
        let _reserved = i32::from_le_bytes(read_exact_array::<4>(&mut f)?);
        let n_tiles = usize::try_from(n_tiles).map_err(|_| io::ErrorKind::InvalidData)?;
        let mut tiles = Vec::with_capacity(n_tiles);
        for _ in 0..n_tiles {
            let n = u64::from_le_bytes(read_exact_array::<8>(&mut f)?);
            tiles.push(read_points(&mut f, n)?);
        }
        let source = read_points(&mut f, n_source)?;
        Ok(Self {
            tiles,
            source,
            guess,
            params: NdtParams {
                trans_epsilon,
                step_size,
                resolution,
                max_iterations,
                outlier_ratio,
                regularization: None,
                num_threads: 1,
            },
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let fx = Fixture {
            tiles: vec![
                (0..10).map(|i| [i as f32, 0.5, -1.0]).collect(),
                (0..4).map(|i| [i as f32 * 0.1, -0.5, 1.0]).collect(),
            ],
            source: (0..3).map(|i| [0.0, i as f32 * 0.25, 2.0]).collect(),
            guess: Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 2.0, 3.0)),
            params: NdtParams {
                trans_epsilon: 0.01,
                step_size: 0.1,
                resolution: 2.0,
                max_iterations: 30,
                outlier_ratio: 0.55,
                regularization: None,
                num_threads: 1,
            },
        };
        assert_eq!(fx.map_len(), 14);
        let dir = std::env::temp_dir().join("ndtfix_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.ndtfix");
        fx.write(&path).unwrap();
        let back = Fixture::read(&path).unwrap();
        assert_eq!(back.tiles, fx.tiles);
        assert_eq!(back.source, fx.source);
        assert_eq!(back.guess, fx.guess);
        assert_eq!(back.params.max_iterations, 30);
        assert_eq!(back.params.resolution, 2.0);

        // bad magic rejected
        std::fs::write(dir.join("bad.ndtfix"), b"NOPE0000rest").unwrap();
        assert!(Fixture::read(&dir.join("bad.ndtfix")).is_err());
    }
}
