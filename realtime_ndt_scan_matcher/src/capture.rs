// Copyright 2024 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Real-drive input capture — the sidecar format behind the node crate's `NDT_CAPTURE_DIR` hook
//! (`plan/ndt_wcet.md` operational-envelope follow-up).
//!
//! Unlike a [`crate::fixture`] file (one self-contained align input), a capture directory records
//! a whole drive against a **dynamically tiled** map without duplicating the map per frame:
//!
//! ```text
//! <dir>/params.bin              trans_epsilon, step_size, resolution, outlier_ratio (4 × f64),
//!                               max_iterations i32, reserved i32           (written once)
//! <dir>/tiles/<hex(id)>.bin     n u64 + n × 3 × f32                        (write-once per id)
//! <dir>/frames/frame_<seq>.bin  guess 16 × f32, n_ids u64,
//!                               n_ids × [len u64, id bytes],               (active tile set)
//!                               n_src u64 + points
//! ```
//!
//! The offline replayers (`examples/wcet_frame.rs --capture`, C++ `ndt_bench_replay --capture`)
//! group frames into **epochs** by their id set and rebuild the map once per epoch, assigning
//! the same deterministic byte ids in sorted-id order on both engines — a self-consistent
//! equal-work comparison
//! (exact equality with the live incremental map build is not claimed). All little-endian;
//! std-only.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use nalgebra::Matrix4;

use crate::fixture::{read_exact_array, read_points, write_points};
use crate::ndt::NdtParams;

/// Sanity caps for the readers (corrupt/hostile file guard).
const MAX_POINTS: u64 = 50_000_000;
const MAX_IDS: u64 = 4096;
const MAX_ID_LEN: u64 = 4096;

/// One captured align input (map referenced by tile ids).
#[derive(Clone, Debug)]
pub struct Frame {
    /// Row-major initial pose guess.
    pub guess: Matrix4<f32>,
    /// Active map tile ids (raw bytes, as reported by the engine) at align time.
    pub ids: Vec<Vec<u8>>,
    /// Source cloud.
    pub source: Vec<[f32; 3]>,
}

/// Hex-encode a tile id for use as a file name.
#[must_use]
#[expect(
    clippy::indexing_slicing,
    reason = "nibble values are provably < 16, indexing a fixed [u8; 16]"
)]
pub fn hex_id(id: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(id.len().saturating_mul(2));
    for b in id {
        s.push(char::from(HEX[usize::from(b >> 4)]));
        s.push(char::from(HEX[usize::from(b & 0x0f)]));
    }
    s
}

/// Write `params.bin` (once; overwrites are harmless — the params are constant per run).
///
/// # Errors
/// Propagates file-IO errors.
pub fn write_params(dir: &Path, p: &NdtParams) -> io::Result<()> {
    let mut f = io::BufWriter::new(File::create(dir.join("params.bin"))?);
    f.write_all(&p.trans_epsilon.to_le_bytes())?;
    f.write_all(&p.step_size.to_le_bytes())?;
    f.write_all(&p.resolution.to_le_bytes())?;
    f.write_all(&p.outlier_ratio.to_le_bytes())?;
    f.write_all(&p.max_iterations.to_le_bytes())?;
    f.write_all(&0_i32.to_le_bytes())?;
    f.flush()
}

/// Read `params.bin` (`regularization = None`, `num_threads = 1` — the serial baseline).
///
/// # Errors
/// Propagates file-IO errors.
pub fn read_params(dir: &Path) -> io::Result<NdtParams> {
    let mut f = io::BufReader::new(File::open(dir.join("params.bin"))?);
    let trans_epsilon = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
    let step_size = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
    let resolution = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
    let outlier_ratio = f64::from_le_bytes(read_exact_array::<8>(&mut f)?);
    let max_iterations = i32::from_le_bytes(read_exact_array::<4>(&mut f)?);
    let _reserved = i32::from_le_bytes(read_exact_array::<4>(&mut f)?);
    Ok(NdtParams {
        trans_epsilon,
        step_size,
        resolution,
        max_iterations,
        outlier_ratio,
        regularization: None,
        num_threads: 1,
    })
}

/// Write a map tile (write-once: returns `Ok` without touching the file if it already exists).
///
/// # Errors
/// Propagates file-IO errors.
pub fn write_tile(dir: &Path, id: &[u8], pts: &[[f32; 3]]) -> io::Result<()> {
    let tiles = dir.join("tiles");
    std::fs::create_dir_all(&tiles)?;
    let path = tiles.join(format!("{}.bin", hex_id(id)));
    if path.exists() {
        return Ok(());
    }
    let mut f = io::BufWriter::new(File::create(path)?);
    let n = u64::try_from(pts.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
    f.write_all(&n.to_le_bytes())?;
    write_points(&mut f, pts)?;
    f.flush()
}

/// Read a tile by its hex file name.
///
/// # Errors
/// Propagates file-IO errors; rejects absurd point counts as `InvalidData`.
pub fn read_tile(dir: &Path, hex: &str) -> io::Result<Vec<[f32; 3]>> {
    let mut f = io::BufReader::new(File::open(dir.join("tiles").join(format!("{hex}.bin")))?);
    let n = u64::from_le_bytes(read_exact_array::<8>(&mut f)?);
    read_points(&mut f, n)
}

/// Write one align frame.
///
/// # Errors
/// Propagates file-IO errors.
pub fn write_frame(
    dir: &Path,
    seq: u64,
    guess: &Matrix4<f32>,
    ids: &[Vec<u8>],
    source: &[[f32; 3]],
) -> io::Result<()> {
    let frames = dir.join("frames");
    std::fs::create_dir_all(&frames)?;
    let mut f = io::BufWriter::new(File::create(frames.join(format!("frame_{seq:06}.bin")))?);
    for r in 0..4 {
        for c in 0..4 {
            f.write_all(&guess[(r, c)].to_le_bytes())?;
        }
    }
    let n_ids = u64::try_from(ids.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
    f.write_all(&n_ids.to_le_bytes())?;
    for id in ids {
        let len = u64::try_from(id.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
        f.write_all(&len.to_le_bytes())?;
        f.write_all(id)?;
    }
    let n_src = u64::try_from(source.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
    f.write_all(&n_src.to_le_bytes())?;
    write_points(&mut f, source)?;
    f.flush()
}

/// Read one frame file.
///
/// # Errors
/// Propagates file-IO errors; rejects absurd counts as `InvalidData`.
#[expect(
    clippy::indexing_slicing,
    reason = "(r, c) ∈ 0..4 index a fixed-size nalgebra Matrix4"
)]
pub fn read_frame(path: &Path) -> io::Result<Frame> {
    let mut f = io::BufReader::new(File::open(path)?);
    let mut guess = Matrix4::<f32>::zeros();
    for r in 0..4 {
        for c in 0..4 {
            guess[(r, c)] = f32::from_le_bytes(read_exact_array::<4>(&mut f)?);
        }
    }
    let n_ids = u64::from_le_bytes(read_exact_array::<8>(&mut f)?);
    if n_ids > MAX_IDS {
        return Err(io::ErrorKind::InvalidData.into());
    }
    let mut ids = Vec::with_capacity(usize::try_from(n_ids).unwrap_or(0));
    for _ in 0..n_ids {
        let len = u64::from_le_bytes(read_exact_array::<8>(&mut f)?);
        if len > MAX_ID_LEN {
            return Err(io::ErrorKind::InvalidData.into());
        }
        let mut id = vec![0_u8; usize::try_from(len).map_err(|_| io::ErrorKind::InvalidData)?];
        f.read_exact(&mut id)?;
        ids.push(id);
    }
    let n_src = u64::from_le_bytes(read_exact_array::<8>(&mut f)?);
    if n_src > MAX_POINTS {
        return Err(io::ErrorKind::InvalidData.into());
    }
    let source = read_points(&mut f, n_src)?;
    Ok(Frame { guess, ids, source })
}

// ---------------------------------------------------------------------------
// Env-gated live hook (`NDT_CAPTURE_DIR`) — called from the engine entry points
// ---------------------------------------------------------------------------

static HOOK_DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
static HOOK_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The capture directory, if `NDT_CAPTURE_DIR` is set (resolved once; `None` disables the hook).
/// The disabled cost on the align path is one `OnceLock` load.
pub(crate) fn hook_dir() -> Option<&'static PathBuf> {
    HOOK_DIR
        .get_or_init(|| {
            let d = std::env::var_os("NDT_CAPTURE_DIR").map(PathBuf::from)?;
            std::fs::create_dir_all(&d).ok()?;
            Some(d)
        })
        .as_ref()
}

/// Best-effort params capture (constant per run; overwrite harmless).
pub(crate) fn hook_params(p: &NdtParams) {
    if let Some(d) = hook_dir() {
        write_params(d, p).ok();
    }
}

/// Best-effort tile capture (write-once per cell id).
pub(crate) fn hook_tile(id: &[u8], points: &[[f32; 3]]) {
    if let Some(d) = hook_dir() {
        write_tile(d, id, points).ok();
    }
}

/// Best-effort align-input capture (one frame per call, monotonic sequence).
pub(crate) fn hook_align(guess: &Matrix4<f32>, ids: &[Vec<u8>], source: &[[f32; 3]]) {
    if let Some(d) = hook_dir() {
        let seq = HOOK_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        write_frame(d, seq, guess, ids, source).ok();
    }
}

/// List `frames/` sorted by sequence number.
///
/// # Errors
/// Propagates directory-IO errors.
pub fn list_frames(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir.join("frames"))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    v.sort();
    Ok(v)
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
        let dir = std::env::temp_dir().join("ndtcap_test");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();

        let params = NdtParams {
            trans_epsilon: 0.01,
            step_size: 0.1,
            resolution: 2.0,
            max_iterations: 30,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1,
        };
        write_params(&dir, &params).unwrap();
        let p2 = read_params(&dir).unwrap();
        assert_eq!(p2.resolution, 2.0);
        assert_eq!(p2.max_iterations, 30);

        let tile: Vec<[f32; 3]> = (0..9).map(|i| [i as f32, 0.5, -1.0]).collect();
        write_tile(&dir, b"grid_1_2", &tile).unwrap();
        // write-once: second call is a no-op, not an error
        write_tile(&dir, b"grid_1_2", &[]).unwrap();
        let back = read_tile(&dir, &hex_id(b"grid_1_2")).unwrap();
        assert_eq!(back, tile);

        let guess = Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 2.0, 3.0));
        let ids = vec![b"grid_1_2".to_vec(), b"grid_9_9".to_vec()];
        let src: Vec<[f32; 3]> = (0..4).map(|i| [0.1 * i as f32, 1.0, 2.0]).collect();
        write_frame(&dir, 7, &guess, &ids, &src).unwrap();
        write_frame(&dir, 8, &guess, &ids, &src).unwrap();

        let frames = list_frames(&dir).unwrap();
        assert_eq!(frames.len(), 2);
        let fr = read_frame(&frames[0]).unwrap();
        assert_eq!(fr.guess, guess);
        assert_eq!(fr.ids, ids);
        assert_eq!(fr.source, src);
    }
}
