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

//! Single voxel-grid covariance (NDT leaves), ported from `MultiVoxelGridCovariance`
//! (`src/ndt_omp/multi_voxel_grid_covariance_omp_impl.hpp`). Builds per-voxel mean / covariance /
//! eigenvalue-regularized inverse covariance. `no_std` + `alloc`; nalgebra internal.
//!
//! Multi-grid id-keyed add/remove and the kdtree/radiusSearch are intentionally out of scope here
//! the C++ engine keeps using its own grid until the full engine swap.

// Numeric kernel: nalgebra f64 matrix operators are the float-math domain the integer-overflow lint
// targets; the voxel-index integer math below uses explicit checked ops where overflow could occur.
// Numeric kernel. Suppressions are scoped per-function (no module-wide `#![allow]`):
// arithmetic_side_effects = nalgebra f64 float math + checked integer voxel-index math;
// as_conversions/cast_* = deliberate f32<->f64<->i64 voxelization conversions (the int32 voxel-count
// guard bounds the floor->i64 narrowing); indexing_slicing = constant indices into fixed `[_; 3]`;
// many_single_char_names = x/y/z geometry notation.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use nalgebra::{Matrix3, Vector3};

use crate::kdtree::{KdSearchError, KdSearchOutcome, KdTree};

const DEFAULT_MIN_POINTS_PER_VOXEL: i32 = 6;

/// A finalized NDT leaf: point count, 3D mean, row-major inverse covariance, and the f32 voxel
/// centroid (used to build the kd-tree, matching the C++ `centroid_`).
#[derive(Clone)]
pub struct Leaf {
    pub n: i32,
    pub mean: [f64; 3],
    pub icov: [f64; 9],
    pub centroid: [f32; 3],
    /// Per-grid integer voxel id, retained only by analysis traces.
    #[cfg(feature = "wcet-trace")]
    pub trace_voxel_id: i64,
    /// Canonical grid ordinal after tile-key sorting, retained only by analysis traces.
    #[cfg(feature = "wcet-trace")]
    pub trace_grid_ordinal: u64,
}

struct Accumulator {
    n: i64,
    sum: Vector3<f64>,
    sum_outer: Matrix3<f64>,
    centroid_sum: [f32; 3],
}

impl Accumulator {
    fn new() -> Self {
        Self {
            n: 0,
            sum: Vector3::zeros(),
            sum_outer: Matrix3::zeros(),
            centroid_sum: [0.0; 3],
        }
    }
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::allow_attributes,
        reason = "nalgebra f64 vector/matrix accumulation"
    )]
    fn add(&mut self, p: Vector3<f64>, raw: [f32; 3]) {
        self.n = self.n.saturating_add(1);
        self.sum += p;
        self.sum_outer += p * p.transpose();
        self.centroid_sum[0] += raw[0];
        self.centroid_sum[1] += raw[1];
        self.centroid_sum[2] += raw[2];
    }
}

/// A built single voxel grid with point->leaf lookup by voxel id.
#[derive(Clone)]
pub struct VoxelGrid {
    leaves: Vec<Leaf>,
    index: BTreeMap<i64, usize>,
    inverse_leaf_size: [f64; 3],
    bbox_min: [i64; 3],
    div_mul: [i64; 3],
}

/// floor(x) as i64. The caller bounds x via the bounding-box / int32 voxel-count guard.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "deliberate floor->i64; the caller's int32 voxel-count guard bounds the magnitude"
)]
fn floor_i64(x: f64) -> i64 {
    libm::floor(x) as i64
}
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "control-plane map build: nalgebra f64 covariance math and deliberate count casts"
)]
#[cfg_attr(
    not(feature = "wcet-trace"),
    expect(unused_variables, reason = "voxel id is retained only by trace builds")
)]
fn finalize_leaf(id: i64, accumulator: &Accumulator, eig_mult: f64) -> Option<Leaf> {
    let n_f = accumulator.n as f64;
    let mean = accumulator.sum / n_f;
    // C++ initializes cov_ to Identity, adding I/(n-1) to the sample covariance.
    let cov = (accumulator.sum_outer + Matrix3::identity() - accumulator.sum * mean.transpose())
        / (n_f - 1.0);
    let icov = compute_icov(&cov, eig_mult)?;
    let n_f32 = accumulator.n as f32;
    Some(Leaf {
        n: i32::try_from(accumulator.n).unwrap_or(i32::MAX),
        mean: [mean.x, mean.y, mean.z],
        icov,
        centroid: [
            accumulator.centroid_sum[0] / n_f32,
            accumulator.centroid_sum[1] / n_f32,
            accumulator.centroid_sum[2] / n_f32,
        ],
        #[cfg(feature = "wcet-trace")]
        trace_voxel_id: id,
        #[cfg(feature = "wcet-trace")]
        trace_grid_ordinal: 0,
    })
}

impl VoxelGrid {
    /// Voxel id for a point under this grid's bbox/leaf layout, or `None` on index overflow.
    #[allow(
        clippy::many_single_char_names,
        clippy::allow_attributes,
        reason = "x/y/z coordinates and a/b/c voxel-index lattice terms"
    )]
    fn leaf_id(&self, x: f64, y: f64, z: f64) -> Option<i64> {
        let i = floor_i64(x * self.inverse_leaf_size[0]).checked_sub(self.bbox_min[0])?;
        let j = floor_i64(y * self.inverse_leaf_size[1]).checked_sub(self.bbox_min[1])?;
        let k = floor_i64(z * self.inverse_leaf_size[2]).checked_sub(self.bbox_min[2])?;
        let a = i.checked_mul(self.div_mul[0])?;
        let b = j.checked_mul(self.div_mul[1])?;
        let c = k.checked_mul(self.div_mul[2])?;
        a.checked_add(b)?.checked_add(c)
    }

    /// Build a voxel grid from `points`, mirroring `MultiVoxelGridCovariance::apply_filter`.
    #[must_use]
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::allow_attributes,
        reason = "control-plane map build: nalgebra f64 matrix math only (all i64 voxel-index arithmetic is explicit checked_*); deliberate count casts; d ∈ 0..3 indexing of fixed [_; 3]"
    )]
    pub fn build(points: &[[f32; 3]], leaf_size: [f64; 3], min_points: i32, eig_mult: f64) -> Self {
        let inverse_leaf_size = [1.0 / leaf_size[0], 1.0 / leaf_size[1], 1.0 / leaf_size[2]];

        let empty = Self {
            leaves: Vec::new(),
            index: BTreeMap::new(),
            inverse_leaf_size,
            bbox_min: [0; 3],
            div_mul: [1, 1, 1],
        };

        // Bounding box over finite points.
        let mut min_p = [f64::INFINITY; 3];
        let mut max_p = [f64::NEG_INFINITY; 3];
        let mut any = false;
        for &[px, py, pz] in points {
            if !(px.is_finite() && py.is_finite() && pz.is_finite()) {
                continue;
            }
            let p = [f64::from(px), f64::from(py), f64::from(pz)];
            for d in 0..3 {
                if p[d] < min_p[d] {
                    min_p[d] = p[d];
                }
                if p[d] > max_p[d] {
                    max_p[d] = p[d];
                }
            }
            any = true;
        }
        if !any {
            return empty;
        }

        // Reject if the voxel count would overflow int32 (matches the C++ guard). All i64 span
        // math is checked_* (integer arithmetic must never rely on a lint suppression): an
        // overflowing span degrades to the same `empty` grid the voxel-count guard produces.
        let mut div_b = [0_i64; 3];
        let mut voxel_count: i64 = 1;
        for d in 0..3 {
            let hi = floor_i64(max_p[d] * inverse_leaf_size[d]);
            let lo = floor_i64(min_p[d] * inverse_leaf_size[d]);
            let Some(span) = hi.checked_sub(lo).and_then(|s| s.checked_add(1)) else {
                return empty;
            };
            div_b[d] = span;
            match voxel_count.checked_mul(span) {
                Some(v) if v <= i64::from(i32::MAX) => voxel_count = v,
                _ => return empty,
            }
        }

        let bbox_min = [
            floor_i64(min_p[0] * inverse_leaf_size[0]),
            floor_i64(min_p[1] * inverse_leaf_size[1]),
            floor_i64(min_p[2] * inverse_leaf_size[2]),
        ];
        let Some(div_mul_2) = div_b[0].checked_mul(div_b[1]) else {
            return empty;
        };
        let div_mul = [1, div_b[0], div_mul_2];

        let mut grid = Self {
            leaves: Vec::new(),
            index: BTreeMap::new(),
            inverse_leaf_size,
            bbox_min,
            div_mul,
        };

        // First pass: accumulate per leaf (ordered map mirrors std::map<int64, Leaf>).
        let mut acc: BTreeMap<i64, Accumulator> = BTreeMap::new();
        for &[px, py, pz] in points {
            if !(px.is_finite() && py.is_finite() && pz.is_finite()) {
                continue;
            }
            let (x, y, z) = (f64::from(px), f64::from(py), f64::from(pz));
            if let Some(id) = grid.leaf_id(x, y, z) {
                acc.entry(id)
                    .or_insert_with(Accumulator::new)
                    .add(Vector3::new(x, y, z), [px, py, pz]);
            }
        }

        let min_points = if min_points <= 0 {
            DEFAULT_MIN_POINTS_PER_VOXEL
        } else {
            min_points
        };

        // Second pass: finalize leaf params.
        for (id, a) in &acc {
            if a.n < i64::from(min_points) {
                continue;
            }
            if let Some(leaf) = finalize_leaf(*id, a, eig_mult) {
                let idx = grid.leaves.len();
                grid.leaves.push(leaf);
                grid.index.insert(*id, idx);
            }
        }
        grid
    }

    /// Leaf whose voxel contains `point`, if any (and it passed the point/eigenvalue filters).
    #[must_use]
    pub fn leaf_at(&self, point: [f32; 3]) -> Option<&Leaf> {
        let [px, py, pz] = point;
        if !(px.is_finite() && py.is_finite() && pz.is_finite()) {
            return None;
        }
        let id = self.leaf_id(f64::from(px), f64::from(py), f64::from(pz))?;
        self.index.get(&id).and_then(|&i| self.leaves.get(i))
    }
}

/// Eigenvalue-regularized inverse covariance (Magnusson eq 6.11), or `None` if the voxel is
/// degenerate (negative/zero eigenvalue or non-finite inverse). Mirrors `computeLeafParams`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "nalgebra f64 eigen/matrix math; order[..] ∈ 0..3 indexes the 3-element eigen results"
)]
fn compute_icov(cov: &Matrix3<f64>, eig_mult: f64) -> Option<[f64; 9]> {
    let se = cov.symmetric_eigen();
    // nalgebra does not guarantee ordering; sort ascending to match Eigen's SelfAdjointEigenSolver.
    let mut order = [0_usize, 1, 2];
    order.sort_by(|&i, &j| se.eigenvalues[i].total_cmp(&se.eigenvalues[j]));
    let mut evals = [
        se.eigenvalues[order[0]],
        se.eigenvalues[order[1]],
        se.eigenvalues[order[2]],
    ];
    let c0: Vector3<f64> = se.eigenvectors.column(order[0]).into_owned();
    let c1: Vector3<f64> = se.eigenvectors.column(order[1]).into_owned();
    let c2: Vector3<f64> = se.eigenvectors.column(order[2]).into_owned();
    let evecs = Matrix3::from_columns(&[c0, c1, c2]);

    if evals[0] < 0.0 || evals[1] < 0.0 || evals[2] <= 0.0 {
        return None;
    }

    let mut cov = *cov;
    let min_covar_eigvalue = eig_mult * evals[2];
    if evals[0] < min_covar_eigvalue {
        evals[0] = min_covar_eigvalue;
        if evals[1] < min_covar_eigvalue {
            evals[1] = min_covar_eigvalue;
        }
        let diag = Matrix3::from_diagonal(&Vector3::new(evals[0], evals[1], evals[2]));
        // Eigenvectors are orthonormal, so the inverse is the transpose.
        cov = evecs * diag * evecs.transpose();
    }

    let icov = cov.try_inverse()?;
    if icov.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Row-major.
    Some([
        icov.m11, icov.m12, icov.m13, icov.m21, icov.m22, icov.m23, icov.m31, icov.m32, icov.m33,
    ])
}

// ---- multi-grid map + kd-tree (the MultiVoxelGridCovariance equivalent) ----

/// Byte-id-keyed collection of per-cloud voxel grids plus a kd-tree over all voxel centroids for radius
/// search. Mirrors `MultiVoxelGridCovariance` (`sid_to_iid_`/`grid_list_` + `KdTreeFLANN`).
#[derive(Clone)]
pub struct VoxelGridMap {
    leaf_size: [f64; 3],
    min_points: i32,
    eig_mult: f64,
    grids: BTreeMap<Vec<u8>, VoxelGrid>,
    flat_leaves: Vec<Leaf>,
    kdtree: Option<KdTree>,
}
/// Failure while flattening tiles and finalizing a staged map for publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MapBuildError {
    /// A control-plane vector reservation failed.
    AllocationFailed,
    /// The total active-leaf count could not be represented.
    ArithmeticOverflow,
    /// Balanced-tree construction failed.
    KdTree(KdSearchError),
    /// The staged map contained more active Gaussian leaves than its admission bound.
    LeafLimitExceeded,
}

impl From<KdSearchError> for MapBuildError {
    fn from(value: KdSearchError) -> Self {
        Self::KdTree(value)
    }
}

impl VoxelGridMap {
    /// A map with no tiles.
    ///
    /// # Arguments
    /// * `leaf_size` — voxel edge lengths `[x, y, z]` in metres (usually the same value on all axes).
    /// * `min_points` — minimum points per voxel for it to contribute a Gaussian (C++ default 6).
    /// * `eig_mult` — eigenvalue-inflation multiplier conditioning each voxel covariance (C++ 0.01).
    #[must_use]
    pub fn new(leaf_size: [f64; 3], min_points: i32, eig_mult: f64) -> Self {
        Self {
            leaf_size,
            min_points,
            eig_mult,
            grids: BTreeMap::new(),
            flat_leaves: Vec::new(),
            kdtree: None,
        }
    }

    /// Build a grid from `points` and register it under the raw cell-id bytes, replacing any
    /// existing tile with the same id. Invalidates the kd-tree.
    ///
    /// # Arguments
    /// * `points` — the tile's map points (`[x, y, z]`, metres).
    /// * `id` — raw tile key; re-adding the same byte sequence replaces that tile.
    pub fn add_target(&mut self, points: &[[f32; 3]], id: &[u8]) {
        let grid = VoxelGrid::build(points, self.leaf_size, self.min_points, self.eig_mult);
        self.grids.insert(id.to_vec(), grid);
        self.invalidate();
    }

    /// Remove the tile registered under `id` (no-op if absent). Invalidates the kd-tree — call
    /// [`Self::create_kdtree`] before searching again.
    pub fn remove_target(&mut self, id: &[u8]) {
        self.grids.remove(id);
        self.invalidate();
    }

    /// Registered raw cell ids in canonical bytewise order.
    pub(crate) fn cell_ids(&self) -> Vec<Vec<u8>> {
        self.grids.keys().cloned().collect()
    }

    /// Whether any target grid is registered (the C++ `hasTarget`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }

    fn invalidate(&mut self) {
        self.kdtree = None;
        self.flat_leaves.clear();
    }

    /// Flatten all grids' leaves in canonical tile order and build the kd-tree over centroids.
    ///
    /// `max_leaves` is an admission check, not a reservation. Temporary vectors reserve the actual
    /// leaf count only. The flattened leaves and tree are replaced only after every check and build
    /// step succeeds; engine-level map updates call this on a private staging state and publish that
    /// state atomically.
    /// # Errors
    /// Returns [`MapBuildError::LeafLimitExceeded`] before tree construction when the map exceeds
    /// `max_leaves`, [`MapBuildError::ArithmeticOverflow`] on checked-count overflow,
    /// [`MapBuildError::AllocationFailed`] on reservation failure, or [`MapBuildError::KdTree`] for
    /// balanced-tree construction failures.
    pub fn try_create_kdtree(&mut self, max_leaves: usize) -> Result<usize, MapBuildError> {
        let total_leaves = self.grids.values().try_fold(0_usize, |total, grid| {
            total
                .checked_add(grid.leaves.len())
                .ok_or(MapBuildError::ArithmeticOverflow)
        })?;
        if total_leaves > max_leaves {
            return Err(MapBuildError::LeafLimitExceeded);
        }
        let mut centroids: Vec<[f32; 3]> = Vec::new();
        centroids
            .try_reserve_exact(total_leaves)
            .map_err(|_| MapBuildError::AllocationFailed)?;
        let mut flat: Vec<Leaf> = Vec::new();
        flat.try_reserve_exact(total_leaves)
            .map_err(|_| MapBuildError::AllocationFailed)?;
        for (grid_ordinal, grid) in self.grids.values().enumerate() {
            for leaf in &grid.leaves {
                #[cfg(not(feature = "wcet-trace"))]
                let _ = grid_ordinal;
                centroids.push(leaf.centroid);
                #[cfg(not(feature = "wcet-trace"))]
                let flat_leaf = leaf.clone();
                #[cfg(feature = "wcet-trace")]
                let mut flat_leaf = leaf.clone();
                #[cfg(feature = "wcet-trace")]
                {
                    flat_leaf.trace_grid_ordinal = u64::try_from(grid_ordinal).unwrap_or(u64::MAX);
                }
                flat.push(flat_leaf);
            }
        }
        let kdtree = KdTree::try_build(&centroids)?;
        self.kdtree = Some(kdtree);
        self.flat_leaves = flat;
        Ok(total_leaves)
    }

    /// Flat indices of leaves whose centroid is within `radius` of `point` (needs `create_kdtree`).
    ///
    /// # Arguments
    /// * `point` — query point (`[x, y, z]`, metres).
    /// * `radius` — search radius in metres.
    /// * `max_nn` — cap on retained neighbors; a further match sets the returned outcome's
    ///   `result_limit_exceeded` field. Zero means unlimited.
    /// * `out` — matching flat leaf indices are appended to this buffer (index into [`Self::leaf`]);
    ///   clear and pre-reserve it before the call for allocation-free use. No-op if the kd-tree is
    ///   not built.
    /// # Errors
    /// Returns [`MapBuildError::KdTree`] containing `KdSearchError::StackCapacityExceeded` if
    /// traversal exceeds the fixed stack, or `KdSearchError::ArithmeticOverflow` if a work counter
    /// cannot be represented.
    pub fn radius_search(
        &self,
        point: [f32; 3],
        radius: f64,
        max_nn: usize,
        out: &mut Vec<usize>,
    ) -> Result<KdSearchOutcome, KdSearchError> {
        match &self.kdtree {
            Some(kt) => kt.radius_search(&point, radius, max_nn, out),
            None => Ok(KdSearchOutcome::default()),
        }
    }

    /// [`Self::radius_search`] that also returns the number of kd-tree nodes visited — the
    /// deterministic traversal-cost counter for the WCET analysis (`plan/ndt_wcet.md`).
    #[cfg(feature = "wcet-count")]
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn radius_search_counted(
        &self,
        point: [f32; 3],
        radius: f64,
        max_nn: usize,
        out: &mut Vec<usize>,
    ) -> Result<KdSearchOutcome, KdSearchError> {
        match &self.kdtree {
            Some(kt) => kt.radius_search_counted(&point, radius, max_nn, out),
            None => Ok(KdSearchOutcome::default()),
        }
    }

    /// Number of searchable leaves (== kd-tree node count once built) — the analytic per-query
    /// traversal bound used by the WCET property checks.
    #[must_use]
    pub fn num_leaves(&self) -> usize {
        self.flat_leaves.len()
    }

    /// The leaf at flat index `idx` (as returned by [`Self::radius_search`]), or `None` if out of
    /// range. Indices are only valid until the next [`Self::create_kdtree`].
    #[must_use]
    pub fn leaf(&self, idx: usize) -> Option<&Leaf> {
        self.flat_leaves.get(idx)
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::unreadable_literal,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    unsafe_code,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    // A single dense voxel: mean is the point centroid; the inverse covariance is finite.
    #[test]
    fn single_voxel_mean_matches_brute_force() {
        // All within voxel (10, 20, 30) for leaf_size 1.0 (each coord stays in [k, k+1)).
        let pts: alloc::vec::Vec<[f32; 3]> = (0..20)
            .map(|i| {
                let f = i as f32 * 0.01;
                [10.0 + f, 20.0 + f, 30.0 + 0.5 * f]
            })
            .collect();
        let grid = VoxelGrid::build(&pts, [1.0, 1.0, 1.0], 6, 0.01);
        let leaf = grid.leaf_at([10.0, 20.0, 30.0]).expect("leaf");
        // Brute-force mean over all points (they all fall in one voxel given leaf_size 1.0).
        let mut m = [0.0_f64; 3];
        for p in &pts {
            for d in 0..3 {
                m[d] += f64::from(p[d]);
            }
        }
        for d in 0..3 {
            m[d] /= pts.len() as f64;
            assert!((leaf.mean[d] - m[d]).abs() < 1e-9);
        }
        assert_eq!(leaf.n, 20);
    }

    #[test]
    fn sparse_voxel_is_dropped() {
        let pts = [[0.0_f32, 0.0, 0.0], [0.1, 0.1, 0.1]]; // only 2 points < min 6
        let grid = VoxelGrid::build(&pts, [1.0, 1.0, 1.0], 6, 0.01);
        assert!(grid.leaf_at([0.0, 0.0, 0.0]).is_none());
    }

    // Robustness against malformed sensor input: non-finite points (NaN/inf, common LiDAR
    // dropouts) must be excluded from the bbox and accumulation, `min_points <= 0` must fall
    // back to the default threshold, and a non-finite query must return None (never index a leaf).
    #[test]
    fn non_finite_points_filtered_and_min_points_defaults() {
        let mut pts: alloc::vec::Vec<[f32; 3]> = (0..6)
            .map(|i| {
                let f = i as f32 * 0.01;
                [0.4 + f, 0.4 + f, 0.4 + f]
            })
            .collect();
        pts.push([f32::NAN, 0.0, 0.0]); // skipped in both the bbox and accumulation passes
        pts.push([f32::INFINITY, 0.0, 0.0]);

        // min_points = 0 -> default (6); the voxel has exactly 6 finite points -> kept.
        let grid = VoxelGrid::build(&pts, [1.0, 1.0, 1.0], 0, 0.01);
        let leaf = grid
            .leaf_at([0.4, 0.4, 0.4])
            .expect("finite cluster kept at default min_points");
        assert_eq!(leaf.n, 6, "non-finite points must not be counted");

        // Brute-force mean over the 6 finite points only.
        let mut m = 0.0_f64;
        for i in 0..6 {
            m += f64::from(0.4 + i as f32 * 0.01);
        }
        m /= 6.0;
        assert!(
            (leaf.mean[0] - m).abs() < 1e-6,
            "mean corrupted by non-finite points"
        );

        // A non-finite query must be rejected, not turned into a garbage voxel index.
        assert!(grid.leaf_at([f32::NAN, 0.4, 0.4]).is_none());
    }

    // Well-conditioned voxel: leaf icov must equal the brute-force sample-covariance inverse.
    // Heavy (50-point eigen + matrix inverse) — skipped under Miri to keep the unsafe-UB run fast.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn icov_matches_brute_force_inverse() {
        let mut pts: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::new();
        // deterministic spread around (5,5,5), all inside voxel [0,10) for leaf_size 10.
        for i in 0..50_i32 {
            let f = i as f32;
            pts.push([
                5.0 + 0.2 * (f * 0.7).sin(),
                5.0 + 0.15 * (f * 1.3).cos(),
                5.0 + 0.1 * (f * 0.31).sin(),
            ]);
        }
        let grid = VoxelGrid::build(&pts, [10.0, 10.0, 10.0], 6, 0.01);
        let leaf = grid.leaf_at([5.0, 5.0, 5.0]).expect("leaf");

        // Brute-force sample covariance + inverse.
        let n = pts.len() as f64;
        let mut mean = Vector3::<f64>::zeros();
        for p in &pts {
            mean += Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        }
        mean /= n;
        let mut cov = Matrix3::<f64>::zeros();
        for p in &pts {
            let d = Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])) - mean;
            cov += d * d.transpose();
        }
        // Match the C++ Identity-initialized covariance accumulator: sample_cov + I/(n-1).
        cov += Matrix3::identity();
        cov /= n - 1.0;
        let bf_icov = cov.try_inverse().expect("inverse");
        let bf = [
            bf_icov.m11,
            bf_icov.m12,
            bf_icov.m13,
            bf_icov.m21,
            bf_icov.m22,
            bf_icov.m23,
            bf_icov.m31,
            bf_icov.m32,
            bf_icov.m33,
        ];
        for k in 0..9 {
            assert!(
                (leaf.icov[k] - bf[k]).abs() <= 1e-6 * bf[k].abs() + 1e-9,
                "k={k} {} vs {}",
                leaf.icov[k],
                bf[k]
            );
        }
    }

    #[test]
    fn empty_cloud_builds_empty_grid() {
        let grid = VoxelGrid::build(&[], [1.0, 1.0, 1.0], 6, 0.01);
        assert!(grid.leaf_at([0.0, 0.0, 0.0]).is_none());
    }

    // 8 points packed inside the single voxel containing (cx,cy,cz) for leaf_size 2.0.
    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> alloc::vec::Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    // ---- VoxelGridMap: public API + state transitions ----

    #[test]
    fn map_add_create_then_radius_search_finds_cluster() {
        let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
        map.add_target(&dense_cluster(1.0, 1.0, 1.0), b"0");
        map.add_target(&dense_cluster(21.0, 1.0, 1.0), b"1");
        map.try_create_kdtree(418_000).expect("build kd-tree");

        let mut hits = alloc::vec::Vec::new();
        map.radius_search([1.0, 1.0, 1.0], 1.5, 0, &mut hits)
            .expect("radius search");
        assert_eq!(hits.len(), 1, "exactly one leaf near cluster A");
        let leaf = map.leaf(hits[0]).expect("leaf");
        assert!((leaf.mean[0] - 1.07).abs() < 0.2 && (leaf.mean[1] - 0.93).abs() < 0.2);

        // A query far from every centroid finds nothing.
        let mut none = alloc::vec::Vec::new();
        map.radius_search([100.0, 100.0, 100.0], 1.5, 0, &mut none)
            .expect("radius search");
        assert!(none.is_empty());
    }

    #[test]
    fn map_remove_target_drops_its_leaves() {
        let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
        map.add_target(&dense_cluster(1.0, 1.0, 1.0), b"0");
        map.add_target(&dense_cluster(21.0, 1.0, 1.0), b"1");
        map.remove_target(b"0");
        map.try_create_kdtree(418_000).expect("build kd-tree");

        let mut a = alloc::vec::Vec::new();
        map.radius_search([1.0, 1.0, 1.0], 1.5, 0, &mut a)
            .expect("radius search");
        assert!(a.is_empty(), "removed grid's leaf must be gone");
        let mut b = alloc::vec::Vec::new();
        map.radius_search([21.0, 1.0, 1.0], 1.5, 0, &mut b)
            .expect("radius search");
        assert_eq!(b.len(), 1, "remaining grid still searchable");
    }

    #[test]
    fn map_search_before_create_kdtree_is_empty() {
        let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
        map.add_target(&dense_cluster(1.0, 1.0, 1.0), b"0");
        // No create_kdtree() yet.
        let mut hits = alloc::vec::Vec::new();
        map.radius_search([1.0, 1.0, 1.0], 1.5, 0, &mut hits)
            .expect("radius search");
        assert!(hits.is_empty());
        assert!(map.leaf(0).is_none(), "no flat leaves before create_kdtree");
    }

    // Reference-model property: kd-tree radius search over the map == brute force over flat leaves.
    // Heavy (9 clusters of eigen + 40 queries) — skipped under Miri to keep the unsafe-UB run fast.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn map_radius_search_matches_brute_force() {
        let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
        let mut id = 0_u64;
        for i in 0..3 {
            for j in 0..3 {
                let (cx, cy) = ((3 * i + 1) as f32, (3 * j + 1) as f32);
                map.add_target(&dense_cluster(cx, cy, 1.0), &id.to_be_bytes());
                id = id.checked_add(1).expect("nine test tiles fit in u64");
            }
        }
        map.try_create_kdtree(418_000).expect("build kd-tree");

        // Deterministic queries spanning the populated region.
        let mut state = 0x9E37_79B9_u64;
        for _ in 0..40 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let qx = ((state >> 33) % 1000) as f32 / 100.0; // [0,10)
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let qy = ((state >> 33) % 1000) as f32 / 100.0;
            let radius = 2.5_f64;

            let mut got = alloc::vec::Vec::new();
            map.radius_search([qx, qy, 1.0], radius, 0, &mut got)
                .expect("radius search");
            got.sort_unstable();

            let r2 = radius * radius;
            let mut expected: alloc::vec::Vec<usize> = (0..map.flat_leaves.len())
                .filter(|&k| {
                    let c = map.flat_leaves[k].centroid;
                    let (dx, dy, dz) = (
                        f64::from(c[0]) - f64::from(qx),
                        f64::from(c[1]) - f64::from(qy),
                        f64::from(c[2]) - 1.0,
                    );
                    (dx * dx) + (dy * dy) + (dz * dz) <= r2
                })
                .collect();
            expected.sort_unstable();
            assert_eq!(got, expected);
        }
    }

    // NDT eigenvalue regularization (compute_icov clamp branch): for a strongly anisotropic
    // (near-planar) voxel the smallest covariance eigenvalue is clamped UP to eig_mult*largest.
    // Oracle is the *invariant* this enforces, not a re-implementation of the clamp: after
    // regularization the largest icov eigenvalue == 1 / (eig_mult * largest_cov_eigenvalue).
    // A bug that skipped clamping would leave a far larger (1/tiny) eigenvalue -> test fails.
    #[cfg_attr(miri, ignore)] // eigendecomposition is heavy under Miri
    #[test]
    fn icov_eigenvalue_regularization_clamps_smallest() {
        // Near-planar cluster inside one voxel: wide spread in x, less in y, zero in z.
        let mut pts: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::new();
        for i in 0..60_i32 {
            let f = i as f32;
            pts.push([
                50.0 + 4.0 * (f * 0.7).sin(),
                50.0 + 1.0 * (f * 1.3).cos(),
                50.0,
            ]);
        }
        let eig_mult = 0.01_f64;
        let grid = VoxelGrid::build(&pts, [100.0, 100.0, 100.0], 6, eig_mult);
        let leaf = grid.leaf_at([50.0, 50.0, 50.0]).expect("leaf");

        // Independent brute-force covariance (Identity-init like C++: +I then /(n-1)).
        let n = pts.len() as f64;
        let mut mean = Vector3::<f64>::zeros();
        for p in &pts {
            mean += Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        }
        mean /= n;
        let mut cov = Matrix3::<f64>::zeros();
        for p in &pts {
            let d = Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])) - mean;
            cov += d * d.transpose();
        }
        cov += Matrix3::identity();
        cov /= n - 1.0;

        let mut evals: alloc::vec::Vec<f64> =
            cov.symmetric_eigen().eigenvalues.iter().copied().collect();
        evals.sort_by(f64::total_cmp);
        let (smallest, largest) = (evals[0], evals[2]);
        // Precondition: clamping must actually fire (else the test is vacuous).
        assert!(
            smallest < eig_mult * largest,
            "cluster must trigger clamping: {smallest} !< {}",
            eig_mult * largest
        );

        let icov = Matrix3::new(
            leaf.icov[0],
            leaf.icov[1],
            leaf.icov[2],
            leaf.icov[3],
            leaf.icov[4],
            leaf.icov[5],
            leaf.icov[6],
            leaf.icov[7],
            leaf.icov[8],
        );
        let max_icov_eig = icov
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let expected = 1.0 / (eig_mult * largest);
        assert!(
            (max_icov_eig - expected).abs() <= 1e-6 * expected,
            "regularized icov max eigenvalue {max_icov_eig} != 1/(eig_mult*largest) {expected}"
        );
        // And it is far below the un-regularized 1/smallest, proving the clamp happened.
        assert!(
            max_icov_eig < 0.5 / smallest,
            "icov eigenvalue was not clamped down"
        );
    }
}
