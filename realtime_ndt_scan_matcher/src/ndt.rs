//! NDT derivative assembly, ported from `multigrid_ndt_omp_impl.hpp`:
//! - [`compute_derivatives`] — the source-point loop that queries the target map and accumulates the
//!   score, 6-gradient, and 6x6 Hessian (`computeDerivatives`, lines 378-533, incl. the
//!   regularization term 486-519).
//! - [`transformation_probability`] — score-only loop (`calculateTransformationProbability`, 1100).
//! - [`nearest_voxel_transformation_likelihood`] — per-point max-cell-score loop
//!   (`calculateNearestVoxelTransformationLikelihood`, 1153).
//!
//! Serial (the `ParReduce` parallel backends are added separately as a pure performance change).
//! Per the zero-alloc policy, the per-point loop reuses an [`AlignWorkspace`] buffer (`clear()` keeps
//! capacity) so steady state allocates nothing. `no_std` + `alloc`; the per-point math is `f64`,
//! point clouds are `f32` (matching the C++ `PointXYZ` → `Vector3d` promotion).

// Numeric kernel: f64 matrix/scalar operators (overflow lint), fixed-array / 0..6 indexing, f32<->f64
// casts inherent to point-cloud math + the regularization (mirrors C++ float arithmetic), and x/y/z
// single-char geometry names.
// Suppressions are scoped per-function (no module-wide `#![allow]`); rationale per the comment above.

use alloc::vec::Vec;

use nalgebra::{Matrix3, Matrix4, Matrix6, SVD, Vector3, Vector6};

use crate::derivatives::{
    AngleDerivatives, compute_angle_derivatives, compute_point_derivatives, update_derivatives,
};
use crate::kdtree::KdSearchError;
use crate::transform::{
    GaussConstants, gauss_constants, matrix_to_euler, se3_matrix_f32, transform_cloud_by_matrix,
    transform_cloud_f32,
};
use crate::voxel_grid::VoxelGridMap;

/// Per-point neighbor cap for the RT-critical radius search and the `K` term of the structural work
/// envelope. The first 64 in traversal order are retained. Search continues only until it finds a
/// 65th in-radius leaf, then returns a non-fatal diagnostic; the engine forces that alignment to
/// non-converged. The cap bounds kernel evaluations, not kd traversal, whose per-query bound is the
/// configured active-leaf limit.
pub const MAX_NEIGHBORS: usize = 64;

/// Fatal failure of the bounded alignment kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlignError {
    /// A checked integer operation or work-counter product overflowed.
    ArithmeticOverflow,
    /// The balanced tree or iterative search exceeded its fixed 64-entry stack contract.
    KdStackCapacityExceeded,
    /// A parameter, guess, source point, or result contained NaN or infinity.
    NonFiniteValue,
    /// Active alignment parameters exceeded the configured `Imax` or the hard limit of 30.
    IterationLimitExceeded,
    /// A work envelope had a zero point/leaf bound or an iteration bound outside `0..=30`.
    InvalidLimits,
    /// The source cloud exceeded the engine's configured `Pmax`.
    SourcePointLimitExceeded,
    /// Allocation or kd-tree construction failed while finalizing a staged map.
    MapBuildFailed,
    /// A staged or committed map exceeded the engine's configured `Lmax`.
    MapLeafLimitExceeded,
    /// Caller-owned scratch was undersized or could not be reserved.
    WorkspaceCapacityExceeded,
}

const fn map_kd_search_error(error: KdSearchError) -> AlignError {
    match error {
        KdSearchError::ArithmeticOverflow => AlignError::ArithmeticOverflow,
        KdSearchError::AllocationFailed | KdSearchError::StackCapacityExceeded => {
            AlignError::KdStackCapacityExceeded
        }
    }
}

/// Non-fatal observations collected without allocation or logging in the align kernel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AlignDiagnostics {
    /// Whether any point query found a 65th in-radius leaf after retaining 64.
    pub neighbor_limit_exceeded: bool,
    /// Number of point queries in this alignment that exceeded the neighbor cap.
    pub neighbor_limit_exceeded_queries: u64,
}

impl AlignDiagnostics {
    fn record_neighbor_limit_exceeded(&mut self) -> Result<(), AlignError> {
        self.neighbor_limit_exceeded = true;
        self.neighbor_limit_exceeded_queries = self
            .neighbor_limit_exceeded_queries
            .checked_add(1)
            .ok_or(AlignError::ArithmeticOverflow)?;
        Ok(())
    }
}

/// Reusable scratch buffers for the per-frame derivative pass and align loop. Hold one per concurrent
/// alignment and reuse it across frames. Construct it with [`AlignWorkspace::try_with_capacity`]
/// before entering the real-time path; alignment rejects rather than grows undersized scratch. The
/// serial backend is allocation-free inside a correctly sized workspace. The parallel backend may
/// allocate worker-local buffers and is outside that claim.
#[derive(Debug)]
pub struct AlignWorkspace {
    neighbor_idx: Vec<usize>,
    trans_cloud: Vec<[f32; 3]>,
    max_points: usize,
    /// Diagnostics for the current/last align.
    pub diagnostics: AlignDiagnostics,
    /// Per-point contributions for the parallel backend (reused across frames; an order-preserving
    /// `collect` fills it). Only the `parallel` feature touches it — the serial backend
    /// stays allocation-free.
    #[cfg(feature = "parallel")]
    contribs: Vec<PointContribution>,
    /// Algorithmic-cost counters for the current/last [`align`] (reset at each align start).
    #[cfg(feature = "wcet-count")]
    pub counters: crate::wcet::WcetCounters,
    /// Per-pass records for cross-language work-shape analysis (reset per align;
    /// serial backend only — the parallel backend poisons it).
    #[cfg(feature = "wcet-trace")]
    pub trace: crate::wcet::AlignTrace,
}

impl AlignWorkspace {
    /// A workspace pre-reserved for clouds of up to `max_points`. In the serial backend, no frame
    /// allocates, including the first.
    /// # Errors
    /// Returns [`AlignError::WorkspaceCapacityExceeded`] if a reservation fails.
    pub fn try_with_capacity(max_points: usize) -> Result<Self, AlignError> {
        let mut neighbor_idx = Vec::new();
        neighbor_idx
            .try_reserve_exact(MAX_NEIGHBORS)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        let mut trans_cloud = Vec::new();
        trans_cloud
            .try_reserve_exact(max_points)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        #[cfg(feature = "parallel")]
        let mut contribs = Vec::new();
        #[cfg(feature = "parallel")]
        contribs
            .try_reserve_exact(max_points)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        Ok(Self {
            neighbor_idx,
            trans_cloud,
            max_points,
            diagnostics: AlignDiagnostics::default(),
            #[cfg(feature = "parallel")]
            contribs,
            #[cfg(feature = "wcet-count")]
            counters: crate::wcet::WcetCounters::new(),
            #[cfg(feature = "wcet-trace")]
            trace: crate::wcet::AlignTrace::new(),
        })
    }

    /// Heap payload bytes reserved by this workspace (allocator metadata excluded).
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn allocated_payload_bytes(&self) -> Result<usize, AlignError> {
        let mut bytes = self
            .neighbor_idx
            .capacity()
            .checked_mul(core::mem::size_of::<usize>())
            .ok_or(AlignError::ArithmeticOverflow)?;
        bytes = bytes
            .checked_add(
                self.trans_cloud
                    .capacity()
                    .checked_mul(core::mem::size_of::<[f32; 3]>())
                    .ok_or(AlignError::ArithmeticOverflow)?,
            )
            .ok_or(AlignError::ArithmeticOverflow)?;
        #[cfg(feature = "parallel")]
        {
            bytes = bytes
                .checked_add(
                    self.contribs
                        .capacity()
                        .checked_mul(core::mem::size_of::<PointContribution>())
                        .ok_or(AlignError::ArithmeticOverflow)?,
                )
                .ok_or(AlignError::ArithmeticOverflow)?;
        }
        Ok(bytes)
    }
}

/// Optional longitudinal-distance regularization toward a reference pose
/// (`setRegularizationPose`). `scale_factor == 0` (the Autoware default) makes it a no-op.
#[derive(Clone, Copy, Debug)]
pub struct Regularization {
    /// Reference pose translation `(x, y)` in the map frame.
    pub pose_xy: [f32; 2],
    pub scale_factor: f32,
}

/// Result of one derivative pass at a pose.
#[derive(Clone, Copy, Debug)]
pub struct Derivatives {
    pub score: f64,
    pub gradient: Vector6<f64>,
    pub hessian: Matrix6<f64>,
    pub nearest_voxel_likelihood: f64,
    pub transform_probability: f64,
}

/// Per-align scoring configuration, fixed across the iteration loop (the C++ NDT object holds these
/// as members): the neighbor-search `resolution`, the Gaussian fitting `gauss` constants, and the
/// optional `reg`ularization.
#[derive(Clone, Copy, Debug)]
pub struct ScoreConfig<'a> {
    pub resolution: f64,
    pub gauss: &'a GaussConstants,
    pub reg: Option<&'a Regularization>,
}

/// Row-major `[f64; 9]` (a `Leaf.icov`) → `Matrix3`.
fn mat3(icov: &[f64; 9]) -> Matrix3<f64> {
    Matrix3::new(
        icov[0], icov[1], icov[2], icov[3], icov[4], icov[5], icov[6], icov[7], icov[8],
    )
}

/// `[f32; 3]` point → `Vector3<f64>` (the C++ `Vector3d(pt.x, pt.y, pt.z)` promotion).
fn vec3(p: [f32; 3]) -> Vector3<f64> {
    Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))
}

/// Per-cell NDT score `-d1 · exp(-d2/2 · x_transᵀ Σ⁻¹ x_trans)` (eq. 6.9). `x_trans` is already
/// `transformed_point − cell.mean`. Shared by the score-only loops; matches the score
/// `update_derivatives` computes internally.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix product (quadratic form)"
)]
fn score_increment(x_trans: &Vector3<f64>, c_inv: &Matrix3<f64>, g: &GaussConstants) -> f64 {
    // x_transᵀ Σ⁻¹ x_trans as a scalar (avoids indexing the 1x1 product matrix).
    let quad = x_trans.dot(&(c_inv * x_trans));
    -g.d1 * libm::exp(-g.d2 * quad * 0.5)
}

/// One source point's independent contribution to the derivative reduction: the score, the local
/// 6-gradient and 6x6 Hessian (folded over the point's neighbor cells from zero), the per-point max
/// cell score (`nearest`), and the neighbor count. Computed independently of every other point, so
/// the serial and parallel backends fold these in identical point-index order and agree
/// **bit-for-bit** — parallel is a pure performance option, never a numeric change.
#[derive(Clone, Copy, Debug)]
struct PointContribution {
    score: f64,
    gradient: Vector6<f64>,
    hessian: Matrix6<f64>,
    nearest: f64,
    neighborhood: usize,
    found: bool,
    neighbor_limit_exceeded: bool,
    /// kd-tree nodes visited by this point's neighbor search (WCET traversal counter).
    #[cfg(feature = "wcet-count")]
    kd_nodes: u64,
}

/// Compute one source point's [`PointContribution`]. `nbr` is caller-owned neighbor-search scratch
/// (the serial backend reuses one buffer; the parallel backend gives each rayon worker its own).
/// Reads the map and writes only `nbr` + the return value, so it is safe to run concurrently for
/// distinct points.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "nalgebra fixed-size f64 matrix math"
)]
fn point_contribution(
    map: &VoxelGridMap,
    sp: [f32; 3],
    tp: [f32; 3],
    ad: &AngleDerivatives,
    cfg: &ScoreConfig,
    nbr: &mut Vec<usize>,
) -> Result<PointContribution, AlignError> {
    nbr.clear();
    #[cfg(feature = "wcet-count")]
    let search = map
        .radius_search_counted(tp, cfg.resolution, MAX_NEIGHBORS, nbr)
        .map_err(map_kd_search_error)?;
    #[cfg(not(feature = "wcet-count"))]
    let search = map
        .radius_search(tp, cfg.resolution, MAX_NEIGHBORS, nbr)
        .map_err(map_kd_search_error)?;
    if nbr.is_empty() {
        return Ok(PointContribution {
            score: 0.0,
            gradient: Vector6::zeros(),
            hessian: Matrix6::zeros(),
            nearest: 0.0,
            neighborhood: 0,
            found: false,
            neighbor_limit_exceeded: search.result_limit_exceeded,
            #[cfg(feature = "wcet-count")]
            kd_nodes: search.nodes_visited,
        });
    }
    let pd = compute_point_derivatives(&vec3(sp), ad);
    let x_trans = vec3(tp);
    let mut gradient = Vector6::zeros();
    let mut hessian = Matrix6::zeros();
    let mut score = 0.0_f64;
    let mut nearest = 0.0_f64;
    for &li in nbr.iter() {
        if let Some(leaf) = map.leaf(li) {
            let mean = Vector3::new(leaf.mean[0], leaf.mean[1], leaf.mean[2]);
            let c_inv = mat3(&leaf.icov);
            let s = update_derivatives(
                &(x_trans - mean),
                &c_inv,
                &pd,
                cfg.gauss,
                &mut gradient,
                &mut hessian,
            );
            score += s;
            if s > nearest {
                nearest = s;
            }
        }
    }
    Ok(PointContribution {
        score,
        gradient,
        hessian,
        nearest,
        neighborhood: nbr.len(),
        found: true,
        neighbor_limit_exceeded: search.result_limit_exceeded,
        #[cfg(feature = "wcet-count")]
        kd_nodes: search.nodes_visited,
    })
}

/// Running reduction of per-point contributions (the C++ thread accumulators). Both backends fold
/// into this in point-index order, so the assembled result is independent of the backend.
struct Reduction {
    score: f64,
    gradient: Vector6<f64>,
    hessian: Matrix6<f64>,
    nearest_voxel_score: f64,
    found: usize,
    total_neighborhood: usize,
}

impl Reduction {
    fn new() -> Self {
        Self {
            score: 0.0,
            gradient: Vector6::zeros(),
            hessian: Matrix6::zeros(),
            nearest_voxel_score: 0.0,
            found: 0,
            total_neighborhood: 0,
        }
    }

    #[allow(
        clippy::arithmetic_side_effects,
        clippy::allow_attributes,
        reason = "nalgebra fixed-size f64 matrix sums; integer counters use saturating_add"
    )]
    fn add(&mut self, c: &PointContribution) {
        self.score += c.score;
        self.gradient += c.gradient;
        self.hessian += c.hessian;
        self.nearest_voxel_score += c.nearest;
        if c.found {
            self.found = self.found.saturating_add(1);
        }
        self.total_neighborhood = self.total_neighborhood.saturating_add(c.neighborhood);
    }
}

/// Apply the optional regularization and assemble [`Derivatives`] from a completed reduction. Shared
/// by both backends; runs after the reduce, so it is backend-independent.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "numeric kernel: nalgebra fixed-size math + deliberate C++-parity f32 regularization casts"
)]
fn finalize(
    mut red: Reduction,
    p: &Vector6<f64>,
    cfg: &ScoreConfig,
    source_len: usize,
) -> Derivatives {
    if let Some(r) = cfg.reg {
        // Mirrors C++ 486-519: float arithmetic, with sin/cos computed in f64 then cast to f32
        // (C++ `static_cast<float>(sin(p(5,0)))`). No-op when scale_factor == 0.
        let dx = r.pose_xy[0] - p[0] as f32;
        let dy = r.pose_xy[1] - p[1] as f32;
        let sin_yaw = libm::sin(p[5]) as f32;
        let cos_yaw = libm::cos(p[5]) as f32;
        let longitudinal = dy * sin_yaw + dx * cos_yaw;
        let w = red.total_neighborhood as f32;
        let sf = r.scale_factor;

        red.score += f64::from(-sf * w * longitudinal * longitudinal);
        red.gradient[0] += f64::from(sf * w * 2.0 * cos_yaw * longitudinal);
        red.gradient[1] += f64::from(sf * w * 2.0 * sin_yaw * longitudinal);
        let h01 = f64::from(-sf * w * 2.0 * cos_yaw * sin_yaw);
        red.hessian[(0, 0)] += f64::from(-sf * w * 2.0 * cos_yaw * cos_yaw);
        red.hessian[(0, 1)] += h01;
        red.hessian[(1, 1)] += f64::from(-sf * w * 2.0 * sin_yaw * sin_yaw);
        red.hessian[(1, 0)] += h01;
    }

    let nearest_voxel_likelihood = if red.found != 0 {
        red.nearest_voxel_score / red.found as f64
    } else {
        0.0
    };
    let transform_probability = if source_len == 0 {
        0.0
    } else {
        red.score / source_len as f64
    };

    Derivatives {
        score: red.score,
        gradient: red.gradient,
        hessian: red.hessian,
        nearest_voxel_likelihood,
        transform_probability,
    }
}

/// Score + gradient + Hessian over the source cloud (`computeDerivatives`), **serial** backend.
/// `source` are the original points (transform derivatives); `trans_cloud` the same points already
/// transformed by the current pose (neighbor search + `x_trans`); `p` the 6-vector of that pose;
/// `cfg` the neighbor radius / Gaussian constants / optional regularization (fixed across the loop).
///
/// WCET contract (RT-critical; the predictable baseline; see `porting_notes/ndt_wcet_audit.md`):
/// `O(P · K)` where `P` = source points (caller-bounded by downsampling) and `K` = neighbors/point ≤
/// `MAX_NEIGHBORS` (the `radius_search` cap); reuses `ws.neighbor_idx` → **no allocation** (measured);
/// no panic/block/logging; per-point math is fixed-size `O(1)`. The map is read-only.
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn compute_derivatives(
    map: &VoxelGridMap,
    source: &[[f32; 3]],
    trans_cloud: &[[f32; 3]],
    p: &Vector6<f64>,
    cfg: &ScoreConfig,
    ws: &mut AlignWorkspace,
) -> Result<Derivatives, AlignError> {
    let ad = compute_angle_derivatives(p);
    let mut red = Reduction::new();
    #[cfg(feature = "wcet-count")]
    {
        ws.counters.derivative_passes = ws.counters.derivative_passes.saturating_add(1);
    }
    #[cfg(feature = "wcet-trace")]
    let mut pass = crate::wcet::PassTrace::new();
    // Per-point-local contributions, folded in point-index order (zip stops at the shorter cloud).
    for (&tp, &sp) in trans_cloud.iter().zip(source.iter()) {
        let c = point_contribution(map, sp, tp, &ad, cfg, &mut ws.neighbor_idx)?;
        if c.neighbor_limit_exceeded {
            ws.diagnostics.record_neighbor_limit_exceeded()?;
        }
        #[cfg(feature = "wcet-count")]
        {
            let k = u64::try_from(c.neighborhood).unwrap_or(u64::MAX);
            ws.counters.points_processed = ws.counters.points_processed.saturating_add(1);
            ws.counters.sum_neighbors = ws.counters.sum_neighbors.saturating_add(k);
            ws.counters.kd_nodes_visited = ws.counters.kd_nodes_visited.saturating_add(c.kd_nodes);
            ws.counters.max_neighbors = ws.counters.max_neighbors.max(k);
        }
        #[cfg(feature = "wcet-trace")]
        {
            pass.points = pass.points.saturating_add(1);
            pass.neighbors = pass
                .neighbors
                .saturating_add(u64::try_from(c.neighborhood).unwrap_or(u64::MAX));
            pass.kd_nodes = pass.kd_nodes.saturating_add(c.kd_nodes);
            trace_point(&mut pass, map, &mut ws.neighbor_idx);
        }
        red.add(&c);
    }
    let d = finalize(red, p, cfg, source.len());
    #[cfg(feature = "wcet-trace")]
    {
        trace_pass_tail(&mut pass, &d);
        ws.trace.push(pass);
    }
    Ok(d)
}

#[cfg(feature = "wcet-trace")]
fn trace_point(pass: &mut crate::wcet::PassTrace, map: &VoxelGridMap, neighbors: &mut [usize]) {
    use sha2::{Digest, Sha256};

    neighbors.sort_unstable_by_key(|idx| {
        map.leaf(*idx).map_or((u64::MAX, i64::MAX), |leaf| {
            (leaf.trace_grid_ordinal, leaf.trace_voxel_id)
        })
    });

    let mut shape = Sha256::new();
    shape.update(b"NDT-SHAPE-POINT-v1\0");
    shape.update(
        u64::try_from(neighbors.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    let mut payload = Sha256::new();
    payload.update(b"NDT-PAYLOAD-POINT-v1\0");
    payload.update(
        u64::try_from(neighbors.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for idx in neighbors {
        if let Some(leaf) = map.leaf(*idx) {
            let grid = leaf.trace_grid_ordinal.to_le_bytes();
            let voxel = leaf.trace_voxel_id.to_le_bytes();
            shape.update(grid);
            shape.update(voxel);
            payload.update(grid);
            payload.update(voxel);
            for value in leaf.mean.iter().chain(leaf.icov.iter()) {
                payload.update(value.to_bits().to_le_bytes());
            }
        }
    }
    let point_shape: [u8; 32] = shape.finalize().into();
    let point_payload: [u8; 32] = payload.finalize().into();
    pass.shape_digest = trace_chain(b"NDT-SHAPE-CHAIN-v1\0", &pass.shape_digest, &point_shape);
    pass.payload_digest = trace_chain(
        b"NDT-PAYLOAD-CHAIN-v1\0",
        &pass.payload_digest,
        &point_payload,
    );
}

#[cfg(feature = "wcet-trace")]
fn trace_chain(domain: &[u8], previous: &[u8; 32], point: &[u8; 32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(previous);
    hash.update(point);
    hash.finalize().into()
}

/// Close a pass record with the values handed to the Newton step (mirrors the C++
/// `ndt_trace::pass_end`): the score bits plus FNV-1a folds of the gradient (index order) and
/// Hessian (row-major (r, c) order) f64 bit patterns.
#[cfg(feature = "wcet-trace")]
#[expect(
    clippy::indexing_slicing,
    reason = "fixed-size nalgebra Vector6/Matrix6 indexing with constant-bounded loops"
)]
fn trace_pass_tail(pass: &mut crate::wcet::PassTrace, d: &Derivatives) {
    pass.score_bits = d.score.to_bits();
    for r in 0..6 {
        pass.grad_hash = crate::wcet::fnv1a_u64(pass.grad_hash, d.gradient[r].to_bits());
    }
    for r in 0..6 {
        for c in 0..6 {
            pass.hess_hash = crate::wcet::fnv1a_u64(pass.hess_hash, d.hessian[(r, c)].to_bits());
        }
    }
}

/// **Parallel** (rayon) backend for [`compute_derivatives`] — bit-identical to the serial version
/// (per-point contributions collected in point-index order, then folded in the same order). NOT the
/// WCET baseline: it allocates `ws.contribs` + per-worker neighbor buffers and adds scheduling
/// jitter, so it is a throughput option only. `align` selects it when `params.num_threads > 1`. It
/// runs on rayon's **process-global** thread pool; the pool's worker count is set separately (see
/// [`crate::init_thread_pool`] or `RAYON_NUM_THREADS`), not by `num_threads`.
#[cfg(feature = "parallel")]
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn compute_derivatives_parallel(
    map: &VoxelGridMap,
    source: &[[f32; 3]],
    trans_cloud: &[[f32; 3]],
    p: &Vector6<f64>,
    cfg: &ScoreConfig,
    ws: &mut AlignWorkspace,
) -> Result<Derivatives, AlignError> {
    use rayon::prelude::*;

    let ad = compute_angle_derivatives(p);
    // `IndexedParallelIterator` preserves order, so `contribs` is in point-index order and the fold
    // below matches the serial backend bit-for-bit. `map_init` gives each worker a reusable buffer.
    let contributions: Result<Vec<PointContribution>, AlignError> = trans_cloud
        .par_iter()
        .zip(source.par_iter())
        .map_init(
            || Vec::<usize>::with_capacity(MAX_NEIGHBORS),
            |nbr, (&tp, &sp)| point_contribution(map, sp, tp, &ad, cfg, nbr),
        )
        .collect();
    ws.contribs = contributions?;

    let mut red = Reduction::new();
    #[cfg(feature = "wcet-count")]
    let mut pass = crate::wcet::WcetCounters::new();
    for c in &ws.contribs {
        if c.neighbor_limit_exceeded {
            ws.diagnostics.record_neighbor_limit_exceeded()?;
        }
        #[cfg(feature = "wcet-count")]
        {
            let k = u64::try_from(c.neighborhood).unwrap_or(u64::MAX);
            pass.points_processed = pass.points_processed.saturating_add(1);
            pass.sum_neighbors = pass.sum_neighbors.saturating_add(k);
            pass.kd_nodes_visited = pass.kd_nodes_visited.saturating_add(c.kd_nodes);
            pass.max_neighbors = pass.max_neighbors.max(k);
        }
        red.add(c);
    }
    #[cfg(feature = "wcet-count")]
    {
        ws.counters.derivative_passes = ws.counters.derivative_passes.saturating_add(1);
        ws.counters.points_processed = ws
            .counters
            .points_processed
            .saturating_add(pass.points_processed);
        ws.counters.sum_neighbors = ws.counters.sum_neighbors.saturating_add(pass.sum_neighbors);
        ws.counters.kd_nodes_visited = ws
            .counters
            .kd_nodes_visited
            .saturating_add(pass.kd_nodes_visited);
        ws.counters.max_neighbors = ws.counters.max_neighbors.max(pass.max_neighbors);
    }
    #[cfg(feature = "wcet-trace")]
    {
        // The work-shape trace is defined on the serial backend only (per-pass neighbor-order
        // hashing); mark any align that went through here as invalid for comparison.
        ws.trace.poisoned = true;
    }
    Ok(finalize(red, p, cfg, source.len()))
}

/// Score-only transformation probability: sum over all neighbor cells of the per-cell score,
/// divided by the cloud size (`calculateTransformationProbability`).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "nalgebra f64 vector math; the score/len average is a deliberate usize->f64 cast"
)]
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn transformation_probability(
    map: &VoxelGridMap,
    trans_cloud: &[[f32; 3]],
    resolution: f64,
    gauss: &GaussConstants,
    ws: &mut AlignWorkspace,
) -> Result<f64, AlignError> {
    let mut score = 0.0_f64;
    for &tp in trans_cloud {
        ws.neighbor_idx.clear();
        let search = map
            .radius_search(tp, resolution, MAX_NEIGHBORS, &mut ws.neighbor_idx)
            .map_err(map_kd_search_error)?;
        if search.result_limit_exceeded {
            ws.diagnostics.record_neighbor_limit_exceeded()?;
        }
        if ws.neighbor_idx.is_empty() {
            continue;
        }
        let x_trans = vec3(tp);
        // Per-point-local sum, then add to the running score — matches `compute_derivatives`'
        // grouping so the two agree (and so serial == parallel stays bit-for-bit).
        let mut pt = 0.0_f64;
        for &li in &ws.neighbor_idx {
            if let Some(leaf) = map.leaf(li) {
                let mean = Vector3::new(leaf.mean[0], leaf.mean[1], leaf.mean[2]);
                pt += score_increment(&(x_trans - mean), &mat3(&leaf.icov), gauss);
            }
        }
        score += pt;
    }
    Ok(if trans_cloud.is_empty() {
        0.0
    } else {
        score / trans_cloud.len() as f64
    })
}

/// Per-point maximum cell score, averaged over points that found a neighbor
/// (`calculateNearestVoxelTransformationLikelihood`).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "nalgebra f64 vector math; the score/found average is a deliberate usize->f64 cast"
)]
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn nearest_voxel_transformation_likelihood(
    map: &VoxelGridMap,
    trans_cloud: &[[f32; 3]],
    resolution: f64,
    gauss: &GaussConstants,
    ws: &mut AlignWorkspace,
) -> Result<f64, AlignError> {
    let mut nearest_voxel_score = 0.0_f64;
    let mut found: usize = 0;
    for &tp in trans_cloud {
        ws.neighbor_idx.clear();
        let search = map
            .radius_search(tp, resolution, MAX_NEIGHBORS, &mut ws.neighbor_idx)
            .map_err(map_kd_search_error)?;
        if search.result_limit_exceeded {
            ws.diagnostics.record_neighbor_limit_exceeded()?;
        }
        if ws.neighbor_idx.is_empty() {
            continue;
        }
        let x_trans = vec3(tp);
        let mut nearest_pt = 0.0_f64;
        for &li in &ws.neighbor_idx {
            if let Some(leaf) = map.leaf(li) {
                let mean = Vector3::new(leaf.mean[0], leaf.mean[1], leaf.mean[2]);
                let s = score_increment(&(x_trans - mean), &mat3(&leaf.icov), gauss);
                if s > nearest_pt {
                    nearest_pt = s;
                }
            }
        }
        nearest_voxel_score += nearest_pt;
        found = found.checked_add(1).ok_or(AlignError::ArithmeticOverflow)?;
    }
    Ok(if found != 0 {
        nearest_voxel_score / found as f64
    } else {
        0.0
    })
}

/// Per-point nearest-voxel score (the C++ `calculateNearestVoxelScoreEachPoint` intensity): for each
/// point in `trans_cloud`, the max cell score over its neighbors, or `0.0` if it has no neighbor.
/// `out` is cleared and filled to `trans_cloud.len()`. The per-cell score is strictly positive
/// (`-d1 > 0`), so `out[i] > 0.0` iff point `i` found a neighbor — the C++ includes exactly those
/// points in its output cloud (with this value as the `PointXYZI` intensity).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "nalgebra f64 vector math; the per-point score is narrowed f64->f32 for the output"
)]
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn nearest_voxel_score_each_point(
    map: &VoxelGridMap,
    trans_cloud: &[[f32; 3]],
    resolution: f64,
    gauss: &GaussConstants,
    ws: &mut AlignWorkspace,
    out: &mut Vec<f32>,
) -> Result<(), AlignError> {
    out.clear();
    out.try_reserve_exact(trans_cloud.len())
        .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
    for &tp in trans_cloud {
        ws.neighbor_idx.clear();
        let search = map
            .radius_search(tp, resolution, MAX_NEIGHBORS, &mut ws.neighbor_idx)
            .map_err(map_kd_search_error)?;
        if search.result_limit_exceeded {
            ws.diagnostics.record_neighbor_limit_exceeded()?;
        }
        let mut nearest_pt = 0.0_f64;
        if !ws.neighbor_idx.is_empty() {
            let x_trans = vec3(tp);
            for &li in &ws.neighbor_idx {
                if let Some(leaf) = map.leaf(li) {
                    let mean = Vector3::new(leaf.mean[0], leaf.mean[1], leaf.mean[2]);
                    let s = score_increment(&(x_trans - mean), &mat3(&leaf.icov), gauss);
                    if s > nearest_pt {
                        nearest_pt = s;
                    }
                }
            }
        }
        out.push(nearest_pt as f32);
    }
    Ok(())
}

/// NDT parameters (`NdtParams` + the `outlier_ratio_` member, default 0.55).
///
/// # Examples
///
/// Start from [`Default`] and override only what you need (struct-update syntax):
///
/// ```
/// use realtime_ndt_scan_matcher::ndt::NdtParams;
///
/// let params = NdtParams {
///     resolution: 2.0,
///     max_iterations: 30,
///     ..NdtParams::default()
/// };
/// assert_eq!(params.outlier_ratio, 0.55); // untouched default
/// ```
#[derive(Clone, Copy, Debug)]
pub struct NdtParams {
    pub trans_epsilon: f64,
    pub step_size: f64,
    pub resolution: f64,
    pub max_iterations: i32,
    pub outlier_ratio: f64,
    pub regularization: Option<Regularization>,
    /// Selects the derivative-reduction backend (mirrors C++ `NdtParams.num_threads`): `> 1` uses
    /// the rayon backend when the `parallel` feature is on; otherwise the serial backend runs. The
    /// result is bit-identical either way, so this only trades WCET predictability for throughput.
    ///
    /// This is a **switch, not a worker count** — the rayon backend runs on the process-global pool,
    /// whose size is set separately via [`crate::init_thread_pool`] or `RAYON_NUM_THREADS` (the node
    /// handle sizes it from this same value at construction).
    pub num_threads: usize,
}

impl Default for NdtParams {
    fn default() -> Self {
        Self {
            trans_epsilon: 0.1,
            step_size: 0.1,
            resolution: 1.0,
            max_iterations: 35,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1,
        }
    }
}

/// Result of [`align`] (`NdtResult`). The `Vec`s are reused across calls ([`align`] clears them).
///
/// # Examples
///
/// Construct an empty slot with [`Default`], hand it to [`align`], then read the fields:
///
/// ```
/// use realtime_ndt_scan_matcher::ndt::AlignResult;
///
/// let out = AlignResult::default();
/// assert_eq!(out.iteration_num, 0);
/// assert!(out.transform_probability_array.is_empty());
/// ```
#[derive(Clone, Debug, Default)]
pub struct AlignResult {
    pub pose: Matrix4<f32>,
    pub transform_probability: f32,
    pub nearest_voxel_likelihood: f32,
    pub iteration_num: i32,
    pub hessian: Matrix6<f64>,
    pub transformation_array: Vec<Matrix4<f32>>,
    pub transform_probability_array: Vec<f32>,
    pub nearest_voxel_likelihood_array: Vec<f32>,
    /// Algorithmic-cost counters of this align (WCET analysis; serial + parallel backends).
    #[cfg(feature = "wcet-count")]
    pub counters: crate::wcet::WcetCounters,
    /// Per-pass work-shape trace of this align (analysis build; serial backend only).
    #[cfg(feature = "wcet-trace")]
    pub trace: crate::wcet::AlignTrace,
}

impl AlignResult {
    /// Allocate the per-pass result traces for an iteration cap.
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn try_with_capacity(max_iterations: usize) -> Result<Self, AlignError> {
        let cap = max_iterations
            .checked_add(1)
            .ok_or(AlignError::ArithmeticOverflow)?;
        let mut result = Self::default();
        result
            .transformation_array
            .try_reserve_exact(cap)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        result
            .transform_probability_array
            .try_reserve_exact(cap)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        result
            .nearest_voxel_likelihood_array
            .try_reserve_exact(cap)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        Ok(result)
    }
}

/// Solve `H · delta = -gradient` via SVD (mirrors the C++ `JacobiSVD(ComputeFullU|ComputeFullV)`).
/// Returns `None` if the solve fails (singular / non-finite).
fn svd_solve(hessian: &Matrix6<f64>, neg_gradient: &Vector6<f64>) -> Option<Vector6<f64>> {
    SVD::new(*hessian, true, true)
        .solve(neg_gradient, 1e-9)
        .ok()
}

/// Transform the cloud by `p` and compute the derivatives there. Moves `ws.trans_cloud` out for the
/// `compute_derivatives` call (so `ws.neighbor_idx` can be borrowed mutably without aliasing) and
/// puts it back — a buffer move, no allocation.
fn derivatives_at(
    map: &VoxelGridMap,
    source: &[[f32; 3]],
    p: &Vector6<f64>,
    cfg: &ScoreConfig,
    ws: &mut AlignWorkspace,
    parallel: bool,
) -> Result<Derivatives, AlignError> {
    let mut trans = core::mem::take(&mut ws.trans_cloud);
    transform_cloud_f32(p, source, &mut trans);
    let d = derivatives_from(map, source, &trans, p, cfg, ws, parallel);
    ws.trans_cloud = trans;
    d
}

/// [`derivatives_at`] over an already-transformed cloud (the initial pass transforms by the guess
/// MATRIX for C++ parity; see `align`).
fn derivatives_from(
    map: &VoxelGridMap,
    source: &[[f32; 3]],
    trans: &[[f32; 3]],
    p: &Vector6<f64>,
    cfg: &ScoreConfig,
    ws: &mut AlignWorkspace,
    parallel: bool,
) -> Result<Derivatives, AlignError> {
    // `parallel` selects the rayon backend (bit-identical to serial) when the feature is built.
    #[cfg(feature = "parallel")]
    let d = if parallel {
        compute_derivatives_parallel(map, source, trans, p, cfg, ws)
    } else {
        compute_derivatives(map, source, trans, p, cfg, ws)
    };
    #[cfg(not(feature = "parallel"))]
    let d = {
        let _ = parallel; // no rayon backend in this build; always serial
        compute_derivatives(map, source, trans, p, cfg, ws)
    };
    d
}

fn validate_align_inputs(
    source: &[[f32; 3]],
    guess: &Matrix4<f32>,
    params: &NdtParams,
    ws: &AlignWorkspace,
    out: &AlignResult,
) -> Result<(), AlignError> {
    if source.len() > ws.max_points {
        return Err(AlignError::SourcePointLimitExceeded);
    }
    if !(0..=30).contains(&params.max_iterations) {
        return Err(AlignError::IterationLimitExceeded);
    }
    let iterations =
        usize::try_from(params.max_iterations).map_err(|_| AlignError::ArithmeticOverflow)?;
    let result_capacity = iterations
        .checked_add(1)
        .ok_or(AlignError::ArithmeticOverflow)?;
    if ws.neighbor_idx.capacity() < MAX_NEIGHBORS
        || ws.trans_cloud.capacity() < source.len()
        || out.transformation_array.capacity() < result_capacity
        || out.transform_probability_array.capacity() < result_capacity
        || out.nearest_voxel_likelihood_array.capacity() < result_capacity
    {
        return Err(AlignError::WorkspaceCapacityExceeded);
    }
    let params_finite = params.trans_epsilon.is_finite()
        && params.step_size.is_finite()
        && params.resolution.is_finite()
        && params.outlier_ratio.is_finite();
    if !params_finite
        || !guess.iter().all(|value| value.is_finite())
        || !source.iter().flatten().all(|value| value.is_finite())
    {
        return Err(AlignError::NonFiniteValue);
    }
    Ok(())
}

/// NDT alignment (`align` / `computeTransformation` + the default-path `computeStepLengthMT`,
/// `use_line_search = false`). `guess` is the initial pose (the C++ `Matrix4f` guess); `source` is
/// the original cloud. Fills `out` (its `Vec`s are reused). The per-iteration cloud transform is f32
/// (C++ `Matrix4f` pipeline).
///
/// WCET contract (RT-critical; serial; see `porting_notes/ndt_wcet_audit.md`):
/// - Outer loop runs at most `params.max_iterations` times (static cap).
/// - Each iteration does one `compute_derivatives` pass (`O(P · K)`: `P` = source points,
///   `K` = neighbors/point) + one 6×6 SVD solve + one f32 cloud transform (`O(P)`).
/// - `K` (evaluated neighbors/point) ≤ [`MAX_NEIGHBORS`]. The caller bounds `P` through workspace
///   capacity; the engine additionally enforces its configured `Pmax`. Kd traversal is bounded by
///   the active map's leaf count, which the engine constrains with `Lmax` before publication.
/// - No panic (the crate's deny-`unwrap`/`expect`/`panic`/`indexing_slicing` lints); no blocking; no
///   logging/formatting; no user callbacks. Only fixed-width float math.
/// - The serial kernel performs **zero allocation** from the first frame when workspace and result
///   capacities cover the declared envelope; undersized buffers return
///   [`AlignError::WorkspaceCapacityExceeded`] (measured in
///   `tests/zero_alloc.rs`).
/// - A query that finds a 65th in-radius leaf retains the first 64 and records an
///   [`AlignDiagnostics`] event. The engine-level verdict wrapper turns this into non-convergence.
/// - rt-core (this runtime path) vs control-plane (map build/update) — the map is read-only here.
///
/// # Arguments
/// * `map` — the target voxel-grid map with its kd-tree already built ([`VoxelGridMap::try_create_kdtree`]).
/// * `source` — the sensor cloud to align, in the `base_link` frame (`[x, y, z]`, metres).
/// * `guess` — initial pose estimate as a 4×4 homogeneous transform (the C++ `Matrix4f` guess).
/// * `params` — alignment parameters (resolution, epsilon, step size, iteration cap, …).
/// * `ws` — caller-owned workspace preallocated with [`AlignWorkspace::try_with_capacity`].
/// * `out` — result slot; its `Vec` fields are cleared and refilled each call.
///
/// # Examples
///
/// ```
/// use realtime_ndt_scan_matcher::ndt::{align, AlignResult, AlignWorkspace, NdtParams};
/// use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;
/// use realtime_ndt_scan_matcher::nalgebra::Matrix4;
///
/// let mut map = VoxelGridMap::new([2.0; 3], 6, 0.01);
/// let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
/// map.add_target(&target, b"0");
/// map.try_create_kdtree(418_000).expect("build kd-tree");
///
/// let params = NdtParams { resolution: 2.0, max_iterations: 30, ..NdtParams::default() };
/// let mut ws = AlignWorkspace::try_with_capacity(target.len()).expect("reserve workspace");
/// let mut out = AlignResult::try_with_capacity(30).expect("reserve result");
/// align(&map, &target, &Matrix4::identity(), &params, &mut ws, &mut out).expect("align");
/// assert!(out.iteration_num >= 0);
/// ```
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::allow_attributes,
    reason = "numeric kernel: nalgebra fixed-size math + deliberate C++-parity f32/usize<->int casts"
)]
/// # Errors
/// Returns [`AlignError::SourcePointLimitExceeded`] or [`AlignError::WorkspaceCapacityExceeded`]
/// for undersized scratch, [`AlignError::IterationLimitExceeded`] outside `0..=30`,
/// [`AlignError::NonFiniteValue`] for non-finite inputs/results, and checked arithmetic or kd-stack
/// errors encountered by the bounded kernel. Singular Newton systems remain ordinary
/// non-convergence rather than errors.
pub fn align(
    map: &VoxelGridMap,
    source: &[[f32; 3]],
    guess: &Matrix4<f32>,
    params: &NdtParams,
    ws: &mut AlignWorkspace,
    out: &mut AlignResult,
) -> Result<AlignDiagnostics, AlignError> {
    validate_align_inputs(source, guess, params, ws, out)?;
    ws.diagnostics = AlignDiagnostics::default();
    let gauss = gauss_constants(params.outlier_ratio, params.resolution);
    let reg = params.regularization.as_ref();
    let cfg = ScoreConfig {
        resolution: params.resolution,
        gauss: &gauss,
        reg,
    };
    #[cfg(feature = "wcet-count")]
    {
        ws.counters = crate::wcet::WcetCounters::new();
    }
    #[cfg(feature = "wcet-trace")]
    {
        ws.trace = crate::wcet::AlignTrace::new();
    }
    // Use the rayon backend when the caller asks for >1 thread (bit-identical to serial; ignored
    // when the `parallel` feature is off).
    let parallel = params.num_threads > 1;
    let len = source.len();

    // Capacities were validated above; clearing retains them for this frame.
    out.transformation_array.clear();
    out.transform_probability_array.clear();
    out.nearest_voxel_likelihood_array.clear();

    // Initial guess -> 6-vector (matches the C++ eulerAngles(0,1,2) extraction for non-gimbal angles).
    let guess_f64 = guess.map(f64::from);
    let mut p = matrix_to_euler(&guess_f64);

    out.transformation_array.push(*guess);

    // Derivatives at the initial guess. C++ PARITY: the initial pass transforms the cloud by the
    // guess MATRIX (`pcl::transformPointCloud(output, output, guess)`, multigrid_ndt_omp_impl
    // :266) and records the guess matrix itself; only SUBSEQUENT passes rebuild the transform
    // from the Euler 6-vector. Rebuilding here instead is not bit-equal for rotated guesses
    // (`R_rebuilt(euler(R)) != R` in f32, amplified by the map's coordinate magnitude) and was
    // the root cause of the real-map iteration divergence (PORT-QUIRK; see the porting notes).
    let mut d = {
        let mut trans = core::mem::take(&mut ws.trans_cloud);
        transform_cloud_by_matrix(guess, source, &mut trans);
        let d = derivatives_from(map, source, &trans, &p, &cfg, ws, parallel);
        ws.trans_cloud = trans;
        d
    }?;
    out.transform_probability_array
        .push(d.transform_probability as f32);
    out.nearest_voxel_likelihood_array
        .push(d.nearest_voxel_likelihood as f32);
    let mut score = d.score;

    let mut iter: i32 = 0;
    let mut nvl = d.nearest_voxel_likelihood;
    // Newton descent direction (negative for maximization), C++ line 307-310; exits if the SVD solve
    // fails (singular / non-finite).
    while let Some(delta_full) = svd_solve(&d.hessian, &(-d.gradient)) {
        let delta_norm = delta_full.norm();

        if delta_norm == 0.0 || delta_norm.is_nan() {
            break;
        }

        // Default-path step length (computeStepLengthMT, use_line_search = false).
        let mut dir = delta_full / delta_norm; // normalize
        let d_phi_0 = -(d.gradient.dot(&dir));
        if d_phi_0 >= 0.0 {
            if d_phi_0 == 0.0 {
                break;
            }
            dir = -dir;
        }
        let step_min = params.trans_epsilon / 2.0;
        // C++ parity: computeStepLengthMT applies `std::min(a_t, step_max)` THEN
        // `std::max(a_t, step_min)` (multigrid_ndt_omp_impl.hpp:953-955) — min-then-max, never
        // panicking. `f64::clamp` would panic when a misconfigured `trans_epsilon / 2 > step_size`
        // (or a NaN bound) makes min > max; min/max yields `step_min` there, exactly like C++
        // (PORT-QUIRK, resolved: see doc/book/src/port/divergences.md). Rust `f64::min`/`f64::max`
        // also match std::min/std::max on single-NaN operands (both return the non-NaN side).
        let a_t = delta_norm.min(params.step_size).max(step_min);

        let x_t = p + dir * a_t;
        d = derivatives_at(map, source, &x_t, &cfg, ws, parallel)?;
        out.transform_probability_array
            .push(d.transform_probability as f32);
        out.nearest_voxel_likelihood_array
            .push(d.nearest_voxel_likelihood as f32);
        score = d.score;
        nvl = d.nearest_voxel_likelihood;

        out.transformation_array.push(se3_matrix_f32(&x_t));
        p = x_t;
        iter = iter.saturating_add(1);

        if iter >= params.max_iterations || a_t.abs() < params.trans_epsilon {
            break;
        }
    }

    out.pose = se3_matrix_f32(&p);
    out.iteration_num = iter;
    out.hessian = d.hessian;
    out.nearest_voxel_likelihood = nvl as f32;
    out.transform_probability = if len == 0 {
        0.0
    } else {
        (score / len as f64) as f32
    };
    #[cfg(feature = "wcet-count")]
    {
        out.counters = ws.counters;
    }
    #[cfg(feature = "wcet-trace")]
    {
        out.trace = ws.trace;
    }
    let result_finite = out.pose.iter().all(|value| value.is_finite())
        && out.transform_probability.is_finite()
        && out.nearest_voxel_likelihood.is_finite()
        && out.hessian.iter().all(|value| value.is_finite());
    if !result_finite {
        return Err(AlignError::NonFiniteValue);
    }
    Ok(ws.diagnostics)
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::unwrap_used,
    clippy::unreadable_literal,
    clippy::needless_range_loop,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::borrow_as_ptr,
    unsafe_code,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;
    use crate::transform::{gauss_constants, transform_point};

    // 8 points packed inside one voxel around (cx,cy,cz) for leaf_size 1.0.
    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> alloc::vec::Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    fn test_map() -> VoxelGridMap {
        let mut map = VoxelGridMap::new([1.0, 1.0, 1.0], 6, 0.01);
        map.add_target(&dense_cluster(0.5, 0.5, 0.5), b"0");
        map.add_target(&dense_cluster(2.5, 0.5, 0.5), b"1");
        map.try_create_kdtree(418_000).expect("build kd-tree");
        map
    }

    fn test_source() -> alloc::vec::Vec<[f32; 3]> {
        alloc::vec![[0.55, 0.5, 0.5], [0.5, 0.45, 0.52], [2.55, 0.5, 0.5]]
    }

    // trans_cloud = transform_point(p, source) stored as f32.
    fn transform_cloud(source: &[[f32; 3]], p: &Vector6<f64>) -> alloc::vec::Vec<[f32; 3]> {
        source
            .iter()
            .map(|s| {
                let t = transform_point(p, &vec3(*s));
                [t[0] as f32, t[1] as f32, t[2] as f32]
            })
            .collect()
    }

    // Pure-f64 reference score: the per-cell score math runs in f64 (smooth in `p`), while neighbors
    // are found with the f32 query (matching `compute_derivatives`' neighbor set). Used as the FD
    // oracle so f32 cloud quantization doesn't add finite-difference noise.
    fn ref_score(
        map: &VoxelGridMap,
        source: &[[f32; 3]],
        p: &Vector6<f64>,
        g: &GaussConstants,
    ) -> f64 {
        let mut s = 0.0_f64;
        let mut ws = AlignWorkspace::try_with_capacity(2_000).expect("reserve workspace");
        for &sp in source {
            let xt = transform_point(p, &vec3(sp));
            let q = [xt[0] as f32, xt[1] as f32, xt[2] as f32];
            ws.neighbor_idx.clear();
            map.radius_search(q, 1.0, 0, &mut ws.neighbor_idx)
                .expect("radius search");
            for &li in &ws.neighbor_idx {
                if let Some(leaf) = map.leaf(li) {
                    let mean = Vector3::new(leaf.mean[0], leaf.mean[1], leaf.mean[2]);
                    s += score_increment(&(xt - mean), &mat3(&leaf.icov), g);
                }
            }
        }
        s
    }

    // Headline oracle: the assembled gradient equals the central finite difference of the full
    // multi-point score; the full 6x6 Hessian equals the second FD. (Before PR #1217 the `h_ang`
    // d1 sign bug made the angle-angle block non-exact, so it was excluded from the FD check; with
    // the sign fixed the whole Hessian matches FD.) Also asserts Hessian symmetry.
    #[test]
    fn compute_derivatives_matches_finite_difference() {
        let map = test_map();
        let source = test_source();
        let g = gauss_constants(0.55, 1.0);
        let p = Vector6::new(0.05, -0.03, 0.02, 0.04, -0.02, 0.03);

        let trans = transform_cloud(&source, &p);
        let mut ws = AlignWorkspace::try_with_capacity(2_000).expect("reserve workspace");
        let d = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: None,
            },
            &mut ws,
        )
        .expect("derivative computation");
        assert!(d.score > 0.0, "score should be positive for a near match");

        // gradient vs central FD of the score.
        let eps = 1e-6;
        for k in 0..6 {
            let mut pp = p;
            let mut pm = p;
            pp[k] += eps;
            pm[k] -= eps;
            let fd = (ref_score(&map, &source, &pp, &g) - ref_score(&map, &source, &pm, &g))
                / (2.0 * eps);
            assert!(
                (d.gradient[k] - fd).abs() < 1e-4,
                "grad[{k}] {} vs fd {}",
                d.gradient[k],
                fd
            );
        }

        // symmetry.
        for i in 0..6 {
            for j in 0..6 {
                assert!((d.hessian[(i, j)] - d.hessian[(j, i)]).abs() < 1e-9);
            }
        }

        // full Hessian vs second central FD (exact since the PR #1217 h_ang d1 sign fix).
        let h = 1e-4;
        for i in 0..6 {
            for j in 0..6 {
                let (mut ppp, mut ppm, mut pmp, mut pmm) = (p, p, p, p);
                ppp[i] += h;
                ppp[j] += h;
                ppm[i] += h;
                ppm[j] -= h;
                pmp[i] -= h;
                pmp[j] += h;
                pmm[i] -= h;
                pmm[j] -= h;
                let fd = (ref_score(&map, &source, &ppp, &g)
                    - ref_score(&map, &source, &ppm, &g)
                    - ref_score(&map, &source, &pmp, &g)
                    + ref_score(&map, &source, &pmm, &g))
                    / (4.0 * h * h);
                assert!(
                    (d.hessian[(i, j)] - fd).abs() < 5e-3,
                    "hess[{i}][{j}] {} vs fd {}",
                    d.hessian[(i, j)],
                    fd
                );
            }
        }
    }

    // The score-only loops are independent code paths; they must agree with compute_derivatives.
    #[test]
    fn score_only_loops_agree_with_compute_derivatives() {
        let map = test_map();
        let source = test_source();
        let g = gauss_constants(0.55, 1.0);
        let p = Vector6::new(0.05, -0.03, 0.02, 0.04, -0.02, 0.03);
        let trans = transform_cloud(&source, &p);

        let mut ws = AlignWorkspace::try_with_capacity(2_000).expect("reserve workspace");
        let d = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: None,
            },
            &mut ws,
        )
        .expect("derivative computation");

        let tp = transformation_probability(&map, &trans, 1.0, &g, &mut ws)
            .expect("transformation probability");
        assert!(
            (tp - d.transform_probability).abs() < 1e-12,
            "{tp} vs {}",
            d.transform_probability
        );

        let nvl = nearest_voxel_transformation_likelihood(&map, &trans, 1.0, &g, &mut ws)
            .expect("nearest voxel likelihood");
        assert!(
            (nvl - d.nearest_voxel_likelihood).abs() < 1e-12,
            "{nvl} vs {}",
            d.nearest_voxel_likelihood
        );
    }

    #[test]
    fn empty_cloud_and_no_neighbor_are_zero() {
        let map = test_map();
        let g = gauss_constants(0.55, 1.0);
        let p = Vector6::zeros();
        let mut ws = AlignWorkspace::try_with_capacity(2_000).expect("reserve workspace");

        // Empty cloud.
        let d = compute_derivatives(
            &map,
            &[],
            &[],
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: None,
            },
            &mut ws,
        )
        .expect("derivative computation");
        assert_eq!(d.score, 0.0);
        assert_eq!(d.transform_probability, 0.0);
        assert_eq!(d.nearest_voxel_likelihood, 0.0);

        // A point far from any voxel contributes nothing and is not counted.
        let source = alloc::vec![[100.0_f32, 100.0, 100.0]];
        let trans = source.clone();
        let d2 = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: None,
            },
            &mut ws,
        )
        .expect("derivative computation");
        assert_eq!(d2.score, 0.0);
        assert_eq!(d2.gradient, Vector6::zeros());
        assert_eq!(d2.nearest_voxel_likelihood, 0.0);

        // The standalone score-only loops honor the same empty / no-neighbor contract (0.0).
        assert_eq!(
            transformation_probability(&map, &[], 1.0, &g, &mut ws),
            Ok(0.0)
        );
        assert_eq!(
            nearest_voxel_transformation_likelihood(&map, &[], 1.0, &g, &mut ws),
            Ok(0.0)
        );
        assert_eq!(
            transformation_probability(&map, &source, 1.0, &g, &mut ws),
            Ok(0.0)
        );
        assert_eq!(
            nearest_voxel_transformation_likelihood(&map, &source, 1.0, &g, &mut ws),
            Ok(0.0)
        );
    }

    // reg = None and a zero-scale Regularization must produce identical results.
    #[test]
    fn zero_scale_regularization_is_noop() {
        let map = test_map();
        let source = test_source();
        let g = gauss_constants(0.55, 1.0);
        let p = Vector6::new(0.05, -0.03, 0.02, 0.04, -0.02, 0.03);
        let trans = transform_cloud(&source, &p);
        let mut ws = AlignWorkspace::try_with_capacity(2_000).expect("reserve workspace");

        let d_none = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: None,
            },
            &mut ws,
        )
        .expect("derivative computation");
        let reg = Regularization {
            pose_xy: [0.0, 0.0],
            scale_factor: 0.0,
        };
        let d_zero = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: Some(&reg),
            },
            &mut ws,
        )
        .expect("derivative computation");
        assert_eq!(d_none.score, d_zero.score);
        assert_eq!(d_none.gradient, d_zero.gradient);
        assert_eq!(d_none.hessian, d_zero.hessian);
    }

    // The complement: a nonzero-scale regularization toward a distant reference pose must actually
    // change the score and gradient (the `finalize` reg term is exercised, not just executed).
    #[test]
    fn nonzero_regularization_changes_derivatives() {
        let map = test_map();
        let source = test_source();
        let g = gauss_constants(0.55, 1.0);
        let p = Vector6::new(0.05, -0.03, 0.02, 0.04, -0.02, 0.03);
        let trans = transform_cloud(&source, &p);
        let mut ws = AlignWorkspace::try_with_capacity(2_000).expect("reserve workspace");

        let d_none = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: None,
            },
            &mut ws,
        )
        .expect("derivative computation");
        let reg = Regularization {
            pose_xy: [5.0, 5.0],
            scale_factor: 1.0,
        };
        let d_reg = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &g,
                reg: Some(&reg),
            },
            &mut ws,
        )
        .expect("derivative computation");
        assert!(
            (d_reg.score - d_none.score).abs() > 1e-9,
            "reg must change the score"
        );
        assert!(
            (d_reg.gradient - d_none.gradient).norm() > 1e-9,
            "reg must change the gradient"
        );
    }

    // ---- align ----

    // Target points spread in 3D (constrains translation), built into a map; the originals are
    // returned so tests can synthesize a source by transforming them.
    fn spread_target() -> (VoxelGridMap, alloc::vec::Vec<[f32; 3]>) {
        let mut pts: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::new();
        for &(cx, cy, cz) in &[
            (0.5_f32, 0.5_f32, 0.5_f32),
            (4.5, 0.5, 0.5),
            (0.5, 4.5, 0.5),
            (0.5, 0.5, 4.5),
            (4.5, 4.5, 4.5),
        ] {
            pts.extend(dense_cluster(cx, cy, cz));
        }
        let mut map = VoxelGridMap::new([1.0, 1.0, 1.0], 6, 0.01);
        map.add_target(&pts, b"0");
        map.try_create_kdtree(418_000).expect("build kd-tree");
        (map, pts)
    }

    fn tight_params() -> NdtParams {
        NdtParams {
            trans_epsilon: 1e-3,
            step_size: 0.2,
            resolution: 1.0,
            max_iterations: 30,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1,
        }
    }

    #[test]
    fn align_degenerate_step_bounds_do_not_panic() {
        // trans_epsilon / 2 > step_size makes f64::clamp's min > max (a panic); the C++-parity
        // min-then-max (multigrid_ndt_omp_impl.hpp:953-955) instead yields step_min. The align
        // must complete without panicking and produce a finite pose.
        let (map, source) = spread_target();
        let mut params = tight_params();
        params.step_size = 0.01;
        params.trans_epsilon = 1.0; // step_min = 0.5 > step_size
        params.max_iterations = 5;
        let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
        let mut result = AlignResult::try_with_capacity(5).expect("reserve result");
        align(
            &map,
            &source,
            &Matrix4::identity(),
            &params,
            &mut ws,
            &mut result,
        )
        .expect("align");
        for v in result.pose.iter() {
            assert!(v.is_finite());
        }
    }

    // ParReduce: the rayon backend must equal the serial backend BIT-FOR-BIT — enabling
    // parallelism is a pure performance change. Exact `==` on every field (not approx).
    #[cfg(feature = "parallel")]
    #[test]
    fn serial_and_parallel_compute_derivatives_are_bit_identical() {
        let (map, source) = spread_target(); // 40 points across 5 clusters -> rayon splits the work
        let g = gauss_constants(0.55, 1.0);
        let p = Vector6::new(0.1, -0.05, 0.03, 0.02, -0.01, 0.04);
        let trans = transform_cloud(&source, &p);
        let cfg = ScoreConfig {
            resolution: 1.0,
            gauss: &g,
            reg: None,
        };
        let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
        let s = compute_derivatives(&map, &source, &trans, &p, &cfg, &mut ws)
            .expect("derivative computation");
        let par = compute_derivatives_parallel(&map, &source, &trans, &p, &cfg, &mut ws)
            .expect("derivative computation");

        assert_eq!(s.score, par.score, "score");
        assert_eq!(s.transform_probability, par.transform_probability, "tp");
        assert_eq!(
            s.nearest_voxel_likelihood, par.nearest_voxel_likelihood,
            "nvl"
        );
        for i in 0..6 {
            assert_eq!(s.gradient[i], par.gradient[i], "gradient[{i}]");
        }
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(s.hessian[(i, j)], par.hessian[(i, j)], "hessian[{i}][{j}]");
            }
        }
    }

    // The same property at the `align` level: num_threads = 1 (serial) vs 8 (rayon) must produce an
    // identical AlignResult (pose / iterations / Hessian / per-iteration arrays).
    #[cfg(feature = "parallel")]
    #[test]
    fn align_serial_equals_parallel_bit_identical() {
        let (map, target) = spread_target();
        let t = [0.2_f32, -0.15, 0.1];
        let source: alloc::vec::Vec<[f32; 3]> = target
            .iter()
            .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
            .collect();
        let guess = Matrix4::identity();

        let run = |num_threads: usize| {
            let mut params = tight_params();
            params.num_threads = num_threads;
            let mut ws =
                AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
            let mut out = AlignResult::try_with_capacity(30).expect("reserve result");
            align(&map, &source, &guess, &params, &mut ws, &mut out).expect("align");
            out
        };
        let serial = run(1);
        let parallel = run(8);

        assert_eq!(serial.iteration_num, parallel.iteration_num, "iterations");
        assert_eq!(serial.pose, parallel.pose, "pose");
        assert_eq!(serial.hessian, parallel.hessian, "hessian");
        assert_eq!(
            serial.transform_probability_array, parallel.transform_probability_array,
            "tp array"
        );
        assert_eq!(
            serial.nearest_voxel_likelihood_array, parallel.nearest_voxel_likelihood_array,
            "nvl array"
        );
    }

    // Recover-known-transform oracle: source = target + t; align maps source back, so the recovered
    // pose's translation ≈ -t (and rotation ≈ identity). Strong independent oracle.
    #[test]
    fn align_recovers_known_translation() {
        let (map, target) = spread_target();
        let t = [0.15_f32, -0.10, 0.05];
        let source: alloc::vec::Vec<[f32; 3]> = target
            .iter()
            .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
            .collect();

        let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
        let mut out = AlignResult::try_with_capacity(30).expect("reserve result");
        align(
            &map,
            &source,
            &Matrix4::identity(),
            &tight_params(),
            &mut ws,
            &mut out,
        )
        .expect("align");

        assert!(
            out.iteration_num <= 30,
            "did not stop within max_iterations"
        );
        // Final translation ≈ -t.
        assert!(
            (out.pose[(0, 3)] - (-t[0])).abs() < 3e-2,
            "tx {}",
            out.pose[(0, 3)]
        );
        assert!(
            (out.pose[(1, 3)] - (-t[1])).abs() < 3e-2,
            "ty {}",
            out.pose[(1, 3)]
        );
        assert!(
            (out.pose[(2, 3)] - (-t[2])).abs() < 3e-2,
            "tz {}",
            out.pose[(2, 3)]
        );
        // Rotation block ≈ identity.
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((out.pose[(i, j)] - want).abs() < 2e-2, "R[{i}][{j}]");
            }
        }
        // Score improved from the (misaligned) guess to convergence.
        let tpa = &out.transform_probability_array;
        assert!(
            tpa.len() == out.iteration_num as usize + 1,
            "array length == iter+1"
        );
        assert!(
            *tpa.last().unwrap() >= *tpa.first().unwrap(),
            "transform_probability did not improve: {tpa:?}"
        );
    }

    // Already aligned (source == target): align stays at identity and converges immediately.
    #[test]
    fn align_identity_when_already_aligned() {
        let (map, target) = spread_target();
        let mut ws = AlignWorkspace::try_with_capacity(target.len()).expect("reserve workspace");
        let mut out = AlignResult::try_with_capacity(30).expect("reserve result");
        align(
            &map,
            &target,
            &Matrix4::identity(),
            &tight_params(),
            &mut ws,
            &mut out,
        )
        .expect("align");
        for i in 0..3 {
            assert!(
                out.pose[(i, 3)].abs() < 2e-2,
                "translation drifted: {}",
                out.pose[(i, 3)]
            );
        }
        assert!(out.transform_probability > 0.0);
    }
}
