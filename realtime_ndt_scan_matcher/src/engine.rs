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

//! Persistent NDT engine handle. This crate exposes the Rust API; the sibling node crate wraps it
//! in the opaque C ABI used by the C++ package.
//!
//! Concurrency: the engine exposes **`&self`-only** methods, and its
//! mutable state lives behind interior mutability in one of three configs:
//!
//! - **std** (default): target map + params in an `ArcSwap<EngineState>` (the read/align path loads
//!   an immutable snapshot lock-free; map-update publishes a fresh state with an atomic store), the
//!   optional regularization in a tiny `ArcSwap<Option<Regularization>>`. Alignment uses a
//!   caller-owned [`MatchScratch`]. `Sync`: a shared `&NdtEngine` is sound across concurrent ROS
//!   callbacks **without** an external mutex.
//! - **`no_std` single-core** (no features): `RefCell` cells instead of `ArcSwap`. Alignment still
//!   requires caller-owned scratch. The state cells make this configuration intentionally `!Sync`.
//! - **`no_std` `mt`** (multi-core kernel): the cells are `awkernel_sync` mutexes whose guards
//!   disable interrupts, so every critical section is a few instructions — an `Arc` refcount bump
//!   (load) or a pointer swap (store); rcu builds the next state OUTSIDE the lock and publishes
//!   with an optimistic `Arc::ptr_eq` retry; old snapshots drop outside the lock. There is **no
//!   engine-owned scratch**: callers own a [`MatchScratch`] per task/thread. `Sync`, like std.
//!
//! Deployment code fixes an immutable [`AlignLimits`] envelope at construction and uses explicit
//! scratch. Map assembly is control-plane work: it may allocate, but a completed map is published
//! atomically only when its active-leaf count is inside the envelope. The serial align-plus-verdict
//! path is allocation-free from its first call when [`MatchScratch::try_for_limits`] succeeds.
//! Parallel reduction remains a throughput option and is not covered by that allocation claim.

use alloc::sync::Arc;

use nalgebra::{Matrix4, Matrix6};

use crate::ndt::{
    AlignDiagnostics, AlignError, AlignResult, AlignWorkspace, NdtParams, Regularization, align,
    nearest_voxel_score_each_point, nearest_voxel_transformation_likelihood,
    transformation_probability,
};
use crate::transform::gauss_constants;
use crate::voxel_grid::{MapBuildError, VoxelGridMap};

// ---- interior-mutability cell (std: lock-free arc-swap; no_std single-core: RefCell<Arc<…>>;
// no_std `mt`: awkernel_sync mutex around the Arc) ----
// `Swap<T>` aliases the backing cell; the `swap_*` free fns give a uniform load/store/rcu API. `load`
// returns an owned `Arc<T>` snapshot (lock-free under std), so callers hold a stable view while a
// concurrent `store`/`rcu` publishes a new version (the old `Arc` lives until its last reader drops).
// Under `mt` the cell is an interrupt-disabling kernel mutex, so every critical section is a few
// instructions (an `Arc` refcount bump or a pointer swap) — never an align, never a deep clone, and
// never the drop of an old snapshot (a last-reference drop deallocates a whole map; all `mt` paths
// drop old `Arc`s after the guard is released, with interrupts re-enabled).

#[cfg(feature = "std")]
type Swap<T> = arc_swap::ArcSwap<T>;
#[cfg(all(not(feature = "std"), not(feature = "mt")))]
type Swap<T> = core::cell::RefCell<Arc<T>>;
#[cfg(all(feature = "mt", not(feature = "std")))]
type Swap<T> = awkernel_sync::mutex::Mutex<Arc<T>>;

#[cfg(feature = "std")]
fn swap_new<T>(v: T) -> Swap<T> {
    arc_swap::ArcSwap::from_pointee(v)
}
#[cfg(all(not(feature = "std"), not(feature = "mt")))]
fn swap_new<T>(v: T) -> Swap<T> {
    core::cell::RefCell::new(Arc::new(v))
}
#[cfg(all(feature = "mt", not(feature = "std")))]
fn swap_new<T: Send + Sync>(v: T) -> Swap<T> {
    awkernel_sync::mutex::Mutex::new(Arc::new(v))
}

#[cfg(feature = "std")]
fn swap_load<T>(c: &Swap<T>) -> Arc<T> {
    c.load_full()
}
#[cfg(all(not(feature = "std"), not(feature = "mt")))]
fn swap_load<T>(c: &Swap<T>) -> Arc<T> {
    c.borrow().clone()
}
#[cfg(all(feature = "mt", not(feature = "std")))]
fn swap_load<T: Send + Sync>(c: &Swap<T>) -> Arc<T> {
    let mut node = awkernel_sync::mcs::MCSNode::new();
    // Critical section: one Arc refcount increment.
    Arc::clone(&c.lock(&mut node))
}

#[cfg(feature = "std")]
fn swap_store<T>(c: &Swap<T>, v: T) {
    c.store(Arc::new(v));
}
#[cfg(all(not(feature = "std"), not(feature = "mt")))]
fn swap_store<T>(c: &Swap<T>, v: T) {
    *c.borrow_mut() = Arc::new(v);
}
#[cfg(all(feature = "mt", not(feature = "std")))]
fn swap_store<T: Send + Sync>(c: &Swap<T>, v: T) {
    swap_store_arc(c, Arc::new(v));
}

/// Read-copy-update: build the next value from the current one and publish it atomically. The
/// closure may run more than once under contention (std/arc-swap CAS retry and the `mt` optimistic
/// retry), so it must be pure.
#[cfg(feature = "std")]
fn swap_rcu<T>(c: &Swap<T>, f: impl Fn(&T) -> T) {
    c.rcu(|cur| Arc::new(f(cur)));
}
#[cfg(all(not(feature = "std"), not(feature = "mt")))]
fn swap_rcu<T>(c: &Swap<T>, f: impl Fn(&T) -> T) {
    let next = {
        let cur = c.borrow();
        Arc::new(f(&cur))
    };
    *c.borrow_mut() = next;
}
/// `mt`: optimistic retry — `f` runs OUTSIDE the lock (it may deep-clone a whole map, which must
/// never happen with interrupts disabled); the short re-lock publishes only if the snapshot is
/// still current (`Arc::ptr_eq`), else it retries against the fresh one. Writers here are rare
/// (params setup / map-update commit), so the loop converges immediately in practice.
#[cfg(all(feature = "mt", not(feature = "std")))]
fn swap_rcu<T: Send + Sync>(c: &Swap<T>, f: impl Fn(&T) -> T) {
    let mut cur = swap_load(c);
    loop {
        let next = Arc::new(f(&cur));
        let published = {
            // The MCS node lives one iteration — the guard borrows it.
            let mut node = awkernel_sync::mcs::MCSNode::new();
            let mut guard = c.lock(&mut node);
            if Arc::ptr_eq(&guard, &cur) {
                // Critical section: one pointer swap; the previous Arc is handed out first.
                Some(core::mem::replace(&mut *guard, next))
            } else {
                cur = Arc::clone(&guard);
                None
            }
        };
        if let Some(old) = published {
            drop(old); // outside the lock — may deallocate the previous state
            return;
        }
        // A stale `next` also drops here, outside the lock.
    }
}

/// Publish an already-built `Arc<T>` snapshot (no deep copy) — the map-update commit shares the
/// staging engine's final state `Arc` into the live engine in one atomic store.
#[cfg(feature = "std")]
fn swap_store_arc<T>(c: &Swap<T>, v: Arc<T>) {
    c.store(v);
}
#[cfg(all(not(feature = "std"), not(feature = "mt")))]
fn swap_store_arc<T>(c: &Swap<T>, v: Arc<T>) {
    *c.borrow_mut() = v;
}
#[cfg(all(feature = "mt", not(feature = "std")))]
fn swap_store_arc<T: Send + Sync>(c: &Swap<T>, v: Arc<T>) {
    let old = {
        let mut node = awkernel_sync::mcs::MCSNode::new();
        let mut guard = c.lock(&mut node);
        // Critical section: one pointer swap; the old Arc drops after the guard is released.
        core::mem::replace(&mut *guard, v)
    };
    drop(old);
}

/// The `covariance` hyper-params that drive [`NdtEngine::estimate_covariance`]: the estimation mode +
/// scaling + the configured 6x6 + the candidate search offsets. Holds owned `Vec` offsets (so it is
/// `Clone`, not `Copy`); set off the hot path via [`NdtEngine::set_covariance_config`].
#[derive(Clone, Debug)]
struct CovarianceConfig {
    estimation_type: i32,
    scale_factor: f64,
    temperature: f64,
    output_pose_covariance: [f64; 36],
    offset_x: alloc::vec::Vec<f64>,
    offset_y: alloc::vec::Vec<f64>,
}

impl CovarianceConfig {
    /// Default: `FIXED_VALUE` (the configured 6x6 only), unit scale, no candidates — the node/example
    /// always sets the real config before estimating.
    fn new() -> Self {
        Self {
            estimation_type: 0,
            scale_factor: 1.0,
            temperature: 1.0,
            output_pose_covariance: [0.0; 36],
            offset_x: alloc::vec::Vec::new(),
            offset_y: alloc::vec::Vec::new(),
        }
    }
}
/// Immutable work-envelope limits for one engine instance.
///
/// These are admission bounds, not observations. `max_source_points` sizes and checks the
/// per-frame workspace. `max_active_leaves` limits the map snapshot that may be published; it does
/// not preallocate that many leaves. `max_iterations` limits Newton updates and must be in
/// `0..=30`. Increasing a bound admits more work and increases the corresponding structural work
/// envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlignLimits {
    /// Maximum source points accepted by one alignment.
    pub max_source_points: usize,
    /// Maximum active Gaussian leaves accepted in a published map.
    pub max_active_leaves: usize,
    /// Maximum Newton iterations, in `0..=30`.
    pub max_iterations: i32,
}

impl AlignLimits {
    /// Validate and construct an engine work envelope.
    ///
    /// Construction also checks that `(max_iterations + 1) * max_source_points * 64` and
    /// `(max_iterations + 1) * max_source_points * max_active_leaves` fit the `u64` work counters.
    /// # Errors
    /// Returns [`AlignError::InvalidLimits`] for zero point/leaf bounds or an iteration bound
    /// outside `0..=30`, and [`AlignError::ArithmeticOverflow`] when either work product does not
    /// fit its counter.
    pub fn new(
        max_source_points: usize,
        max_active_leaves: usize,
        max_iterations: i32,
    ) -> Result<Self, AlignError> {
        if max_source_points == 0 || max_active_leaves == 0 || !(0..=30).contains(&max_iterations) {
            return Err(AlignError::InvalidLimits);
        }
        let passes = u64::try_from(max_iterations)
            .map_err(|_| AlignError::ArithmeticOverflow)?
            .checked_add(1)
            .ok_or(AlignError::ArithmeticOverflow)?;
        let points =
            u64::try_from(max_source_points).map_err(|_| AlignError::ArithmeticOverflow)?;
        let leaves =
            u64::try_from(max_active_leaves).map_err(|_| AlignError::ArithmeticOverflow)?;
        passes
            .checked_mul(points)
            .and_then(|value| value.checked_mul(64))
            .ok_or(AlignError::ArithmeticOverflow)?;
        passes
            .checked_mul(points)
            .and_then(|value| value.checked_mul(leaves))
            .ok_or(AlignError::ArithmeticOverflow)?;

        Ok(Self {
            max_source_points,
            max_active_leaves,
            max_iterations,
        })
    }
}

/// The atomically-swappable engine state: the target map and active params. Immutable once
/// published — map-update builds a fresh `EngineState` and stores it.
#[derive(Clone)]
struct EngineState {
    map: VoxelGridMap,
    params: NdtParams,
    /// The `score_estimation` scalars that gate the convergence verdict (read on `align_outcome`).
    /// Part of the swapped state so `set_convergence_params` is lock-free like `set_params`.
    conv: ConvergenceParams,
    /// The `covariance` hyper-params read on `estimate_covariance` (swapped like `conv`).
    cov_config: CovarianceConfig,
}

/// Per-align caller-owned scratch: the reused workspace, last result, and convergence history.
/// Own one instance per task or thread and pass it explicitly to every alignment. Reuse is required
/// for predictable serial execution; the `parallel` backend is a throughput option and may allocate
/// worker-local storage.
///
/// # Examples
///
/// Own one scratch per task/thread and reuse it across frames (the only align path under `mt`):
///
/// ```
/// use realtime_ndt_scan_matcher::engine::{MatchScratch, NdtEngine};
/// use realtime_ndt_scan_matcher::nalgebra::Matrix4;
///
/// let engine = NdtEngine::new(2.0, 6, 0.01, 64, 64, 30)
///     .expect("valid work envelope");
/// engine.set_params(0.01, 0.1, 2.0, 30, 0.55, 1);
/// let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
/// engine.add_target(&target, b"map");
/// engine.create_kdtree().expect("build kd-tree");
///
/// let mut scratch = MatchScratch::try_for_limits(engine.limits()).expect("reserve scratch");
/// // Frame 1, then frame 2 reuse the same preallocated scratch.
/// engine.align(&Matrix4::identity(), &target, &mut scratch).expect("align");
/// let next_guess = scratch.result_ref().pose;
/// engine.align(&next_guess, &target, &mut scratch).expect("align");
/// assert!(scratch.result_ref().iteration_num >= 0);
/// ```
pub struct MatchScratch {
    workspace: AlignWorkspace,
    last: AlignResult,
    oscillation_positions: alloc::vec::Vec<[f64; 3]>,
}

impl MatchScratch {
    /// Scratch pre-reserved for clouds of up to `max_points` and up to `max_iterations` Newton
    /// iterations. In the serial backend, this makes the align-plus-verdict path allocation-free
    /// including its first frame. It reserves the point workspace, neighbor indices, per-iteration
    /// result arrays, and convergence history (`max_iterations + 1` entries: the initial pose plus
    /// one per iteration).
    /// # Errors
    /// Returns [`AlignError::ArithmeticOverflow`] if the result-slot count overflows and
    /// [`AlignError::WorkspaceCapacityExceeded`] if any reservation fails.
    pub fn try_with_capacity(max_points: usize, max_iterations: usize) -> Result<Self, AlignError> {
        let cap = max_iterations
            .checked_add(1)
            .ok_or(AlignError::ArithmeticOverflow)?;
        let mut last = AlignResult::default();
        last.transformation_array
            .try_reserve_exact(cap)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        last.transform_probability_array
            .try_reserve_exact(cap)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        last.nearest_voxel_likelihood_array
            .try_reserve_exact(cap)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        let mut oscillation_positions = alloc::vec::Vec::new();
        oscillation_positions
            .try_reserve_exact(cap)
            .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
        Ok(Self {
            workspace: AlignWorkspace::try_with_capacity(max_points)?,
            last,
            oscillation_positions,
        })
    }

    /// Allocate all serial align-plus-verdict buffers from an engine work envelope.
    /// # Errors
    /// Returns [`AlignError::InvalidLimits`] if the iteration limit cannot be converted, or the
    /// reservation errors documented by [`Self::try_with_capacity`].
    pub fn try_for_limits(limits: AlignLimits) -> Result<Self, AlignError> {
        let max_iterations =
            usize::try_from(limits.max_iterations).map_err(|_| AlignError::InvalidLimits)?;
        Self::try_with_capacity(limits.max_source_points, max_iterations)
    }

    /// Heap payload bytes reserved by this scratch, excluding allocator metadata and any
    /// parallel-backend worker-local allocations.
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn allocated_payload_bytes(&self) -> Result<usize, AlignError> {
        let mut bytes = self.workspace.allocated_payload_bytes()?;
        bytes = bytes
            .checked_add(
                self.last
                    .transformation_array
                    .capacity()
                    .checked_mul(core::mem::size_of::<Matrix4<f32>>())
                    .ok_or(AlignError::ArithmeticOverflow)?,
            )
            .ok_or(AlignError::ArithmeticOverflow)?;
        bytes = bytes
            .checked_add(
                self.last
                    .transform_probability_array
                    .capacity()
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or(AlignError::ArithmeticOverflow)?,
            )
            .ok_or(AlignError::ArithmeticOverflow)?;
        bytes = bytes
            .checked_add(
                self.last
                    .nearest_voxel_likelihood_array
                    .capacity()
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or(AlignError::ArithmeticOverflow)?,
            )
            .ok_or(AlignError::ArithmeticOverflow)?;
        bytes = bytes
            .checked_add(
                self.oscillation_positions
                    .capacity()
                    .checked_mul(core::mem::size_of::<[f64; 3]>())
                    .ok_or(AlignError::ArithmeticOverflow)?,
            )
            .ok_or(AlignError::ArithmeticOverflow)?;
        Ok(bytes)
    }

    /// Borrow the last result without cloning its iteration arrays.
    #[must_use]
    pub const fn result_ref(&self) -> &AlignResult {
        &self.last
    }

    /// Borrow the last per-iteration score traces without allocation.
    #[must_use]
    pub fn score_slices(&self) -> (&[f32], &[f32]) {
        (
            &self.last.transform_probability_array,
            &self.last.nearest_voxel_likelihood_array,
        )
    }
}

/// Persistent NDT engine: a `Swap` cell of the target map + params and the
/// optional regularization. `&self`-only + `Sync` (std and `mt`; the plain `no_std` single-core
/// build is `!Sync`). Every configuration uses caller-owned scratch; see the module docs.
///
/// # Examples
///
/// Configure a bounded engine, load a target map, and align via caller-owned [`MatchScratch`] (the
/// universal path available in every build configuration):
///
/// ```
/// use realtime_ndt_scan_matcher::engine::{MatchScratch, NdtEngine};
/// use realtime_ndt_scan_matcher::nalgebra::Matrix4;
///
/// let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
/// let engine = NdtEngine::new(2.0, 6, 0.01, target.len(), target.len(), 30)
///     .expect("valid work envelope");
/// // trans_epsilon, step_size, resolution, max_iterations, outlier_ratio, num_threads
/// engine.set_params(0.01, 0.1, 2.0, 30, 0.55, 1);
///
/// engine.add_target(&target, b"map");
/// engine.create_kdtree().expect("build kd-tree");
///
/// let mut scratch = MatchScratch::try_for_limits(engine.limits()).expect("reserve scratch");
/// engine.align(&Matrix4::identity(), &target, &mut scratch).expect("align");
/// assert!(scratch.result_ref().iteration_num >= 0);
/// ```
pub struct NdtEngine {
    state: Swap<EngineState>,
    /// Optional longitudinal regularization, swapped lock-free (set per sensor frame before align,
    /// read at align time on the same thread). A dedicated tiny cell so setting it never clones the
    /// map (it is not part of `EngineState`).
    reg: Swap<Option<Regularization>>,
    min_points: i32,
    eig_mult: f64,
    limits: AlignLimits,
}

/// Compile-time proof that the engine is shareable across threads in the two concurrent configs
/// (std: arc-swap; `mt`: `awkernel_sync` cells; both use caller-owned scratch). The
/// plain `no_std` single-core engine stays intentionally `!Sync` (`RefCell`), so it is excluded.
#[cfg(any(feature = "std", feature = "mt"))]
const fn assert_send_sync<T: Send + Sync>() {}
#[cfg(any(feature = "std", feature = "mt"))]
const _: () = assert_send_sync::<NdtEngine>();

impl Clone for NdtEngine {
    fn clone(&self) -> Self {
        // Deep-copy the published state + regularization (the map-update double-buffer clones the
        // whole engine). Caller-owned scratch is not engine state.
        let st = swap_load(&self.state);
        Self {
            state: swap_new((*st).clone()),
            reg: swap_new(*swap_load(&self.reg)),
            min_points: self.min_points,
            eig_mult: self.eig_mult,
            limits: self.limits,
        }
    }
}

impl NdtEngine {
    /// Clone the engine's configuration (params, convergence + covariance config, regularization) but
    /// with an **empty** map (no tiles). Used by the map-update rebuild path to start
    /// a staging engine from scratch (mirrors the C++ `need_rebuild`: fresh `NdtType` + `setParams`),
    /// as opposed to [`Clone`] which deep-copies the current map.
    #[must_use]
    pub fn clone_empty(&self) -> Self {
        let st = self.load_state();
        Self {
            state: swap_new(EngineState {
                map: VoxelGridMap::new([st.params.resolution; 3], self.min_points, self.eig_mult),
                params: st.params,
                conv: st.conv,
                cov_config: st.cov_config.clone(),
            }),
            reg: swap_new(*swap_load(&self.reg)),
            min_points: self.min_points,
            eig_mult: self.eig_mult,
            limits: self.limits,
        }
    }

    /// Construct an engine whose runtime work envelope is fixed for its lifetime.
    ///
    /// `max_source_points` is the per-align source admission bound, `max_active_leaves` is the
    /// published-map admission bound, and `max_iterations` is constrained to `0..=30`. The leaf
    /// bound does not reserve map storage; memory follows the leaves actually built. Choose it from
    /// the largest deployment map plus an operational margin.
    ///
    /// # Arguments
    /// * `resolution` — voxel/leaf size in metres; also the neighbor search radius.
    /// * `min_points` — minimum points per voxel for it to contribute a Gaussian.
    /// * `eig_mult` — eigenvalue-inflation multiplier conditioning each voxel covariance.
    /// * `max_source_points` — maximum accepted source points per alignment.
    /// * `max_active_leaves` — maximum active Gaussian leaves in a published map.
    /// * `max_iterations` — maximum optimizer iterations, in `0..=30`.
    ///
    /// # Errors
    /// Returns the validation errors documented by [`AlignLimits::new`].
    pub fn new(
        resolution: f64,
        min_points: i32,
        eig_mult: f64,
        max_source_points: usize,
        max_active_leaves: usize,
        max_iterations: i32,
    ) -> Result<Self, AlignError> {
        let limits = AlignLimits::new(max_source_points, max_active_leaves, max_iterations)?;
        Ok(Self {
            state: swap_new(EngineState {
                map: VoxelGridMap::new([resolution; 3], min_points, eig_mult),
                params: NdtParams {
                    resolution,
                    max_iterations,
                    ..NdtParams::default()
                },
                conv: ConvergenceParams::default(),
                cov_config: CovarianceConfig::new(),
            }),
            reg: swap_new(None),
            min_points,
            eig_mult,
            limits,
        })
    }

    /// Return the immutable work envelope configured for this engine.
    #[must_use]
    pub const fn limits(&self) -> AlignLimits {
        self.limits
    }

    /// The current published state snapshot (lock-free under std).
    fn load_state(&self) -> Arc<EngineState> {
        swap_load(&self.state)
    }

    /// Update the alignment params (the C++ `setParams`). The regularization is preserved — it is
    /// set separately via [`Self::set_regularization`], mirroring `setRegularizationPose`. Mirroring
    /// the C++ (which applies `resolution` as the leaf size at `addTarget`), the empty map is rebuilt
    /// at the new resolution — the node always calls `set_params` before `add_target`.
    ///
    /// # Arguments
    /// * `trans_epsilon` — translation convergence tolerance (`setTransformationEpsilon`).
    /// * `step_size` — More-Thuente line-search step size (`setStepSize`).
    /// * `resolution` — voxel/leaf size in metres (`setResolution`); rebuilds the map if still empty.
    /// * `max_iterations` — optimizer iteration cap (`setMaximumIterations`).
    /// * `outlier_ratio` — Gaussian mixture outlier ratio (default 0.55).
    /// * `num_threads` — derivative-reduction worker count; `> 1` uses the rayon backend when the
    ///   `parallel` feature is on (bit-identical to serial).
    pub fn set_params(
        &self,
        trans_epsilon: f64,
        step_size: f64,
        resolution: f64,
        max_iterations: i32,
        outlier_ratio: f64,
        num_threads: usize,
    ) {
        #[cfg(feature = "std")]
        crate::capture::hook_params(&NdtParams {
            trans_epsilon,
            step_size,
            resolution,
            max_iterations,
            outlier_ratio,
            regularization: None,
            num_threads: 1,
        });
        swap_rcu(&self.state, |s| {
            let mut n = s.clone();
            n.params.trans_epsilon = trans_epsilon;
            n.params.step_size = step_size;
            n.params.resolution = resolution;
            n.params.max_iterations = max_iterations;
            n.params.outlier_ratio = outlier_ratio;
            n.params.num_threads = num_threads;
            if n.map.is_empty() {
                n.map = VoxelGridMap::new([resolution; 3], self.min_points, self.eig_mult);
            }
            n
        });
    }

    /// Set the `score_estimation` scalars that gate the convergence verdict (the C++
    /// `converged_param_*`). Lock-free (swapped into the published state like [`Self::set_params`]).
    pub fn set_convergence_params(
        &self,
        converged_param_type: i32,
        converged_param_transform_probability: f64,
        converged_param_nearest_voxel_transformation_likelihood: f64,
    ) {
        swap_rcu(&self.state, |s| {
            let mut n = s.clone();
            n.conv = ConvergenceParams {
                converged_param_type,
                converged_param_transform_probability,
                converged_param_nearest_voxel_transformation_likelihood,
            };
            n
        });
    }

    /// Align from `guess` and derive the full [`AlignOutcome`] with caller-owned scratch.
    /// # Errors
    /// Returns the source, iteration, numeric, arithmetic, kd-stack, and workspace errors documented
    /// by [`Self::align`].
    pub fn align_outcome(
        &self,
        guess: &Matrix4<f32>,
        source: &[[f32; 3]],
        scratch: &mut MatchScratch,
    ) -> Result<AlignOutcome, AlignError> {
        let conv = self.load_state().conv;
        run_align(self, guess, source, &conv, scratch)
    }

    /// Set the `covariance` hyper-params read on [`Self::estimate_covariance`] (estimation mode +
    /// scaling + the configured 6x6 + the candidate search offsets). Lock-free (swapped into the
    /// published state like [`Self::set_params`]).
    pub fn set_covariance_config(
        &self,
        estimation_type: i32,
        scale_factor: f64,
        temperature: f64,
        output_pose_covariance: [f64; 36],
        offset_x: &[f64],
        offset_y: &[f64],
    ) {
        swap_rcu(&self.state, |s| {
            let mut n = s.clone();
            n.cov_config = CovarianceConfig {
                estimation_type,
                scale_factor,
                temperature,
                output_pose_covariance,
                offset_x: offset_x.to_vec(),
                offset_y: offset_y.to_vec(),
            };
            n
        });
    }

    /// Estimate the 6x6 pose covariance from an align result, reading the engine's own
    /// `CovarianceConfig`. The self-contained counterpart of [`estimate_pose_covariance`] (which takes
    /// the params explicitly for the C++ FFI) — what the portable `ScanMatcher` calls after a match.
    /// `map_to_base_link_rot3x3` is derived from `result_pose`'s rotation block (the C++ builds the same
    /// from the result quaternion). `scratch` backs the `MULTI_NDT*` candidate re-aligns/scores.
    ///
    /// # Arguments
    /// * `result_pose` — the converged align pose; its rotation block seeds the covariance rotation.
    /// * `hessian` — the align's 6×6 Hessian (from [`MatchScratch::result_ref`]'s `AlignResult`).
    /// * `initial_pose` — the initial guess the align started from (the candidate-search origin).
    /// * `source` — the same sensor cloud that was aligned (`base_link` frame).
    /// * `main_nvtl` — the main result's nearest-voxel transformation likelihood (softmax reference).
    /// * `scratch` — caller-owned workspace reused for the covariance candidate re-aligns/scores.
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn estimate_covariance(
        &self,
        result_pose: &Matrix4<f32>,
        hessian: &Matrix6<f64>,
        initial_pose: &Matrix4<f32>,
        source: &[[f32; 3]],
        main_nvtl: f32,
        scratch: &mut MatchScratch,
    ) -> Result<CovEstimationResult, AlignError> {
        let cfg = self.load_state().cov_config.clone();
        // Hessian → row-major [f64; 36]. nalgebra stores column-major, so the transpose's slice is the
        // row-major layout (lengths are both 36 — `copy_from_slice` cannot mismatch here).
        let mut hess = [0.0_f64; 36];
        hess.copy_from_slice(hessian.transpose().as_slice());
        // map→base_link 3x3 rotation (row-major) from the result pose's rotation block (explicit literal
        // — fixed nalgebra indices, no computed index/arithmetic).
        let rot3x3 = [
            f64::from(result_pose[(0, 0)]),
            f64::from(result_pose[(0, 1)]),
            f64::from(result_pose[(0, 2)]),
            f64::from(result_pose[(1, 0)]),
            f64::from(result_pose[(1, 1)]),
            f64::from(result_pose[(1, 2)]),
            f64::from(result_pose[(2, 0)]),
            f64::from(result_pose[(2, 1)]),
            f64::from(result_pose[(2, 2)]),
        ];
        let params = CovEstimationParams {
            estimation_type: cfg.estimation_type,
            scale_factor: cfg.scale_factor,
            temperature: cfg.temperature,
            main_nvtl,
            output_pose_covariance: cfg.output_pose_covariance,
            map_to_base_link_rot3x3: rot3x3,
        };
        estimate_pose_covariance(
            self,
            result_pose,
            &hess,
            initial_pose,
            source,
            &cfg.offset_x,
            &cfg.offset_y,
            &params,
            scratch,
        )
    }

    /// Set (or, when `scale == 0`, clear) the longitudinal regularization toward `(x, y)`.
    #[allow(
        clippy::float_cmp,
        clippy::allow_attributes,
        reason = "exact == 0 is the C++ disable sentinel (setRegularizationPose vs unset)"
    )]
    pub fn set_regularization(&self, x: f32, y: f32, scale: f32) {
        let reg = if scale == 0.0 {
            None
        } else {
            Some(Regularization {
                pose_xy: [x, y],
                scale_factor: scale,
            })
        };
        swap_store(&self.reg, reg);
    }

    /// Add a target map tile keyed by raw cell-id bytes. Tile ids are ordered bytewise, so the
    /// finalized map is independent of update arrival order. Needs a following
    /// [`Self::create_kdtree`].
    ///
    /// # Arguments
    /// * `points` — the tile's map points in the map frame (`[x, y, z]`, metres).
    /// * `id` — raw tile key; re-adding the same byte sequence replaces that tile.
    pub fn add_target(&self, points: &[[f32; 3]], id: &[u8]) {
        #[cfg(feature = "std")]
        crate::capture::hook_tile(id, points);
        swap_rcu(&self.state, |s| {
            let mut n = s.clone();
            n.map.add_target(points, id);
            n
        });
    }

    /// Remove the map tile registered under the raw cell-id bytes; no-op if absent.
    pub fn remove_target(&self, id: &[u8]) {
        swap_rcu(&self.state, |s| {
            let mut n = s.clone();
            n.map.remove_target(id);
            n
        });
    }

    /// Rebuild the kd-tree over the current tiles' centroids (the C++ `createVoxelKdtree`).
    ///
    /// The engine finalizes a staging copy and publishes it only on success. A leaf-limit or build
    /// failure therefore leaves the previously published map intact. The effective bound is the
    /// smaller of `max_leaves` and this engine's [`AlignLimits::max_active_leaves`].
    /// # Errors
    /// Returns [`AlignError::MapLeafLimitExceeded`] when the staged map exceeds the effective leaf
    /// bound, [`AlignError::ArithmeticOverflow`] on checked-count overflow, or
    /// [`AlignError::MapBuildFailed`] when allocation or kd-tree construction fails.
    pub fn try_create_kdtree(&self, max_leaves: usize) -> Result<usize, AlignError> {
        let max_leaves = max_leaves.min(self.limits.max_active_leaves);
        let current = self.load_state();
        let mut next = (*current).clone();
        let leaves = next
            .map
            .try_create_kdtree(max_leaves)
            .map_err(|error| match error {
                MapBuildError::ArithmeticOverflow => AlignError::ArithmeticOverflow,
                MapBuildError::LeafLimitExceeded => AlignError::MapLeafLimitExceeded,
                MapBuildError::AllocationFailed | MapBuildError::KdTree(_) => {
                    AlignError::MapBuildFailed
                }
            })?;
        swap_store_arc(&self.state, Arc::new(next));
        Ok(leaves)
    }

    /// Rebuild using this engine's configured active-leaf limit.
    /// # Errors
    /// Returns the errors documented by [`Self::try_create_kdtree`].
    pub fn create_kdtree(&self) -> Result<usize, AlignError> {
        self.try_create_kdtree(self.limits.max_active_leaves)
    }

    /// Atomically publish `src`'s map/params/id state into this engine in **one** store — the
    /// map-update commit. The node's timer builds a fresh, fully-finalized map (tiles added/removed +
    /// kd-tree built) on a private staging engine, then commits it here so concurrent aligns switch to
    /// the complete new map in a single atomic step (never a partially-built / kd-tree-less map). The
    /// state `Arc` is shared, not deep-copied; the old snapshot lives until its last reader drops it.
    /// # Errors
    /// Returns [`AlignError::MapLeafLimitExceeded`] without changing this engine if `src` exceeds
    /// this engine's active-leaf limit.
    pub fn commit_from(&self, src: &NdtEngine) -> Result<(), AlignError> {
        if src.load_state().map.num_leaves() > self.limits.max_active_leaves {
            return Err(AlignError::MapLeafLimitExceeded);
        }
        swap_store_arc(&self.state, swap_load(&src.state));
        Ok(())
    }

    /// Number of active Gaussian leaves in the published map snapshot.
    #[must_use]
    pub fn active_leaf_count(&self) -> usize {
        self.load_state().map.num_leaves()
    }
    #[must_use]
    pub fn has_target(&self) -> bool {
        !self.load_state().map.is_empty()
    }

    /// The registered tile cell-ids, `BTreeMap`-sorted (each the raw cell-id bytes). Backs the node
    /// crate's `get_current_map_ids` C-ABI shim, which cannot reach the engine-private state.
    #[must_use]
    pub fn map_ids(&self) -> alloc::vec::Vec<alloc::vec::Vec<u8>> {
        self.load_state().map.cell_ids()
    }

    /// Align `source` from `guess` into the caller-owned `scratch` (result lands in
    /// [`MatchScratch::result_ref`]) — the core align entry point, and the only one under `mt`.
    /// Reusing scratch created by [`MatchScratch::try_for_limits`] makes the serial kernel
    /// allocation-free from the first frame.
    ///
    /// # Arguments
    /// * `guess` — initial pose estimate as a 4×4 homogeneous transform (the C++ `Matrix4f` guess).
    /// * `source` — the sensor cloud to align, in the `base_link` frame (`[x, y, z]`, metres).
    /// * `scratch` — caller-owned per-align workspace + result slot; reuse it across frames.
    /// # Errors
    /// Returns [`AlignError::SourcePointLimitExceeded`] when `source` exceeds `Pmax`,
    /// [`AlignError::IterationLimitExceeded`] when active parameters exceed `Imax`, and the numeric,
    /// arithmetic, stack, or workspace errors produced by [`crate::ndt::align`]. A 65th in-radius
    /// neighbor is non-fatal and is reported in the returned [`AlignDiagnostics`].
    pub fn align(
        &self,
        guess: &Matrix4<f32>,
        source: &[[f32; 3]],
        scratch: &mut MatchScratch,
    ) -> Result<AlignDiagnostics, AlignError> {
        // Env-gated real-drive input capture (NDT_CAPTURE_DIR): one OnceLock load when disabled.
        #[cfg(feature = "std")]
        if crate::capture::hook_dir().is_some() {
            crate::capture::hook_align(guess, &self.map_ids(), source);
        }
        let st = self.load_state();
        if source.len() > self.limits.max_source_points {
            return Err(AlignError::SourcePointLimitExceeded);
        }
        if st.params.max_iterations > self.limits.max_iterations {
            return Err(AlignError::IterationLimitExceeded);
        }
        // Fold the per-call regularization into a local params copy (the free `align` reads
        // `params.regularization`); the engine's stored params keep `regularization: None`.
        let params = NdtParams {
            regularization: *swap_load(&self.reg),
            ..st.params
        };
        align(
            &st.map,
            source,
            guess,
            &params,
            &mut scratch.workspace,
            &mut scratch.last,
        )
    }

    /// Score `cloud` (already in the target frame) without aligning, using the caller-owned
    /// `scratch` workspace — the C++ `calculateTransformationProbability`.
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn calc_transformation_probability(
        &self,
        cloud: &[[f32; 3]],
        scratch: &mut MatchScratch,
    ) -> Result<f64, AlignError> {
        let st = self.load_state();
        let gauss = gauss_constants(st.params.outlier_ratio, st.params.resolution);
        transformation_probability(
            &st.map,
            cloud,
            st.params.resolution,
            &gauss,
            &mut scratch.workspace,
        )
    }

    /// Nearest-voxel likelihood of `cloud` without aligning, using the caller-owned `scratch`
    /// workspace — the C++ `calculateNearestVoxelTransformationLikelihood`.
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn calc_nearest_voxel_likelihood(
        &self,
        cloud: &[[f32; 3]],
        scratch: &mut MatchScratch,
    ) -> Result<f64, AlignError> {
        let st = self.load_state();
        let gauss = gauss_constants(st.params.outlier_ratio, st.params.resolution);
        nearest_voxel_transformation_likelihood(
            &st.map,
            cloud,
            st.params.resolution,
            &gauss,
            &mut scratch.workspace,
        )
    }

    /// Per-point nearest-voxel score (the C++ `calculateNearestVoxelScoreEachPoint`) using the
    /// caller-owned `scratch` workspace; `out[i] > 0` iff point `i` found a neighbor. `out` is
    /// filled to `cloud.len()`.
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn nearest_voxel_score_each_point(
        &self,
        cloud: &[[f32; 3]],
        out: &mut alloc::vec::Vec<f32>,
        scratch: &mut MatchScratch,
    ) -> Result<(), AlignError> {
        let st = self.load_state();
        let gauss = gauss_constants(st.params.outlier_ratio, st.params.resolution);
        nearest_voxel_score_each_point(
            &st.map,
            cloud,
            st.params.resolution,
            &gauss,
            &mut scratch.workspace,
            out,
        )
    }

    /// The configured iteration cap (the C++ `getMaximumIterations`).
    #[must_use]
    pub fn max_iterations(&self) -> i32 {
        self.load_state().params.max_iterations
    }
}

// --- sensor-callback align orchestrator (std-gated node glue) ---
// Folds align + oscillation count + the convergence verdict into one Rust call against the live
// engine, so the C++ sensor callback no longer drives align through separate C++ glue plus the
// C++ `count_oscillation` and `evaluate_convergence` FFI. Reuses the existing engine align,
// `helper::count_oscillation`, and `convergence::evaluate_convergence` — all no_std, so this is part
// of the no_std rlib (the C-ABI `extern "C"` wrapper below stays `std`-gated).

/// The `score_estimation` scalars that gate convergence (mirrors `AwConvergenceInput`'s param fields).
#[derive(Clone, Copy, Debug, Default)]
pub struct ConvergenceParams {
    pub converged_param_type: i32,
    pub converged_param_transform_probability: f64,
    pub converged_param_nearest_voxel_transformation_likelihood: f64,
}

/// Result of [`run_align`]: alignment scalars, convergence verdict, and non-fatal diagnostics.
///
/// `diagnostics.neighbor_limit_exceeded` means at least one query retained 64 neighbors and found a
/// 65th. In that case `verdict.is_converged` is forced to `false`; the pose remains available for
/// logging and analysis but must not be published as a converged localization result.
#[derive(Clone, Copy, Debug)]
pub struct AlignOutcome {
    pub pose: Matrix4<f32>,
    pub transform_probability: f32,
    pub nearest_voxel_likelihood: f32,
    pub iteration_num: i32,
    pub max_iterations: i32,
    pub oscillation_num: i32,
    pub verdict: crate::convergence::ConvergenceVerdict,
    pub diagnostics: AlignDiagnostics,
}

/// Align the live engine from `guess`, derive the oscillation count and convergence verdict, and
/// keep the full result in caller-owned scratch. The whole align plus verdict reads one scratch
/// session, so concurrent callers cannot observe cross-call state. With `num_threads == 1` and
/// scratch created by [`MatchScratch::try_for_limits`], the path performs no heap allocation from
/// its first call. A neighbor-limit diagnostic forces a non-converged verdict.
/// # Errors
/// Returns the errors documented by [`NdtEngine::align`], plus
/// [`AlignError::WorkspaceCapacityExceeded`] if convergence-history capacity is insufficient.
pub fn run_align(
    engine: &NdtEngine,
    guess: &Matrix4<f32>,
    source: &[[f32; 3]],
    conv: &ConvergenceParams,
    scratch: &mut MatchScratch,
) -> Result<AlignOutcome, AlignError> {
    let diagnostics = engine.align(guess, source, scratch)?;
    let max_iterations = engine.max_iterations();
    if scratch.oscillation_positions.capacity() < scratch.last.transformation_array.len() {
        return Err(AlignError::WorkspaceCapacityExceeded);
    }
    scratch.oscillation_positions.clear();
    for matrix in &scratch.last.transformation_array {
        scratch.oscillation_positions.push([
            f64::from(matrix[(0, 3)]),
            f64::from(matrix[(1, 3)]),
            f64::from(matrix[(2, 3)]),
        ]);
    }
    let oscillation_num = crate::helper::count_oscillation(&scratch.oscillation_positions);
    let result = &scratch.last;
    let mut verdict =
        crate::convergence::evaluate_convergence(&crate::convergence::ConvergenceInput {
            iteration_num: result.iteration_num,
            max_iterations,
            oscillation_num,
            transform_probability: f64::from(result.transform_probability),
            nearest_voxel_transformation_likelihood: f64::from(result.nearest_voxel_likelihood),
            converged_param_type: conv.converged_param_type,
            converged_param_transform_probability: conv.converged_param_transform_probability,
            converged_param_nearest_voxel_transformation_likelihood: conv
                .converged_param_nearest_voxel_transformation_likelihood,
        });
    // Rust-only degenerate guard (documented divergence, doc/book/src/port/divergences.md): a
    // non-finite result pose is forced to non-converged so no consumer publishes NaN/Inf downstream
    // (the node gates every pose/TF publish on `is_converged`). C++ has no such gate; on the valid
    // domain the align pose is always finite, so this branch never fires there.
    if diagnostics.neighbor_limit_exceeded || !result.pose.iter().all(|v| v.is_finite()) {
        verdict.is_converged = false;
    }
    Ok(AlignOutcome {
        pose: result.pose,
        transform_probability: result.transform_probability,
        nearest_voxel_likelihood: result.nearest_voxel_likelihood,
        iteration_num: result.iteration_num,
        max_iterations,
        oscillation_num,
        verdict,
        diagnostics,
    })
}

// --- sensor-callback covariance orchestrator ---
// Folds the whole covariance block of callback_sensor_points_main (rotate the configured 6x6 cov,
// dispatch on the estimation type against the LIVE engine map, scale, and adjust) into one Rust call,
// so the C++ node no longer calls the templated estimate_xy_covariance_by_multi_ndt[_score],
// adjust_diagonal_covariance, propose_poses_to_search, or the rotate_covariance helper twin. Reuses
// crate::{helper,covariance,cov_estimate} — all no_std, so this orchestrator is part of the no_std rlib
// (only the C-ABI `Aw*` structs + the `extern "C"` wrapper below stay `std`-gated).

/// Params for [`estimate_pose_covariance`] (the `covariance` hyper-params + the result-pose rotation).
#[derive(Clone, Copy, Debug)]
pub struct CovEstimationParams {
    /// `0` = `FIXED_VALUE`, `1` = `LAPLACE_APPROXIMATION`, `2` = `MULTI_NDT`, `3` = `MULTI_NDT_SCORE`.
    pub estimation_type: i32,
    pub scale_factor: f64,
    pub temperature: f64,
    /// The main result's nearest-voxel likelihood (used by the `MULTI_NDT_SCORE` softmax).
    pub main_nvtl: f32,
    pub output_pose_covariance: [f64; 36],
    /// Row-major 3x3 map-from-`base_link` rotation, built C++-side from the result-pose quaternion.
    pub map_to_base_link_rot3x3: [f64; 9],
}

/// Result of [`estimate_pose_covariance`]: the full 6x6 output covariance + the debug pose arrays the
/// node publishes (`publish_kind`: `0` none, `1` `MULTI_NDT` publishes both, `2` `MULTI_NDT_SCORE`
/// publishes only the initial poses). Each pose vec is `[main, then per-candidate]`.
#[derive(Clone, Debug)]
pub struct CovEstimationResult {
    pub ndt_covariance: [f64; 36],
    pub publish_kind: i32,
    pub multi_ndt_result_poses: alloc::vec::Vec<Matrix4<f32>>,
    pub multi_initial_poses: alloc::vec::Vec<Matrix4<f32>>,
}

/// Pure covariance orchestrator, ported verbatim from `callback_sensor_points_main`'s covariance block
/// and the `estimate_covariance` method. Runs the `MULTI_NDT`/`MULTI_NDT_SCORE` re-align/score against
/// the live engine map (a loaded snapshot) — never a rebuilt one-tile map. No new math (wires ports).
/// `scratch` backs the candidate re-aligns/scores and is always caller-owned.
#[expect(
    clippy::too_many_arguments,
    reason = "covariance orchestrator wires the engine + result/initial poses + hessian + source + offsets + params + scratch; grouping would only relocate the same inputs"
)]
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn estimate_pose_covariance(
    engine: &NdtEngine,
    result_pose: &Matrix4<f32>,
    hessian: &[f64; 36],
    initial_pose: &Matrix4<f32>,
    source: &[[f32; 3]],
    offset_x: &[f64],
    offset_y: &[f64],
    params: &CovEstimationParams,
    scratch: &mut MatchScratch,
) -> Result<CovEstimationResult, AlignError> {
    use alloc::vec::Vec;

    let mut ndt_covariance = crate::helper::rotate_covariance(
        &params.output_pose_covariance,
        &params.map_to_base_link_rot3x3,
    );

    // FIXED_VALUE: the rotated configured covariance only (no 2x2 estimate/adjust — C++ short-circuit).
    if params.estimation_type == 0 {
        return Ok(CovEstimationResult {
            ndt_covariance,
            publish_kind: 0,
            multi_ndt_result_poses: Vec::new(),
            multi_initial_poses: Vec::new(),
        });
    }

    let st = engine.load_state();
    let main_ndt = AlignResult {
        pose: *result_pose,
        nearest_voxel_likelihood: params.main_nvtl,
        ..AlignResult::default()
    };
    let mut publish_kind = 0_i32;
    let mut multi_ndt_result_poses: Vec<Matrix4<f32>> = Vec::new();
    let mut multi_initial_poses: Vec<Matrix4<f32>> = Vec::new();

    let est: [f64; 4] = match params.estimation_type {
        // LAPLACE_APPROXIMATION
        1 => crate::covariance::laplace_xy_covariance(hessian),
        // MULTI_NDT: re-align each candidate against the live map; publish result + initial poses.
        2 => {
            let poses =
                crate::cov_estimate::propose_poses_to_search(result_pose, offset_x, offset_y);
            let r = crate::cov_estimate::estimate_xy_covariance_by_multi_ndt(
                &main_ndt,
                &poses,
                &st.map,
                source,
                &st.params,
                &mut scratch.workspace,
            )?;
            publish_kind = 1;
            multi_ndt_result_poses.push(*result_pose);
            multi_ndt_result_poses.extend_from_slice(&r.candidate_result_poses);
            multi_initial_poses.push(*initial_pose);
            multi_initial_poses.extend_from_slice(&poses);
            r.covariance
        }
        // MULTI_NDT_SCORE: score each candidate (no re-align); publish only the initial poses.
        3 => {
            let poses =
                crate::cov_estimate::propose_poses_to_search(result_pose, offset_x, offset_y);
            let r = crate::cov_estimate::estimate_xy_covariance_by_multi_ndt_score(
                &main_ndt,
                &poses,
                &st.map,
                source,
                &st.params,
                params.temperature,
                &mut scratch.workspace,
            )?;
            publish_kind = 2;
            multi_initial_poses.push(*initial_pose);
            multi_initial_poses.extend_from_slice(&poses);
            r.covariance
        }
        // Unknown type: C++ returns Identity * output_pose_covariance[0] (still scaled+adjusted below).
        _ => {
            let c0 = params.output_pose_covariance[0];
            [c0, 0.0, 0.0, c0]
        }
    };

    // scale (C++ `* scale_factor`) then adjust_diagonal_covariance with the result-pose 2x2 rotation.
    let scaled = [
        est[0] * params.scale_factor,
        est[1] * params.scale_factor,
        est[2] * params.scale_factor,
        est[3] * params.scale_factor,
    ];
    let rot2 = [
        f64::from(result_pose[(0, 0)]),
        f64::from(result_pose[(0, 1)]),
        f64::from(result_pose[(1, 0)]),
        f64::from(result_pose[(1, 1)]),
    ];
    let adj = crate::covariance::adjust_diagonal_covariance(
        &scaled,
        &rot2,
        params.output_pose_covariance[0],
        params.output_pose_covariance[7],
    );
    // C++: ndt_covariance[0+6*0]=adj(0,0), [1+6*1]=adj(1,1), [1+6*0]=adj(1,0), [0+6*1]=adj(0,1).
    ndt_covariance[0] = adj[0];
    ndt_covariance[7] = adj[3];
    ndt_covariance[1] = adj[2];
    ndt_covariance[6] = adj[1];

    Ok(CovEstimationResult {
        ndt_covariance,
        publish_kind,
        multi_ndt_result_poses,
        multi_initial_poses,
    })
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    unsafe_code,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::borrow_as_ptr,
    clippy::too_many_lines,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;
    use crate::ndt::{NdtParams, align};
    use alloc::vec::Vec;

    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    fn test_scratch(source_len: usize) -> MatchScratch {
        MatchScratch::try_with_capacity(source_len, 30).expect("reserve test scratch")
    }

    #[test]
    fn align_limits_reject_invalid_and_overflowing_envelopes() {
        assert_eq!(AlignLimits::new(0, 1, 1), Err(AlignError::InvalidLimits));
        assert_eq!(AlignLimits::new(1, 1, 31), Err(AlignError::InvalidLimits));
        assert_eq!(
            AlignLimits::new(usize::MAX, usize::MAX, 30),
            Err(AlignError::ArithmeticOverflow)
        );
    }

    #[test]
    fn bounded_engine_rejects_source_and_map_without_partial_publish() {
        let engine = NdtEngine::new(1.0, 6, 0.01, 1, 1, 30).expect("bounded engine");
        let source = [[0.0_f32; 3]; 2];
        let mut scratch =
            MatchScratch::try_with_capacity(source.len(), 30).expect("reserve scratch");
        assert_eq!(
            engine.align(&Matrix4::identity(), &source, &mut scratch),
            Err(AlignError::SourcePointLimitExceeded)
        );

        engine.add_target(&dense_cluster(0.5, 0.5, 0.5), b"0");
        engine.add_target(&dense_cluster(4.5, 0.5, 0.5), b"1");
        assert_eq!(
            engine.create_kdtree(),
            Err(AlignError::MapLeafLimitExceeded)
        );
        assert_eq!(engine.active_leaf_count(), 0);
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn serial_deployment_scratch_payload_is_fixed() {
        let limits = AlignLimits::new(2_000, 418_000, 30).expect("deployment limits");
        let scratch = MatchScratch::try_for_limits(limits).expect("reserve deployment scratch");
        assert_eq!(scratch.allocated_payload_bytes(), Ok(27_488));
    }

    fn configured(engine: &mut NdtEngine) {
        engine.set_params(0.01, 0.1, 1.0, 30, 0.55, 1);
    }

    // Building the map incrementally through the engine + aligning equals a one-shot ndt::align on a
    // map with the same two tiles (the engine is a thin stateful wrapper — bit-identical).
    #[test]
    fn engine_align_matches_one_shot() {
        let tile_a = dense_cluster(0.5, 0.5, 0.5);
        let tile_b = dense_cluster(4.5, 0.5, 0.5);
        let source: Vec<[f32; 3]> = tile_a
            .iter()
            .chain(tile_b.iter())
            .map(|p| [p[0] + 0.1, p[1] - 0.05, p[2]])
            .collect();
        let guess = Matrix4::<f32>::identity();

        let mut engine = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut engine);
        engine.add_target(&tile_a, b"0");
        engine.add_target(&tile_b, b"1");
        engine.create_kdtree().expect("build kd-tree");
        assert!(engine.has_target());
        let mut scratch = test_scratch(source.len());
        engine.align(&guess, &source, &mut scratch).expect("align");
        let got = scratch.result_ref();

        let mut map = VoxelGridMap::new([1.0; 3], 6, 0.01);
        map.add_target(&tile_a, b"0");
        map.add_target(&tile_b, b"1");
        map.try_create_kdtree(418_000).expect("build kd-tree");
        let params = NdtParams {
            trans_epsilon: 0.01,
            step_size: 0.1,
            resolution: 1.0,
            max_iterations: 30,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1,
        };
        let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
        let mut want = AlignResult::try_with_capacity(30).expect("reserve result");
        align(&map, &source, &guess, &params, &mut ws, &mut want).expect("align");

        assert_eq!(got.iteration_num, want.iteration_num);
        assert_eq!(got.pose, want.pose);
        assert_eq!(got.hessian, want.hessian);
        assert_eq!(got.transform_probability, want.transform_probability);
        assert_eq!(got.nearest_voxel_likelihood, want.nearest_voxel_likelihood);
    }

    fn two_tile_engine() -> (NdtEngine, Vec<[f32; 3]>) {
        let tile_a = dense_cluster(0.5, 0.5, 0.5);
        let tile_b = dense_cluster(4.5, 0.5, 0.5);
        let source: Vec<[f32; 3]> = tile_a
            .iter()
            .chain(tile_b.iter())
            .map(|p| [p[0] + 0.1, p[1] - 0.05, p[2]])
            .collect();
        let mut engine = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut engine);
        engine.add_target(&tile_a, b"0");
        engine.add_target(&tile_b, b"1");
        engine.create_kdtree().expect("build kd-tree");
        (engine, source)
    }

    const TP_PARAMS: ConvergenceParams = ConvergenceParams {
        converged_param_type: 0, // TRANSFORM_PROBABILITY
        converged_param_transform_probability: 0.0,
        converged_param_nearest_voxel_transformation_likelihood: 0.0,
    };

    // The verdict gate: a non-finite result pose must never report converged, even when every
    // score/iteration criterion passes. A NaN guess propagates into the result pose (the
    // derivatives treat NaN points as no-ops, so iteration/oscillation/score all "pass").
    #[test]
    fn run_align_gates_non_finite_pose_to_not_converged() {
        let (engine, source) = two_tile_engine();
        let conv = ConvergenceParams {
            converged_param_type: 1, // NVTL
            converged_param_transform_probability: 0.0,
            // Negative threshold: score 0.0 clears it, so only the pose gate can veto.
            converged_param_nearest_voxel_transformation_likelihood: -1.0,
        };
        let mut nan_guess = Matrix4::<f32>::identity();
        nan_guess[(0, 3)] = f32::NAN;
        let mut scratch =
            MatchScratch::try_with_capacity(source.len(), 30).expect("reserve align scratch");
        assert!(
            matches!(
                run_align(&engine, &nan_guess, &source, &conv, &mut scratch),
                Err(AlignError::NonFiniteValue)
            ),
            "NaN guess must be rejected"
        );
        // Valid domain unchanged: the same conv params with a finite guess converge.
        let ok = run_align(&engine, &Matrix4::identity(), &source, &conv, &mut scratch)
            .expect("finite align");
        assert!(ok.pose.iter().all(|v| v.is_finite()));
        assert!(ok.verdict.is_converged);
    }

    // run_align == composing engine.align + result + helper::count_oscillation + evaluate_convergence
    // by hand (the orchestrator adds no new math, just folds the existing ports).
    #[test]
    fn run_align_matches_manual_composition() {
        let (engine, source) = two_tile_engine();
        let guess = Matrix4::<f32>::identity();
        let mut scratch = test_scratch(source.len());
        let outcome =
            run_align(&engine, &guess, &source, &TP_PARAMS, &mut scratch).expect("align outcome");

        // Manual reference on a fresh, identically-built engine.
        let (ref_engine, ref_source) = two_tile_engine();
        let mut ref_scratch = test_scratch(ref_source.len());
        ref_engine
            .align(&guess, &ref_source, &mut ref_scratch)
            .expect("align");
        let max_it = ref_engine.max_iterations();
        let r = ref_scratch.result_ref();
        let positions: Vec<[f64; 3]> = r
            .transformation_array
            .iter()
            .map(|m| {
                [
                    f64::from(m[(0, 3)]),
                    f64::from(m[(1, 3)]),
                    f64::from(m[(2, 3)]),
                ]
            })
            .collect();
        let osc = crate::helper::count_oscillation(&positions);
        let verdict =
            crate::convergence::evaluate_convergence(&crate::convergence::ConvergenceInput {
                iteration_num: r.iteration_num,
                max_iterations: max_it,
                oscillation_num: osc,
                transform_probability: f64::from(r.transform_probability),
                nearest_voxel_transformation_likelihood: f64::from(r.nearest_voxel_likelihood),
                converged_param_type: 0,
                converged_param_transform_probability: 0.0,
                converged_param_nearest_voxel_transformation_likelihood: 0.0,
            });

        assert_eq!(outcome.pose, r.pose);
        assert_eq!(outcome.transform_probability, r.transform_probability);
        assert_eq!(outcome.nearest_voxel_likelihood, r.nearest_voxel_likelihood);
        assert_eq!(outcome.iteration_num, r.iteration_num);
        assert_eq!(outcome.max_iterations, max_it);
        assert_eq!(outcome.oscillation_num, osc);
        assert_eq!(outcome.verdict, verdict);
    }

    // --- covariance orchestrator tests ---

    const ROT3X3_ID: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    const COV_OX: [f64; 4] = [0.3, -0.3, 0.0, 0.0];
    const COV_OY: [f64; 4] = [0.0, 0.0, 0.3, -0.3];

    fn out_cov() -> [f64; 36] {
        let mut c = [0.0_f64; 36];
        c[0] = 0.25;
        c[7] = 0.36;
        c[14] = 0.01;
        c[21] = 0.01;
        c[28] = 0.01;
        c[35] = 0.01;
        c
    }

    fn cov_params(estimation_type: i32, scale_factor: f64) -> CovEstimationParams {
        CovEstimationParams {
            estimation_type,
            scale_factor,
            temperature: 0.1,
            main_nvtl: 1.0,
            output_pose_covariance: out_cov(),
            map_to_base_link_rot3x3: ROT3X3_ID,
        }
    }

    // Align an engine once and return (engine, source, result_pose) for the cov tests.
    fn aligned_engine() -> (NdtEngine, Vec<[f32; 3]>, Matrix4<f32>) {
        let (engine, source) = two_tile_engine();
        let mut scratch = test_scratch(source.len());
        engine
            .align(&Matrix4::<f32>::identity(), &source, &mut scratch)
            .expect("align");
        let pose = scratch.result_ref().pose;
        (engine, source, pose)
    }

    fn rot2_of(pose: &Matrix4<f32>) -> [f64; 4] {
        [
            f64::from(pose[(0, 0)]),
            f64::from(pose[(0, 1)]),
            f64::from(pose[(1, 0)]),
            f64::from(pose[(1, 1)]),
        ]
    }

    #[test]
    fn estimate_pose_covariance_fixed_just_rotates() {
        let (engine, source, result_pose) = aligned_engine();
        let mut scratch = test_scratch(source.len());
        let r = estimate_pose_covariance(
            &engine,
            &result_pose,
            &[0.0; 36],
            &Matrix4::identity(),
            &source,
            &COV_OX,
            &COV_OY,
            &cov_params(0, 1.0),
            &mut scratch,
        )
        .expect("pose covariance estimate");
        assert_eq!(r.publish_kind, 0);
        assert!(r.multi_ndt_result_poses.is_empty() && r.multi_initial_poses.is_empty());
        // FIXED → just the rotated configured covariance (identity rotation → unchanged).
        assert_eq!(
            r.ndt_covariance,
            crate::helper::rotate_covariance(&out_cov(), &ROT3X3_ID)
        );
    }

    #[test]
    fn estimate_pose_covariance_multi_ndt_matches_composition() {
        let (engine, source, result_pose) = aligned_engine();
        let mut scratch = test_scratch(source.len());
        let r = estimate_pose_covariance(
            &engine,
            &result_pose,
            &[0.0; 36],
            &Matrix4::identity(),
            &source,
            &COV_OX,
            &COV_OY,
            &cov_params(2, 2.0),
            &mut scratch,
        )
        .expect("pose covariance estimate");

        // Manual reference on a fresh identical engine.
        let (ref_engine, ref_source) = two_tile_engine();
        let ref_state = ref_engine.load_state();
        let poses = crate::cov_estimate::propose_poses_to_search(&result_pose, &COV_OX, &COV_OY);
        let main_ndt = AlignResult {
            pose: result_pose,
            ..AlignResult::default()
        };
        let mut ws = AlignWorkspace::try_with_capacity(ref_source.len())
            .expect("reserve covariance workspace");
        let mr = crate::cov_estimate::estimate_xy_covariance_by_multi_ndt(
            &main_ndt,
            &poses,
            &ref_state.map,
            &ref_source,
            &ref_state.params,
            &mut ws,
        )
        .expect("multi-NDT covariance estimate");
        let scaled = [
            mr.covariance[0] * 2.0,
            mr.covariance[1] * 2.0,
            mr.covariance[2] * 2.0,
            mr.covariance[3] * 2.0,
        ];
        let adj = crate::covariance::adjust_diagonal_covariance(
            &scaled,
            &rot2_of(&result_pose),
            out_cov()[0],
            out_cov()[7],
        );
        let mut expected = crate::helper::rotate_covariance(&out_cov(), &ROT3X3_ID);
        expected[0] = adj[0];
        expected[7] = adj[3];
        expected[1] = adj[2];
        expected[6] = adj[1];

        assert_eq!(r.ndt_covariance, expected);
        assert_eq!(r.publish_kind, 1);
        assert_eq!(r.multi_ndt_result_poses.len(), poses.len() + 1);
        assert_eq!(r.multi_initial_poses.len(), poses.len() + 1);
        // The first published pose is the main result / initial pose.
        assert_eq!(r.multi_ndt_result_poses[0], result_pose);
        assert_eq!(r.multi_initial_poses[0], Matrix4::<f32>::identity());
    }

    #[test]
    fn estimate_pose_covariance_score_publishes_initial_only() {
        let (engine, source, result_pose) = aligned_engine();
        let mut scratch = test_scratch(source.len());
        let r = estimate_pose_covariance(
            &engine,
            &result_pose,
            &[0.0; 36],
            &Matrix4::identity(),
            &source,
            &COV_OX,
            &COV_OY,
            &cov_params(3, 1.0),
            &mut scratch,
        )
        .expect("pose covariance estimate");
        assert_eq!(r.publish_kind, 2);
        assert!(r.multi_ndt_result_poses.is_empty());
        assert_eq!(r.multi_initial_poses.len(), COV_OX.len() + 1);
    }

    // clone() is an independent deep copy: mutating the original's map after cloning must not affect
    // the clone's alignment.
    #[test]
    fn engine_clone_is_independent() {
        let tile_a = dense_cluster(0.5, 0.5, 0.5);
        let tile_b = dense_cluster(4.5, 0.5, 0.5);
        let source: Vec<[f32; 3]> = tile_a.iter().map(|p| [p[0] + 0.1, p[1], p[2]]).collect();
        let guess = Matrix4::<f32>::identity();

        let mut original = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut original);
        original.add_target(&tile_a, b"0");
        original.add_target(&tile_b, b"1");
        original.create_kdtree().expect("build kd-tree");

        let mut scratch = test_scratch(source.len());
        let clone = original.clone();
        clone.align(&guess, &source, &mut scratch).expect("align");
        let clone_before = scratch.result_ref().pose;

        // Mutate the original after the clone; the clone must be unaffected.
        original.remove_target(b"1");
        original.create_kdtree().expect("build kd-tree");
        original
            .align(&guess, &source, &mut scratch)
            .expect("align");

        clone.align(&guess, &source, &mut scratch).expect("align");
        assert_eq!(
            scratch.result_ref().pose,
            clone_before,
            "clone shares state with original"
        );
    }

    #[test]
    fn engine_remove_target_and_has_target() {
        let engine = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        assert!(!engine.has_target());
        engine.add_target(&dense_cluster(0.5, 0.5, 0.5), b"0");
        engine.add_target(&dense_cluster(4.5, 0.5, 0.5), b"1");
        assert!(engine.has_target());
        engine.remove_target(b"0");
        engine.remove_target(b"1");
        assert!(!engine.has_target());
    }

    // --- engine-owned canonical cell-id map ---

    fn ids_of(engine: &NdtEngine) -> Vec<Vec<u8>> {
        engine.map_ids()
    }

    #[test]
    fn cell_id_add_remove_and_sorted_ids() {
        let tile = dense_cluster(0.5, 0.5, 0.5);
        let mut engine = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut engine);
        engine.add_target(&tile, b"beta");
        engine.add_target(&tile, b"alpha");
        // BTreeMap-sorted, deterministic.
        assert_eq!(ids_of(&engine), vec![b"alpha".to_vec(), b"beta".to_vec()]);
        assert!(engine.has_target());

        engine.remove_target(b"alpha");
        assert_eq!(ids_of(&engine), vec![b"beta".to_vec()]);
        // Removing an unknown id is a no-op.
        engine.remove_target(b"missing");
        assert_eq!(ids_of(&engine), vec![b"beta".to_vec()]);
    }

    #[test]
    fn cell_id_replaces_existing_tile() {
        let tile = dense_cluster(0.5, 0.5, 0.5);
        let mut engine = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut engine);
        engine.add_target(&tile, b"0");
        engine.add_target(&tile, b"0"); // same cell-id replaces the tile
        assert_eq!(ids_of(&engine), vec![b"0".to_vec()]);
    }

    #[test]
    fn clone_carries_cell_ids() {
        let tile = dense_cluster(0.5, 0.5, 0.5);
        let mut engine = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut engine);
        engine.add_target(&tile, b"0");
        engine.add_target(&tile, b"1");
        let clone = engine.clone();
        // Mutating the original must not affect the clone's cell-id map.
        engine.remove_target(b"0");
        assert_eq!(ids_of(&engine), vec![b"1".to_vec()]);
        assert_eq!(ids_of(&clone), vec![b"0".to_vec(), b"1".to_vec()]);
    }

    fn assert_same_canonical_map(left: &NdtEngine, right: &NdtEngine) {
        assert_eq!(left.map_ids(), right.map_ids());
        let left_state = left.load_state();
        let right_state = right.load_state();
        assert_eq!(left_state.map.num_leaves(), right_state.map.num_leaves());
        for idx in 0..left_state.map.num_leaves() {
            let left_leaf = left_state.map.leaf(idx);
            let right_leaf = right_state.map.leaf(idx);
            assert!(left_leaf.is_some(), "left leaf {idx} missing");
            assert!(right_leaf.is_some(), "right leaf {idx} missing");
            let (Some(left_leaf), Some(right_leaf)) = (left_leaf, right_leaf) else {
                return;
            };
            assert_eq!(left_leaf.n, right_leaf.n);
            assert_eq!(
                left_leaf.mean.map(f64::to_bits),
                right_leaf.mean.map(f64::to_bits)
            );
            assert_eq!(
                left_leaf.icov.map(f64::to_bits),
                right_leaf.icov.map(f64::to_bits)
            );
            assert_eq!(
                left_leaf.centroid.map(f32::to_bits),
                right_leaf.centroid.map(f32::to_bits)
            );

            let mut left_neighbors = Vec::new();
            let mut right_neighbors = Vec::new();
            left_state
                .map
                .radius_search(
                    left_leaf.centroid,
                    left_state.params.resolution,
                    0,
                    &mut left_neighbors,
                )
                .expect("radius search");
            right_state
                .map
                .radius_search(
                    right_leaf.centroid,
                    right_state.params.resolution,
                    0,
                    &mut right_neighbors,
                )
                .expect("radius search");
            assert_eq!(left_neighbors, right_neighbors);
        }
    }

    fn canonical_tiles() -> [([u8; 5], Vec<[f32; 3]>); 3] {
        [
            (*b"gamma", dense_cluster(8.5, 0.5, 0.5)),
            (*b"alpha", dense_cluster(0.5, 0.5, 0.5)),
            (*b"bravo", dense_cluster(4.5, 0.5, 0.5)),
        ]
    }

    fn cell_engine(tiles: &[([u8; 5], Vec<[f32; 3]>); 3], order: [usize; 3]) -> NdtEngine {
        let mut engine = NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut engine);
        for index in order {
            let (id, points) = &tiles[index];
            engine.add_target(points, id);
        }
        engine.create_kdtree().expect("build kd-tree");
        engine
    }

    #[test]
    fn cell_tile_permutations_build_identical_map_and_alignment() {
        let tiles = canonical_tiles();
        let baseline = cell_engine(&tiles, [0, 1, 2]);
        let source: Vec<[f32; 3]> = tiles
            .iter()
            .flat_map(|(_, points)| points.iter().copied())
            .collect();
        let guess = Matrix4::<f32>::identity();
        let mut baseline_scratch = test_scratch(source.len());
        baseline
            .align(&guess, &source, &mut baseline_scratch)
            .expect("align");
        let baseline_result = baseline_scratch.result_ref();

        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let candidate = cell_engine(&tiles, order);
            assert_same_canonical_map(&baseline, &candidate);

            let mut scratch = test_scratch(source.len());
            candidate
                .align(&guess, &source, &mut scratch)
                .expect("align");
            let result = scratch.result_ref();
            assert_eq!(result.pose, baseline_result.pose);
            assert_eq!(result.hessian, baseline_result.hessian);
            assert_eq!(result.iteration_num, baseline_result.iteration_num);
            assert_eq!(
                result.transform_probability.to_bits(),
                baseline_result.transform_probability.to_bits()
            );
            assert_eq!(
                result.nearest_voxel_likelihood.to_bits(),
                baseline_result.nearest_voxel_likelihood.to_bits()
            );
        }
    }

    #[test]
    fn cell_tile_update_history_builds_identical_final_map() {
        let tiles = canonical_tiles();
        let baseline = cell_engine(&tiles, [0, 1, 2]);

        let mut historical =
            NdtEngine::new(1.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        configured(&mut historical);
        for index in [2, 0] {
            let (id, points) = &tiles[index];
            historical.add_target(points, id);
            historical.create_kdtree().expect("build kd-tree");
        }
        let (alpha_id, alpha_points) = &tiles[1];
        historical.add_target(alpha_points, alpha_id);
        historical.create_kdtree().expect("build kd-tree");
        historical.remove_target(alpha_id);
        historical.create_kdtree().expect("build kd-tree");
        historical.add_target(alpha_points, alpha_id);
        historical.create_kdtree().expect("build kd-tree");

        assert_same_canonical_map(&baseline, &historical);
    }

    // #2: set_regularization must fold into the align path and shift the converged pose.
    #[test]
    fn set_regularization_changes_align_result() {
        let (engine, source) = two_tile_engine();
        let guess = Matrix4::<f32>::identity();
        let mut scratch = test_scratch(source.len());

        engine.set_regularization(0.0, 0.0, 0.0); // cleared (scale 0 => None)
        engine.align(&guess, &source, &mut scratch).expect("align");
        let pose_none = scratch.result_ref().pose;

        engine.set_regularization(5.0, 5.0, 10.0); // strong pull toward (5, 5)
        engine.align(&guess, &source, &mut scratch).expect("align");
        let pose_reg = scratch.result_ref().pose;

        assert!(
            (pose_reg - pose_none).norm() > 1e-4_f32,
            "regularization must fold into align and shift the result"
        );
    }

    // #3: the self-contained estimate_covariance wrapper must equal a manual composition — pinning
    // its rot3x3 extraction from the result pose + the covariance-config plumbing.
    #[test]
    fn estimate_covariance_wrapper_matches_manual_composition() {
        let (engine, source, _pose) = aligned_engine();
        engine.set_covariance_config(2, 2.0, 0.1, out_cov(), &COV_OX, &COV_OY);
        // A result pose with a clear yaw so the rot3x3 extraction is actually exercised.
        let result_pose =
            crate::transform::se3_matrix_f32(&nalgebra::Vector6::new(0.1, 0.2, 0.0, 0.0, 0.0, 0.5));
        let mut scratch = test_scratch(source.len());

        let wrapped = engine
            .estimate_covariance(
                &result_pose,
                &Matrix6::zeros(),
                &Matrix4::identity(),
                &source,
                1.0,
                &mut scratch,
            )
            .expect("wrapped covariance estimate");

        // Manual composition: the wrapper builds rot3x3 (row-major) from the pose and reads the same
        // params from the engine's covariance config.
        let rot3x3 = [
            f64::from(result_pose[(0, 0)]),
            f64::from(result_pose[(0, 1)]),
            f64::from(result_pose[(0, 2)]),
            f64::from(result_pose[(1, 0)]),
            f64::from(result_pose[(1, 1)]),
            f64::from(result_pose[(1, 2)]),
            f64::from(result_pose[(2, 0)]),
            f64::from(result_pose[(2, 1)]),
            f64::from(result_pose[(2, 2)]),
        ];
        let params = CovEstimationParams {
            estimation_type: 2,
            scale_factor: 2.0,
            temperature: 0.1,
            main_nvtl: 1.0,
            output_pose_covariance: out_cov(),
            map_to_base_link_rot3x3: rot3x3,
        };
        let manual = estimate_pose_covariance(
            &engine,
            &result_pose,
            &[0.0; 36],
            &Matrix4::identity(),
            &source,
            &COV_OX,
            &COV_OY,
            &params,
            &mut scratch,
        )
        .expect("pose covariance estimate");
        assert_eq!(wrapped.ndt_covariance, manual.ndt_covariance);
    }
}
