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

//! Pure-Rust core of the Autoware NDT (Normal Distributions Transform) localization engine.
//!
//! This crate is **ROS-free** and `no_std`-capable (heap only, via `alloc`). It carries **no**
//! `extern "C"` — the C ABI and all ROS integration live in the sibling `autoware_ndt_scan_matcher_rs`
//! node crate, which wraps this crate's public Rust API. The numeric kernels mirror the C++
//! `Matrix4f` / `MultiVoxelGridCovariance` pipeline and are reusable under ROS, a bare-metal kernel,
//! or an async runtime.
//!
//! # Entry points
//!
//! - [`engine::NdtEngine`] — the persistent, `&self`-only NDT handle: load a target map, build the
//!   kd-tree, then align sensor clouds. This is the primary API.
//! - [`scan_matcher::ScanMatcher`] — portable node orchestration over [`engine::NdtEngine`] and the
//!   [`host`] ports (map update + single-scan match).
//! - [`ndt::align`] — the RT-critical, WCET-bounded alignment kernel the engine drives.
//! - [`tpe::TreeStructuredParzenEstimator`] — the align-service pose-search sampler.
//!
//! Deployment code constructs an engine with [`engine::NdtEngine::new`]. Its immutable
//! work envelope contains the maximum source-point count (`Pmax`), maximum published active-leaf
//! count (`Lmax`), and maximum Newton iterations (`Imax`, constrained to `0..=30`). `Pmax` sizes and
//! checks caller-owned scratch; `Lmax` is a map-admission bound, not a reservation of `Lmax` leaves.
//! Increasing either bound admits more work but proportionally weakens the structural work bound.
//!
//! Pose guesses and result matrices use [`nalgebra`] types (`Matrix4<f32>`, `Matrix6<f64>`). The
//! exact `nalgebra` version this crate is built against is re-exported as [`nalgebra`] so callers can
//! construct those matrices without independently pinning a (possibly mismatched) version.
//!
//! # Cargo features
//!
//! | Feature | Default | Effect |
//! |---|---|---|
//! | `std` | yes | Host build: lock-free `ArcSwap` engine state + caller-owned [`engine::MatchScratch`]; the engine is `Sync`. |
//! | `parallel` | yes | rayon-backed derivative reduction (implies `std`); bit-identical to serial, a pure throughput option. |
//! | `mt` | no | Multi-core `no_std` (kernel): `awkernel_sync` mutex cells + caller-owned [`engine::MatchScratch`]; engine is `Sync`. Ignored when `std` is on. |
//!
//! With **no** features (`--no-default-features`) the engine is a single-core `no_std` build
//! (`RefCell` state cells, caller-owned scratch, intentionally `!Sync`). See the [`engine`] module docs for
//! the full interior-mutability matrix.
//!
//! # Example
//!
//! Load a one-tile target map, build the kd-tree, and align a source cloud from an identity guess:
//!
//! ```
//! use realtime_ndt_scan_matcher::engine::{MatchScratch, NdtEngine};
//! use realtime_ndt_scan_matcher::nalgebra::Matrix4;
//!
//! // Empty engine: 2.0 m voxels; `MultiVoxelGridCovariance` defaults (min 6 points / eig 0.01).
//! let engine = NdtEngine::new(2.0, 6, 0.01, 64, 64, 30)
//!     .expect("valid work envelope");
//!
//! // Register a target map tile (id 0) and build the kd-tree over the voxel centroids.
//! let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
//! engine.add_target(&target, b"0");
//! engine.create_kdtree().expect("map fits the leaf limit");
//! assert!(engine.has_target());
//!
//! // Preallocate once, then reuse this scratch on every frame.
//! let source = target.clone();
//! let mut scratch = MatchScratch::try_for_limits(engine.limits()).expect("reserve scratch");
//! engine
//!     .align(&Matrix4::identity(), &source, &mut scratch)
//!     .expect("input is inside the work envelope");
//! let result = scratch.result_ref();
//! assert!(result.iteration_num >= 0);
//! ```

// no_std for the awkernel/no_std build; `std` (default) for dev and the ROS-node build. The
// `mt` feature makes the no_std engine multi-core-safe (awkernel_sync cells + caller-owned
// `MatchScratch`); without it the no_std build is single-core (`RefCell`, `!Sync`).
// Test builds always use std (the test harness + `Vec`/etc. need it), regardless of features.
#![cfg_attr(not(any(test, feature = "std")), no_std)]

// Heap types (Vec/Box/BTreeMap) for the engine data structures. The allocator is provided by std
// (host) or the kernel; the no_std rlib defers it to the final binary.
extern crate alloc;

pub mod convergence;
pub mod cov_estimate;
pub mod covariance;
pub mod derivatives;
pub mod engine;
pub mod helper;
// Portable node orchestration: the `Host` port traits + the `no_std` scan matcher over the engine
// (reusable on ROS / bare-metal / the Tokio example).
pub mod host;
mod kdtree;
pub mod ndt;
// no_std port of autoware_localization_util::SmartPoseBuffer (time-ordered pose interpolation buffer);
// reused by the node crate's Rust-owned pose buffers.
pub mod pose_buffer;
pub mod scan_matcher;
pub mod tpe;
pub mod transform;
pub mod voxel_grid;
// Real-drive input capture (the NDT_CAPTURE_DIR sidecar format); std-only file IO.
#[cfg(feature = "std")]
pub mod capture;
// Frozen WCET benchmark fixtures (capture-once, replay-everywhere); std-only file IO.
#[cfg(feature = "std")]
pub mod fixture;
// Deterministic algorithmic-cost counters for the WCET analysis (plan/ndt_wcet.md); opt-in.
#[cfg(feature = "wcet-count")]
pub mod wcet;

/// Re-export of the exact [`nalgebra`] version this crate is built against.
///
/// The engine's pose API is expressed in `nalgebra` matrices (`Matrix4<f32>` guesses/results,
/// `Matrix6<f64>` Hessians/covariances). Construct and read them through this re-export so the type
/// identities match the engine's — a locally pinned `nalgebra` of a different version would be a
/// distinct, incompatible type.
pub use nalgebra;

/// Per-worker CPU-affinity plan for the `parallel` pool, resolved once from the environment.
///
/// Two environment variables control it, mirroring OpenMP's `GOMP_CPU_AFFINITY` (the C++ engine's
/// knob), in precedence order:
///  1. `NDT_RAYON_CPU_AFFINITY` — an explicit CPU list in `GOMP_CPU_AFFINITY` syntax
///     (whitespace/comma-separated tokens, each `N`, `N-M`, or `N-M:S`; e.g. `"2 4 6 8"`,
///     `"2,4,6,8"`, `"2-8:2"`). Worker `i` is pinned to the `i`-th listed CPU, wrapping when there
///     are more workers than list entries (as GOMP does).
///  2. `NDT_PIN_RAYON_WORKERS` (set to anything) — auto: pin worker `i` to the `i`-th CPU allowed
///     in the process's current affinity mask (the caller's `taskset`/cpuset).
///  3. neither set — no pinning (the scheduler's default; production behavior is unchanged).
///
/// Pinning matters on isolated cores: `isolcpus` disables the kernel's automatic load balancing,
/// so unpinned rayon workers pile onto a single core. Explicit affinity spreads them one-per-core,
/// exactly as an OpenMP deployment does with `GOMP_CPU_AFFINITY`.
#[cfg(feature = "parallel")]
enum PinPlan {
    None,
    Cpuset,
    List(alloc::vec::Vec<usize>),
}

#[cfg(feature = "parallel")]
impl PinPlan {
    fn from_env() -> Self {
        if let Some(list) = std::env::var("NDT_RAYON_CPU_AFFINITY")
            .ok()
            .map(|s| parse_cpu_affinity(&s))
            .filter(|v| !v.is_empty())
        {
            Self::List(list)
        } else if std::env::var_os("NDT_PIN_RAYON_WORKERS").is_some() {
            Self::Cpuset
        } else {
            Self::None
        }
    }

    fn apply(&self, idx: usize) {
        match self {
            Self::None => {}
            Self::Cpuset => pin_worker_to_cpuset(idx),
            Self::List(cpus) => {
                // Wrap round-robin when there are more workers than list entries (as GOMP does);
                // `checked_rem` yields `None` for an empty list, so no panic and no bare `%`.
                if let Some(&cpu) = idx.checked_rem(cpus.len()).and_then(|m| cpus.get(m)) {
                    set_thread_affinity_single(cpu);
                }
            }
        }
    }
}

/// Parse a `GOMP_CPU_AFFINITY`-style CPU list: whitespace/comma-separated tokens, each `N`,
/// `N-M` (inclusive range), or `N-M:S` (range with stride). Malformed tokens are skipped
/// (best-effort, never panics); the result is capped at [`libc::CPU_SETSIZE`] entries.
#[cfg(feature = "parallel")]
#[must_use]
fn parse_cpu_affinity(s: &str) -> alloc::vec::Vec<usize> {
    #[cfg(target_os = "linux")]
    #[expect(
        clippy::as_conversions,
        reason = "libc::CPU_SETSIZE is a small positive c_int"
    )]
    let cap = libc::CPU_SETSIZE as usize;
    #[cfg(not(target_os = "linux"))]
    let cap = 4096usize;

    let mut out: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
    for tok in s.split([' ', '\t', ',', '\n']).filter(|t| !t.is_empty()) {
        // Split off an optional ":stride".
        let (range, stride) = match tok.split_once(':') {
            Some((r, st)) => (r, st.parse::<usize>().ok().filter(|&x| x >= 1).unwrap_or(1)),
            None => (tok, 1),
        };
        let (lo, hi) = match range.split_once('-') {
            Some((a, b)) => match (a.parse::<usize>(), b.parse::<usize>()) {
                (Ok(a), Ok(b)) if b >= a => (a, b),
                _ => continue,
            },
            None => match range.parse::<usize>() {
                Ok(v) => (v, v),
                Err(_) => continue,
            },
        };
        let mut cpu = lo;
        while cpu <= hi {
            if out.len() >= cap {
                return out;
            }
            out.push(cpu);
            cpu = cpu.saturating_add(stride);
        }
    }
    out
}

/// Size rayon's process-global worker pool used by the `parallel` derivative reduction.
///
/// The parallel backend runs on rayon's process-global thread pool; this sizes that pool to
/// `num_threads` workers. It is orthogonal to [`ndt::NdtParams::num_threads`], which only selects
/// serial vs parallel (`> 1`). If never called, the pool defaults to the number of logical CPUs
/// (or `RAYON_NUM_THREADS`, if set). Worker CPU affinity follows the internal `PinPlan` (see its
/// documentation for the
/// `NDT_RAYON_CPU_AFFINITY` / `NDT_PIN_RAYON_WORKERS` environment variables).
///
/// Best-effort and idempotent — call it **once, early** (before any align). Returns `true` iff this
/// call initialized the pool; `false` if the pool was already initialized (a previous call, or a
/// prior parallel run initialized it lazily) or `num_threads == 0`. Never panics.
///
/// # Arguments
/// * `num_threads` — desired global worker count; `0` is a no-op returning `false`.
#[cfg(feature = "parallel")]
#[must_use]
pub fn init_thread_pool(num_threads: usize) -> bool {
    if num_threads == 0 {
        return false;
    }
    // Per-worker CPU pinning, resolved once from the environment (see `PinPlan`). Default (no
    // env set) preserves the prior unpinned behavior.
    let plan = PinPlan::from_env();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |idx| plan.apply(idx))
        .build_global()
        .is_ok()
}

/// Pin the calling (rayon worker) thread to a single CPU. Best-effort: any failure leaves the
/// thread unpinned (the scheduler default), which is safe. Linux-only; a no-op elsewhere.
#[cfg(all(feature = "parallel", target_os = "linux"))]
#[expect(
    unsafe_code,
    reason = "libc sched_setaffinity + CPU_* macros; best-effort worker pinning at pool init"
)]
fn set_thread_affinity_single(cpu: usize) {
    // SAFETY: `cpu_set_t` is a plain bitmask POD; `sched_setaffinity` takes a valid pointer and
    // the struct size per sched_setaffinity(2), and `CPU_SET` indexes the local set. No Rust
    // aliasing or lifetime invariant crosses the C boundary.
    unsafe {
        let mut one: libc::cpu_set_t = core::mem::zeroed();
        libc::CPU_SET(cpu, &mut one);
        let _ = libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &raw const one);
    }
}

/// Pin the calling (rayon worker) thread to the `idx`-th CPU allowed in the process's current
/// affinity mask. Best-effort; Linux-only.
#[cfg(all(feature = "parallel", target_os = "linux"))]
#[expect(
    unsafe_code,
    reason = "libc sched_getaffinity + CPU_ISSET; reads the process cpuset to find the idx-th CPU"
)]
fn pin_worker_to_cpuset(idx: usize) {
    // SAFETY: `cpu_set_t` is a plain bitmask POD; `sched_getaffinity` takes a valid pointer and
    // the struct size per sched_setaffinity(2), and `CPU_ISSET` indexes the local set.
    unsafe {
        let mut allowed: libc::cpu_set_t = core::mem::zeroed();
        if libc::sched_getaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &raw mut allowed)
            != 0
        {
            return;
        }
        #[expect(
            clippy::as_conversions,
            reason = "libc::CPU_SETSIZE is a small positive c_int bound"
        )]
        let setsize = libc::CPU_SETSIZE as usize;
        let mut seen: usize = 0;
        for cpu in 0..setsize {
            if libc::CPU_ISSET(cpu, &allowed) {
                if seen == idx {
                    set_thread_affinity_single(cpu);
                    return;
                }
                seen = seen.saturating_add(1);
            }
        }
    }
}

/// Non-Linux fallback: pinning is unavailable, so these are no-ops.
#[cfg(all(feature = "parallel", not(target_os = "linux")))]
fn set_thread_affinity_single(_cpu: usize) {}

/// Non-Linux fallback: pinning is unavailable, so these are no-ops.
#[cfg(all(feature = "parallel", not(target_os = "linux")))]
fn pin_worker_to_cpuset(_idx: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "parallel")]
    #[test]
    fn init_thread_pool_zero_is_noop_and_pool_is_usable() {
        // Zero threads is always a deterministic no-op.
        assert!(!init_thread_pool(0), "0 threads must be a no-op");
        // Best-effort: `sized` is true only if THIS call initialized the process-global pool — a
        // prior test in the same binary may have already done so — so assert the pool is usable
        // rather than the flag value.
        let sized = init_thread_pool(4);
        assert!(
            rayon::current_num_threads() >= 1,
            "global pool usable after init (sized={sized})"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parse_cpu_affinity_forms() {
        // Whitespace, comma, and mixed separators -> the same explicit list.
        assert_eq!(parse_cpu_affinity("2 4 6 8"), vec![2, 4, 6, 8]);
        assert_eq!(parse_cpu_affinity("2,4,6,8"), vec![2, 4, 6, 8]);
        assert_eq!(parse_cpu_affinity(" 2 , 4\t6\n8 "), vec![2, 4, 6, 8]);
        // Inclusive range and range-with-stride (GOMP syntax).
        assert_eq!(parse_cpu_affinity("2-5"), vec![2, 3, 4, 5]);
        assert_eq!(parse_cpu_affinity("2-8:2"), vec![2, 4, 6, 8]);
        assert_eq!(parse_cpu_affinity("0-3 8"), vec![0, 1, 2, 3, 8]);
        // Single CPU.
        assert_eq!(parse_cpu_affinity("2"), vec![2]);
        // Malformed tokens are skipped best-effort; empty input -> empty list.
        assert_eq!(parse_cpu_affinity(""), Vec::<usize>::new());
        assert_eq!(parse_cpu_affinity("x 4 -1 5-"), vec![4]);
        // Reversed range (b < a) is dropped, not panicked.
        assert_eq!(parse_cpu_affinity("5-2"), Vec::<usize>::new());
        // Zero/invalid stride falls back to 1.
        assert_eq!(parse_cpu_affinity("0-2:0"), vec![0, 1, 2]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[expect(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "test code may index and panic freely"
    )]
    fn pin_plan_precedence_and_wrap() {
        // List wraps when there are more workers than entries (GOMP round-robin): worker idx maps
        // to list[idx % len]. (Actual affinity is asserted by the on-host sanity run.)
        let plan = PinPlan::List(vec![2, 4]);
        if let PinPlan::List(v) = &plan {
            let mapped: Vec<usize> = (0..4).map(|idx| v[idx % v.len()]).collect();
            assert_eq!(mapped, vec![2, 4, 2, 4]);
        } else {
            panic!("expected List");
        }
    }
}
