//! Zero-allocation gate for the EKF per-event path (the runtime allocation policy).
//! A counting global allocator asserts that, after warmup, `predict_with_delay` and the
//! pose/twist `measurement_update_*` methods perform **no heap allocation** with tracing
//! handled through a preallocated event buffer: the construction/first-tick phase may size
//! the scratch pools (extended-dimension buffers at init; per-measurement-dimension buffers
//! on their first sighting), but the steady-state event path must reuse them.
//!
//! This lives in its own integration-test binary so the global allocator does not perturb
//! the unit tests (same pattern as `realtime_ndt_scan_matcher/tests/zero_alloc.rs`).

#![allow(
    unsafe_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "test code"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use realtime_ekf_localizer::ekf_module::{EkfDiagnosticInfo, EkfModule};
use realtime_ekf_localizer::hyper_parameters::HyperParameters;
use realtime_ekf_localizer::msg::{PoseWithCovariance, Transform, TwistWithCovariance, cov_idx};
use realtime_ekf_localizer::tf2_math::quaternion_from_rpy;
use realtime_ekf_localizer::trace::TraceEvent;

/// A pass-through allocator that counts allocations while `ENABLED` is set. The default
/// `GlobalAlloc::realloc` routes through `alloc`, so `Vec`/`VecDeque` growth is counted too.
struct Counting;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);

// SAFETY: delegates every call to the System allocator unchanged; only adds an atomic counter.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::SeqCst) {
            ALLOCS.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: same layout contract as the System allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr/layout came from System.alloc above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `f` with allocation counting enabled; return how many allocations it performed.
fn count_allocs<R>(f: impl FnOnce() -> R) -> u64 {
    let before = ALLOCS.load(Ordering::SeqCst);
    ENABLED.store(true, Ordering::SeqCst);
    let _r = f();
    ENABLED.store(false, Ordering::SeqCst);
    ALLOCS.load(Ordering::SeqCst).saturating_sub(before)
}

const SEC: i64 = 1_000_000_000;

fn pose(x: f64, y: f64, yaw: f64, stamp_ns: i64) -> PoseWithCovariance {
    let mut covariance = [0.0_f64; 36];
    covariance[cov_idx::X_X] = 0.04;
    covariance[cov_idx::Y_Y] = 0.04;
    covariance[cov_idx::Z_Z] = 0.09;
    covariance[cov_idx::ROLL_ROLL] = 0.001;
    covariance[cov_idx::PITCH_PITCH] = 0.001;
    covariance[cov_idx::YAW_YAW] = 0.01;
    PoseWithCovariance {
        stamp_ns,
        position: [x, y, 0.3],
        orientation: quaternion_from_rpy(0.0, 0.0, yaw),
        covariance,
    }
}

fn twist(vx: f64, wz: f64, stamp_ns: i64) -> TwistWithCovariance {
    let mut covariance = [0.0_f64; 36];
    covariance[0] = 0.01;
    covariance[35] = 0.005;
    TwistWithCovariance {
        stamp_ns,
        linear: [vx, 0.0, 0.0],
        angular: [0.0, 0.0, wz],
        covariance,
    }
}

/// One combined test: the `ENABLED` flag is process-global, so warmup and measurement live
/// in one sequential `#[test]`.
#[test]
fn steady_state_event_path_is_allocation_free() {
    let params = HyperParameters::default();
    let mut module = EkfModule::new(params).unwrap();
    module
        .initialize(&pose(10.0, 20.0, 0.1, 100 * SEC), &Transform::identity())
        .unwrap();
    for _ in 0..50 {
        module.accumulate_delay_time(0.02);
    }

    // Preallocated trace-event buffer (the FFI handle owns the equivalent); `clear()` keeps
    // capacity, so pushes after warmup do not allocate.
    let mut events: Vec<TraceEvent> = Vec::with_capacity(8);
    let mut diag = EkfDiagnosticInfo::default();

    // Warmup: sizes the per-measurement-dimension scratch pools (dims 3 and 2) and the
    // event buffer.
    let mut now = 100 * SEC;
    for i in 0..3_i64 {
        now += 20_000_000;
        events.clear();
        module
            .predict_with_delay_traced(0.02, now, &mut events)
            .unwrap();
        let accepted = module
            .measurement_update_pose(
                &pose(10.0 + 0.01 * i as f64, 20.0, 0.1, now - 30_000_000),
                now,
                &mut diag,
                &mut events,
            )
            .unwrap();
        assert!(accepted);
        let accepted = module
            .measurement_update_twist(
                &twist(1.0, 0.05, now - 5_000_000),
                now,
                &mut diag,
                &mut events,
            )
            .unwrap();
        assert!(accepted);
    }

    // Steady state: predict + accepted pose update + accepted twist update + rejected
    // (delay-gate and Mahalanobis-path) updates — zero allocations per event.
    for i in 0..10_i64 {
        now += 20_000_000;
        events.clear();
        let n = count_allocs(|| {
            module.accumulate_delay_time(0.02);
            module
                .predict_with_delay_traced(0.02, now, &mut events)
                .unwrap();
            let ok_pose = module
                .measurement_update_pose(
                    &pose(10.0 + 0.01 * i as f64, 20.0, 0.1, now - 30_000_000),
                    now,
                    &mut diag,
                    &mut events,
                )
                .unwrap();
            let ok_twist = module
                .measurement_update_twist(
                    &twist(1.0, 0.05, now - 5_000_000),
                    now,
                    &mut diag,
                    &mut events,
                )
                .unwrap();
            // Delay-gate rejection path.
            let rejected = module
                .measurement_update_pose(
                    &pose(10.0, 20.0, 0.1, now - 100 * SEC),
                    now,
                    &mut diag,
                    &mut events,
                )
                .unwrap();
            assert!(ok_pose && ok_twist && !rejected);
        });
        assert_eq!(
            n, 0,
            "event path allocated {n} times at steady-state tick {i}"
        );
    }

    // Getter path (per-tick outputs) is allocation-free too.
    let n = count_allocs(|| {
        let _pose = module.get_current_pose(false).unwrap();
        let _biased = module.get_current_pose(true).unwrap();
        let _twist = module.get_current_twist().unwrap();
        let _pc = module.get_current_pose_covariance().unwrap();
        let _tc = module.get_current_twist_covariance().unwrap();
        let _b = module.get_yaw_bias().unwrap();
    });
    assert_eq!(n, 0, "getter path allocated {n} times");
}
