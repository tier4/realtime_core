//! Behavior tests for the `EkfLocalizerCore` tick orchestration (the port of the
//! `timer_callback` control flow): initialization gating, measured-dt semantics (clamp,
//! jump-back), queue drains with aging/smoothing, the queue-size caps, and the
//! zero-velocity-twist covariance override.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    reason = "test code"
)]

use realtime_ekf_localizer::hyper_parameters::HyperParameters;
use realtime_ekf_localizer::msg::{PoseWithCovariance, Transform, TwistWithCovariance, cov_idx};
use realtime_ekf_localizer::tf2_math::quaternion_from_rpy;
use realtime_ekf_localizer::tick::EkfLocalizerCore;
use realtime_ekf_localizer::trace::{EventKind, TraceEvent};

const SEC: i64 = 1_000_000_000;
const TICK: i64 = 20_000_000;

fn params() -> HyperParameters {
    HyperParameters::default()
}

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

fn init_core(t0: i64) -> EkfLocalizerCore {
    let mut core = EkfLocalizerCore::new(params()).unwrap();
    core.initialize(&pose(10.0, 20.0, 0.1, t0), &Transform::identity())
        .unwrap();
    core
}

#[test]
fn tick_before_initialpose_is_a_noop() {
    let mut core = EkfLocalizerCore::new(params()).unwrap();
    let mut trace: Vec<TraceEvent> = Vec::new();
    // C++: measurements before activation/initial pose are dropped; the tick early-returns.
    core.enqueue_pose(pose(1.0, 2.0, 0.0, SEC));
    assert_eq!(core.pose_queue_size(), 0);
    let out = core.tick(SEC, &mut trace).unwrap();
    assert!(out.is_none());
    assert!(trace.is_empty());
}

#[test]
fn smoothing_replays_measurements_across_ticks() {
    let t0 = 100 * SEC;
    let mut core = init_core(t0);
    let mut trace: Vec<TraceEvent> = Vec::new();

    // One pose (smoothing 5), one twist (smoothing 2).
    core.enqueue_pose(pose(10.1, 20.05, 0.1, t0 + TICK - 5_000_000));
    core.enqueue_twist(twist(2.0, 0.05, t0 + TICK - 5_000_000));

    // Tick 1: predict + 1 pose + 1 twist.
    let out = core.tick(t0 + TICK, &mut trace).unwrap().unwrap();
    let kinds: Vec<EventKind> = trace.iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        vec![EventKind::Predict, EventKind::Pose, EventKind::Twist]
    );
    // Aging: pose age 1 < 5 and twist age 1 < 2, so both are re-enqueued.
    assert_eq!(core.pose_queue_size(), 1);
    assert_eq!(core.twist_queue_size(), 1);
    assert!(out.twist.vx > 0.0);

    // Tick 2: the same measurements replay; the twist reaches max age and is dropped.
    trace.clear();
    core.tick(t0 + 2 * TICK, &mut trace).unwrap().unwrap();
    assert_eq!(
        trace.iter().filter(|e| e.kind == EventKind::Pose).count(),
        1
    );
    assert_eq!(
        trace.iter().filter(|e| e.kind == EventKind::Twist).count(),
        1
    );
    assert_eq!(core.pose_queue_size(), 1);
    assert_eq!(core.twist_queue_size(), 0);

    // Ticks 3..5: the pose replays until age 5, then disappears.
    for k in 3..=5 {
        trace.clear();
        core.tick(t0 + k * TICK, &mut trace).unwrap().unwrap();
        assert_eq!(
            trace.iter().filter(|e| e.kind == EventKind::Pose).count(),
            1
        );
    }
    assert_eq!(core.pose_queue_size(), 0);
}

#[test]
fn queue_caps_drop_oldest() {
    let t0 = 100 * SEC;
    let mut core = init_core(t0);

    // max_pose_queue_size = 5: the 6th push evicts the oldest.
    for i in 0..6 {
        core.enqueue_pose(pose(10.0, 20.0, 0.1, t0 + i));
    }
    assert_eq!(core.pose_queue_size(), 5);

    // max_twist_queue_size = 2: the 3rd push evicts the oldest.
    for i in 0..3 {
        core.enqueue_twist(twist(1.0, 0.0, t0 + i));
    }
    assert_eq!(core.twist_queue_size(), 2);
}

#[test]
fn dt_clamps_at_ten_seconds() {
    let t0 = 100 * SEC;
    let mut core = init_core(t0);
    let mut trace: Vec<TraceEvent> = Vec::new();
    core.tick(t0 + TICK, &mut trace).unwrap();
    trace.clear();
    // 15 s gap: the measured dt must clamp to 10 s (recorded in the predict row's delay_s).
    core.tick(t0 + TICK + 15 * SEC, &mut trace).unwrap();
    let predict = &trace[0];
    assert_eq!(predict.kind, EventKind::Predict);
    assert_eq!(predict.delay_s, 10.0);
}

#[test]
fn jump_back_keeps_previous_dt() {
    let t0 = 100 * SEC;
    let mut core = init_core(t0);
    let mut trace: Vec<TraceEvent> = Vec::new();
    core.tick(t0 + TICK, &mut trace).unwrap();
    core.tick(t0 + 2 * TICK, &mut trace).unwrap();
    trace.clear();
    // Jump back: dt is NOT re-measured (warn-only in C++); the previous dt is reused.
    core.tick(t0 + TICK, &mut trace).unwrap();
    assert_eq!(trace[0].kind, EventKind::Predict);
    assert_eq!(trace[0].delay_s, 0.02);
}

#[test]
fn slow_twist_covariance_override() {
    let t0 = 100 * SEC;
    let mut p = params();
    p.threshold_observable_velocity_mps = 0.2;
    let mut core = EkfLocalizerCore::new(p).unwrap();
    core.initialize(&pose(0.0, 0.0, 0.0, t0), &Transform::identity())
        .unwrap();
    let mut trace: Vec<TraceEvent> = Vec::new();

    // |vx| = 0.05 < 0.2: the vx covariance is overridden to 10000, so the accepted update
    // barely moves the velocity estimate.
    core.enqueue_twist(twist(0.05, 0.0, t0 + TICK - 1_000_000));
    let out = core.tick(t0 + TICK, &mut trace).unwrap().unwrap();
    let twist_ev = trace.iter().find(|e| e.kind == EventKind::Twist).unwrap();
    assert!(twist_ev.accepted);
    assert!(
        out.twist.vx.abs() < 0.05 * 0.05,
        "vx barely moves: {}",
        out.twist.vx
    );
}

#[test]
fn no_update_counts_and_outputs() {
    let t0 = 100 * SEC;
    let mut core = init_core(t0);
    let mut trace: Vec<TraceEvent> = Vec::new();

    // Empty queues: no pose/twist events, counters climb.
    core.tick(t0 + TICK, &mut trace).unwrap().unwrap();
    core.tick(t0 + 2 * TICK, &mut trace).unwrap().unwrap();
    assert_eq!(core.pose_diag_info.no_update_count, 2);
    assert_eq!(core.twist_diag_info.no_update_count, 2);

    // An accepted pose resets the pose counter only.
    core.enqueue_pose(pose(10.0, 20.0, 0.1, t0 + 3 * TICK - 5_000_000));
    let out = core.tick(t0 + 3 * TICK, &mut trace).unwrap().unwrap();
    assert_eq!(core.pose_diag_info.no_update_count, 0);
    assert_eq!(core.twist_diag_info.no_update_count, 3);

    // Outputs carry the module state: biased and unbiased pose differ only by yaw bias.
    assert_eq!(out.pose.position[0], out.biased_pose.position[0]);
    assert!(out.pose_covariance[cov_idx::X_X] > 0.0);
    assert!(out.twist_covariance[cov_idx::X_X] > 0.0);
    assert!(out.yaw_bias.abs() < 1.0);
}
