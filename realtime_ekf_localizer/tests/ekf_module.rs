//! Transcription of the C++ `test_ekf_module.cpp` suite (delay-table search/accumulation,
//! roll/pitch/height delay compensation, and the pose/twist measurement-update decision
//! chains), judged with the port-equivalence contract's policies.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "test code"
)]

use realtime_ekf_localizer::ekf_module::{EkfDiagnosticInfo, EkfModule};
use realtime_ekf_localizer::hyper_parameters::HyperParameters;
use realtime_ekf_localizer::msg::{PoseWithCovariance, Transform, TwistWithCovariance, cov_idx};
use realtime_ekf_localizer::tf2_math::{get_yaw, quaternion_from_rpy};
use realtime_ekf_localizer::trace::TraceEvent;

const SEC: i64 = 1_000_000_000;

/// C++ `make_params()`.
fn make_params() -> HyperParameters {
    HyperParameters {
        ekf_rate: 50.0,
        ekf_dt: 1.0 / 50.0,
        enable_yaw_bias_estimation: true,
        extend_state_step: 50,
        pose_additional_delay: 0.0,
        pose_gate_dist: 10000.0,
        pose_smoothing_steps: 5,
        max_pose_queue_size: 5,
        twist_additional_delay: 0.0,
        twist_gate_dist: 10000.0,
        twist_smoothing_steps: 2,
        max_twist_queue_size: 5,
        proc_stddev_vx_c: 10.0,
        proc_stddev_wz_c: 5.0,
        proc_stddev_yaw_c: 0.005,
        z_filter_proc_dev: 1.0,
        roll_filter_proc_dev: 0.01,
        pitch_filter_proc_dev: 0.01,
        threshold_observable_velocity_mps: 0.0,
    }
}

/// C++ `make_pose()`.
fn make_pose(x: f64, y: f64, yaw: f64, stamp_ns: i64) -> PoseWithCovariance {
    let mut covariance = [0.0_f64; 36];
    covariance[cov_idx::X_X] = 1.0;
    covariance[cov_idx::Y_Y] = 1.0;
    covariance[cov_idx::Z_Z] = 1.0;
    covariance[cov_idx::ROLL_ROLL] = 0.01;
    covariance[cov_idx::PITCH_PITCH] = 0.01;
    covariance[cov_idx::YAW_YAW] = 0.01;
    PoseWithCovariance {
        stamp_ns,
        position: [x, y, 0.0],
        orientation: quaternion_from_rpy(0.0, 0.0, yaw),
        covariance,
    }
}

/// C++ `make_twist()`.
fn make_twist(vx: f64, wz: f64, stamp_ns: i64) -> TwistWithCovariance {
    let mut covariance = [0.0_f64; 36];
    covariance[cov_idx::X_X] = 1.0;
    covariance[cov_idx::YAW_YAW] = 1.0;
    TwistWithCovariance {
        stamp_ns,
        linear: [vx, 0.0, 0.0],
        angular: [0.0, 0.0, wz],
        covariance,
    }
}

fn make_module(params: &HyperParameters) -> EkfModule {
    EkfModule::new(params.clone()).unwrap()
}

/// C++ `MeasurementUpdatePose::reset_module()` / `MeasurementUpdateTwist::reset_module()`.
fn reset_module(params: &HyperParameters) -> EkfModule {
    let mut module = make_module(params);
    let t0 = 100 * SEC;
    let initial_pose = make_pose(0.0, 0.0, 0.0, t0);
    module
        .initialize(&initial_pose, &Transform::identity())
        .unwrap();
    for _ in 0..params.extend_state_step {
        module.accumulate_delay_time(params.ekf_dt);
    }
    module.predict_with_delay(params.ekf_dt).unwrap();
    module
}

// ---------------------------------------------------------------------------
// find_closest_delay_time_index
// ---------------------------------------------------------------------------

#[test]
fn find_closest_delay_time_index() {
    let params = make_params();
    let mut module = make_module(&params);

    let dt = 0.1;
    for _ in 0..params.extend_state_step {
        module.accumulate_delay_time(dt);
    }
    // After extend_state_step accumulations the table is [0, dt, 2*dt, ..., (n-1)*dt].

    assert_eq!(module.find_closest_delay_time_index(-1.0), 0);
    assert_eq!(module.find_closest_delay_time_index(0.0), 0);
    assert_eq!(module.find_closest_delay_time_index(0.14), 1);
    assert_eq!(module.find_closest_delay_time_index(0.16), 2);

    let beyond = params.extend_state_step as f64 * dt + 1.0;
    assert_eq!(
        module.find_closest_delay_time_index(beyond),
        params.extend_state_step
    );
}

#[test]
fn find_closest_delay_time_index_empty_table() {
    let mut params = make_params();
    params.extend_state_step = 0;
    let module = make_module(&params);

    assert_eq!(module.find_closest_delay_time_index(-1.0), 0);
    assert_eq!(module.find_closest_delay_time_index(0.0), 0);
    assert_eq!(module.find_closest_delay_time_index(1.0), 0);
    assert_eq!(module.find_closest_delay_time_index(1.0e15), 0);
}

#[test]
fn accumulate_delay_time() {
    let mut params = make_params();
    params.extend_state_step = 4;
    let mut module = make_module(&params);

    let dt = 0.2;

    module.accumulate_delay_time(dt);
    assert_eq!(module.find_closest_delay_time_index(0.0), 0);

    module.accumulate_delay_time(dt);
    assert_eq!(module.find_closest_delay_time_index(dt), 1);

    module.accumulate_delay_time(dt);
    assert_eq!(module.find_closest_delay_time_index(2.0 * dt), 2);

    assert_eq!(
        module.find_closest_delay_time_index(2.0e15),
        params.extend_state_step
    );
}

// ---------------------------------------------------------------------------
// compensate_rph_with_delay
// ---------------------------------------------------------------------------

#[test]
fn compensate_rph_with_delay_zero_angular_velocity() {
    let params = make_params();
    let module = make_module(&params);

    let stamp = 100 * SEC;
    let pose = make_pose(1.0, 2.0, 0.3, stamp);

    let delay_time = 0.2;
    let compensated = module
        .compensate_rph_with_delay(&pose, [0.0, 0.0, 0.0], delay_time)
        .unwrap();

    for i in 0..4 {
        assert!((compensated.orientation[i] - pose.orientation[i]).abs() < 1e-9);
    }
    assert!(
        ((compensated.stamp_ns - stamp) as f64 / 1e9 - delay_time).abs() < 1e-9,
        "stamp shifted by delay"
    );
    assert!((compensated.position[2] - pose.position[2]).abs() < 1e-9);
}

#[test]
fn compensate_rph_with_delay_non_zero_angular_velocity() {
    let params = make_params();
    let module = make_module(&params);

    let pose = make_pose(0.0, 0.0, 0.0, 100 * SEC);
    let compensated = module
        .compensate_rph_with_delay(&pose, [0.0, 0.0, 1.0], 0.5)
        .unwrap();

    let yaw = get_yaw(&compensated.orientation);
    assert!((yaw - 0.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// measurement_update_pose
// ---------------------------------------------------------------------------

#[test]
fn pose_accepts_valid_measurement() {
    let params = make_params();
    let mut module = reset_module(&params);
    let t_curr = 100 * SEC;

    let pose = make_pose(1.0, 2.0, 0.0, t_curr);

    let pose_before = module.get_current_pose(false).unwrap();
    let cov_before = module.get_current_pose_covariance().unwrap();
    assert_eq!(pose_before.position[0], 0.0);
    assert_eq!(pose_before.position[1], 0.0);

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_pose(&pose, t_curr, &mut diag, &mut trace)
        .unwrap();

    assert!(ok);
    assert!(diag.is_passed_delay_gate);
    assert!(diag.is_passed_mahalanobis_gate);

    let pose_after = module.get_current_pose(false).unwrap();
    assert!(pose_after.position[0] > 0.0);
    assert!(pose_after.position[0] < 1.0);
    assert!(pose_after.position[1] > 0.0);
    assert!(pose_after.position[1] < 2.0);

    let cov_after = module.get_current_pose_covariance().unwrap();
    assert!(cov_after[cov_idx::X_X] < cov_before[cov_idx::X_X]);
    assert!(cov_after[cov_idx::Y_Y] < cov_before[cov_idx::Y_Y]);

    // Trace: one accepted pose event.
    assert_eq!(trace.len(), 1);
    assert!(trace[0].accepted);
}

#[test]
fn pose_rejects_on_delay_gate() {
    let params = make_params();
    let mut module = reset_module(&params);

    let t_curr = 1000 * SEC;
    let t_old = 100 * SEC;
    let pose = make_pose(0.0, 0.0, 0.0, t_old);

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_pose(&pose, t_curr, &mut diag, &mut trace)
        .unwrap();

    assert!(!ok);
    assert!(!diag.is_passed_delay_gate);
    assert_eq!(trace.len(), 1);
    assert!(!trace[0].delay_gate);
}

#[test]
fn pose_rejects_on_nan() {
    let params = make_params();
    let mut module = reset_module(&params);
    let t_curr = 100 * SEC;
    let mut pose = make_pose(0.0, 0.0, 0.0, t_curr);
    pose.position[0] = f64::NAN;

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_pose(&pose, t_curr, &mut diag, &mut trace)
        .unwrap();

    assert!(!ok);
    // The NaN gate is reached after the delay gate, so the delay gate is still passed.
    assert!(diag.is_passed_delay_gate);
    assert!(trace[0].delay_gate);
    assert!(!trace[0].accepted);
    assert!(trace[0].mahalanobis.is_nan());
}

#[test]
fn pose_rejects_on_inf() {
    let params = make_params();
    let mut module = reset_module(&params);
    let t_curr = 100 * SEC;
    let mut pose = make_pose(0.0, 0.0, 0.0, t_curr);
    pose.position[1] = f64::INFINITY;

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_pose(&pose, t_curr, &mut diag, &mut trace)
        .unwrap();
    assert!(!ok);
}

#[test]
fn pose_rejects_on_mahalanobis_gate() {
    let mut params = make_params();
    params.pose_gate_dist = 1e-6;
    let mut module = reset_module(&params);

    let t_curr = 100 * SEC;
    let pose = make_pose(1000.0, 1000.0, 0.0, t_curr);

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_pose(&pose, t_curr, &mut diag, &mut trace)
        .unwrap();

    assert!(!ok);
    assert!(diag.is_passed_delay_gate);
    assert!(!diag.is_passed_mahalanobis_gate);
    assert!(diag.mahalanobis_distance > 0.0);
    assert!(!trace[0].mahalanobis_gate);
}

// ---------------------------------------------------------------------------
// measurement_update_twist
// ---------------------------------------------------------------------------

#[test]
fn twist_accepts_valid_measurement() {
    let params = make_params();
    let mut module = reset_module(&params);
    let t_curr = 100 * SEC;

    let twist = make_twist(3.0, 1.0, t_curr);

    let twist_before = module.get_current_twist().unwrap();
    let cov_before = module.get_current_twist_covariance().unwrap();
    assert_eq!(twist_before.vx, 0.0);
    assert_eq!(twist_before.wz, 0.0);

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_twist(&twist, t_curr, &mut diag, &mut trace)
        .unwrap();

    assert!(ok);
    assert!(diag.is_passed_delay_gate);
    assert!(diag.is_passed_mahalanobis_gate);

    let twist_after = module.get_current_twist().unwrap();
    assert!(twist_after.vx > 0.0);
    assert!(twist_after.vx < 3.0);
    assert!(twist_after.wz > 0.0);
    assert!(twist_after.wz < 1.0);

    let cov_after = module.get_current_twist_covariance().unwrap();
    assert!(cov_after[cov_idx::X_X] < cov_before[cov_idx::X_X]);
}

#[test]
fn twist_rejects_on_delay_gate() {
    let params = make_params();
    let mut module = reset_module(&params);

    let twist = make_twist(0.0, 0.0, 100 * SEC);
    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_twist(&twist, 1000 * SEC, &mut diag, &mut trace)
        .unwrap();

    assert!(!ok);
    assert!(!diag.is_passed_delay_gate);
}

#[test]
fn twist_rejects_on_nan() {
    let params = make_params();
    let mut module = reset_module(&params);
    let t_curr = 100 * SEC;
    let mut twist = make_twist(0.0, 0.0, t_curr);
    twist.linear[0] = f64::NAN;

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_twist(&twist, t_curr, &mut diag, &mut trace)
        .unwrap();

    assert!(!ok);
    assert!(diag.is_passed_delay_gate);
}

#[test]
fn twist_rejects_on_inf() {
    let params = make_params();
    let mut module = reset_module(&params);
    let t_curr = 100 * SEC;
    let mut twist = make_twist(0.0, 0.0, t_curr);
    twist.angular[2] = f64::INFINITY;

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_twist(&twist, t_curr, &mut diag, &mut trace)
        .unwrap();
    assert!(!ok);
}

#[test]
fn twist_rejects_on_mahalanobis_gate() {
    let mut params = make_params();
    params.twist_gate_dist = 1e-6;
    let mut module = reset_module(&params);

    let t_curr = 100 * SEC;
    let twist = make_twist(1000.0, 1000.0, t_curr);

    let mut diag = EkfDiagnosticInfo::default();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let ok = module
        .measurement_update_twist(&twist, t_curr, &mut diag, &mut trace)
        .unwrap();

    assert!(!ok);
    assert!(diag.is_passed_delay_gate);
    assert!(!diag.is_passed_mahalanobis_gate);
    assert!(diag.mahalanobis_distance > 0.0);
}
