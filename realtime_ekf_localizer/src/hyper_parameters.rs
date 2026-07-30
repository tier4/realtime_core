//! Port of `hyper_parameters.hpp` — the plain tuned-parameter struct. Parameter-server loading
//! stays on the ROS side; the port carries only the fields that feed the filter math and the
//! queue/tick control flow (diagnostics-threshold and logging fields are out of the
//! port-equivalence contract's scope).

/// EKF localizer hyper-parameters (defaults = `config/ekf_localizer.param.yaml`).
#[derive(Clone, Debug)]
pub struct HyperParameters {
    /// EKF update frequency `predict_frequency` \[Hz\].
    pub ekf_rate: f64,
    /// EKF update period `1 / max(ekf_rate, 0.1)` \[s\].
    pub ekf_dt: f64,
    /// Enable yaw-bias estimation (initial YAWB variance).
    pub enable_yaw_bias_estimation: bool,
    /// Number of delay-compensation steps (extended state = `6 * extend_state_step`).
    pub extend_state_step: usize,
    /// Additional artificial delay applied to pose measurements \[s\].
    pub pose_additional_delay: f64,
    /// Mahalanobis gate threshold for pose updates.
    pub pose_gate_dist: f64,
    /// Pose smoothing steps (measurement replay count and R scale).
    pub pose_smoothing_steps: usize,
    /// Maximum pose queue size.
    pub max_pose_queue_size: usize,
    /// Additional artificial delay applied to twist measurements \[s\].
    pub twist_additional_delay: f64,
    /// Mahalanobis gate threshold for twist updates.
    pub twist_gate_dist: f64,
    /// Twist smoothing steps.
    pub twist_smoothing_steps: usize,
    /// Maximum twist queue size.
    pub max_twist_queue_size: usize,
    /// vx process noise (continuous) \[m/s\].
    pub proc_stddev_vx_c: f64,
    /// wz process noise (continuous) \[rad/s\].
    pub proc_stddev_wz_c: f64,
    /// yaw process noise (continuous) \[rad\].
    pub proc_stddev_yaw_c: f64,
    /// z `Simple1DFilter` process stddev.
    pub z_filter_proc_dev: f64,
    /// roll `Simple1DFilter` process stddev.
    pub roll_filter_proc_dev: f64,
    /// pitch `Simple1DFilter` process stddev.
    pub pitch_filter_proc_dev: f64,
    /// Twists slower than this get their vx covariance forced to 10000 (unobservable).
    pub threshold_observable_velocity_mps: f64,
}

impl Default for HyperParameters {
    /// Defaults from `config/ekf_localizer.param.yaml`.
    fn default() -> Self {
        Self {
            ekf_rate: 50.0,
            ekf_dt: 1.0 / 50.0,
            enable_yaw_bias_estimation: true,
            extend_state_step: 50,
            pose_additional_delay: 0.0,
            pose_gate_dist: 49.5,
            pose_smoothing_steps: 5,
            max_pose_queue_size: 5,
            twist_additional_delay: 0.0,
            twist_gate_dist: 46.1,
            twist_smoothing_steps: 2,
            max_twist_queue_size: 2,
            proc_stddev_vx_c: 10.0,
            proc_stddev_wz_c: 5.0,
            proc_stddev_yaw_c: 0.005,
            z_filter_proc_dev: 5.0,
            roll_filter_proc_dev: 0.1,
            pitch_filter_proc_dev: 0.1,
            threshold_observable_velocity_mps: 0.0,
        }
    }
}
