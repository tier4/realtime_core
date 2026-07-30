//! Port of `ekf_module.cpp` — the EKF measurement/prediction module, ROS-free: plain-struct
//! measurements ([`crate::msg`]), `i64` nanosecond timestamps, and trace events returned as
//! records ([`crate::trace`]) instead of a CSV stream. The filter math, gate order, and the
//! observable side effects mirror the C++ line-for-line; warn-only logging branches are
//! dropped (out of the port-equivalence contract's scope).

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::{DMatrix, Matrix6, Vector6};
use realtime_kalman_filter::{KalmanError, TimeDelayKalmanFilter};

use crate::covariance::{
    ekf_covariance_to_pose_message_covariance, ekf_covariance_to_twist_message_covariance,
};
use crate::hyper_parameters::HyperParameters;
use crate::mahalanobis::{MahalanobisScratch, mahalanobis_in};
use crate::measurement::{
    pose_measurement_covariance, pose_measurement_matrix, twist_measurement_covariance,
    twist_measurement_matrix,
};
use crate::msg::{
    PoseOutput, PoseWithCovariance, Transform, TwistOutput, TwistWithCovariance, cov_idx,
};
use crate::numeric::{has_inf, has_nan};
use crate::simple_1d_filter::Simple1DFilter;
use crate::state_transition::{
    create_state_transition_matrix, idx, normalize_yaw, predict_next_state,
    process_noise_covariance,
};
use crate::tf2_math::{
    get_rpy, get_yaw, quaternion_from_axis_angle, quaternion_multiply, quaternion_normalize,
    vector3_length, vector3_normalized,
};
use crate::trace::{EventKind, TraceEvent};

/// Per-measurement-channel diagnostic aggregation (port of `EKFDiagnosticInfo`).
#[derive(Clone, Copy, Debug)]
pub struct EkfDiagnosticInfo {
    /// Consecutive ticks without an accepted update.
    pub no_update_count: usize,
    /// Queue size at the start of the tick.
    pub queue_size: usize,
    /// `false` when any measurement this tick failed the delay gate.
    pub is_passed_delay_gate: bool,
    /// Maximum observed delay.
    pub delay_time: f64,
    /// Delay-table threshold (last accumulated entry).
    pub delay_time_threshold: f64,
    /// `false` when any measurement this tick failed the Mahalanobis gate.
    pub is_passed_mahalanobis_gate: bool,
    /// Maximum observed Mahalanobis distance.
    pub mahalanobis_distance: f64,
}

impl Default for EkfDiagnosticInfo {
    fn default() -> Self {
        Self {
            no_update_count: 0,
            queue_size: 0,
            is_passed_delay_gate: true,
            delay_time: 0.0,
            delay_time_threshold: 0.0,
            is_passed_mahalanobis_gate: true,
            mahalanobis_distance: 0.0,
        }
    }
}

/// Preallocated per-event buffers for the RT-critical predict/update path (all sized at
/// construction; the event path performs no heap allocation in steady state).
#[derive(Clone, Debug)]
struct ModuleScratch {
    /// `6×1` predicted state handed to the delay filter.
    x_next_dm: DMatrix<f64>,
    /// `6×6` state-transition matrix.
    a_dm: DMatrix<f64>,
    /// `6×6` process-noise covariance.
    q_dm: DMatrix<f64>,
    /// `3×1` pose measurement vector.
    y3: DMatrix<f64>,
    /// `3×1` pose prediction at the delayed block.
    y_ekf3: DMatrix<f64>,
    /// `3×3` pose covariance block.
    p_y3: DMatrix<f64>,
    /// `3×6` pose measurement matrix.
    c3: DMatrix<f64>,
    /// `3×3` pose measurement covariance.
    r3: DMatrix<f64>,
    /// Pose Mahalanobis buffers.
    mh3: MahalanobisScratch,
    /// `2×1` twist measurement vector.
    y2: DMatrix<f64>,
    /// `2×1` twist prediction at the delayed block.
    y_ekf2: DMatrix<f64>,
    /// `2×2` twist covariance block.
    p_y2: DMatrix<f64>,
    /// `2×6` twist measurement matrix.
    c2: DMatrix<f64>,
    /// `2×2` twist measurement covariance.
    r2: DMatrix<f64>,
    /// Twist Mahalanobis buffers.
    mh2: MahalanobisScratch,
}

impl ModuleScratch {
    fn new() -> Self {
        Self {
            x_next_dm: DMatrix::zeros(6, 1),
            a_dm: DMatrix::zeros(6, 6),
            q_dm: DMatrix::zeros(6, 6),
            y3: DMatrix::zeros(3, 1),
            y_ekf3: DMatrix::zeros(3, 1),
            p_y3: DMatrix::zeros(3, 3),
            c3: DMatrix::zeros(3, 6),
            r3: DMatrix::zeros(3, 3),
            mh3: MahalanobisScratch::new(3),
            y2: DMatrix::zeros(2, 1),
            y_ekf2: DMatrix::zeros(2, 1),
            p_y2: DMatrix::zeros(2, 2),
            c2: DMatrix::zeros(2, 6),
            r2: DMatrix::zeros(2, 2),
            mh2: MahalanobisScratch::new(2),
        }
    }
}

/// The EKF module (port of `EKFModule`): a 6-state time-delay Kalman filter plus three scalar
/// filters for z/roll/pitch.
///
/// The per-event methods (`predict_with_delay`, `measurement_update_pose`,
/// `measurement_update_twist` and the getters they feed) are allocation-free in steady state:
/// every dynamic-matrix temporary lives in [`ModuleScratch`] / the delay filter's own scratch,
/// and state reads borrow the filter storage instead of copying it to the heap.
#[derive(Clone, Debug)]
pub struct EkfModule {
    kalman_filter: TimeDelayKalmanFilter,
    dim_x: usize,
    accumulated_delay_times: Vec<f64>,
    params: HyperParameters,
    z_filter: Simple1DFilter,
    roll_filter: Simple1DFilter,
    pitch_filter: Simple1DFilter,
    last_angular_velocity: [f64; 3],
    ekf_dt: f64,
    scratch: ModuleScratch,
}

/// `std::max(a, b)` for f64 with the C++ semantics: `(a < b) ? b : a`. Differs from
/// `f64::max` for NaN inputs (C++ returns the first argument when the comparison is false).
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

impl EkfModule {
    /// Construct with the C++ constructor's initial state: zero state, huge diagonal prior
    /// (`1e15` for x/y, `50` for yaw/wz, `1000` for vx, yaw-bias `50` only when estimation is
    /// enabled), delay table filled with `1e15`.
    ///
    /// # Errors
    /// Propagates [`KalmanError`] from the initial filter setup (unreachable for
    /// `extend_state_step >= 1`; a zero `extend_state_step` skips the filter init, matching
    /// the degenerate C++ configuration where every measurement is delay-gate rejected).
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::allow_attributes,
        reason = "f64 math; constant indices into fixed-size Matrix6"
    )]
    pub fn new(params: HyperParameters) -> Result<Self, KalmanError> {
        let dim_x = 6_usize;
        let mut kalman_filter = TimeDelayKalmanFilter::new();

        let x = DMatrix::<f64>::zeros(dim_x, 1);
        let mut p = DMatrix::<f64>::identity(dim_x, dim_x) * 1.0e15; // for x & y
        p[(idx::YAW, idx::YAW)] = 50.0; // for yaw
        if params.enable_yaw_bias_estimation {
            p[(idx::YAWB, idx::YAWB)] = 50.0; // for yaw bias
        }
        p[(idx::VX, idx::VX)] = 1000.0; // for vx
        p[(idx::WZ, idx::WZ)] = 50.0; // for wz

        if params.extend_state_step > 0 {
            kalman_filter.init(&x, &p, params.extend_state_step)?;
        }

        let mut z_filter = Simple1DFilter::new();
        let mut roll_filter = Simple1DFilter::new();
        let mut pitch_filter = Simple1DFilter::new();
        z_filter.set_proc_var(params.z_filter_proc_dev * params.z_filter_proc_dev);
        roll_filter.set_proc_var(params.roll_filter_proc_dev * params.roll_filter_proc_dev);
        pitch_filter.set_proc_var(params.pitch_filter_proc_dev * params.pitch_filter_proc_dev);

        Ok(Self {
            kalman_filter,
            dim_x,
            accumulated_delay_times: vec![1.0e15; params.extend_state_step],
            params,
            z_filter,
            roll_filter,
            pitch_filter,
            last_angular_velocity: [0.0; 3],
            ekf_dt: 0.0,
            scratch: ModuleScratch::new(),
        })
    }

    /// Re-initialize the filter from an initial pose (port of `EKFModule::initialize`).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] from the filter init (unreachable for valid configs).
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::allow_attributes,
        reason = "f64 math; constant indices into fixed-size matrices/arrays"
    )]
    pub fn initialize(
        &mut self,
        initial_pose: &PoseWithCovariance,
        transform: &Transform,
    ) -> Result<(), KalmanError> {
        let mut x = DMatrix::<f64>::zeros(self.dim_x, 1);
        let mut p = DMatrix::<f64>::zeros(self.dim_x, self.dim_x);

        x[(idx::X, 0)] = initial_pose.position[0] + transform.translation[0];
        x[(idx::Y, 0)] = initial_pose.position[1] + transform.translation[1];
        x[(idx::YAW, 0)] = get_yaw(&initial_pose.orientation) + get_yaw(&transform.rotation);
        x[(idx::YAWB, 0)] = 0.0;
        x[(idx::VX, 0)] = 0.0;
        x[(idx::WZ, 0)] = 0.0;

        p[(idx::X, idx::X)] = initial_pose.covariance[cov_idx::X_X];
        p[(idx::Y, idx::Y)] = initial_pose.covariance[cov_idx::Y_Y];
        p[(idx::YAW, idx::YAW)] = initial_pose.covariance[cov_idx::YAW_YAW];

        if self.params.enable_yaw_bias_estimation {
            p[(idx::YAWB, idx::YAWB)] = 0.0001;
        }
        p[(idx::VX, idx::VX)] = 0.01;
        p[(idx::WZ, idx::WZ)] = 0.01;

        if self.params.extend_state_step > 0 {
            self.kalman_filter
                .init(&x, &p, self.params.extend_state_step)?;
        }

        let z = initial_pose.position[2];
        let (roll, pitch, _yaw) = get_rpy(&initial_pose.orientation);

        let z_var = initial_pose.covariance[cov_idx::Z_Z];
        let roll_var = initial_pose.covariance[cov_idx::ROLL_ROLL];
        let pitch_var = initial_pose.covariance[cov_idx::PITCH_PITCH];

        self.z_filter.init(z, z_var);
        self.roll_filter.init(roll, roll_var);
        self.pitch_filter.init(pitch, pitch_var);
        Ok(())
    }

    /// Current pose estimate (port of `get_current_pose`; `get_biased_yaw` selects the raw
    /// yaw state instead of `yaw + yaw_bias`).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] when the filter is uninitialized.
    pub fn get_current_pose(&self, get_biased_yaw: bool) -> Result<PoseOutput, KalmanError> {
        let z = self.z_filter.get_x();
        let roll = self.roll_filter.get_x();
        let pitch = self.pitch_filter.get_x();

        let x = self.kalman_filter.x_element(idx::X)?;
        let y = self.kalman_filter.x_element(idx::Y)?;
        let biased_yaw = self.kalman_filter.x_element(idx::YAW)?;
        let yaw_bias = self.kalman_filter.x_element(idx::YAWB)?;
        let yaw = biased_yaw + yaw_bias;

        let orientation = if get_biased_yaw {
            crate::tf2_math::quaternion_from_rpy(roll, pitch, biased_yaw)
        } else {
            crate::tf2_math::quaternion_from_rpy(roll, pitch, yaw)
        };
        Ok(PoseOutput {
            position: [x, y, z],
            orientation,
        })
    }

    /// Current twist estimate (port of `get_current_twist`).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] when the filter is uninitialized.
    pub fn get_current_twist(&self) -> Result<TwistOutput, KalmanError> {
        Ok(TwistOutput {
            vx: self.kalman_filter.x_element(idx::VX)?,
            wz: self.kalman_filter.x_element(idx::WZ)?,
        })
    }

    /// Latest 6×6 covariance as a fixed-size (stack) matrix — borrows the filter storage,
    /// no heap allocation.
    fn latest_p6(&self) -> Result<Matrix6<f64>, KalmanError> {
        let p = self.kalman_filter.get_p_ex();
        if p.nrows() < 6 || p.ncols() < 6 {
            return Err(KalmanError::DimensionMismatch);
        }
        Ok(Matrix6::from_iterator(
            p.view((0, 0), (6, 6)).iter().copied(),
        ))
    }

    /// Pose message covariance (port of `get_current_pose_covariance`; z/roll/pitch variances
    /// come from the scalar filters).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] when the filter is uninitialized.
    #[allow(
        clippy::indexing_slicing,
        clippy::allow_attributes,
        reason = "constant indices into [f64; 36]"
    )]
    pub fn get_current_pose_covariance(&self) -> Result<[f64; 36], KalmanError> {
        let mut cov = ekf_covariance_to_pose_message_covariance(&self.latest_p6()?);
        cov[cov_idx::Z_Z] = self.z_filter.get_var();
        cov[cov_idx::ROLL_ROLL] = self.roll_filter.get_var();
        cov[cov_idx::PITCH_PITCH] = self.pitch_filter.get_var();
        Ok(cov)
    }

    /// Twist message covariance (port of `get_current_twist_covariance`).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] when the filter is uninitialized.
    pub fn get_current_twist_covariance(&self) -> Result<[f64; 36], KalmanError> {
        Ok(ekf_covariance_to_twist_message_covariance(
            &self.latest_p6()?,
        ))
    }

    /// Estimated yaw bias (port of `get_yaw_bias`).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] when the filter is uninitialized.
    pub fn get_yaw_bias(&self) -> Result<f64, KalmanError> {
        self.kalman_filter.x_element(idx::YAWB)
    }

    /// Find the delay-table index closest to `target_value` (port of
    /// `find_closest_delay_time_index`, including the empty-table and beyond-last guards).
    #[must_use]
    pub fn find_closest_delay_time_index(&self, target_value: f64) -> usize {
        let table = &self.accumulated_delay_times;
        let Some(last) = table.last() else {
            return 0; // empty table: "no delay slots"
        };

        // If target_value is too large, return last index + 1.
        if target_value > *last {
            return table.len();
        }

        // std::lower_bound: first element not less than target.
        let lower = table.partition_point(|v| *v < target_value);

        if lower == 0 {
            return 0;
        }
        if lower == table.len() {
            return table.len().saturating_sub(1);
        }
        // Compare the target with the lower bound and the previous element.
        let prev_idx = lower.saturating_sub(1);
        let prev = table.get(prev_idx).copied().unwrap_or(f64::NAN);
        let low = table.get(lower).copied().unwrap_or(f64::NAN);
        let is_closer_to_prev = (target_value - prev) < (low - target_value);
        if is_closer_to_prev { prev_idx } else { lower }
    }

    /// Shift the delay table one slot and accumulate `dt` onto the older entries (port of
    /// `accumulate_delay_time`).
    pub fn accumulate_delay_time(&mut self, dt: f64) {
        let table = &mut self.accumulated_delay_times;
        // Shift the delay times to the right (std::copy_backward of [begin, end-1) to end).
        if let Some(prefix) = table.len().checked_sub(1) {
            table.copy_within(0..prefix, 1);
        }
        // Add a new element (=0), and add delay time to the previous elements.
        if let Some(front) = table.first_mut() {
            *front = 0.0;
        }
        for v in table.iter_mut().skip(1) {
            *v += dt;
        }
    }

    /// EKF prediction step (port of `predict_with_delay`): nonlinear next state, linearized
    /// `A`, discretized `Q`, then the time-delay block prediction. Records `ekf_dt`.
    ///
    /// # Errors
    /// Propagates [`KalmanError`] (the C++ ignores the inner `bool`; dimensions are valid by
    /// construction here, so an error indicates an uninitialized/degenerate filter).
    pub fn predict_with_delay(&mut self, dt: f64) -> Result<(), KalmanError> {
        let x_curr = self.latest_x6()?;

        let proc_cov_vx_d = libm::pow(self.params.proc_stddev_vx_c * dt, 2.0);
        let proc_cov_wz_d = libm::pow(self.params.proc_stddev_wz_c * dt, 2.0);
        let proc_cov_yaw_d = libm::pow(self.params.proc_stddev_yaw_c * dt, 2.0);

        let x_next = predict_next_state(&x_curr, dt);
        let a = create_state_transition_matrix(&x_curr, dt);
        let q = process_noise_covariance(proc_cov_yaw_d, proc_cov_vx_d, proc_cov_wz_d);

        // Stage the stack-computed model matrices in the preallocated dynamic buffers
        // (column-major layouts match; no heap allocation).
        self.scratch.x_next_dm.copy_from(&x_next);
        self.scratch.a_dm.copy_from(&a);
        self.scratch.q_dm.copy_from(&q);
        self.kalman_filter.predict_with_delay(
            &self.scratch.x_next_dm,
            &self.scratch.a_dm,
            &self.scratch.q_dm,
        )?;
        self.ekf_dt = dt;
        Ok(())
    }

    /// Latest 6-vector state as a fixed-size (stack) vector — borrows the filter storage,
    /// no heap allocation.
    fn latest_x6(&self) -> Result<Vector6<f64>, KalmanError> {
        let x = self.kalman_filter.get_x_ex();
        if x.nrows() < 6 || x.ncols() != 1 {
            return Err(KalmanError::DimensionMismatch);
        }
        Ok(Vector6::from_column_slice(
            x.as_slice().get(0..6).ok_or(KalmanError::EmptyMatrix)?,
        ))
    }

    /// Snapshot of the trace-extension state (latest x, P diagonal, scalar filters).
    fn trace_state(&self) -> ([f64; 6], [f64; 6], [f64; 3], [f64; 3]) {
        let mut state = [f64::NAN; 6];
        let mut p_diag = [f64::NAN; 6];
        // Borrow the filter storage directly (no heap): an uninitialized filter leaves the
        // NaN defaults exactly as the previous owning-getter path did on its error branch.
        let x = self.kalman_filter.get_x_ex();
        for (i, dst) in state.iter_mut().enumerate() {
            *dst = x.as_slice().get(i).copied().unwrap_or(f64::NAN);
        }
        let p = self.kalman_filter.get_p_ex();
        for (i, dst) in p_diag.iter_mut().enumerate() {
            *dst = p.get((i, i)).copied().unwrap_or(f64::NAN);
        }
        let filters = [
            self.z_filter.get_x(),
            self.roll_filter.get_x(),
            self.pitch_filter.get_x(),
        ];
        let filter_vars = [
            self.z_filter.get_var(),
            self.roll_filter.get_var(),
            self.pitch_filter.get_var(),
        ];
        (state, p_diag, filters, filter_vars)
    }

    /// Build a trace event with the post-event state snapshot appended.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the C++ trace_pose_update signature"
    )]
    fn make_trace_event(
        &self,
        kind: EventKind,
        current_ns: i64,
        measurement_ns: i64,
        delay_s: f64,
        delay_step: usize,
        obs: [f64; 3],
        pred: [f64; 3],
        mahalanobis_distance: f64,
        delay_gate: bool,
        mahalanobis_gate: bool,
        accepted: bool,
    ) -> TraceEvent {
        let (state, p_diag, filters, filter_vars) = self.trace_state();
        let innovation = [
            obs[0] - pred[0],
            obs[1] - pred[1],
            match kind {
                EventKind::Pose => normalize_yaw(obs[2] - pred[2]),
                EventKind::Predict | EventKind::Twist => f64::NAN,
            },
        ];
        TraceEvent {
            kind,
            current_ns,
            measurement_ns,
            delay_s,
            delay_step,
            obs,
            pred,
            innovation,
            mahalanobis: mahalanobis_distance,
            delay_gate,
            mahalanobis_gate,
            accepted,
            state,
            p_diag,
            filters,
            filter_vars,
        }
    }

    /// Pose measurement update (port of `measurement_update_pose`). Returns whether the
    /// update was applied; pushes one trace event describing the decision chain.
    ///
    /// # Errors
    /// Propagates [`KalmanError`] only for internal invariant failures (state reads on an
    /// uninitialized filter); gate rejections are `Ok(false)`.
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::too_many_lines,
        clippy::allow_attributes,
        reason = "f64 math; index arithmetic bounded by delay_step < extend_state_step; length mirrors the C++ decision chain"
    )]
    pub fn measurement_update_pose(
        &mut self,
        pose: &PoseWithCovariance,
        t_curr_ns: i64,
        pose_diag_info: &mut EkfDiagnosticInfo,
        trace: &mut Vec<TraceEvent>,
    ) -> Result<bool, KalmanError> {
        let dim_x = self.dim_x;
        let nan = f64::NAN;
        let obs_x = pose.position[0];
        let obs_y = pose.position[1];

        // Calculate delay step. (t_curr - stamp).seconds() = double(ns diff) / 1e9.
        #[expect(
            clippy::cast_precision_loss,
            clippy::as_conversions,
            reason = "rclcpp Duration::seconds() is double(nanoseconds)/1e9; mirrored deliberately"
        )]
        let mut delay_time = (t_curr_ns.saturating_sub(pose.stamp_ns)) as f64 / 1.0e9
            + self.params.pose_additional_delay;
        // (negative delay is warn-only in C++)
        delay_time = cpp_max(delay_time, 0.0);

        let delay_step = self.find_closest_delay_time_index(delay_time);
        let raw_yaw = get_yaw(&pose.orientation);

        pose_diag_info.delay_time = cpp_max(delay_time, pose_diag_info.delay_time);
        pose_diag_info.delay_time_threshold =
            self.accumulated_delay_times.last().copied().unwrap_or(0.0);
        if delay_step >= self.params.extend_state_step {
            pose_diag_info.is_passed_delay_gate = false;
            trace.push(self.make_trace_event(
                EventKind::Pose,
                t_curr_ns,
                pose.stamp_ns,
                delay_time,
                delay_step,
                [obs_x, obs_y, raw_yaw],
                [nan, nan, nan],
                nan,
                false,
                false,
                false,
            ));
            return Ok(false);
        }

        // Offset the yaw so its difference from the internally-held yaw stays within 2π.
        let mut yaw = get_yaw(&pose.orientation);
        let ekf_yaw = self.kalman_filter.x_element(
            delay_step
                .checked_mul(dim_x)
                .and_then(|b| b.checked_add(idx::YAW))
                .ok_or(KalmanError::IndexOutOfRange)?,
        )?;
        let yaw_error = normalize_yaw(yaw - ekf_yaw);
        yaw = yaw_error + ekf_yaw;

        // Measurement vector (staged in the preallocated buffer).
        self.scratch.y3[(0, 0)] = obs_x;
        self.scratch.y3[(1, 0)] = obs_y;
        self.scratch.y3[(2, 0)] = yaw;

        if has_nan(&self.scratch.y3) || has_inf(&self.scratch.y3) {
            trace.push(self.make_trace_event(
                EventKind::Pose,
                t_curr_ns,
                pose.stamp_ns,
                delay_time,
                delay_step,
                [obs_x, obs_y, yaw],
                [nan, nan, nan],
                nan,
                true,
                false,
                false,
            ));
            return Ok(false);
        }

        // Gate.
        let base = delay_step
            .checked_mul(dim_x)
            .ok_or(KalmanError::IndexOutOfRange)?;
        let y_ekf_x = self.kalman_filter.x_element(
            base.checked_add(idx::X)
                .ok_or(KalmanError::IndexOutOfRange)?,
        )?;
        let y_ekf_y = self.kalman_filter.x_element(
            base.checked_add(idx::Y)
                .ok_or(KalmanError::IndexOutOfRange)?,
        )?;
        self.scratch.y_ekf3[(0, 0)] = y_ekf_x;
        self.scratch.y_ekf3[(1, 0)] = y_ekf_y;
        self.scratch.y_ekf3[(2, 0)] = ekf_yaw;
        let p_curr = self.kalman_filter.get_p_ex();
        if p_curr.nrows() < 3 || p_curr.ncols() < 3 {
            return Err(KalmanError::DimensionMismatch);
        }
        self.scratch.p_y3.copy_from(&p_curr.view((0, 0), (3, 3)));

        let distance = mahalanobis_in(
            &self.scratch.y_ekf3,
            &self.scratch.y3,
            &self.scratch.p_y3,
            &mut self.scratch.mh3,
        );
        pose_diag_info.mahalanobis_distance =
            cpp_max(distance, pose_diag_info.mahalanobis_distance);
        if distance > self.params.pose_gate_dist {
            pose_diag_info.is_passed_mahalanobis_gate = false;
            trace.push(self.make_trace_event(
                EventKind::Pose,
                t_curr_ns,
                pose.stamp_ns,
                delay_time,
                delay_step,
                [obs_x, obs_y, yaw],
                [y_ekf_x, y_ekf_y, ekf_yaw],
                distance,
                true,
                false,
                false,
            ));
            return Ok(false);
        }

        let c = pose_measurement_matrix();
        let r = pose_measurement_covariance(&pose.covariance, self.params.pose_smoothing_steps);
        self.scratch.c3.copy_from(&c);
        self.scratch.r3.copy_from(&r);

        // The C++ discards updateWithDelay's bool (ekf_module.cpp:369): a failed LLT or a
        // non-finite gain leaves the state untouched and the update still counts as accepted.
        let _update_applied_inside_kf = self
            .kalman_filter
            .update_with_delay(
                &self.scratch.y3,
                &self.scratch.c3,
                &self.scratch.r3,
                delay_step,
            )
            .is_ok();

        // Update the z/roll/pitch filters with delay compensation.
        let pose_with_rph_delay_compensation =
            self.compensate_rph_with_delay(pose, self.last_angular_velocity, delay_time)?;
        self.update_simple_1d_filters(
            &pose_with_rph_delay_compensation,
            self.params.pose_smoothing_steps,
        );

        trace.push(self.make_trace_event(
            EventKind::Pose,
            t_curr_ns,
            pose.stamp_ns,
            delay_time,
            delay_step,
            [obs_x, obs_y, yaw],
            [y_ekf_x, y_ekf_y, ekf_yaw],
            distance,
            true,
            true,
            true,
        ));
        Ok(true)
    }

    /// Delay-compensate roll/pitch/height of a pose measurement using the last angular
    /// velocity (port of `compensate_rph_with_delay`).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] from the vx state read.
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        clippy::allow_attributes,
        reason = "f64 math; constant indices; the stamp shift mirrors rclcpp Duration::from_seconds's truncating cast"
    )]
    pub fn compensate_rph_with_delay(
        &self,
        pose: &PoseWithCovariance,
        last_angular_velocity: [f64; 3],
        delay_time: f64,
    ) -> Result<PoseWithCovariance, KalmanError> {
        let delta_orientation = if vector3_length(&last_angular_velocity) > 0.0 {
            quaternion_from_axis_angle(
                &vector3_normalized(&last_angular_velocity),
                vector3_length(&last_angular_velocity) * delay_time,
            )
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };

        let prev_orientation = pose.orientation;
        let curr_orientation =
            quaternion_normalize(&quaternion_multiply(&prev_orientation, &delta_orientation));

        let mut pose_with_delay = pose.clone();
        // rclcpp::Time + Duration::from_seconds(delay_time): the offset is the truncating
        // int64 cast of delay_time * 1e9.
        pose_with_delay.stamp_ns = pose.stamp_ns.saturating_add((delay_time * 1.0e9) as i64);
        pose_with_delay.orientation = curr_orientation;

        let (_roll, pitch, _yaw) = get_rpy(&curr_orientation);
        let vx = self.kalman_filter.x_element(idx::VX)?;
        let delta_z = vx * delay_time * libm::sin(-pitch);
        pose_with_delay.position[2] += delta_z;

        Ok(pose_with_delay)
    }

    /// Twist measurement update (port of `measurement_update_twist`). Returns whether the
    /// update was applied; pushes one trace event (the port's twist/predict trace extension).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] only for internal invariant failures.
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::too_many_lines,
        clippy::allow_attributes,
        reason = "f64 math; index arithmetic bounded by delay_step < extend_state_step; length mirrors the C++ decision chain"
    )]
    pub fn measurement_update_twist(
        &mut self,
        twist: &TwistWithCovariance,
        t_curr_ns: i64,
        twist_diag_info: &mut EkfDiagnosticInfo,
        trace: &mut Vec<TraceEvent>,
    ) -> Result<bool, KalmanError> {
        let dim_x = self.dim_x;
        let nan = f64::NAN;

        self.last_angular_velocity = [0.0, 0.0, 0.0];

        // Calculate delay step.
        #[expect(
            clippy::cast_precision_loss,
            clippy::as_conversions,
            reason = "rclcpp Duration::seconds() is double(nanoseconds)/1e9; mirrored deliberately"
        )]
        let mut delay_time = (t_curr_ns.saturating_sub(twist.stamp_ns)) as f64 / 1.0e9
            + self.params.twist_additional_delay;
        delay_time = cpp_max(delay_time, 0.0);

        let delay_step = self.find_closest_delay_time_index(delay_time);

        twist_diag_info.delay_time = cpp_max(delay_time, twist_diag_info.delay_time);
        twist_diag_info.delay_time_threshold =
            self.accumulated_delay_times.last().copied().unwrap_or(0.0);
        if delay_step >= self.params.extend_state_step {
            twist_diag_info.is_passed_delay_gate = false;
            trace.push(self.make_trace_event(
                EventKind::Twist,
                t_curr_ns,
                twist.stamp_ns,
                delay_time,
                delay_step,
                [twist.linear[0], twist.angular[2], nan],
                [nan, nan, nan],
                nan,
                false,
                false,
                false,
            ));
            return Ok(false);
        }

        // Measurement vector (staged in the preallocated buffer).
        self.scratch.y2[(0, 0)] = twist.linear[0];
        self.scratch.y2[(1, 0)] = twist.angular[2];

        if has_nan(&self.scratch.y2) || has_inf(&self.scratch.y2) {
            trace.push(self.make_trace_event(
                EventKind::Twist,
                t_curr_ns,
                twist.stamp_ns,
                delay_time,
                delay_step,
                [twist.linear[0], twist.angular[2], nan],
                [nan, nan, nan],
                nan,
                true,
                false,
                false,
            ));
            return Ok(false);
        }

        let base = delay_step
            .checked_mul(dim_x)
            .ok_or(KalmanError::IndexOutOfRange)?;
        let y_ekf_vx = self.kalman_filter.x_element(
            base.checked_add(idx::VX)
                .ok_or(KalmanError::IndexOutOfRange)?,
        )?;
        let y_ekf_wz = self.kalman_filter.x_element(
            base.checked_add(idx::WZ)
                .ok_or(KalmanError::IndexOutOfRange)?,
        )?;
        self.scratch.y_ekf2[(0, 0)] = y_ekf_vx;
        self.scratch.y_ekf2[(1, 0)] = y_ekf_wz;
        let p_curr = self.kalman_filter.get_p_ex();
        if p_curr.nrows() < 6 || p_curr.ncols() < 6 {
            return Err(KalmanError::DimensionMismatch);
        }
        self.scratch.p_y2.copy_from(&p_curr.view((4, 4), (2, 2)));

        let distance = mahalanobis_in(
            &self.scratch.y_ekf2,
            &self.scratch.y2,
            &self.scratch.p_y2,
            &mut self.scratch.mh2,
        );
        twist_diag_info.mahalanobis_distance =
            cpp_max(distance, twist_diag_info.mahalanobis_distance);
        if distance > self.params.twist_gate_dist {
            twist_diag_info.is_passed_mahalanobis_gate = false;
            trace.push(self.make_trace_event(
                EventKind::Twist,
                t_curr_ns,
                twist.stamp_ns,
                delay_time,
                delay_step,
                [twist.linear[0], twist.angular[2], nan],
                [y_ekf_vx, y_ekf_wz, nan],
                distance,
                true,
                false,
                false,
            ));
            return Ok(false);
        }

        let c = twist_measurement_matrix();
        let r = twist_measurement_covariance(&twist.covariance, self.params.twist_smoothing_steps);
        self.scratch.c2.copy_from(&c);
        self.scratch.r2.copy_from(&r);

        // The C++ discards updateWithDelay's bool here as well.
        let _update_applied_inside_kf = self
            .kalman_filter
            .update_with_delay(
                &self.scratch.y2,
                &self.scratch.c2,
                &self.scratch.r2,
                delay_step,
            )
            .is_ok();

        self.last_angular_velocity = twist.angular;

        trace.push(self.make_trace_event(
            EventKind::Twist,
            t_curr_ns,
            twist.stamp_ns,
            delay_time,
            delay_step,
            [twist.linear[0], twist.angular[2], nan],
            [y_ekf_vx, y_ekf_wz, nan],
            distance,
            true,
            true,
            true,
        ));
        Ok(true)
    }

    /// Prediction step plus its trace event (used by the tick orchestration; the C++ node
    /// traces this from the same call site).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] from [`EkfModule::predict_with_delay`].
    pub fn predict_with_delay_traced(
        &mut self,
        dt: f64,
        t_curr_ns: i64,
        trace: &mut Vec<TraceEvent>,
    ) -> Result<(), KalmanError> {
        self.predict_with_delay(dt)?;
        let nan = f64::NAN;
        trace.push(self.make_trace_event(
            EventKind::Predict,
            t_curr_ns,
            0,
            dt,
            0,
            [nan, nan, nan],
            [nan, nan, nan],
            nan,
            true,
            true,
            true,
        ));
        Ok(())
    }

    /// Update the z/roll/pitch scalar filters from a (delay-compensated) pose measurement
    /// (port of `update_simple_1d_filters`).
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::cast_precision_loss,
        clippy::as_conversions,
        clippy::allow_attributes,
        reason = "f64 math; constant indices into [f64; 36]; smoothing-step cast mirrors the C++"
    )]
    fn update_simple_1d_filters(&mut self, pose: &PoseWithCovariance, smoothing_step: usize) {
        let z = pose.position[2];
        let (roll, pitch, _yaw) = get_rpy(&pose.orientation);

        let z_var = pose.covariance[cov_idx::Z_Z] * smoothing_step as f64;
        let roll_var = pose.covariance[cov_idx::ROLL_ROLL] * smoothing_step as f64;
        let pitch_var = pose.covariance[cov_idx::PITCH_PITCH] * smoothing_step as f64;

        self.z_filter.update(z, z_var, self.ekf_dt);
        self.roll_filter.update(roll, roll_var, self.ekf_dt);
        self.pitch_filter.update(pitch, pitch_var, self.ekf_dt);
    }

    /// Delay-table threshold (`accumulated_delay_times.back()`; diagnostics).
    #[must_use]
    pub fn delay_time_threshold(&self) -> f64 {
        self.accumulated_delay_times.last().copied().unwrap_or(0.0)
    }
}
