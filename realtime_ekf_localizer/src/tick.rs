//! Port of the `EKFLocalizer::timer_callback` / measurement-callback control flow as pure
//! logic: a fake-clock `tick(now_ns)` plus `enqueue_pose` / `enqueue_twist`, with the queue
//! aging, dt measurement (10 s clamp, jump-back guard), and the drain loops transcribed from
//! `ekf_localizer.cpp`. Publishing, diagnostics topics, and TF are out of scope (contract §2).

use alloc::vec::Vec;

use realtime_kalman_filter::KalmanError;

use crate::aged_object_queue::AgedObjectQueue;
use crate::ekf_module::{EkfDiagnosticInfo, EkfModule};
use crate::hyper_parameters::HyperParameters;
use crate::msg::{PoseOutput, PoseWithCovariance, Transform, TwistOutput, TwistWithCovariance};
use crate::trace::TraceEvent;

/// Outputs of one tick (the values the C++ node publishes).
#[derive(Clone, Debug)]
pub struct TickOutput {
    /// Unbiased pose (`get_current_pose(..., false)`).
    pub pose: PoseOutput,
    /// Biased pose (`get_current_pose(..., true)`).
    pub biased_pose: PoseOutput,
    /// Twist estimate.
    pub twist: TwistOutput,
    /// Pose message covariance.
    pub pose_covariance: [f64; 36],
    /// Twist message covariance.
    pub twist_covariance: [f64; 36],
    /// Estimated yaw bias.
    pub yaw_bias: f64,
}

/// The EKF localizer core: EKF module + measurement queues + tick orchestration.
#[derive(Clone, Debug)]
pub struct EkfLocalizerCore {
    params: HyperParameters,
    ekf_module: EkfModule,
    ekf_dt: f64,
    last_predict_ns: Option<i64>,
    pose_queue: AgedObjectQueue<PoseWithCovariance>,
    twist_queue: AgedObjectQueue<TwistWithCovariance>,
    /// Pose-channel diagnostics (public: mirrors the C++ member the node reads).
    pub pose_diag_info: EkfDiagnosticInfo,
    /// Twist-channel diagnostics.
    pub twist_diag_info: EkfDiagnosticInfo,
    is_set_initialpose: bool,
}

impl EkfLocalizerCore {
    /// Construct with the node constructor's state (`ekf_dt` from params, empty queues).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] from the EKF module construction.
    pub fn new(params: HyperParameters) -> Result<Self, KalmanError> {
        let ekf_module = EkfModule::new(params.clone())?;
        Ok(Self {
            ekf_dt: params.ekf_dt,
            pose_queue: AgedObjectQueue::new(
                params.pose_smoothing_steps,
                params.max_pose_queue_size,
            ),
            twist_queue: AgedObjectQueue::new(
                params.twist_smoothing_steps,
                params.max_twist_queue_size,
            ),
            params,
            ekf_module,
            last_predict_ns: None,
            pose_diag_info: EkfDiagnosticInfo::default(),
            twist_diag_info: EkfDiagnosticInfo::default(),
            is_set_initialpose: false,
        })
    }

    /// Initial-pose callback (port of `callback_initial_pose`; the transform is the
    /// `pose_frame_id`→message-frame TF, identity in the replay harness).
    ///
    /// # Errors
    /// Propagates [`KalmanError`] from the filter re-initialization.
    pub fn initialize(
        &mut self,
        initial_pose: &PoseWithCovariance,
        transform: &Transform,
    ) -> Result<(), KalmanError> {
        self.ekf_module.initialize(initial_pose, transform)?;
        self.is_set_initialpose = true;
        Ok(())
    }

    /// Pose measurement callback (port of `callback_pose_with_covariance`).
    pub fn enqueue_pose(&mut self, pose: PoseWithCovariance) {
        // C++: `if (!is_activated_ && !is_set_initialpose_) return;` — the replay treats
        // activation and initial pose as one flag.
        if !self.is_set_initialpose {
            return;
        }
        self.pose_queue.push(pose);
        if self.pose_queue.exceeded() {
            let _dropped: Option<PoseWithCovariance> = self.pose_queue.pop();
        }
    }

    /// Twist measurement callback (port of `callback_twist_with_covariance`, including the
    /// unobservable-velocity covariance override).
    #[allow(
        clippy::indexing_slicing,
        clippy::allow_attributes,
        reason = "constant index into [f64; 36]"
    )]
    pub fn enqueue_twist(&mut self, twist: TwistWithCovariance) {
        let mut twist = twist;
        // Ignore the twist if velocity is too small (must not include "equal").
        if libm::fabs(twist.linear[0]) < self.params.threshold_observable_velocity_mps {
            twist.covariance[0] = 10000.0;
        }
        self.twist_queue.push(twist);
        if self.twist_queue.exceeded() {
            let _dropped: Option<TwistWithCovariance> = self.twist_queue.pop();
        }
    }

    /// Measured-dt update (port of `update_predict_frequency`): jump-back is warn-only (no dt
    /// change, no accumulation); otherwise the dt is measured, clamped to 10 s, and the delay
    /// table accumulates. `last_predict_ns` always advances.
    fn update_predict_frequency(&mut self, current_ns: i64) {
        if let Some(last) = self.last_predict_ns {
            if current_ns < last {
                // "Detected jump back in time" — warn only.
            } else {
                #[expect(
                    clippy::cast_precision_loss,
                    clippy::as_conversions,
                    reason = "rclcpp Duration::seconds() is double(nanoseconds)/1e9; mirrored deliberately"
                )]
                let mut dt = (current_ns.saturating_sub(last)) as f64 / 1.0e9;
                if dt > 10.0 {
                    dt = 10.0;
                }
                // (the too-slow warning branch has no state effect)
                self.ekf_dt = dt;
                self.ekf_module.accumulate_delay_time(self.ekf_dt);
            }
        }
        self.last_predict_ns = Some(current_ns);
    }

    /// One timer tick (port of the active-path body of `timer_callback`). Appends the
    /// predict/pose/twist trace events for this tick to `trace`.
    ///
    /// # Errors
    /// Propagates [`KalmanError`] for internal invariant failures (uninitialized filter).
    /// Returns `Ok(None)` when no initial pose has been set (the C++ early-return).
    pub fn tick(
        &mut self,
        current_ns: i64,
        trace: &mut Vec<TraceEvent>,
    ) -> Result<Option<TickOutput>, KalmanError> {
        if !self.is_set_initialpose {
            return Ok(None);
        }

        // Update predict frequency with the measured timer rate.
        self.update_predict_frequency(current_ns);

        // Predict model in EKF.
        self.ekf_module
            .predict_with_delay_traced(self.ekf_dt, current_ns, trace)?;

        let mut pose_is_updated = false;
        if !self.pose_queue.empty() {
            // These flags are initialized true before their checks in measurement_update_pose.
            self.pose_diag_info.is_passed_delay_gate = true;
            self.pose_diag_info.is_passed_mahalanobis_gate = true;
            // Save the initial size because the queue size can change in the loop.
            let n = self.pose_queue.size();
            for _ in 0..n {
                let Some(pose) = self.pose_queue.pop_increment_age() else {
                    break;
                };
                let is_updated = self.ekf_module.measurement_update_pose(
                    &pose,
                    current_ns,
                    &mut self.pose_diag_info,
                    trace,
                )?;
                pose_is_updated = pose_is_updated || is_updated;
            }
        }
        self.pose_diag_info.no_update_count = if pose_is_updated {
            0
        } else {
            self.pose_diag_info.no_update_count.saturating_add(1)
        };

        let mut twist_is_updated = false;
        if !self.twist_queue.empty() {
            self.twist_diag_info.is_passed_delay_gate = true;
            self.twist_diag_info.is_passed_mahalanobis_gate = true;
            let n = self.twist_queue.size();
            for _ in 0..n {
                let Some(twist) = self.twist_queue.pop_increment_age() else {
                    break;
                };
                let is_updated = self.ekf_module.measurement_update_twist(
                    &twist,
                    current_ns,
                    &mut self.twist_diag_info,
                    trace,
                )?;
                twist_is_updated = twist_is_updated || is_updated;
            }
        }
        self.twist_diag_info.no_update_count = if twist_is_updated {
            0
        } else {
            self.twist_diag_info.no_update_count.saturating_add(1)
        };

        Ok(Some(TickOutput {
            pose: self.ekf_module.get_current_pose(false)?,
            biased_pose: self.ekf_module.get_current_pose(true)?,
            twist: self.ekf_module.get_current_twist()?,
            pose_covariance: self.ekf_module.get_current_pose_covariance()?,
            twist_covariance: self.ekf_module.get_current_twist_covariance()?,
            yaw_bias: self.ekf_module.get_yaw_bias()?,
        }))
    }

    /// Access the EKF module (tests / harness inspection).
    #[must_use]
    pub fn ekf_module(&self) -> &EkfModule {
        &self.ekf_module
    }

    /// Mutable access to the EKF module (unit-test scaffolding parity with the C++ tests,
    /// which call `EKFModule` methods directly).
    #[must_use]
    pub fn ekf_module_mut(&mut self) -> &mut EkfModule {
        &mut self.ekf_module
    }

    /// Current pose queue size.
    #[must_use]
    pub fn pose_queue_size(&self) -> usize {
        self.pose_queue.size()
    }

    /// Current twist queue size.
    #[must_use]
    pub fn twist_queue_size(&self) -> usize {
        self.twist_queue.size()
    }
}
