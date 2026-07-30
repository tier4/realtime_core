//! ROS-free plain-struct stand-ins for the `geometry_msgs` types the EKF consumes, plus the
//! `XYZRPY_COV_IDX` covariance index constants (`autoware_utils_geometry/msg/covariance.hpp`).
//! Timestamps are `i64` nanoseconds (the value `rclcpp::Time::nanoseconds()` returns).

/// Row-major indices into the ROS 6×6 pose/twist covariance (x, y, z, roll, pitch, yaw).
pub mod cov_idx {
    /// (x, x)
    pub const X_X: usize = 0;
    /// (x, y)
    pub const X_Y: usize = 1;
    /// (x, yaw)
    pub const X_YAW: usize = 5;
    /// (y, x)
    pub const Y_X: usize = 6;
    /// (y, y)
    pub const Y_Y: usize = 7;
    /// (y, yaw)
    pub const Y_YAW: usize = 11;
    /// (z, z)
    pub const Z_Z: usize = 14;
    /// (roll, roll)
    pub const ROLL_ROLL: usize = 21;
    /// (pitch, pitch)
    pub const PITCH_PITCH: usize = 28;
    /// (yaw, x)
    pub const YAW_X: usize = 30;
    /// (yaw, y)
    pub const YAW_Y: usize = 31;
    /// (yaw, yaw)
    pub const YAW_YAW: usize = 35;
}

/// Quaternion as `[x, y, z, w]` (the `geometry_msgs` field order).
pub type Quaternion = [f64; 4];

/// `geometry_msgs::msg::PoseWithCovarianceStamped`, reduced to the fields the EKF reads.
#[derive(Clone, Debug)]
pub struct PoseWithCovariance {
    /// Header stamp in nanoseconds.
    pub stamp_ns: i64,
    /// Position `[x, y, z]`.
    pub position: [f64; 3],
    /// Orientation quaternion `[x, y, z, w]`.
    pub orientation: Quaternion,
    /// Row-major 6×6 covariance (x, y, z, roll, pitch, yaw).
    pub covariance: [f64; 36],
}

/// `geometry_msgs::msg::TwistWithCovarianceStamped`, reduced to the fields the EKF reads.
#[derive(Clone, Debug)]
pub struct TwistWithCovariance {
    /// Header stamp in nanoseconds.
    pub stamp_ns: i64,
    /// Linear velocity `[x, y, z]` (only `x` feeds the filter).
    pub linear: [f64; 3],
    /// Angular velocity `[x, y, z]` (`z` feeds the filter; the full vector feeds the
    /// roll/pitch/height delay compensation).
    pub angular: [f64; 3],
    /// Row-major 6×6 covariance (vx, vy, vz, wx, wy, wz).
    pub covariance: [f64; 36],
}

/// `geometry_msgs::msg::TransformStamped`, reduced to the fields `EKFModule::initialize` reads.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    /// Translation `[x, y, z]`.
    pub translation: [f64; 3],
    /// Rotation quaternion `[x, y, z, w]`.
    pub rotation: Quaternion,
}

impl Transform {
    /// Identity transform (zero translation, unit quaternion) — what the replay harness feeds.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// A stamped pose output (`get_current_pose` result: position + RPY-built quaternion).
#[derive(Clone, Copy, Debug)]
pub struct PoseOutput {
    /// Position `[x, y, z]`.
    pub position: [f64; 3],
    /// Orientation quaternion `[x, y, z, w]`.
    pub orientation: Quaternion,
}

/// A twist output (`get_current_twist` result).
#[derive(Clone, Copy, Debug)]
pub struct TwistOutput {
    /// Longitudinal velocity.
    pub vx: f64,
    /// Yaw rate.
    pub wz: f64,
}
