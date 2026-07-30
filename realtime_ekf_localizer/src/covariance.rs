//! Port of `covariance.cpp` — mapping the EKF 6×6 state covariance into the ROS pose/twist
//! message covariance layout.

use nalgebra::Matrix6;

use crate::msg::cov_idx;
use crate::state_transition::idx;

/// EKF covariance → pose message covariance (x/y/yaw block; other entries zero).
#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "constant indices into fixed-size Matrix6 / [f64; 36]"
)]
#[must_use]
pub fn ekf_covariance_to_pose_message_covariance(p: &Matrix6<f64>) -> [f64; 36] {
    let mut covariance = [0.0_f64; 36];

    covariance[cov_idx::X_X] = p[(idx::X, idx::X)];
    covariance[cov_idx::X_Y] = p[(idx::X, idx::Y)];
    covariance[cov_idx::X_YAW] = p[(idx::X, idx::YAW)];
    covariance[cov_idx::Y_X] = p[(idx::Y, idx::X)];
    covariance[cov_idx::Y_Y] = p[(idx::Y, idx::Y)];
    covariance[cov_idx::Y_YAW] = p[(idx::Y, idx::YAW)];
    covariance[cov_idx::YAW_X] = p[(idx::YAW, idx::X)];
    covariance[cov_idx::YAW_Y] = p[(idx::YAW, idx::Y)];
    covariance[cov_idx::YAW_YAW] = p[(idx::YAW, idx::YAW)];

    covariance
}

/// EKF covariance → twist message covariance (vx/wz block; other entries zero).
#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "constant indices into fixed-size Matrix6 / [f64; 36]"
)]
#[must_use]
pub fn ekf_covariance_to_twist_message_covariance(p: &Matrix6<f64>) -> [f64; 36] {
    let mut covariance = [0.0_f64; 36];

    covariance[cov_idx::X_X] = p[(idx::VX, idx::VX)];
    covariance[cov_idx::X_YAW] = p[(idx::VX, idx::WZ)];
    covariance[cov_idx::YAW_X] = p[(idx::WZ, idx::VX)];
    covariance[cov_idx::YAW_YAW] = p[(idx::WZ, idx::WZ)];

    covariance
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    // Transcription of test_covariance.cpp.

    #[test]
    fn pose_message_covariance_mapping() {
        let mut p = Matrix6::<f64>::zeros();
        p[(0, 0)] = 1.0;
        p[(0, 1)] = 2.0;
        p[(0, 2)] = 3.0;
        p[(1, 0)] = 4.0;
        p[(1, 1)] = 5.0;
        p[(1, 2)] = 6.0;
        p[(2, 0)] = 7.0;
        p[(2, 1)] = 8.0;
        p[(2, 2)] = 9.0;

        let covariance = ekf_covariance_to_pose_message_covariance(&p);
        assert_eq!(covariance[0], 1.0);
        assert_eq!(covariance[1], 2.0);
        assert_eq!(covariance[5], 3.0);
        assert_eq!(covariance[6], 4.0);
        assert_eq!(covariance[7], 5.0);
        assert_eq!(covariance[11], 6.0);
        assert_eq!(covariance[30], 7.0);
        assert_eq!(covariance[31], 8.0);
        assert_eq!(covariance[35], 9.0);

        let zeros = ekf_covariance_to_pose_message_covariance(&Matrix6::zeros());
        assert!(zeros.iter().all(|&e| e == 0.0));
    }

    #[test]
    fn twist_message_covariance_mapping() {
        let mut p = Matrix6::<f64>::zeros();
        p[(4, 4)] = 1.0;
        p[(4, 5)] = 2.0;
        p[(5, 4)] = 3.0;
        p[(5, 5)] = 4.0;

        let covariance = ekf_covariance_to_twist_message_covariance(&p);
        assert_eq!(covariance[0], 1.0);
        assert_eq!(covariance[5], 2.0);
        assert_eq!(covariance[30], 3.0);
        assert_eq!(covariance[35], 4.0);

        let zeros = ekf_covariance_to_twist_message_covariance(&Matrix6::zeros());
        assert!(zeros.iter().all(|&e| e == 0.0));
    }
}
