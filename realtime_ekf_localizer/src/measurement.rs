//! Port of `measurement.cpp` — measurement matrices and measurement-noise covariances for the
//! pose (`x, y, yaw`) and twist (`vx, wz`) updates.

use nalgebra::{Matrix2, Matrix3, SMatrix};

use crate::msg::cov_idx;
use crate::state_transition::idx;

/// Pose measurement matrix `C` (3×6): observes `x`, `y`, `yaw`.
#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "constant indices into a fixed-size matrix"
)]
#[must_use]
pub fn pose_measurement_matrix() -> SMatrix<f64, 3, 6> {
    let mut c = SMatrix::<f64, 3, 6>::zeros();
    c[(0, idx::X)] = 1.0;
    c[(1, idx::Y)] = 1.0;
    c[(2, idx::YAW)] = 1.0;
    c
}

/// Twist measurement matrix `C` (2×6): observes `vx`, `wz`.
#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "constant indices into a fixed-size matrix"
)]
#[must_use]
pub fn twist_measurement_matrix() -> SMatrix<f64, 2, 6> {
    let mut c = SMatrix::<f64, 2, 6>::zeros();
    c[(0, idx::VX)] = 1.0;
    c[(1, idx::WZ)] = 1.0;
    c
}

/// Pose measurement covariance `R` (3×3) from the ROS 6×6 covariance, scaled by the smoothing
/// step count (port of `pose_measurement_covariance`).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::allow_attributes,
    reason = "f64 math; constant indices into [f64; 36]; smoothing-step cast mirrors the C++ static_cast<double>"
)]
#[must_use]
pub fn pose_measurement_covariance(covariance: &[f64; 36], smoothing_step: usize) -> Matrix3<f64> {
    let r = Matrix3::new(
        covariance[cov_idx::X_X],
        covariance[cov_idx::X_Y],
        covariance[cov_idx::X_YAW],
        covariance[cov_idx::Y_X],
        covariance[cov_idx::Y_Y],
        covariance[cov_idx::Y_YAW],
        covariance[cov_idx::YAW_X],
        covariance[cov_idx::YAW_Y],
        covariance[cov_idx::YAW_YAW],
    );
    r * smoothing_step as f64
}

/// Twist measurement covariance `R` (2×2) from the ROS 6×6 covariance, scaled by the smoothing
/// step count (port of `twist_measurement_covariance`).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::allow_attributes,
    reason = "f64 math; constant indices into [f64; 36]; smoothing-step cast mirrors the C++ static_cast<double>"
)]
#[must_use]
pub fn twist_measurement_covariance(covariance: &[f64; 36], smoothing_step: usize) -> Matrix2<f64> {
    let r = Matrix2::new(
        covariance[cov_idx::X_X],
        covariance[cov_idx::X_YAW],
        covariance[cov_idx::YAW_X],
        covariance[cov_idx::YAW_YAW],
    );
    r * smoothing_step as f64
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
    use nalgebra::{Matrix2, Matrix3, SMatrix};

    // Transcription of test_measurement.cpp.

    #[test]
    fn pose_measurement_matrix_layout() {
        let m = pose_measurement_matrix();
        let expected = SMatrix::<f64, 3, 6>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]);
        assert_eq!((m - expected).norm(), 0.0);
    }

    #[test]
    fn twist_measurement_matrix_layout() {
        let m = twist_measurement_matrix();
        let expected = SMatrix::<f64, 2, 6>::from_row_slice(&[
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!((m - expected).norm(), 0.0);
    }

    #[test]
    fn pose_measurement_covariance_mapping() {
        let mut covariance = [0.0_f64; 36];
        covariance[0] = 1.0;
        covariance[1] = 2.0;
        covariance[5] = 3.0;
        covariance[6] = 4.0;
        covariance[7] = 5.0;
        covariance[11] = 6.0;
        covariance[30] = 7.0;
        covariance[31] = 8.0;
        covariance[35] = 9.0;

        let m = pose_measurement_covariance(&covariance, 2);
        let expected = Matrix3::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
        assert_eq!((m - expected).norm(), 0.0);

        let zeros = [0.0_f64; 36];
        assert_eq!(pose_measurement_covariance(&zeros, 2).norm(), 0.0);
    }

    #[test]
    fn twist_measurement_covariance_mapping() {
        let mut covariance = [0.0_f64; 36];
        covariance[0] = 1.0;
        covariance[5] = 2.0;
        covariance[30] = 3.0;
        covariance[35] = 4.0;

        let m = twist_measurement_covariance(&covariance, 2);
        let expected = Matrix2::new(2.0, 4.0, 6.0, 8.0);
        assert_eq!((m - expected).norm(), 0.0);

        let zeros = [0.0_f64; 36];
        assert_eq!(twist_measurement_covariance(&zeros, 2).norm(), 0.0);
    }
}
