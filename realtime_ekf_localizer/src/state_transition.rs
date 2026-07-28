//! Port of `state_transition.cpp` — the nonlinear process model, its linearization, and the
//! process-noise covariance for the 6-state EKF (`x, y, yaw, yaw_bias, vx, wz`).

use nalgebra::{Matrix6, Vector6};

/// State-vector indices (port of `state_index.hpp` `IDX`).
pub mod idx {
    /// Position x.
    pub const X: usize = 0;
    /// Position y.
    pub const Y: usize = 1;
    /// Biased yaw (the filter state holds `yaw` such that published yaw = `yaw + yaw_bias`).
    pub const YAW: usize = 2;
    /// Yaw bias.
    pub const YAWB: usize = 3;
    /// Longitudinal velocity.
    pub const VX: usize = 4;
    /// Yaw rate.
    pub const WZ: usize = 5;
}

/// Wrap an angle into (-π, π] via `atan2(sin(yaw), cos(yaw))` (port of `normalize_yaw`).
#[must_use]
pub fn normalize_yaw(yaw: f64) -> f64 {
    libm::atan2(libm::sin(yaw), libm::cos(yaw))
}

/// Nonlinear state transition (port of `predict_next_state`):
///
/// ```text
/// x_{k+1}   = x_k + vx_k * cos(yaw_k + b_k) * dt
/// y_{k+1}   = y_k + vx_k * sin(yaw_k + b_k) * dt
/// yaw_{k+1} = normalize(yaw_k + wz_k * dt)
/// b, vx, wz unchanged
/// ```
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "f64 math; constant indices into fixed-size Vector6"
)]
#[must_use]
pub fn predict_next_state(x_curr: &Vector6<f64>, dt: f64) -> Vector6<f64> {
    let x = x_curr[idx::X];
    let y = x_curr[idx::Y];
    let yaw = x_curr[idx::YAW];
    let yaw_bias = x_curr[idx::YAWB];
    let vx = x_curr[idx::VX];
    let wz = x_curr[idx::WZ];

    let mut x_next = Vector6::zeros();
    x_next[idx::X] = x + vx * libm::cos(yaw + yaw_bias) * dt;
    x_next[idx::Y] = y + vx * libm::sin(yaw + yaw_bias) * dt;
    x_next[idx::YAW] = normalize_yaw(yaw + wz * dt);
    x_next[idx::YAWB] = yaw_bias;
    x_next[idx::VX] = vx;
    x_next[idx::WZ] = wz;
    x_next
}

/// Linearized state-transition matrix `A = df/dx` (port of `create_state_transition_matrix`).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "f64 math; constant indices into fixed-size Matrix6"
)]
#[must_use]
pub fn create_state_transition_matrix(x_curr: &Vector6<f64>, dt: f64) -> Matrix6<f64> {
    let yaw = x_curr[idx::YAW];
    let yaw_bias = x_curr[idx::YAWB];
    let vx = x_curr[idx::VX];

    let mut a = Matrix6::identity();
    a[(idx::X, idx::YAW)] = -vx * libm::sin(yaw + yaw_bias) * dt;
    a[(idx::X, idx::YAWB)] = -vx * libm::sin(yaw + yaw_bias) * dt;
    a[(idx::X, idx::VX)] = libm::cos(yaw + yaw_bias) * dt;
    a[(idx::Y, idx::YAW)] = vx * libm::cos(yaw + yaw_bias) * dt;
    a[(idx::Y, idx::YAWB)] = vx * libm::cos(yaw + yaw_bias) * dt;
    a[(idx::Y, idx::VX)] = libm::sin(yaw + yaw_bias) * dt;
    a[(idx::YAW, idx::WZ)] = dt;
    a
}

/// Process-noise covariance `Q` (port of `process_noise_covariance`; only yaw/vx/wz carry
/// process noise — x/y receive theirs through the model).
#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "constant indices into fixed-size Matrix6"
)]
#[must_use]
pub fn process_noise_covariance(
    proc_cov_yaw_d: f64,
    proc_cov_vx_d: f64,
    proc_cov_wz_d: f64,
) -> Matrix6<f64> {
    let mut q = Matrix6::zeros();
    q[(idx::X, idx::X)] = 0.0;
    q[(idx::Y, idx::Y)] = 0.0;
    q[(idx::YAW, idx::YAW)] = proc_cov_yaw_d;
    q[(idx::YAWB, idx::YAWB)] = 0.0;
    q[(idx::VX, idx::VX)] = proc_cov_vx_d;
    q[(idx::WZ, idx::WZ)] = proc_cov_wz_d;
    q
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
    use core::f64::consts::PI;

    // Transcription of test_state_transition.cpp.

    #[test]
    fn normalize_yaw_wraps() {
        let tolerance = 1e-6;
        assert!((normalize_yaw(PI * 4.0 / 3.0) - (-PI * 2.0 / 3.0)).abs() < tolerance);
        assert!((normalize_yaw(-PI * 4.0 / 3.0) - (PI * 2.0 / 3.0)).abs() < tolerance);
        assert!((normalize_yaw(PI * 9.0 / 2.0) - (PI * 1.0 / 2.0)).abs() < tolerance);
        assert!((normalize_yaw(PI * 4.0) - 0.0).abs() < tolerance);
    }

    #[test]
    fn predict_next_state_formula() {
        let x_curr =
            Vector6::from_column_slice(&[2.0, 3.0, PI / 2.0, PI / 4.0, 10.0, 2.0 * PI / 3.0]);
        let dt = 0.5;
        let x_next = predict_next_state(&x_curr, dt);

        let tolerance = 1e-10;
        assert!(
            (x_next[0] - (2.0 + 10.0 * libm::cos(PI / 2.0 + PI / 4.0) * 0.5)).abs() < tolerance
        );
        assert!(
            (x_next[1] - (3.0 + 10.0 * libm::sin(PI / 2.0 + PI / 4.0) * 0.5)).abs() < tolerance
        );
        assert!((x_next[2] - normalize_yaw(PI / 2.0 + PI / 3.0)).abs() < tolerance);
        assert!((x_next[3] - x_curr[3]).abs() < tolerance);
        assert!((x_next[4] - x_curr[4]).abs() < tolerance);
        assert!((x_next[5] - x_curr[5]).abs() < tolerance);
    }

    #[test]
    fn state_transition_matrix_numerical_approximation() {
        // df = A * dx approximates f(x + dx) - f(x).
        {
            let dt = 0.1;
            let dx = Vector6::repeat(0.1);
            let x = Vector6::zeros();
            let a = create_state_transition_matrix(&x, dt);
            let df = predict_next_state(&(x + dx), dt) - predict_next_state(&x, dt);
            assert!((df - a * dx).norm() < 2e-3);
        }
        {
            let dt = 0.1;
            let dx = Vector6::repeat(0.1);
            let x = Vector6::from_column_slice(&[0.1, 0.2, 0.1, 0.4, 0.1, 0.3]);
            let a = create_state_transition_matrix(&x, dt);
            let df = predict_next_state(&(x + dx), dt) - predict_next_state(&x, dt);
            assert!((df - a * dx).norm() < 5e-3);
        }
    }

    #[test]
    fn process_noise_covariance_layout() {
        let q = process_noise_covariance(1.0, 2.0, 3.0);
        assert_eq!(q[(2, 2)], 1.0);
        assert_eq!(q[(4, 4)], 2.0);
        assert_eq!(q[(5, 5)], 3.0);
        assert_eq!(process_noise_covariance(0.0, 0.0, 0.0).norm(), 0.0);
    }
}
