//! Pure ports of the `pclomp` covariance helpers from `src/ndt_omp/estimate_covariance.cpp`.
//!
//! All public fns take/return plain arrays/slices (row-major for matrices) so the FFI boundary is
//! unambiguous; nalgebra is used internally. `no_std`-clean (fixed-size matrices, `libm`).
//! `test_estimate_covariance` is the differential oracle.

// nalgebra's fixed-size f64 matrix operators (+, *, unary -) are the float-math domain the
// integer-overflow lint targets; they cannot integer-overflow.
// Numeric kernel. Suppressions are scoped per-function (no module-wide `#![allow]`):
// arithmetic_side_effects = nalgebra f64 float math (cannot integer-overflow); doc_markdown =
// domain terms (base_link, etc.) that are not code.

use nalgebra::{Matrix2, Vector2};

fn mat2(row_major: &[f64; 4]) -> Matrix2<f64> {
    let &[m00, m01, m10, m11] = row_major;
    Matrix2::new(m00, m01, m10, m11)
}

fn to_row_major(m: &Matrix2<f64>) -> [f64; 4] {
    [m.m11, m.m12, m.m21, m.m22]
}

/// Temperature-scaled softmax of `scores` written into `out` (must be the same length).
///
/// A non-positive or non-finite `temperature` falls back to **uniform weights** (`1/n`) instead of
/// dividing by zero — the C++ (`estimate_covariance.cpp` `calc_weight_vec`) divides unguarded and
/// would poison the weights with NaN/Inf. Documented divergence (degenerate config only). Valid
/// temperatures are bit-identical to C++.
pub fn calc_weight_vec(scores: &[f64], temperature: f64, out: &mut [f64]) {
    if !(temperature > 0.0 && temperature.is_finite()) {
        let n = scores.len().min(out.len());
        if n > 0 {
            #[expect(
                clippy::cast_precision_loss,
                clippy::as_conversions,
                reason = "n is a small pose-candidate count, exactly representable in f64"
            )]
            let w = 1.0 / (n as f64);
            for slot in out.iter_mut() {
                *slot = w;
            }
        }
        return;
    }
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (slot, &s) in out.iter_mut().zip(scores.iter()) {
        let e = libm::exp((s - max_score) / temperature);
        *slot = e;
        sum += e;
    }
    for slot in out.iter_mut() {
        *slot /= sum;
    }
}

/// Weighted mean and covariance of 2D points. `poses2d_flat` is `[x0,y0, x1,y1, ...]`
/// (`2 * weights.len()` long). Returns `(mean[x,y], cov row-major 2x2)`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix math"
)]
#[must_use]
pub fn calculate_weighted_mean_and_cov(
    poses2d_flat: &[f64],
    weights: &[f64],
) -> ([f64; 2], [f64; 4]) {
    let mut mean = Vector2::zeros();
    for (chunk, &w) in poses2d_flat.chunks_exact(2).zip(weights.iter()) {
        let [px, py] = chunk else { continue };
        mean += w * Vector2::new(*px, *py);
    }
    let mut cov = Matrix2::zeros();
    for (chunk, &w) in poses2d_flat.chunks_exact(2).zip(weights.iter()) {
        let [px, py] = chunk else { continue };
        let diff = Vector2::new(*px, *py) - mean;
        cov += w * (diff * diff.transpose());
    }
    ([mean.x, mean.y], to_row_major(&cov))
}

/// `-inverse` of the top-left 2x2 block of a row-major 6x6 Hessian. NaN-filled if singular.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix math"
)]
#[must_use]
pub fn laplace_xy_covariance(hessian_row_major: &[f64; 36]) -> [f64; 4] {
    // top-left 2x2 of the row-major 6x6: indices 0,1 (row 0) and 6,7 (row 1).
    let &[h00, h01, _, _, _, _, h10, h11, ..] = hessian_row_major;
    let hessian_xy = Matrix2::new(h00, h01, h10, h11);
    match hessian_xy.try_inverse() {
        Some(inv) => to_row_major(&(-inv)),
        None => [f64::NAN; 4],
    }
}

/// `R^T * C * R` — rotate a map-frame covariance into base_link. `rot` is the row-major 2x2 yaw block.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::doc_markdown,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix math; base_link is a domain term, not code"
)]
#[must_use]
pub fn rotate_covariance_to_base_link(cov: &[f64; 4], rot: &[f64; 4]) -> [f64; 4] {
    let r = mat2(rot);
    to_row_major(&(r.transpose() * mat2(cov) * r))
}

/// `R * C * R^T` — rotate a base_link-frame covariance into map.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix math"
)]
#[must_use]
pub fn rotate_covariance_to_map(cov: &[f64; 4], rot: &[f64; 4]) -> [f64; 4] {
    let r = mat2(rot);
    to_row_major(&(r * mat2(cov) * r.transpose()))
}

/// Clamp the base_link-frame diagonal to floors, then rotate back to map.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix math"
)]
#[must_use]
pub fn adjust_diagonal_covariance(
    cov: &[f64; 4],
    rot: &[f64; 4],
    fixed_cov00: f64,
    fixed_cov11: f64,
) -> [f64; 4] {
    let base = mat2(&rotate_covariance_to_base_link(cov, rot));
    let clamped = Matrix2::new(
        base.m11.max(fixed_cov00),
        base.m12,
        base.m21,
        base.m22.max(fixed_cov11),
    );
    let r = mat2(rot);
    to_row_major(&(r * clamped * r.transpose()))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    unsafe_code,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    #[test]
    fn calc_weight_vec_degenerate_temperature_is_uniform() {
        // Degenerate temperature (C++-unguarded /temperature): uniform 1/n fallback, no NaN/Inf.
        let scores = [1.0, 2.0, 3.0, 4.0];
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut out = [-1.0_f64; 4];
            calc_weight_vec(&scores, t, &mut out);
            for &w in &out {
                assert_eq!(w, 0.25, "temperature {t}");
            }
        }
    }

    #[test]
    fn calc_weight_vec_valid_temperature_unchanged() {
        // The guard must not perturb the valid domain: softmax normalized, best score dominant.
        let scores = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0_f64; 4];
        calc_weight_vec(&scores, 0.1, &mut out);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(out[3] > out[0]);
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn calc_weight_vec_uniform_for_equal_scores() {
        let scores = [1.5_f64; 4];
        let mut out = [0.0_f64; 4];
        calc_weight_vec(&scores, 1.0, &mut out);
        for w in out {
            assert!(close(w, 0.25));
        }
    }

    #[test]
    fn calc_weight_vec_known_softmax() {
        let scores = [0.0, libm::log(3.0)];
        let mut out = [0.0_f64; 2];
        calc_weight_vec(&scores, 1.0, &mut out);
        assert!(close(out[0], 0.25));
        assert!(close(out[1], 0.75));
    }

    #[test]
    fn weighted_mean_and_cov_known_values() {
        let poses = [0.0, 0.0, 2.0, 0.0];
        let weights = [0.5, 0.5];
        let (mean, cov) = calculate_weighted_mean_and_cov(&poses, &weights);
        assert!(close(mean[0], 1.0) && close(mean[1], 0.0));
        assert!(
            close(cov[0], 1.0) && close(cov[1], 0.0) && close(cov[2], 0.0) && close(cov[3], 0.0)
        );
    }

    #[test]
    fn laplace_negative_inverse_hessian() {
        let mut h = [0.0_f64; 36];
        h[0] = -2.0; // (0,0)
        h[7] = -4.0; // (1,1)
        h[14] = -100.0; // (2,2)
        h[21] = 7.0; // (3,3)
        let cov = laplace_xy_covariance(&h);
        assert!(
            close(cov[0], 0.5) && close(cov[3], 0.25) && close(cov[1], 0.0) && close(cov[2], 0.0)
        );
    }

    #[test]
    fn rotate_map_and_base_link_are_inverses() {
        let theta = 0.3_f64;
        let (c, s) = (libm::cos(theta), libm::sin(theta));
        let rot = [c, -s, s, c];
        let cov = [4.0, 1.0, 1.0, 9.0];
        let map = rotate_covariance_to_map(&cov, &rot);
        let exp00 = c * c * 4.0 - 2.0 * c * s + s * s * 9.0;
        let exp11 = s * s * 4.0 + 2.0 * c * s + c * c * 9.0;
        assert!(close(map[0], exp00) && close(map[3], exp11));
        let round_trip = rotate_covariance_to_base_link(&map, &rot);
        for (a, b) in round_trip.iter().zip(cov.iter()) {
            assert!(close(*a, *b));
        }
    }

    #[test]
    fn adjust_diagonal_clamps_and_keeps() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let clamped = adjust_diagonal_covariance(&[1.0, 0.0, 0.0, 1.0], &identity, 4.0, 9.0);
        assert!(close(clamped[0], 4.0) && close(clamped[3], 9.0));
        let kept = adjust_diagonal_covariance(&[10.0, 0.0, 0.0, 20.0], &identity, 4.0, 9.0);
        assert!(close(kept[0], 10.0) && close(kept[3], 20.0));
    }

    // Error path: a singular top-left 2x2 Hessian block has no inverse, so the Laplace covariance
    // is NaN-filled (the documented contract). Oracle: every entry is NaN.
    #[test]
    fn laplace_singular_hessian_is_nan() {
        let mut h = [0.0_f64; 36];
        // top-left block [[1,1],[1,1]] is rank-1 (det 0) -> not invertible.
        h[0] = 1.0;
        h[1] = 1.0;
        h[6] = 1.0;
        h[7] = 1.0;
        let cov = laplace_xy_covariance(&h);
        assert!(
            cov.iter().all(|v| v.is_nan()),
            "singular Hessian -> all NaN"
        );
    }
}
