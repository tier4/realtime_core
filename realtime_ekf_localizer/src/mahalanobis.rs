//! Port of `mahalanobis.cpp` — the Mahalanobis gate distance. The C++ inverts the covariance
//! with Eigen's `.inverse()`, which for runtime sizes 2 and 3 uses the analytic cofactor
//! formulas; this port mirrors those exact expressions (and their behavior on singular input:
//! `1/0 = inf`, propagating to a NaN distance rather than an error, so the `distance > gate`
//! comparison stays `false` exactly as in C++).

use nalgebra::DMatrix;

/// Eigen `compute_inverse<..., 2>`: analytic 2×2 inverse (no singularity check).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "f64 math; constant indices into a dimension-checked 2x2 matrix"
)]
fn inverse2(m: &DMatrix<f64>) -> DMatrix<f64> {
    let det = m[(0, 0)] * m[(1, 1)] - m[(1, 0)] * m[(0, 1)];
    let invdet = 1.0 / det;
    DMatrix::from_row_slice(
        2,
        2,
        &[
            m[(1, 1)] * invdet,
            -m[(0, 1)] * invdet,
            -m[(1, 0)] * invdet,
            m[(0, 0)] * invdet,
        ],
    )
}

/// Eigen `cofactor_3x3<i, j>`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "f64 math; indices are `% 3` of constants into a dimension-checked 3x3 matrix"
)]
fn cofactor3(m: &DMatrix<f64>, i: usize, j: usize) -> f64 {
    let i1 = (i + 1) % 3;
    let i2 = (i + 2) % 3;
    let j1 = (j + 1) % 3;
    let j2 = (j + 2) % 3;
    m[(i1, j1)] * m[(i2, j2)] - m[(i1, j2)] * m[(i2, j1)]
}

/// Eigen `compute_inverse<..., 3>`: analytic cofactor 3×3 inverse (no singularity check).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "f64 math; constant indices into a dimension-checked 3x3 matrix"
)]
fn inverse3(m: &DMatrix<f64>) -> DMatrix<f64> {
    let c00 = cofactor3(m, 0, 0);
    let c10 = cofactor3(m, 1, 0);
    let c20 = cofactor3(m, 2, 0);
    let det = c00 * m[(0, 0)] + c10 * m[(0, 1)] + c20 * m[(0, 2)];
    let invdet = 1.0 / det;
    let mut out = DMatrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            out[(j, i)] = cofactor3(m, i, j) * invdet;
        }
    }
    out
}

/// Matrix inverse dispatching on runtime size like Eigen's dynamic `.inverse()` (analytic for
/// n ≤ 3, LU otherwise). Singular small matrices produce inf/NaN entries, as in Eigen.
#[must_use]
fn eigen_like_inverse(c: &DMatrix<f64>) -> DMatrix<f64> {
    match c.nrows() {
        2 if c.ncols() == 2 => inverse2(c),
        3 if c.ncols() == 3 => inverse3(c),
        _ => c
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::from_element(c.nrows(), c.ncols(), f64::NAN)),
    }
}

/// Squared Mahalanobis distance `dᵀ C⁻¹ d` with `d = x - y` (port of `squared_mahalanobis`).
#[expect(clippy::arithmetic_side_effects, reason = "nalgebra f64 matrix math")]
#[must_use]
pub fn squared_mahalanobis(x: &DMatrix<f64>, y: &DMatrix<f64>, c: &DMatrix<f64>) -> f64 {
    let d = x - y;
    (eigen_like_inverse(c) * &d).dot(&d)
}

/// Mahalanobis distance (port of `mahalanobis`). NaN propagates (never an error), matching
/// the C++ gate semantics on degenerate covariance.
#[must_use]
pub fn mahalanobis(x: &DMatrix<f64>, y: &DMatrix<f64>, c: &DMatrix<f64>) -> f64 {
    libm::sqrt(squared_mahalanobis(x, y, c))
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

    const TOLERANCE: f64 = 1e-8;

    fn v2(a: f64, b: f64) -> DMatrix<f64> {
        DMatrix::from_column_slice(2, 1, &[a, b])
    }

    // Transcription of test_mahalanobis.cpp.
    #[test]
    fn squared_mahalanobis_smoke() {
        {
            let x = v2(0.0, 1.0);
            let y = v2(3.0, 2.0);
            let c = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 10.0]);
            assert!((squared_mahalanobis(&x, &y, &c) - 1.0).abs() < TOLERANCE);
        }
        {
            let x = v2(4.0, 1.0);
            let y = v2(1.0, 5.0);
            let c = DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 5.0]);
            assert!((squared_mahalanobis(&x, &y, &c) - 5.0).abs() < TOLERANCE);
        }
    }

    #[test]
    fn mahalanobis_smoke() {
        {
            let x = v2(0.0, 1.0);
            let y = v2(3.0, 2.0);
            let c = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 10.0]);
            assert!((mahalanobis(&x, &y, &c) - 1.0).abs() < TOLERANCE);
        }
        {
            let x = v2(4.0, 1.0);
            let y = v2(1.0, 5.0);
            let c = DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 5.0]);
            assert!((mahalanobis(&x, &y, &c) - libm::sqrt(5.0)).abs() < TOLERANCE);
        }
    }

    /// 3×3 analytic inverse agrees with the exact inverse of a well-conditioned matrix.
    #[test]
    fn inverse3_matches_known_inverse() {
        let m = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let inv = eigen_like_inverse(&m);
        let prod = &m * &inv;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod[(i, j)] - expected).abs() < 1e-12);
            }
        }
    }

    /// Singular covariance: distance is NaN (C++ semantics — `NaN > gate` is false).
    #[test]
    fn singular_covariance_gives_nan_not_error() {
        let x = v2(0.0, 1.0);
        let y = v2(3.0, 2.0);
        let c = DMatrix::zeros(2, 2);
        assert!(
            squared_mahalanobis(&x, &y, &c).is_nan()
                || squared_mahalanobis(&x, &y, &c).is_infinite()
        );
    }
}
