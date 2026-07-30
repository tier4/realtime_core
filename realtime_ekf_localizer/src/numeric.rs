//! Port of `numeric.hpp` — NaN/Inf detection on measurement vectors.

use nalgebra::DMatrix;

/// `true` when any element is ±Inf (port of `has_inf`).
#[must_use]
pub fn has_inf(v: &DMatrix<f64>) -> bool {
    v.iter().any(|e| e.is_infinite())
}

/// `true` when any element is NaN (port of `has_nan`).
#[must_use]
pub fn has_nan(v: &DMatrix<f64>) -> bool {
    v.iter().any(|e| e.is_nan())
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

    // Transcription of test_numeric.cpp.

    fn v3(a: f64, b: f64, c: f64) -> DMatrix<f64> {
        DMatrix::from_column_slice(3, 1, &[a, b, c])
    }

    #[test]
    fn has_nan_detects_nan_only() {
        let empty = DMatrix::<f64>::zeros(0, 1);
        assert!(!has_nan(&empty));
        assert!(!has_nan(&v3(0.0, 0.0, 1.0)));
        assert!(!has_nan(&v3(1e16, 0.0, 1.0)));
        assert!(!has_nan(&v3(0.0, 1.0, f64::INFINITY)));
        assert!(has_nan(&v3(f64::NAN, 1.0, 0.0)));
    }

    #[test]
    fn has_inf_detects_inf_only() {
        let empty = DMatrix::<f64>::zeros(0, 1);
        assert!(!has_inf(&empty));
        assert!(!has_inf(&v3(0.0, 0.0, 1.0)));
        assert!(!has_inf(&v3(1e16, 0.0, 1.0)));
        assert!(!has_inf(&v3(f64::NAN, 1.0, 0.0)));
        assert!(has_inf(&v3(0.0, 1.0, f64::INFINITY)));
    }
}
