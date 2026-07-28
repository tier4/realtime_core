//! Port of `autoware_kalman_filter/src/kalman_filter.cpp` — the base `KalmanFilter` on
//! dynamically-sized (`DMatrix`) state, matching the C++ `Eigen::MatrixXd` implementation
//! operation-for-operation (including the LLT-solved Kalman gain and its rejection guards).
//!
//! Every C++ `return false` maps to a typed [`KalmanError`]; the matrix expressions keep the
//! C++ evaluation order (`AP = A*P` then `AP*Aᵀ` then `+= Q`, etc.) so the float rounding
//! matches Eigen as closely as the backends allow.

use nalgebra::{Cholesky, DMatrix, Dyn};

/// Rejection/failure reasons for the fallible filter operations. Mirrors the C++ `bool`
/// results: any variant corresponds to a `return false` path in the C++ sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KalmanError {
    /// An input matrix has zero rows or columns (`init` guards).
    EmptyMatrix,
    /// Matrix dimensions are inconsistent for the requested operation.
    DimensionMismatch,
    /// The innovation covariance `S` is not positive definite (LLT decomposition failed).
    NotPositiveDefinite,
    /// The computed Kalman gain contains NaN or ±Inf.
    NonFiniteGain,
    /// `delay_step` is outside `[0, max_delay_step)` (`TimeDelayKalmanFilter` update guard).
    InvalidDelayStep,
    /// A state-element index is out of range (`getXelement` equivalent).
    IndexOutOfRange,
}

/// Base Kalman filter with dynamically-sized matrices (port of the C++ `KalmanFilter` class).
#[derive(Clone, Debug, Default)]
pub struct KalmanFilter {
    pub(crate) x: DMatrix<f64>,
    pub(crate) a: DMatrix<f64>,
    pub(crate) b: DMatrix<f64>,
    pub(crate) c: DMatrix<f64>,
    pub(crate) q: DMatrix<f64>,
    pub(crate) r: DMatrix<f64>,
    pub(crate) p: DMatrix<f64>,
}

fn is_empty(m: &DMatrix<f64>) -> bool {
    m.ncols() == 0 || m.nrows() == 0
}

impl KalmanFilter {
    /// No-initialization constructor (all matrices 0×0, as default-constructed `MatrixXd`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Full initialization (state, model and noise matrices).
    ///
    /// # Errors
    /// [`KalmanError::EmptyMatrix`] when any input matrix has zero rows or columns.
    #[allow(
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        clippy::allow_attributes,
        reason = "mirrors the C++ init(x,A,B,C,Q,R,P) signature and naming"
    )]
    pub fn init(
        &mut self,
        x: &DMatrix<f64>,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        c: &DMatrix<f64>,
        q: &DMatrix<f64>,
        r: &DMatrix<f64>,
        p: &DMatrix<f64>,
    ) -> Result<(), KalmanError> {
        if is_empty(x)
            || is_empty(a)
            || is_empty(b)
            || is_empty(c)
            || is_empty(q)
            || is_empty(r)
            || is_empty(p)
        {
            return Err(KalmanError::EmptyMatrix);
        }
        self.x = x.clone();
        self.a = a.clone();
        self.b = b.clone();
        self.c = c.clone();
        self.q = q.clone();
        self.r = r.clone();
        self.p = p.clone();
        Ok(())
    }

    /// State-only initialization (`init(x, P0)` in C++).
    ///
    /// # Errors
    /// [`KalmanError::EmptyMatrix`] when `x` or `p0` has zero rows or columns.
    pub fn init_state(&mut self, x: &DMatrix<f64>, p0: &DMatrix<f64>) -> Result<(), KalmanError> {
        if is_empty(x) || is_empty(p0) {
            return Err(KalmanError::EmptyMatrix);
        }
        self.x = x.clone();
        self.p = p0.clone();
        Ok(())
    }

    /// Set the process-model matrix `A`.
    pub fn set_a(&mut self, a: &DMatrix<f64>) {
        self.a = a.clone();
    }

    /// Set the input matrix `B`.
    pub fn set_b(&mut self, b: &DMatrix<f64>) {
        self.b = b.clone();
    }

    /// Set the measurement-model matrix `C`.
    pub fn set_c(&mut self, c: &DMatrix<f64>) {
        self.c = c.clone();
    }

    /// Set the process-noise covariance `Q`.
    pub fn set_q(&mut self, q: &DMatrix<f64>) {
        self.q = q.clone();
    }

    /// Set the measurement-noise covariance `R`.
    pub fn set_r(&mut self, r: &DMatrix<f64>) {
        self.r = r.clone();
    }

    /// Current estimated state (C++ `getX`).
    #[must_use]
    pub fn get_x(&self) -> &DMatrix<f64> {
        &self.x
    }

    /// Current estimation covariance (C++ `getP`).
    #[must_use]
    pub fn get_p(&self) -> &DMatrix<f64> {
        &self.p
    }

    /// `x[i]` in the column-major linear order (C++ `getXelement`).
    ///
    /// # Errors
    /// [`KalmanError::IndexOutOfRange`] when `i` is outside the state vector.
    pub fn x_element(&self, i: usize) -> Result<f64, KalmanError> {
        self.x
            .as_slice()
            .get(i)
            .copied()
            .ok_or(KalmanError::IndexOutOfRange)
    }

    /// Prediction with an externally-computed next state: `x = x_next`, `P = A P Aᵀ + Q`.
    ///
    /// # Errors
    /// [`KalmanError::DimensionMismatch`] when the matrix dimensions are inconsistent
    /// (the C++ `return false` guard).
    #[expect(clippy::arithmetic_side_effects, reason = "nalgebra f64 matrix math")]
    pub fn predict_with_state(
        &mut self,
        x_next: &DMatrix<f64>,
        a: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) -> Result<(), KalmanError> {
        if self.x.nrows() != x_next.nrows()
            || a.ncols() != self.p.nrows()
            || q.ncols() != q.nrows()
            || a.nrows() != q.ncols()
        {
            return Err(KalmanError::DimensionMismatch);
        }
        self.x = x_next.clone();
        // P = A P Aᵀ + Q, evaluated through the same AP temporary as the C++.
        let ap = a * &self.p;
        self.p = &ap * a.transpose();
        self.p += q;
        Ok(())
    }

    /// Prediction with the stored process noise: `predict(x_next, A)` in C++.
    ///
    /// # Errors
    /// [`KalmanError::DimensionMismatch`] when the matrix dimensions are inconsistent.
    pub fn predict_with_state_default_q(
        &mut self,
        x_next: &DMatrix<f64>,
        a: &DMatrix<f64>,
    ) -> Result<(), KalmanError> {
        let q = self.q.clone();
        self.predict_with_state(x_next, a, &q)
    }

    /// Prediction from an input vector: `x_next = A x + B u`, then the covariance update.
    ///
    /// # Errors
    /// [`KalmanError::DimensionMismatch`] when the matrix dimensions are inconsistent.
    #[expect(clippy::arithmetic_side_effects, reason = "nalgebra f64 matrix math")]
    pub fn predict_with_input(
        &mut self,
        u: &DMatrix<f64>,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) -> Result<(), KalmanError> {
        if a.ncols() != self.x.nrows() || b.ncols() != u.nrows() {
            return Err(KalmanError::DimensionMismatch);
        }
        let x_next = a * &self.x + b * u;
        self.predict_with_state(&x_next, a, q)
    }

    /// Prediction from an input vector with the stored `A`, `B`, `Q` (C++ `predict(u)`).
    ///
    /// # Errors
    /// [`KalmanError::DimensionMismatch`] when the matrix dimensions are inconsistent.
    pub fn predict(&mut self, u: &DMatrix<f64>) -> Result<(), KalmanError> {
        let (a, b, q) = (self.a.clone(), self.b.clone(), self.q.clone());
        self.predict_with_input(u, &a, &b, &q)
    }

    /// Measurement update with an externally-computed predicted output `y_pred`.
    ///
    /// The Kalman gain is obtained through a Cholesky (LLT) solve of the innovation covariance
    /// `S = C P Cᵀ + R`, exactly as the C++: a non-positive-definite `S` or a non-finite gain
    /// rejects the update, leaving the state untouched.
    ///
    /// # Errors
    /// [`KalmanError::DimensionMismatch`], [`KalmanError::NotPositiveDefinite`], or
    /// [`KalmanError::NonFiniteGain`] on the corresponding C++ `return false` guard.
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::many_single_char_names,
        clippy::allow_attributes,
        reason = "nalgebra f64 matrix math; C++ kernel naming (y, C, R, S, K)"
    )]
    pub fn update_with_pred(
        &mut self,
        y: &DMatrix<f64>,
        y_pred: &DMatrix<f64>,
        c: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> Result<(), KalmanError> {
        if self.p.ncols() != c.ncols()
            || r.nrows() != r.ncols()
            || r.nrows() != c.nrows()
            || y.nrows() != y_pred.nrows()
            || y.nrows() != c.nrows()
        {
            return Err(KalmanError::DimensionMismatch);
        }
        let pct = &self.p * c.transpose();

        // Innovation covariance S = C P Cᵀ + R.
        let s = r + c * &pct;

        // K = PCT S⁻¹ via LLT: solve S Kᵀ = PCTᵀ (S symmetric), then transpose back.
        let llt: Cholesky<f64, Dyn> = Cholesky::new(s).ok_or(KalmanError::NotPositiveDefinite)?;
        let k = llt.solve(&pct.transpose()).transpose();

        if k.iter().any(|v| !v.is_finite()) {
            return Err(KalmanError::NonFiniteGain);
        }

        self.x += &k * (y - y_pred);
        let cp = c * &self.p;
        self.p -= &k * &cp;
        Ok(())
    }

    /// Measurement update computing `y_pred = C x` (C++ `update(y, C, R)`).
    ///
    /// # Errors
    /// As [`KalmanFilter::update_with_pred`].
    #[expect(clippy::arithmetic_side_effects, reason = "nalgebra f64 matrix math")]
    pub fn update_with_model(
        &mut self,
        y: &DMatrix<f64>,
        c: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> Result<(), KalmanError> {
        if c.ncols() != self.x.nrows() {
            return Err(KalmanError::DimensionMismatch);
        }
        let y_pred = c * &self.x;
        self.update_with_pred(y, &y_pred, c, r)
    }

    /// Measurement update with the stored `C`, `R` (C++ `update(y)`).
    ///
    /// # Errors
    /// As [`KalmanFilter::update_with_pred`].
    #[expect(clippy::arithmetic_side_effects, reason = "nalgebra f64 matrix math")]
    pub fn update(&mut self, y: &DMatrix<f64>) -> Result<(), KalmanError> {
        let (c, r) = (self.c.clone(), self.r.clone());
        self.update_with_pred(y, &(&c * &self.x), &c, &r)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    fn dm(rows: usize, cols: usize, data: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(rows, cols, data)
    }

    /// Transcription of the C++ `TEST(kalman_filter, kf)`.
    #[test]
    fn kf_predict_update_roundtrip() {
        let mut kf = KalmanFilter::new();

        let x_t = dm(2, 1, &[1.0, 2.0]);
        let p_t = dm(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let q_t = dm(2, 2, &[0.01, 0.0, 0.0, 0.01]);
        let r_t = dm(2, 2, &[0.09, 0.0, 0.0, 0.09]);
        let c_t = dm(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let a_t = dm(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_t = dm(2, 2, &[1.0, 0.0, 0.0, 1.0]);

        kf.init(&x_t, &a_t, &b_t, &c_t, &q_t, &r_t, &p_t).unwrap();

        let u_t = dm(2, 1, &[0.1, 0.1]);
        kf.predict(&u_t).unwrap();

        let x_predict = kf.get_x().clone();
        let p_predict = kf.get_p().clone();
        assert!((x_predict[(0, 0)] - 1.1).abs() < 1e-5);
        assert!((x_predict[(1, 0)] - 2.1).abs() < 1e-5);
        assert!((p_predict[(0, 0)] - 1.01).abs() < 1e-5);
        assert!((p_predict[(1, 1)] - 1.01).abs() < 1e-5);

        let y_t = dm(2, 1, &[1.05, 2.05]);
        kf.update(&y_t).unwrap();

        let x_update = kf.get_x().clone();
        let p_update = kf.get_p().clone();
        assert!((x_update[(0, 0)] - 1.0540909090909092).abs() < 1e-5);
        assert!((x_update[(1, 0)] - 2.0540909090909087).abs() < 1e-5);
        assert!((p_update[(0, 0)] - 0.08263636363636362).abs() < 1e-5);
        assert!((p_update[(1, 1)] - 0.08263636363636362).abs() < 1e-5);

        // Explicit-A predict overload on a freshly-initialized state.
        let mut kf_new = KalmanFilter::new();
        kf_new
            .init(&x_t, &a_t, &b_t, &c_t, &q_t, &r_t, &p_t)
            .unwrap();
        kf_new.init_state(&x_t, &p_t).unwrap();
        kf_new.set_a(&a_t);
        kf_new.set_b(&b_t);
        kf_new.set_c(&c_t);
        kf_new.set_q(&q_t);
        kf_new.set_r(&r_t);

        let x_next = dm(2, 1, &[1.1, 2.1]);
        kf_new.predict_with_state_default_q(&x_next, &a_t).unwrap();
        let p_predict = kf_new.get_p().clone();
        assert!((p_predict[(0, 0)] - 1.01).abs() < 1e-5);
        assert!((p_predict[(1, 1)] - 1.01).abs() < 1e-5);
    }

    #[test]
    fn init_rejects_empty_matrices() {
        let mut kf = KalmanFilter::new();
        let empty = DMatrix::<f64>::zeros(0, 0);
        let x = dm(2, 1, &[1.0, 2.0]);
        let i2 = DMatrix::<f64>::identity(2, 2);
        assert_eq!(
            kf.init(&x, &empty, &i2, &i2, &i2, &i2, &i2),
            Err(KalmanError::EmptyMatrix)
        );
        assert_eq!(kf.init_state(&x, &empty), Err(KalmanError::EmptyMatrix));
        assert_eq!(kf.init_state(&empty, &i2), Err(KalmanError::EmptyMatrix));
    }

    #[test]
    fn predict_rejects_dimension_mismatch() {
        let mut kf = KalmanFilter::new();
        let x = dm(2, 1, &[1.0, 2.0]);
        let i2 = DMatrix::<f64>::identity(2, 2);
        kf.init_state(&x, &i2).unwrap();

        let x3 = dm(3, 1, &[1.0, 2.0, 3.0]);
        assert_eq!(
            kf.predict_with_state(&x3, &i2, &i2),
            Err(KalmanError::DimensionMismatch)
        );
        let a_bad = DMatrix::<f64>::identity(2, 3);
        assert_eq!(
            kf.predict_with_state(&x, &a_bad, &i2),
            Err(KalmanError::DimensionMismatch)
        );
        let q_bad = DMatrix::<f64>::zeros(2, 3);
        assert_eq!(
            kf.predict_with_state(&x, &i2, &q_bad),
            Err(KalmanError::DimensionMismatch)
        );
    }

    #[test]
    fn update_rejects_dimension_mismatch_and_non_pd() {
        let mut kf = KalmanFilter::new();
        let x = dm(2, 1, &[1.0, 2.0]);
        let i2 = DMatrix::<f64>::identity(2, 2);
        kf.init_state(&x, &i2).unwrap();

        // C with the wrong column count.
        let c_bad = DMatrix::<f64>::identity(2, 3);
        let y = dm(2, 1, &[1.0, 2.0]);
        assert_eq!(
            kf.update_with_model(&y, &c_bad, &i2),
            Err(KalmanError::DimensionMismatch)
        );

        // Negative-definite R makes S non-PD: the LLT must fail and reject the update.
        let r_neg = dm(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
        let before = kf.get_x().clone();
        assert_eq!(
            kf.update_with_model(&y, &i2, &r_neg),
            Err(KalmanError::NotPositiveDefinite)
        );
        assert_eq!(kf.get_x(), &before);
    }

    #[test]
    fn x_element_bounds() {
        let mut kf = KalmanFilter::new();
        let x = dm(2, 1, &[1.0, 2.0]);
        let i2 = DMatrix::<f64>::identity(2, 2);
        kf.init_state(&x, &i2).unwrap();
        assert_eq!(kf.x_element(0), Ok(1.0));
        assert_eq!(kf.x_element(1), Ok(2.0));
        assert_eq!(kf.x_element(2), Err(KalmanError::IndexOutOfRange));
    }
}
