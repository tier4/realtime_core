//! Port of `autoware_kalman_filter/src/time_delay_kalman_filter.cpp` — the delay-augmented
//! Kalman filter used by the EKF localizer. The extended state stacks `max_delay_step` copies
//! of the `dim_x`-dimensional state; `predict_with_delay` shifts the stack and updates the
//! covariance with the sparse time-delay `A` structure, and `update_with_delay` applies a
//! measurement against the block at `delay_step` (sparse-`C` optimized, LLT-solved gain),
//! keeping the C++ expression order so the float rounding tracks Eigen.

use nalgebra::{Cholesky, DMatrix, Dyn};

use crate::kalman_filter::{KalmanError, KalmanFilter};

/// Kalman filter with delayed-measurement support (port of the C++ `TimeDelayKalmanFilter`).
///
/// The C++ class inherits `KalmanFilter` but only uses its `x_`/`P_` storage; the Rust port
/// composes the base filter for the same effect.
#[derive(Clone, Debug, Default)]
pub struct TimeDelayKalmanFilter {
    base: KalmanFilter,
    max_delay_step: usize,
    dim_x: usize,
    dim_x_ex: usize,
}

impl TimeDelayKalmanFilter {
    /// No-initialization constructor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the extended state: every delay block is a copy of `x` and the block
    /// diagonal of `P` is `p0` (cross-block covariance zero).
    ///
    /// Errors when `x` is not a non-empty column vector, `p0` is not square of the same
    /// dimension, or `max_delay_step` is zero (the C++ silently builds a 0-dimensional
    /// filter in these cases; the port rejects them instead).
    ///
    /// # Errors
    /// [`KalmanError::InvalidDelayStep`] for `max_delay_step == 0`;
    /// [`KalmanError::DimensionMismatch`] for malformed `x`/`p0`.
    pub fn init(
        &mut self,
        x: &DMatrix<f64>,
        p0: &DMatrix<f64>,
        max_delay_step: usize,
    ) -> Result<(), KalmanError> {
        if max_delay_step == 0 {
            return Err(KalmanError::InvalidDelayStep);
        }
        let dim_x = x.nrows();
        if dim_x == 0 || x.ncols() != 1 || p0.nrows() != dim_x || p0.ncols() != dim_x {
            return Err(KalmanError::DimensionMismatch);
        }
        let dim_x_ex = dim_x
            .checked_mul(max_delay_step)
            .ok_or(KalmanError::DimensionMismatch)?;

        self.max_delay_step = max_delay_step;
        self.dim_x = dim_x;
        self.dim_x_ex = dim_x_ex;

        let mut x_ex = DMatrix::<f64>::zeros(dim_x_ex, 1);
        let mut p_ex = DMatrix::<f64>::zeros(dim_x_ex, dim_x_ex);
        for i in 0..max_delay_step {
            let offset = i.checked_mul(dim_x).ok_or(KalmanError::DimensionMismatch)?;
            x_ex.view_mut((offset, 0), (dim_x, 1)).copy_from(x);
            p_ex.view_mut((offset, offset), (dim_x, dim_x))
                .copy_from(p0);
        }
        self.base.x = x_ex;
        self.base.p = p_ex;
        Ok(())
    }

    /// Latest-time state block (C++ `getLatestX`).
    ///
    /// # Errors
    /// [`KalmanError::EmptyMatrix`] before a successful [`TimeDelayKalmanFilter::init`].
    pub fn get_latest_x(&self) -> Result<DMatrix<f64>, KalmanError> {
        if self.dim_x == 0 || self.base.x.nrows() < self.dim_x {
            return Err(KalmanError::EmptyMatrix);
        }
        Ok(self.base.x.view((0, 0), (self.dim_x, 1)).into_owned())
    }

    /// Latest-time covariance block (C++ `getLatestP`).
    ///
    /// # Errors
    /// [`KalmanError::EmptyMatrix`] before a successful [`TimeDelayKalmanFilter::init`].
    pub fn get_latest_p(&self) -> Result<DMatrix<f64>, KalmanError> {
        if self.dim_x == 0 || self.base.p.nrows() < self.dim_x || self.base.p.ncols() < self.dim_x {
            return Err(KalmanError::EmptyMatrix);
        }
        Ok(self
            .base
            .p
            .view((0, 0), (self.dim_x, self.dim_x))
            .into_owned())
    }

    /// Element `i` of the extended state (C++ `getXelement` on the base class).
    ///
    /// # Errors
    /// [`KalmanError::IndexOutOfRange`] when `i` is outside the extended state.
    pub fn x_element(&self, i: usize) -> Result<f64, KalmanError> {
        self.base.x_element(i)
    }

    /// Full extended state (C++ public base-class `getX`).
    #[must_use]
    pub fn get_x_ex(&self) -> &DMatrix<f64> {
        &self.base.x
    }

    /// Full extended covariance (C++ public base-class `getP`).
    #[must_use]
    pub fn get_p_ex(&self) -> &DMatrix<f64> {
        &self.base.p
    }

    /// Prediction with the time-delay model: shift the state stack down one block, place
    /// `x_next` on top, and update `P` with the sparse block structure
    /// (`P00 = A P00 Aᵀ + Q`, `P0j = A P0j`, `Pi0 = Pi0 Aᵀ`, older blocks shifted).
    ///
    /// # Errors
    /// [`KalmanError::DimensionMismatch`] for malformed inputs (where the C++ has no guard
    /// and Eigen would assert).
    #[expect(clippy::arithmetic_side_effects, reason = "nalgebra f64 matrix math")]
    pub fn predict_with_delay(
        &mut self,
        x_next: &DMatrix<f64>,
        a: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) -> Result<(), KalmanError> {
        let dim_x = self.dim_x;
        let dim_x_ex = self.dim_x_ex;
        // The C++ performs no dimension validation here (Eigen would assert); the port
        // rejects malformed inputs instead of proceeding into out-of-bounds block reads.
        if dim_x == 0
            || x_next.nrows() != dim_x
            || x_next.ncols() != 1
            || a.nrows() != dim_x
            || a.ncols() != dim_x
            || q.nrows() != dim_x
            || q.ncols() != dim_x
        {
            return Err(KalmanError::DimensionMismatch);
        }
        let d_dim_x = dim_x_ex
            .checked_sub(dim_x)
            .ok_or(KalmanError::DimensionMismatch)?;

        // Slide states in the time direction.
        let mut x_tmp = DMatrix::<f64>::zeros(dim_x_ex, 1);
        x_tmp.view_mut((0, 0), (dim_x, 1)).copy_from(x_next);
        x_tmp
            .view_mut((dim_x, 0), (d_dim_x, 1))
            .copy_from(&self.base.x.view((0, 0), (d_dim_x, 1)));
        self.base.x = x_tmp;

        // Update P with the delayed-measurement A-matrix structure.
        let p = &self.base.p;
        let mut p_tmp = DMatrix::<f64>::zeros(dim_x_ex, dim_x_ex);
        p_tmp
            .view_mut((0, 0), (dim_x, dim_x))
            .copy_from(&(a * p.view((0, 0), (dim_x, dim_x)) * a.transpose() + q));
        p_tmp
            .view_mut((0, dim_x), (dim_x, d_dim_x))
            .copy_from(&(a * p.view((0, 0), (dim_x, d_dim_x))));
        p_tmp
            .view_mut((dim_x, 0), (d_dim_x, dim_x))
            .copy_from(&(p.view((0, 0), (d_dim_x, dim_x)) * a.transpose()));
        p_tmp
            .view_mut((dim_x, dim_x), (d_dim_x, d_dim_x))
            .copy_from(&p.view((0, 0), (d_dim_x, d_dim_x)));
        self.base.p = p_tmp;

        Ok(())
    }

    /// Measurement update against the state block delayed by `delay_step`.
    ///
    /// Rejections (typed, matching the C++ `return false` guards in order): invalid delay
    /// step, `C` column mismatch, `y`/`C` row mismatch, non-column `y`, non-square `R`,
    /// `R`/`C` row mismatch, non-positive-definite innovation covariance, non-finite gain.
    ///
    /// # Errors
    /// [`KalmanError::InvalidDelayStep`], [`KalmanError::DimensionMismatch`],
    /// [`KalmanError::NotPositiveDefinite`], or [`KalmanError::NonFiniteGain`], matching the
    /// C++ guard order.
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::many_single_char_names,
        clippy::allow_attributes,
        reason = "nalgebra f64 matrix math; C++ kernel naming (y, C, R, S, K)"
    )]
    pub fn update_with_delay(
        &mut self,
        y: &DMatrix<f64>,
        c: &DMatrix<f64>,
        r: &DMatrix<f64>,
        delay_step: usize,
    ) -> Result<(), KalmanError> {
        // The C++ takes a signed int and rejects `delay_step < 0`; `usize` makes the negative
        // case unrepresentable, so only the upper bound remains.
        if delay_step >= self.max_delay_step {
            return Err(KalmanError::InvalidDelayStep);
        }
        let dim_x = self.dim_x;
        if c.ncols() != dim_x {
            return Err(KalmanError::DimensionMismatch);
        }
        if y.nrows() != c.nrows() {
            return Err(KalmanError::DimensionMismatch);
        }
        if y.ncols() != 1 {
            return Err(KalmanError::DimensionMismatch);
        }
        if r.nrows() != r.ncols() {
            return Err(KalmanError::DimensionMismatch);
        }
        if r.nrows() != c.nrows() {
            return Err(KalmanError::DimensionMismatch);
        }

        let start_idx = dim_x
            .checked_mul(delay_step)
            .ok_or(KalmanError::DimensionMismatch)?;

        // Innovation e = y - C x_d (sparse-C_ex optimization: only the delayed block).
        let x_d = self.base.x.view((start_idx, 0), (dim_x, 1));
        let e = y - c * x_d;

        // Innovation covariance S = C P_dd Cᵀ + R (P_dd: diagonal block at the delay).
        let p_dd = self.base.p.view((start_idx, start_idx), (dim_x, dim_x));
        let s = r + c * p_dd * c.transpose();

        // Kalman gain numerator P_CT = P_*d Cᵀ (column block of P at the delay).
        let p_star_d = self.base.p.columns(start_idx, dim_x);
        let p_ct = p_star_d * c.transpose();

        // K = P_CT S⁻¹ via LLT: solve S Kᵀ = P_CTᵀ, transpose back.
        let llt: Cholesky<f64, Dyn> = Cholesky::new(s).ok_or(KalmanError::NotPositiveDefinite)?;
        let k_transposed = llt.solve(&p_ct.transpose());
        let k = k_transposed.transpose();

        if k.iter().any(|v| !v.is_finite()) {
            return Err(KalmanError::NonFiniteGain);
        }

        // Update state and covariance.
        self.base.x += &k * &e;
        self.base.p -= &p_ct * k.transpose();

        Ok(())
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
    use alloc::vec::Vec;

    // Test constants from the C++ test_time_delay_kalman_filter.cpp.
    const DIM_X: usize = 3;
    const MAX_DELAY_STEP: usize = 5;
    const DIM_X_EX: usize = DIM_X * MAX_DELAY_STEP;
    const EPS: f64 = 1e-5;
    const INITIAL_COV: f64 = 0.1;
    const PROCESS_NOISE: f64 = 0.01;
    const MEASUREMENT_NOISE: f64 = 0.001;
    const STATE_TRANSITION_SCALE: f64 = 2.0;
    const OBSERVATION_SCALE: f64 = 0.5;

    struct Fixture {
        kf: TimeDelayKalmanFilter,
        x_ex_gt: DMatrix<f64>,
        p_ex_gt: DMatrix<f64>,
        a: DMatrix<f64>,
        q: DMatrix<f64>,
        c: DMatrix<f64>,
        r: DMatrix<f64>,
        x_t: DMatrix<f64>,
        x_next: DMatrix<f64>,
    }

    fn fixture() -> Fixture {
        let x_t = DMatrix::from_column_slice(DIM_X, 1, &[1.0, 2.0, 3.0]);
        let p_t = DMatrix::<f64>::identity(DIM_X, DIM_X) * INITIAL_COV;

        let mut x_ex_gt = DMatrix::<f64>::zeros(DIM_X_EX, 1);
        let mut p_ex_gt = DMatrix::<f64>::zeros(DIM_X_EX, DIM_X_EX);
        for i in 0..MAX_DELAY_STEP {
            x_ex_gt.view_mut((i * DIM_X, 0), (DIM_X, 1)).copy_from(&x_t);
            p_ex_gt
                .view_mut((i * DIM_X, i * DIM_X), (DIM_X, DIM_X))
                .copy_from(&p_t);
        }

        let mut kf = TimeDelayKalmanFilter::new();
        kf.init(&x_t, &p_t, MAX_DELAY_STEP).unwrap();

        Fixture {
            kf,
            x_ex_gt,
            p_ex_gt,
            a: DMatrix::<f64>::identity(DIM_X, DIM_X) * STATE_TRANSITION_SCALE,
            q: DMatrix::<f64>::identity(DIM_X, DIM_X) * PROCESS_NOISE,
            c: DMatrix::<f64>::identity(DIM_X, DIM_X) * OBSERVATION_SCALE,
            r: DMatrix::<f64>::identity(DIM_X, DIM_X) * MEASUREMENT_NOISE,
            x_t,
            x_next: DMatrix::from_column_slice(DIM_X, 1, &[2.0, 4.0, 6.0]),
        }
    }

    /// Ground-truth predict on the naive extended representation (mirrors the C++ helper).
    fn ground_truth_predict(
        x_ex: &mut DMatrix<f64>,
        p_ex: &mut DMatrix<f64>,
        x_next: &DMatrix<f64>,
        a: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) {
        let d = DIM_X_EX - DIM_X;
        let mut x_shifted = DMatrix::<f64>::zeros(DIM_X_EX, 1);
        x_shifted
            .view_mut((DIM_X, 0), (d, 1))
            .copy_from(&x_ex.view((0, 0), (d, 1)));
        x_shifted.view_mut((0, 0), (DIM_X, 1)).copy_from(x_next);
        *x_ex = x_shifted;

        let mut p_tmp = DMatrix::<f64>::zeros(DIM_X_EX, DIM_X_EX);
        p_tmp
            .view_mut((0, 0), (DIM_X, DIM_X))
            .copy_from(&(a * p_ex.view((0, 0), (DIM_X, DIM_X)) * a.transpose() + q));
        p_tmp
            .view_mut((0, DIM_X), (DIM_X, d))
            .copy_from(&(a * p_ex.view((0, 0), (DIM_X, d))));
        p_tmp
            .view_mut((DIM_X, 0), (d, DIM_X))
            .copy_from(&(p_ex.view((0, 0), (d, DIM_X)) * a.transpose()));
        p_tmp
            .view_mut((DIM_X, DIM_X), (d, d))
            .copy_from(&p_ex.view((0, 0), (d, d)));
        *p_ex = p_tmp;
    }

    /// Ground-truth update via the dense extended-C formulation (mirrors the C++ helper).
    fn ground_truth_update(
        x_ex: &mut DMatrix<f64>,
        p_ex: &mut DMatrix<f64>,
        y: &DMatrix<f64>,
        c: &DMatrix<f64>,
        r: &DMatrix<f64>,
        delay_step: usize,
    ) {
        let dim_y = y.nrows();
        let mut c_ex = DMatrix::<f64>::zeros(dim_y, DIM_X_EX);
        c_ex.view_mut((0, delay_step * DIM_X), (dim_y, DIM_X))
            .copy_from(c);

        let pct = &*p_ex * c_ex.transpose();
        let k = &pct * (r + &c_ex * &pct).try_inverse().unwrap();
        let y_pred = &c_ex * &*x_ex;

        *x_ex += &k * (y - y_pred);
        *p_ex -= &k * (&c_ex * &*p_ex);
    }

    fn assert_latest_matches(kf: &TimeDelayKalmanFilter, x_gt: &DMatrix<f64>, p_gt: &DMatrix<f64>) {
        let x = kf.get_latest_x().unwrap();
        let p = kf.get_latest_p().unwrap();
        let x_expected = x_gt.view((0, 0), (DIM_X, 1)).into_owned();
        let p_expected = p_gt.view((0, 0), (DIM_X, DIM_X)).into_owned();
        assert!(
            (x - &x_expected).norm() <= EPS * x_expected.norm().max(1.0),
            "latest x mismatch"
        );
        assert!(
            (p - &p_expected).norm() <= EPS * p_expected.norm().max(1.0),
            "latest P mismatch"
        );
    }

    #[test]
    fn initialization() {
        let f = fixture();
        let x = f.kf.get_latest_x().unwrap();
        let p = f.kf.get_latest_p().unwrap();
        assert!((x - &f.x_t).norm() < EPS);
        let p_t = DMatrix::<f64>::identity(DIM_X, DIM_X) * INITIAL_COV;
        assert!((p - p_t).norm() < EPS);
    }

    #[test]
    fn prediction_matches_ground_truth() {
        let mut f = fixture();
        ground_truth_predict(&mut f.x_ex_gt, &mut f.p_ex_gt, &f.x_next, &f.a, &f.q);
        f.kf.predict_with_delay(&f.x_next, &f.a, &f.q).unwrap();
        assert_latest_matches(&f.kf, &f.x_ex_gt, &f.p_ex_gt);
    }

    #[test]
    fn update_with_delay_matches_ground_truth() {
        for delay_step in [0usize, 2, MAX_DELAY_STEP - 1] {
            let mut f = fixture();
            ground_truth_predict(&mut f.x_ex_gt, &mut f.p_ex_gt, &f.x_next, &f.a, &f.q);
            f.kf.predict_with_delay(&f.x_next, &f.a, &f.q).unwrap();

            let y = DMatrix::from_column_slice(DIM_X, 1, &[1.05, 2.05, 3.05]);
            ground_truth_update(&mut f.x_ex_gt, &mut f.p_ex_gt, &y, &f.c, &f.r, delay_step);
            f.kf.update_with_delay(&y, &f.c, &f.r, delay_step).unwrap();

            assert_latest_matches(&f.kf, &f.x_ex_gt, &f.p_ex_gt);
        }
    }

    #[test]
    fn multiple_predictions_before_update() {
        let mut f = fixture();
        for i in 0..3usize {
            let s = (i + 1) as f64;
            let x_pred = DMatrix::from_column_slice(DIM_X, 1, &[2.0 * s, 4.0 * s, 6.0 * s]);
            ground_truth_predict(&mut f.x_ex_gt, &mut f.p_ex_gt, &x_pred, &f.a, &f.q);
            f.kf.predict_with_delay(&x_pred, &f.a, &f.q).unwrap();
        }
        let y = DMatrix::from_column_slice(DIM_X, 1, &[1.0, 2.0, 3.0]);
        ground_truth_update(&mut f.x_ex_gt, &mut f.p_ex_gt, &y, &f.c, &f.r, 2);
        f.kf.update_with_delay(&y, &f.c, &f.r, 2).unwrap();
        assert_latest_matches(&f.kf, &f.x_ex_gt, &f.p_ex_gt);
    }

    #[test]
    fn zero_initial_state() {
        let x_zero = DMatrix::<f64>::zeros(DIM_X, 1);
        let p_zero = DMatrix::<f64>::identity(DIM_X, DIM_X) * INITIAL_COV;
        let mut kf = TimeDelayKalmanFilter::new();
        kf.init(&x_zero, &p_zero, MAX_DELAY_STEP).unwrap();
        assert!(kf.get_latest_x().unwrap().norm() < EPS);
        assert!((kf.get_latest_p().unwrap() - p_zero).norm() < EPS);
    }

    #[test]
    fn update_rejections() {
        let f = fixture();
        let mut kf = f.kf;
        let y = DMatrix::from_column_slice(DIM_X, 1, &[1.0, 2.0, 3.0]);

        // delay_step == max_delay_step (out of range).
        assert_eq!(
            kf.update_with_delay(&y, &f.c, &f.r, MAX_DELAY_STEP),
            Err(KalmanError::InvalidDelayStep)
        );

        // C with the wrong number of columns.
        let c_wrong = DMatrix::<f64>::identity(DIM_X, DIM_X + 1);
        assert_eq!(
            kf.update_with_delay(&y, &c_wrong, &f.r, 0),
            Err(KalmanError::DimensionMismatch)
        );

        // Non-column y.
        let y_not_column = DMatrix::<f64>::zeros(DIM_X, 2);
        assert_eq!(
            kf.update_with_delay(&y_not_column, &f.c, &f.r, 0),
            Err(KalmanError::DimensionMismatch)
        );

        // Non-square R.
        let r_non_square = DMatrix::<f64>::zeros(DIM_X, DIM_X + 1);
        assert_eq!(
            kf.update_with_delay(&y, &f.c, &r_non_square, 0),
            Err(KalmanError::DimensionMismatch)
        );

        // Square R with mismatched dimension.
        let r_wrong = DMatrix::<f64>::identity(DIM_X + 1, DIM_X + 1);
        assert_eq!(
            kf.update_with_delay(&y, &f.c, &r_wrong, 0),
            Err(KalmanError::DimensionMismatch)
        );

        // Negative-definite R -> S non-PD -> LLT failure.
        let r_negative = DMatrix::<f64>::identity(DIM_X, DIM_X) * -1.0;
        assert_eq!(
            kf.update_with_delay(&y, &f.c, &r_negative, 0),
            Err(KalmanError::NotPositiveDefinite)
        );
    }

    #[test]
    fn x_element_reads_extended_state() {
        let f = fixture();
        let values: Vec<f64> = (0..DIM_X_EX).map(|i| f.kf.x_element(i).unwrap()).collect();
        for i in 0..MAX_DELAY_STEP {
            assert_eq!(values[i * DIM_X], 1.0);
            assert_eq!(values[i * DIM_X + 1], 2.0);
            assert_eq!(values[i * DIM_X + 2], 3.0);
        }
        assert_eq!(f.kf.x_element(DIM_X_EX), Err(KalmanError::IndexOutOfRange));
    }
}
