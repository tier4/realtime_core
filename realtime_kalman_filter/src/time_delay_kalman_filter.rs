//! Port of `autoware_kalman_filter/src/time_delay_kalman_filter.cpp` — the delay-augmented
//! Kalman filter used by the EKF localizer. The extended state stacks `max_delay_step` copies
//! of the `dim_x`-dimensional state; `predict_with_delay` shifts the stack and updates the
//! covariance with the sparse time-delay `A` structure, and `update_with_delay` applies a
//! measurement against the block at `delay_step` (sparse-`C` optimized, LLT-solved gain),
//! keeping the C++ expression order so the float rounding tracks Eigen.
//!
//! # Real-time path
//!
//! `predict_with_delay` and `update_with_delay` are the RT-critical per-event path. They run
//! **allocation-free in steady state**: all extended-dimension temporaries live in a scratch
//! pool sized at [`TimeDelayKalmanFilter::init`], measurement-dimension buffers are pooled
//! per dimension on first use, products run in place (`gemm`/`transpose_to` into scratch —
//! the same nalgebra kernels the owning operators dispatch to, so the rounding is unchanged),
//! and the innovation-covariance Cholesky is a hand-rolled in-place transcription of
//! nalgebra 0.33's `Cholesky::new`/`solve_mut` loops (same axpy/dot order, same
//! zero-or-negative-or-NaN diagonal rejection).

use alloc::vec::Vec;

use nalgebra::DMatrix;

use crate::kalman_filter::{KalmanError, KalmanFilter};

/// Preallocated temporaries for the RT-critical predict/update path. Extended-dimension
/// buffers are sized at `init`; per-measurement-dimension buffers are created on the first
/// update with that dimension (steady state: no allocation).
#[derive(Clone, Debug, Default)]
struct Scratch {
    /// `N×1` spare state stack (fully rewritten each predict, then swapped with the live x).
    x_tmp: DMatrix<f64>,
    /// `N×N` spare covariance (fully rewritten each predict, then swapped with the live P).
    p_tmp: DMatrix<f64>,
    /// `n×n` materialized `Aᵀ`.
    a_t: DMatrix<f64>,
    /// `n×n` `A·P00` intermediate.
    ap: DMatrix<f64>,
    /// `N×N` fully-accumulated `P_CT·Kᵀ` product: the covariance downdate must subtract the
    /// complete product once (as the previous owning `-=` did), not fold the rank-`m`
    /// accumulation into `P` column-by-column, or the rounding changes.
    pkt: DMatrix<f64>,
    /// Per-measurement-dimension update buffers, indexed by `dim_y`.
    update: Vec<Option<UpdateScratch>>,
}

/// Update-path temporaries for one measurement dimension `m` (`n` = state dim, `N` = n·steps).
#[derive(Clone, Debug)]
struct UpdateScratch {
    /// `m×1` `C·x_d`.
    cxd: DMatrix<f64>,
    /// `m×1` innovation.
    e: DMatrix<f64>,
    /// `n×m` materialized `Cᵀ`.
    c_t: DMatrix<f64>,
    /// `m×n` `C·P_dd` intermediate.
    cp_dd: DMatrix<f64>,
    /// `m×m` innovation covariance, factored in place (lower triangle = L after the LLT).
    s_mat: DMatrix<f64>,
    /// `N×m` Kalman-gain numerator `P_*d·Cᵀ`.
    p_ct: DMatrix<f64>,
    /// `m×N` `Kᵀ` (the LLT solve output).
    k_t: DMatrix<f64>,
    /// `N×m` Kalman gain.
    k: DMatrix<f64>,
    /// `N×1` `K·e`.
    ke: DMatrix<f64>,
}

impl UpdateScratch {
    fn new(dim_y: usize, dim_x: usize, dim_x_ex: usize) -> Self {
        Self {
            cxd: DMatrix::zeros(dim_y, 1),
            e: DMatrix::zeros(dim_y, 1),
            c_t: DMatrix::zeros(dim_x, dim_y),
            cp_dd: DMatrix::zeros(dim_y, dim_x),
            s_mat: DMatrix::zeros(dim_y, dim_y),
            p_ct: DMatrix::zeros(dim_x_ex, dim_y),
            k_t: DMatrix::zeros(dim_y, dim_x_ex),
            k: DMatrix::zeros(dim_x_ex, dim_y),
            ke: DMatrix::zeros(dim_x_ex, 1),
        }
    }
}

/// In-place LLT factorization of the lower triangle of `l` — a transcription of nalgebra
/// 0.33's `Cholesky::new_internal` column loop (axpy update order, then sqrt of the
/// diagonal), so the factor's rounding is bit-identical to `Cholesky::new`.
///
/// WCET contract: no allocation; no panic (indices bounded by `nrows()` by construction);
/// loops bounded by `m²·m` for the `m×m` input (m = measurement dimension, 2/3 in the EKF);
/// fails closed with [`KalmanError::NotPositiveDefinite`] on a zero/negative/NaN diagonal
/// (the same predicate as nalgebra's `is_zero`/`try_sqrt` pair).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "f64 math; indices bounded by nrows() by construction"
)]
fn llt_factor_in_place(l: &mut DMatrix<f64>) -> Result<(), KalmanError> {
    let n = l.nrows();
    for j in 0..n {
        for k in 0..j {
            // col_j[j..] += (-l[(j,k)]) * col_k[j..]  (nalgebra's axpy with b = 1).
            let factor = -l[(j, k)];
            for row in j..n {
                l[(row, j)] += factor * l[(row, k)];
            }
        }
        let diag = l[(j, j)];
        // nalgebra: `is_zero()` fails, `try_sqrt` fails for negative and NaN — i.e. anything
        // for which `diag > 0.0` is false.
        if diag <= 0.0 || diag.is_nan() {
            return Err(KalmanError::NotPositiveDefinite);
        }
        let denom = nalgebra::ComplexField::sqrt(diag);
        l[(j, j)] = denom;
        for row in (j + 1)..n {
            l[(row, j)] /= denom;
        }
    }
    Ok(())
}

/// In-place solve of `L·Lᵀ·X = B` for every column of `b`, given the factor produced by
/// [`llt_factor_in_place`] — a transcription of nalgebra's `Cholesky::solve_mut`
/// (forward substitution with column axpy, then the reverse dot-product back substitution),
/// preserving its floating-point operation order.
///
/// WCET contract: no allocation; no panic (indices bounded by the factor/RHS dimensions by
/// construction); work bounded by `ncols(b)·m²`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "f64 math; indices bounded by nrows()/ncols() by construction"
)]
fn llt_solve_in_place(l: &DMatrix<f64>, b: &mut DMatrix<f64>) {
    let n = l.nrows();
    for col in 0..b.ncols() {
        // Forward: L z = b  (solve_lower_triangular_vector_unchecked_mut).
        for i in 0..n {
            let coeff = b[(i, col)] / l[(i, i)];
            b[(i, col)] = coeff;
            for row in (i + 1)..n {
                b[(row, col)] += -coeff * l[(row, i)];
            }
        }
        // Backward: Lᵀ x = z  (xx_solve_lower_triangular_vector_unchecked_mut).
        for i in (0..n).rev() {
            let mut dot = 0.0;
            for row in (i + 1)..n {
                dot += l[(row, i)] * b[(row, col)];
            }
            b[(i, col)] = (b[(i, col)] - dot) / l[(i, i)];
        }
    }
}

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
    scratch: Scratch,
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

        // Preallocate the RT-path scratch (the only allocations besides the state itself);
        // per-measurement-dimension update buffers are pooled lazily on first use.
        self.scratch.x_tmp = DMatrix::zeros(dim_x_ex, 1);
        self.scratch.p_tmp = DMatrix::zeros(dim_x_ex, dim_x_ex);
        self.scratch.a_t = DMatrix::zeros(dim_x, dim_x);
        self.scratch.ap = DMatrix::zeros(dim_x, dim_x);
        self.scratch.pkt = DMatrix::zeros(dim_x_ex, dim_x_ex);
        self.scratch.update.clear();
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
    /// `x_next` on top, and update `P` with the sparse time-delay `A` structure
    /// (`P00 = A P00 Aᵀ + Q`, `P0j = A P0j`, `Pi0 = Pi0 Aᵀ`, older blocks shifted).
    ///
    /// WCET contract (RT-critical): no heap allocation (all temporaries preallocated at
    /// `init`); no blocking/logging; no panic for validated inputs; work bounded by the
    /// `N×N` extended covariance (`N = dim_x·max_delay_step`, fixed at `init`).
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
        if self.scratch.x_tmp.nrows() != dim_x_ex || self.scratch.p_tmp.nrows() != dim_x_ex {
            return Err(KalmanError::EmptyMatrix); // init() not run for these dimensions
        }
        let Self { base, scratch, .. } = self;

        // Slide states in the time direction: rewrite the spare buffer fully, then swap
        // (allocation-free; the retired buffer becomes next cycle's spare).
        scratch.x_tmp.view_mut((0, 0), (dim_x, 1)).copy_from(x_next);
        scratch
            .x_tmp
            .view_mut((dim_x, 0), (d_dim_x, 1))
            .copy_from(&base.x.view((0, 0), (d_dim_x, 1)));
        core::mem::swap(&mut base.x, &mut scratch.x_tmp);

        // Update P with the delayed-measurement A-matrix structure. Every block of p_tmp is
        // written below, so no zeroing is needed; the in-place gemm/transpose_to calls
        // dispatch to the same kernels as the previous owning `*`/`.transpose()` chain, with
        // identical accumulation order.
        a.transpose_to(&mut scratch.a_t);
        scratch
            .ap
            .gemm(1.0, a, &base.p.view((0, 0), (dim_x, dim_x)), 0.0);
        {
            // Top-left: A·P00·Aᵀ + Q.
            let mut tl = scratch.p_tmp.view_mut((0, 0), (dim_x, dim_x));
            tl.gemm(1.0, &scratch.ap, &scratch.a_t, 0.0);
            tl += q;
        }
        scratch.p_tmp.view_mut((0, dim_x), (dim_x, d_dim_x)).gemm(
            1.0,
            a,
            &base.p.view((0, 0), (dim_x, d_dim_x)),
            0.0,
        );
        scratch.p_tmp.view_mut((dim_x, 0), (d_dim_x, dim_x)).gemm(
            1.0,
            &base.p.view((0, 0), (d_dim_x, dim_x)),
            &scratch.a_t,
            0.0,
        );
        scratch
            .p_tmp
            .view_mut((dim_x, dim_x), (d_dim_x, d_dim_x))
            .copy_from(&base.p.view((0, 0), (d_dim_x, d_dim_x)));
        core::mem::swap(&mut base.p, &mut scratch.p_tmp);

        Ok(())
    }

    /// Measurement update against the state block delayed by `delay_step`.
    ///
    /// WCET contract (RT-critical): allocation-free in steady state (per-measurement-
    /// dimension scratch is created on the first update with that dimension, never after);
    /// no blocking/logging; no panic for validated inputs; work bounded by `N·m²` + one
    /// `N×m·m×N` product (`N` fixed at `init`, `m = y.nrows()`).
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
        let dim_x_ex = self.dim_x_ex;
        let dim_y = c.nrows();

        // Fetch (or lazily create — first sighting of this measurement dimension only) the
        // update scratch pool; steady state performs no allocation.
        if self.scratch.update.len() <= dim_y {
            self.scratch
                .update
                .resize(dim_y.saturating_add(1), Option::<UpdateScratch>::None);
        }
        let Self { base, scratch, .. } = self;
        let us = scratch
            .update
            .get_mut(dim_y)
            .ok_or(KalmanError::DimensionMismatch)?
            .get_or_insert_with(|| UpdateScratch::new(dim_y, dim_x, dim_x_ex));

        // Innovation e = y - C x_d (sparse-C_ex optimization: only the delayed block).
        us.cxd
            .gemm(1.0, c, &base.x.view((start_idx, 0), (dim_x, 1)), 0.0);
        us.e.copy_from(y);
        us.e -= &us.cxd;

        // Innovation covariance S = C P_dd Cᵀ + R (P_dd: diagonal block at the delay).
        c.transpose_to(&mut us.c_t);
        us.cp_dd.gemm(
            1.0,
            c,
            &base.p.view((start_idx, start_idx), (dim_x, dim_x)),
            0.0,
        );
        us.s_mat.gemm(1.0, &us.cp_dd, &us.c_t, 0.0);
        us.s_mat += r;

        // Kalman gain numerator P_CT = P_*d Cᵀ (column block of P at the delay).
        us.p_ct
            .gemm(1.0, &base.p.columns(start_idx, dim_x), &us.c_t, 0.0);

        // K = P_CT S⁻¹ via LLT: factor S in place, solve S Kᵀ = P_CTᵀ in place.
        llt_factor_in_place(&mut us.s_mat)?;
        us.p_ct.transpose_to(&mut us.k_t);
        llt_solve_in_place(&us.s_mat, &mut us.k_t);
        us.k_t.transpose_to(&mut us.k);

        if us.k.iter().any(|v| !v.is_finite()) {
            return Err(KalmanError::NonFiniteGain);
        }

        // Update state and covariance. The P downdate accumulates the full P_CT·Kᵀ product
        // into scratch first and subtracts it once — bit-equal to the previous owning `-=`
        // (a beta = 1 gemm would instead fold the rank-m terms into P one axpy at a time,
        // changing the rounding).
        us.ke.gemm(1.0, &us.k, &us.e, 0.0);
        base.x += &us.ke;
        scratch.pkt.gemm(1.0, &us.p_ct, &us.k_t, 0.0);
        base.p -= &scratch.pkt;

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
