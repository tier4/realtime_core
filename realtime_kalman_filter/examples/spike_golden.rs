//! Step-2 numeric-spike driver: runs the Rust `TimeDelayKalmanFilter` through the same inputs
//! as the C++ golden-vector generator (`test_time_delay_kalman_filter.cpp` constants) and dumps
//! the full extended state/covariance as `case,kind,i,j,value` lines at 17 significant digits.
//! Compared against the C++ output to measure the Eigen-vs-nalgebra relative error that fixes
//! the port-equivalence contract tolerance.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    clippy::print_stdout,
    clippy::many_single_char_names,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "test code (spike example)"
)]

use nalgebra::DMatrix;
use realtime_kalman_filter::TimeDelayKalmanFilter;

const DIM_X: usize = 3;
const MAX_DELAY_STEP: usize = 5;
const INITIAL_COV: f64 = 0.1;
const PROCESS_NOISE: f64 = 0.01;
const MEASUREMENT_NOISE: f64 = 0.001;
const STATE_TRANSITION_SCALE: f64 = 2.0;
const OBSERVATION_SCALE: f64 = 0.5;

fn dump(name: &str, kf: &TimeDelayKalmanFilter) {
    let dim_ex = DIM_X * MAX_DELAY_STEP;
    for i in 0..dim_ex {
        println!("{name},x,{i},0,{:.17e}", kf.x_element(i).unwrap());
    }
    let p = kf.get_p_ex();
    for i in 0..dim_ex {
        for j in 0..dim_ex {
            println!("{name},P,{i},{j},{:.17e}", p[(i, j)]);
        }
    }
}

fn make_initialized() -> TimeDelayKalmanFilter {
    let x_t = DMatrix::from_column_slice(DIM_X, 1, &[1.0, 2.0, 3.0]);
    let p_t = DMatrix::<f64>::identity(DIM_X, DIM_X) * INITIAL_COV;
    let mut kf = TimeDelayKalmanFilter::new();
    kf.init(&x_t, &p_t, MAX_DELAY_STEP).unwrap();
    kf
}

fn main() {
    let a = DMatrix::<f64>::identity(DIM_X, DIM_X) * STATE_TRANSITION_SCALE;
    let q = DMatrix::<f64>::identity(DIM_X, DIM_X) * PROCESS_NOISE;
    let c = DMatrix::<f64>::identity(DIM_X, DIM_X) * OBSERVATION_SCALE;
    let r = DMatrix::<f64>::identity(DIM_X, DIM_X) * MEASUREMENT_NOISE;
    let x_next = DMatrix::from_column_slice(DIM_X, 1, &[2.0, 4.0, 6.0]);

    {
        let kf = make_initialized();
        dump("init", &kf);
    }
    {
        let mut kf = make_initialized();
        kf.predict_with_delay(&x_next, &a, &q).unwrap();
        dump("predict1", &kf);
    }
    for delay in [0usize, 2, 4] {
        let mut kf = make_initialized();
        kf.predict_with_delay(&x_next, &a, &q).unwrap();
        let y = DMatrix::from_column_slice(DIM_X, 1, &[1.05, 2.05, 3.05]);
        kf.update_with_delay(&y, &c, &r, delay).unwrap();
        dump(&format!("update_d{delay}"), &kf);
    }
    {
        let mut kf = make_initialized();
        for i in 0..3usize {
            let s = (i + 1) as f64;
            let x_pred = DMatrix::from_column_slice(DIM_X, 1, &[2.0 * s, 4.0 * s, 6.0 * s]);
            kf.predict_with_delay(&x_pred, &a, &q).unwrap();
        }
        let y = DMatrix::from_column_slice(DIM_X, 1, &[1.0, 2.0, 3.0]);
        kf.update_with_delay(&y, &c, &r, 2).unwrap();
        dump("multi_predict_update", &kf);
    }
}
