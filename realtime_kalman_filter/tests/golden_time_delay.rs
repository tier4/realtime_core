//! Differential golden-vector test: pins the Rust `TimeDelayKalmanFilter` against the frozen
//! C++ (Eigen) dump produced by `porting_notes/ekf_conformance/spike/golden_gen.cpp` from the
//! `test_time_delay_kalman_filter.cpp` input constants. Comparison uses the frozen
//! port-equivalence contract tolerance `rel_tol = 1e-9` (measured spike worst case: 3.61e-15).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::many_single_char_names,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "test code"
)]

use std::collections::HashMap;

use nalgebra::DMatrix;
use realtime_kalman_filter::TimeDelayKalmanFilter;

const DIM_X: usize = 3;
const MAX_DELAY_STEP: usize = 5;
const DIM_X_EX: usize = DIM_X * MAX_DELAY_STEP;
const REL_TOL: f64 = 1e-9;

const GOLDEN: &str = include_str!("data/time_delay_golden_cpp.csv");

type Key = (String, String, usize, usize);

fn parse_golden() -> HashMap<Key, f64> {
    let mut map = HashMap::new();
    for line in GOLDEN.lines() {
        let mut it = line.split(',');
        let case = it.next().unwrap().to_owned();
        let kind = it.next().unwrap().to_owned();
        let i: usize = it.next().unwrap().parse().unwrap();
        let j: usize = it.next().unwrap().parse().unwrap();
        let v: f64 = it.next().unwrap().parse().unwrap();
        map.insert((case, kind, i, j), v);
    }
    map
}

fn assert_case(golden: &HashMap<Key, f64>, case: &str, kf: &TimeDelayKalmanFilter) {
    let mut checked = 0usize;
    for i in 0..DIM_X_EX {
        let got = kf.x_element(i).unwrap();
        let want = golden[&(case.to_owned(), "x".to_owned(), i, 0)];
        assert_rel(case, "x", i, 0, want, got);
        checked += 1;
    }
    let p = kf.get_p_ex();
    for i in 0..DIM_X_EX {
        for j in 0..DIM_X_EX {
            let want = golden[&(case.to_owned(), "P".to_owned(), i, j)];
            assert_rel(case, "P", i, j, want, p[(i, j)]);
            checked += 1;
        }
    }
    assert_eq!(checked, DIM_X_EX + DIM_X_EX * DIM_X_EX);
}

fn assert_rel(case: &str, kind: &str, i: usize, j: usize, want: f64, got: f64) {
    let denom = want.abs().max(got.abs());
    let rel = if denom == 0.0 {
        0.0
    } else {
        (want - got).abs() / denom
    };
    assert!(
        rel <= REL_TOL,
        "{case},{kind},{i},{j}: cpp={want:.17e} rust={got:.17e} rel={rel:.3e}"
    );
}

fn make_initialized() -> TimeDelayKalmanFilter {
    let x_t = DMatrix::from_column_slice(DIM_X, 1, &[1.0, 2.0, 3.0]);
    let p_t = DMatrix::<f64>::identity(DIM_X, DIM_X) * 0.1;
    let mut kf = TimeDelayKalmanFilter::new();
    kf.init(&x_t, &p_t, MAX_DELAY_STEP).unwrap();
    kf
}

#[test]
fn matches_cpp_golden_vectors() {
    let golden = parse_golden();

    let a = DMatrix::<f64>::identity(DIM_X, DIM_X) * 2.0;
    let q = DMatrix::<f64>::identity(DIM_X, DIM_X) * 0.01;
    let c = DMatrix::<f64>::identity(DIM_X, DIM_X) * 0.5;
    let r = DMatrix::<f64>::identity(DIM_X, DIM_X) * 0.001;
    let x_next = DMatrix::from_column_slice(DIM_X, 1, &[2.0, 4.0, 6.0]);

    let kf = make_initialized();
    assert_case(&golden, "init", &kf);

    let mut kf = make_initialized();
    kf.predict_with_delay(&x_next, &a, &q).unwrap();
    assert_case(&golden, "predict1", &kf);

    for delay in [0usize, 2, 4] {
        let mut kf = make_initialized();
        kf.predict_with_delay(&x_next, &a, &q).unwrap();
        let y = DMatrix::from_column_slice(DIM_X, 1, &[1.05, 2.05, 3.05]);
        kf.update_with_delay(&y, &c, &r, delay).unwrap();
        assert_case(&golden, &format!("update_d{delay}"), &kf);
    }

    let mut kf = make_initialized();
    for i in 0..3usize {
        let s = (i + 1) as f64;
        let x_pred = DMatrix::from_column_slice(DIM_X, 1, &[2.0 * s, 4.0 * s, 6.0 * s]);
        kf.predict_with_delay(&x_pred, &a, &q).unwrap();
    }
    let y = DMatrix::from_column_slice(DIM_X, 1, &[1.0, 2.0, 3.0]);
    kf.update_with_delay(&y, &c, &r, 2).unwrap();
    assert_case(&golden, "multi_predict_update", &kf);
}
