# realtime_kalman_filter

`realtime_kalman_filter` is the ROS-free Rust counterpart of the C++ `autoware_kalman_filter`
package (`common/autoware_kalman_filter`). It provides two filters on dynamically-sized matrices:
[`KalmanFilter`](src/kalman_filter.rs), the 1:1 port of `kalman_filter.cpp` (LLT-solved gain), and
[`TimeDelayKalmanFilter`](src/time_delay_kalman_filter.rs), the port of
`time_delay_kalman_filter.cpp` — the delay-augmented state the EKF localizer runs on. It is used by
`realtime_ekf_localizer` and shared through the `realtime_core` workspace.

The crate contains no ROS message types, `rclcpp`, C ABI, or `extern "C"` interface. Those
integration layers belong to the consuming node crate.

## Design goals

- **1:1 numeric parity with the C++ Eigen package.** Matrix math is nalgebra `DMatrix`, mirroring
  `Eigen::MatrixXd`; the golden-vector test pins the Rust filter against a frozen C++ dump within
  the port-equivalence contract tolerance (`rel_tol = 1e-9`).
- **Extended state and delay compensation.** `TimeDelayKalmanFilter` holds the
  `DIM_X × MAX_DELAY_STEP` augmented state and fuses measurements at an arbitrary delay step.
- **Checked failures, no hidden panics.** Every C++ `return false` guard (dimension mismatch,
  non-invertible innovation) becomes a typed [`KalmanError`](src/kalman_filter.rs) returned to the
  caller.
- `no_std` + `alloc`: the only dependency is `nalgebra` (with `libm` for the no_std Cholesky path).

## Usage

Initialize a delay filter, predict, and fuse a delayed measurement (from the crate's
`golden_time_delay` test):

```rust
use nalgebra::DMatrix;
use realtime_kalman_filter::TimeDelayKalmanFilter;

let dim_x = 3;
let max_delay_step = 5;

let x0 = DMatrix::from_column_slice(dim_x, 1, &[1.0, 2.0, 3.0]);
let p0 = DMatrix::<f64>::identity(dim_x, dim_x) * 0.1;
let mut kf = TimeDelayKalmanFilter::new();
kf.init(&x0, &p0, max_delay_step)?;

// Predict one step with state-transition A and process noise Q.
let a = DMatrix::<f64>::identity(dim_x, dim_x) * 2.0;
let q = DMatrix::<f64>::identity(dim_x, dim_x) * 0.01;
let x_next = DMatrix::from_column_slice(dim_x, 1, &[2.0, 4.0, 6.0]);
kf.predict_with_delay(&x_next, &a, &q)?;

// Fuse a measurement delayed by 2 steps into the extended state.
let c = DMatrix::<f64>::identity(dim_x, dim_x) * 0.5;
let r = DMatrix::<f64>::identity(dim_x, dim_x) * 0.001;
let y = DMatrix::from_column_slice(dim_x, 1, &[1.05, 2.05, 3.05]);
kf.update_with_delay(&y, &c, &r, 2)?;

let latest = kf.get_latest_x()?;
# Ok::<(), realtime_kalman_filter::KalmanError>(())
```

## Build and validation

The crate has no Cargo features. Run the normal host checks from this directory:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`tests/golden_time_delay.rs` is the differential golden-vector gate against the frozen C++ dump.

## Workspace

Part of the [`realtime_core`](../) workspace. The workspace documentation book is published at
<https://tier4.github.io/realtime_core/>.

Licensed under either of the Apache License, Version 2.0 or the MIT license, at your option
(`MIT OR Apache-2.0`).
