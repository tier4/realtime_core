# The Kalman filter crate

`realtime_kalman_filter` is the ROS-free Rust counterpart of the C++ `autoware_kalman_filter`
package (`common/autoware_kalman_filter`): `no_std` + `alloc`, matrix math on nalgebra `DMatrix`
mirroring `Eigen::MatrixXd`. It holds two filters:

- `kalman_filter` (`realtime_kalman_filter/src/kalman_filter.rs`) — the `KalmanFilter` port
  (`kalman_filter.cpp`), an LLT-solved gain over dynamically-sized matrices.
- `time_delay_kalman_filter` (`realtime_kalman_filter/src/time_delay_kalman_filter.rs`) — the
  `TimeDelayKalmanFilter` port (`time_delay_kalman_filter.cpp`), the delay-augmented
  `DIM_X × MAX_DELAY_STEP` extended state the EKF localizer runs on, fusing measurements at an
  arbitrary delay step.

Every C++ `return false` guard becomes a typed `KalmanError` returned to the caller (no hidden
panics). Numeric parity is pinned by a differential golden-vector test against a frozen C++ (Eigen)
dump within the port-equivalence contract tolerance (`rel_tol = 1e-9`).

The [EKF localizer crate](ekf-localizer.md) is the primary consumer.
