//! Pure-Rust, ROS-free, `no_std`-capable (heap only, via `alloc`) Kalman filter, the 1:1 port of
//! the C++ package `autoware_kalman_filter` (`common/autoware_kalman_filter`). [`kalman_filter`]
//! ports `kalman_filter.cpp` (`KalmanFilter` on dynamically-sized matrices, LLT-solved gain);
//! [`time_delay_kalman_filter`] ports `time_delay_kalman_filter.cpp` (`TimeDelayKalmanFilter`,
//! the delay-augmented state used by the EKF localizer). Matrix math is nalgebra `DMatrix`,
//! mirroring the C++ `Eigen::MatrixXd`; every C++ `return false` guard becomes a typed
//! [`KalmanError`](kalman_filter::KalmanError).

#![no_std]

extern crate alloc;

pub mod kalman_filter;
pub mod time_delay_kalman_filter;

pub use kalman_filter::{KalmanError, KalmanFilter};
pub use time_delay_kalman_filter::TimeDelayKalmanFilter;
