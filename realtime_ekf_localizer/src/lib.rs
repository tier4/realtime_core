//! Pure-Rust, ROS-free, `no_std`-capable (heap only, via `alloc`) EKF localizer core, the 1:1
//! port of the C++ package `autoware_ekf_localizer` (`localization/autoware_ekf_localizer`).
//! Modules land with Step 1/3 of the port plan; the crate skeleton exists so the numeric spike
//! (Step 2, `realtime_kalman_filter`) builds inside the workspace.

#![no_std]

extern crate alloc;
