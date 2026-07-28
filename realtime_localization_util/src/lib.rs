//! Shared, ROS-free, `no_std`-capable (heap only, via `alloc`) localization utilities for the
//! `realtime_core` Rust ports of Autoware Core. The [`pose_buffer`] and [`tpe`] modules are
//! operation-for-operation ports of the C++ `autoware_localization_util` shared library
//! (`SmartPoseBuffer` and `TreeStructuredParzenEstimator`), so this crate is the 1:1 Rust
//! counterpart of that C++ package; [`transform`] carries the SE3 transform and Gaussian fitting
//! kernels shared by the NDT engine. Consumed by `realtime_ndt_scan_matcher` and intended for
//! reuse by further localization ports (EKF localizer, pose initializer).

#![no_std]

extern crate alloc;

pub mod pose_buffer;
pub mod tpe;
pub mod transform;
