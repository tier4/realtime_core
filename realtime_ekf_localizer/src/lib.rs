//! Pure-Rust, ROS-free, `no_std`-capable (heap only, via `alloc`) EKF localizer core, the 1:1
//! port of the C++ package `autoware_ekf_localizer` (`localization/autoware_ekf_localizer`).
//!
//! Module map (C++ file → Rust module):
//! - `state_transition.cpp` (+ `state_index.hpp`) → [`state_transition`]
//! - `mahalanobis.cpp` → [`mahalanobis`]
//! - `measurement.cpp` → [`measurement`]
//! - `covariance.cpp` → [`covariance`]
//! - `numeric.hpp` → [`numeric`]
//! - `aged_object_queue.hpp` → [`aged_object_queue`]
//! - `Simple1DFilter` (`ekf_module.hpp`) → [`simple_1d_filter`]
//! - `hyper_parameters.hpp` → [`hyper_parameters`] (plain struct; no parameter server)
//! - `ekf_module.cpp` → [`ekf_module`] (ROS-free: plain-struct measurements, `i64` ns time)
//! - `EKFLocalizer::timer_callback` control flow → [`tick`] (queue drain + predict orchestration)
//! - tf2 quaternion/Euler helpers the EKF calls → [`tf2_math`] (expression-exact ports)
//! - trace CSV rows (`AUTOWARE_EKF_POSE_TRACE` format + predict/twist extension) → [`trace`]
//!
//! Behavior equivalence with the C++ is governed by the port-equivalence contract
//! (`porting_notes/ekf_port_contract.md` in the workspace repo): decisions (gates, delay
//! steps, event order) match exactly; f64 chains match within `rel_tol = 1e-9`.

#![no_std]

extern crate alloc;

pub mod aged_object_queue;
pub mod covariance;
pub mod ekf_module;
pub mod hyper_parameters;
pub mod mahalanobis;
pub mod measurement;
pub mod msg;
pub mod numeric;
pub mod simple_1d_filter;
pub mod state_transition;
pub mod tf2_math;
pub mod tick;
pub mod trace;

pub use ekf_module::{EkfDiagnosticInfo, EkfModule};
pub use hyper_parameters::HyperParameters;
pub use tick::{EkfLocalizerCore, TickOutput};
