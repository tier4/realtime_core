# realtime_ekf_localizer

`realtime_ekf_localizer` is the ROS-free Rust counterpart of the C++ `autoware_ekf_localizer`
package (`localization/autoware_ekf_localizer`). It is the portable EKF localizer core: the math
leaves (state transition, Mahalanobis gating, covariance, delay-compensated pose/twist updates,
tf2 quaternion/Euler helpers), the ROS-free [`EkfModule`](src/ekf_module.rs), and the
[`EkfLocalizerCore`](src/tick.rs) tick orchestration that ports the C++ `timer_callback` control
flow (queue drain + predict/update). It depends on
[`realtime_kalman_filter`](../realtime_kalman_filter/README.md) for the delay-augmented filter.

The crate contains no ROS message types, `rclcpp`, C ABI, or `extern "C"` interface. Measurements
cross as plain structs and time is a bare `i64` nanosecond stamp; the ROS node shell and the C ABI
belong to the consuming node crate (see the fork's `autoware_ekf_localizer_rs`).

## Design goals

- **Behavior equivalence with the C++**, governed by the port-equivalence contract
  (`porting_notes/ekf_port_contract.md` in the workspace repo): decisions (gates, delay steps,
  event order) match exactly; f64 chains match within `rel_tol = 1e-9`. The conformance traces are
  byte-identical across the native Rust replay and the C ABI backend.
- **Real-time hardened.** The event path runs on preallocated scratch — a counting-allocator gate
  (`tests/zero_alloc.rs`) asserts zero allocations per event after warmup for predict and
  accepted/rejected pose/twist updates.
- **ROS-free and no_std + alloc.** Modules map 1:1 to the C++ files (`state_transition`,
  `mahalanobis`, `measurement`, `covariance`, `numeric`, `aged_object_queue`, `simple_1d_filter`,
  `hyper_parameters`, `ekf_module`, `tick`, `tf2_math`, `trace`); dependencies are
  `realtime_kalman_filter`, `nalgebra`, and `libm`.

## Usage

Anchor the filter at an initial pose, then drive one predict/update tick (adapted from the crate's
`tick` test):

```rust
use realtime_ekf_localizer::hyper_parameters::HyperParameters;
use realtime_ekf_localizer::msg::{PoseWithCovariance, Transform, cov_idx};
use realtime_ekf_localizer::tf2_math::quaternion_from_rpy;
use realtime_ekf_localizer::tick::EkfLocalizerCore;
use realtime_ekf_localizer::trace::TraceEvent;

let mut core = EkfLocalizerCore::new(HyperParameters::default())?;

let mut covariance = [0.0_f64; 36];
covariance[cov_idx::X_X] = 0.04;
covariance[cov_idx::Y_Y] = 0.04;
covariance[cov_idx::YAW_YAW] = 0.01;
let initial = PoseWithCovariance {
    stamp_ns: 0,
    position: [10.0, 20.0, 0.0],
    orientation: quaternion_from_rpy(0.0, 0.0, 0.1),
    covariance,
};
core.initialize(&initial, &Transform::identity())?;

// A 50 Hz tick (20 ms). Trace events are appended to `trace` for conformance checking.
let mut trace: Vec<TraceEvent> = Vec::new();
if let Some(out) = core.tick(20_000_000, &mut trace)? {
    let _x = out.pose;
    let _vx = out.twist.vx;
}
# Ok::<(), realtime_kalman_filter::KalmanError>(())
```

## Build and validation

The crate has no Cargo features. Run the normal host checks from this directory:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The tests cover the `EkfModule` decision chains (`tests/ekf_module.rs`), the tick orchestration
(`tests/tick.rs`), and the zero-allocation event-path gate (`tests/zero_alloc.rs`).

## Workspace

Part of the [`realtime_core`](../) workspace. See
[FFI integration and real-time results](../mdbook/src/ekf/conformance.md) in the workspace book
(published at <https://tier4.github.io/realtime_core/>) for the conformance and
real-time-hardening results.

Licensed under either of the Apache License, Version 2.0 or the MIT license, at your option
(`MIT OR Apache-2.0`).
