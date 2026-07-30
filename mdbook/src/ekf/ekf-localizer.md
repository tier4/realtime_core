# The EKF localizer crate

`realtime_ekf_localizer` is the ROS-free Rust counterpart of the C++ `autoware_ekf_localizer`
package (`localization/autoware_ekf_localizer`): `no_std` + `alloc`, depending on
[`realtime_kalman_filter`](kalman-filter.md) for the delay-augmented filter. Its modules map 1:1 to
the C++ files:

- Math leaves — `state_transition`, `mahalanobis`, `measurement`, `covariance`, `numeric`,
  `aged_object_queue`, `simple_1d_filter`, and the `tf2_math` quaternion/Euler helpers
  (expression-exact ports of the tf2 formulas the EKF calls).
- `hyper_parameters` — a plain struct (no parameter server).
- `ekf_module` (`EkfModule`) — the ROS-free module: plain-struct measurements, `i64` nanosecond
  time, delay-table search/accumulation, and the pose/twist measurement-update decision chains.
- `tick` (`EkfLocalizerCore`) — the port of the C++ `timer_callback` control flow: queue drain
  with aging/smoothing, the queue-size caps, and predict orchestration.
- `trace` — the extended `AUTOWARE_EKF_POSE_TRACE` CSV rows used for conformance.

Behavior equivalence with the C++ is governed by the port-equivalence contract
(`porting_notes/ekf_port_contract.md` in the workspace repo): decisions match exactly, f64 chains
match within `rel_tol = 1e-9`. The event path is real-time hardened onto preallocated scratch, with
a counting-allocator test asserting zero allocations per event after warmup. The C ABI that exposes
this core to the C++ package, and the measured integration results, are covered in
[FFI integration and real-time results](conformance.md).
