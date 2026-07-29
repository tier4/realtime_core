# FFI integration and real-time results

The Rust EKF core is integrated into the C++ `autoware_ekf_localizer` package behind
`-DEKF_USE_RUST=ON`: the `autoware_ekf_localizer_rs` staticlib (in the fork, branch `ekf_in_rust`)
exposes the core over a C ABI that mirrors the C++ `EKFModule` 1:1, and a thin `ekf_module_rs.cpp`
adapter drives it. This chapter distills §8 of the port-equivalence contract
(`porting_notes/ekf_port_contract.md`); the numbers below are the contract's frozen measured
values, quoted literally.

## Conformance — byte-identity

- The C++ test suite passes unchanged under both backends: **152/152** colcon tests with
  `EKF_USE_RUST=ON` (and `OFF` unaffected), plus **178** workspace tests.
- Over the frozen 13-scenario corpus, the Rust backend reaches **100 % decision agreement** with
  the frozen C++ fixtures at the same per-scenario numerics as the native run (worst synthetic
  relative error `3.0e-10`, real-data within the §3 policy).
- Every trace is **byte-identical** to the native Rust replay (`cmp`) — same core, same trace
  writer, exercised through the C ABI.

## Real-time hardening — allocation-free event path

The event path (both `realtime_kalman_filter` and `realtime_ekf_localizer`) was reworked onto
preallocated scratch: extended-dim buffers sized at init, in-place gemm through the same nalgebra
kernels, and a hand-rolled in-place LLT transcribed from nalgebra's `Cholesky` loops (verified
bit-equal on randomized factors). The FFI layer skips CSV serialization entirely when no trace
stream is open.

- **Invariance held**: the 13-scenario corpus still passes and the traces are byte-identical to
  the pre-hardening Rust build.
- A counting-allocator gate (`realtime_ekf_localizer/tests/zero_alloc.rs`) asserts **0
  allocations per event** after warmup for predict and accepted/rejected pose/twist updates
  (measured residual `≈0.5` allocs/event is the replay harness's line parsing, not the filter).
- Rust compute improved **10–15 %**, dropping the Rust-vs-C++ compute ratio from **1.42×** to
  **1.25–1.29×**.

## Corrected overflow-checks attribution

§8.3 first attributed roughly half of the remaining FFI-backend excess to `overflow-checks`. The
refined finding (§8.5) is that this tax lives in **nalgebra's** internal `copy_from`/`gemm`/`axpy`
loops — bounded loop-counter arithmetic that cannot overflow in practice — **not** in the
`realtime_*` crates, whose own integer arithmetic already uses `checked_*`. It is therefore removed
safely with a **per-package profile override** setting `overflow-checks = false` for `nalgebra`
only, applied in both `realtime_core/Cargo.toml` and the vendored FFI crate's manifest (the profile
Corrosion compiles the static lib with). Every `realtime_*` crate keeps `overflow-checks = true`, so
its integer arithmetic still fails loudly.

The attribution splits as **~79 %** of the overflow-checks tax inside nalgebra (removed) and
**~21 %** in the still-checked crates (kept) — a deliberate safety/perf split, not a blanket flag
flip.

## Result

With the nalgebra override in place, the shipping FFI-Rust EKF backend runs within
**1.06–1.09×** of the header-only, fully-inlined C++/Eigen backend (down from 1.26×), with no
numeric change and the crates' overflow safety intact. The residual gap is the ISA-matched
nalgebra-vs-Eigen kernel cost (the naive rank-m covariance downdate) plus the inherent
FFI/`geometry_msgs` marshaling; closing it further would break trace byte-identity or require
`-march`/target-cpu tuning on both backends, out of scope for the port.

The closed-loop smoke (C++ NDT + hardened Rust EKF, 600 s) produced **50.00 Hz** EKF output with
inter-pose gap p99 = 20.0 ms and zero gaps over 100 ms — indistinguishable from the C++ baseline.
