# Differential testing

The ON-vs-OFF oracle in practice: drive identical inputs through the reference C++ path
(`NDT_USE_RUST=OFF`) and the Rust path (`ON`) and diff the observables. This is the authoritative
correctness mechanism (see Behavior equivalence and verification).

## Building both paths

The Rust differential/FFI gtests are registered only under `-DNDT_USE_RUST=ON` (they link the crate
alongside the C++ engine in one test binary). Build that way, then `colcon test`; the always-on
C++/integration suites run in both backends. This harness lives in the consuming ROS node crate,
which depends on this engine — the exact build/test commands are documented there.

## Layers

- **Unit / property gtests** — the pure kernels checked against a reference or a brute-force oracle:
  `test_voxel_grid` (Rust `VoxelGridMap` vs C++ `MultiVoxelGridCovariance`), `test_ndt_engine` (Rust align vs the C++ NDT engine),
  `test_convergence_verdict`, `test_map_update_verdict`, …
- **FFI direct-call tests** — because `cargo llvm-cov` sees only Rust, the `extern "C"` shims (in the
  node crate) get Rust-side direct-call tests too.
- **Integration** — the `standard_sequence_*` launch/gtests drive the full node on a fixed sequence
  in both backends.

## Observables compared

Estimated pose, covariance, transform probability, NVTL, iteration count, convergence status,
diagnostics, service responses, and map-update behavior — to the tolerances in
Verification, with any documented [upstream divergence](divergences.md) mirrored
so the diff stays green.

> Source: the consuming ROS package's C++ gtests and `standard_sequence_*` launch tests, and its
> `NDT_USE_RUST` test registrations. This engine crate's own tests are `tests/{concurrency,
> wcet_bounds, zero_alloc}.rs` plus the in-`src` numeric/finite-difference unit tests.
