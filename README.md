# realtime_core

A Rust workspace for **real-time, ROS-free ports of Autoware Core components**. The code here is
designed to run unchanged under ROS 2, an async runtime, or a bare-metal `no_std` kernel, with a
zero-warning / panic-free hardening standard and predictable worst-case execution time (WCET) on the
real-time paths.

## Crates

| Crate | Description |
|-------|-------------|
| [`realtime_ndt_scan_matcher`](realtime_ndt_scan_matcher/README.md) | Pure-Rust, ROS-free, `no_std`-capable core of the Autoware NDT (Normal Distributions Transform) localization engine. The numeric kernels mirror the C++ `Matrix4f` / `MultiVoxelGridCovariance` pipeline. The C ABI and ROS integration live in the sibling `autoware_ndt_scan_matcher_rs` node crate, not here. |
| [`realtime_localization_util`](realtime_localization_util/README.md) | Shared localization utilities: the `SmartPoseBuffer` and TPE (`TreeStructuredParzenEstimator`) ports — the Rust counterpart of the C++ `autoware_localization_util` shared library — plus the SE3 transform kernels. Re-exported by the NDT engine crate under its original module paths; intended for reuse by further localization ports. |
| [`realtime_kdtree`](realtime_kdtree/README.md) | Static 3-D kd-tree with bounded, iterative radius search (a `pcl::KdTreeFLANN` replacement). `no_std` + `alloc`, dependency-free. |
| [`realtime_port_instrument`](realtime_port_instrument/README.md) | Deterministic algorithmic-cost counters and the cross-language FNV/SHA trace ABI for WCET analysis of C++→Rust ports (the engine crate's `wcet-count` / `wcet-trace` features forward to its `count` / `trace`). |
| [`realtime_kalman_filter`](realtime_kalman_filter/README.md) | Pure-Rust, ROS-free, `no_std`-capable Kalman filter (the `KalmanFilter` + `TimeDelayKalmanFilter` ports of the C++ `autoware_kalman_filter`), on dynamically-sized nalgebra matrices with delay-augmented state. Used by `realtime_ekf_localizer`. |
| [`realtime_ekf_localizer`](realtime_ekf_localizer/README.md) | Pure-Rust, ROS-free, `no_std`-capable EKF localizer core (the port of the C++ `autoware_ekf_localizer`): state transition, gating, delay-compensated pose/twist updates, and tick orchestration. Depends on `realtime_kalman_filter`; the C ABI and ROS node live in the fork's `autoware_ekf_localizer_rs`, not here. |

## Building and testing

```sh
# Build / test the whole workspace (std, the default configuration)
cargo build
cargo test

# Lint gate (rust-hardening: zero warnings, no panics, no lossy casts)
cargo clippy --all-targets -- -D warnings
```

### `realtime_ndt_scan_matcher` build configurations

The engine crate selects its interior-mutability and concurrency backend by feature:

| Configuration | Backend |
|---------------|---------|
| `std` (default) | `arc-swap` lock-free engine-state double-buffer; engine is `Sync` |
| `--no-default-features` | single-core `no_std` (`RefCell` cells); engine is `!Sync` |
| `--no-default-features --features mt,<lock-backend>` | multi-core `no_std` (kernel-grade mutex cells); engine is `Sync` |

`parallel` (default-on) adds the rayon-backed derivative reduction, bit-for-bit identical to the
serial WCET baseline. See the crate documentation for the full feature matrix.

## Documentation

The workspace design / real-time / porting book is published at
**<https://tier4.github.io/realtime_core/>**.

It is rendered from source into [`docs/`](docs/)
(which GitHub Pages serves; open `docs/index.html` to browse the same content locally). Regenerate
it after editing the sources under [`mdbook/src/`](mdbook/src/):

```sh
./gen_docs.sh
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
