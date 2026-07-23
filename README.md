# realtime_core

A Rust workspace for **real-time, ROS-free ports of Autoware Core components**. The code here is
designed to run unchanged under ROS 2, an async runtime, or a bare-metal `no_std` kernel, with a
zero-warning / panic-free hardening standard and predictable worst-case execution time (WCET) on the
real-time paths.

## Crates

| Crate | Description |
|-------|-------------|
| [`realtime_ndt_scan_matcher`](realtime_ndt_scan_matcher/) | Pure-Rust, ROS-free, `no_std`-capable core of the Autoware NDT (Normal Distributions Transform) localization engine. The numeric kernels mirror the C++ `Matrix4f` / `MultiVoxelGridCovariance` pipeline. The C ABI and ROS integration live in the sibling `autoware_ndt_scan_matcher_rs` node crate, not here. |

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

The `realtime_ndt_scan_matcher` design / real-time / porting book is rendered to
[`docs/realtime_ndt_scan_matcher/`](docs/realtime_ndt_scan_matcher/) (open `index.html`). Regenerate it
after editing the sources under `realtime_ndt_scan_matcher/doc/book/src/`:

```sh
./realtime_ndt_scan_matcher/gen_docs.sh
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
