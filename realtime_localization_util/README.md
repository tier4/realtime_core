# realtime_localization_util

`realtime_localization_util` is the ROS-free Rust counterpart of the C++
`autoware_localization_util` shared library. It provides the localization utility modules shared
by the `realtime_core` ports — today by `realtime_ndt_scan_matcher`, which re-exports them under
its original module paths — and is intended for reuse by further localization ports.

The crate contains no ROS message types, `rclcpp`, C ABI, or `extern "C"` interface. Those
integration layers belong to the consuming node crate.

## Modules

| Module | Contents |
|---|---|
| [`pose_buffer`](src/pose_buffer.rs) | Port of `SmartPoseBuffer`: a time-ordered buffer of stamped poses-with-covariance with twist-based linear interpolation at a query time (tf2 ZYX RPY conventions) |
| [`transform`](src/transform.rs) | SE(3) transforms over the 6-vector `[tx, ty, tz, roll, pitch, yaw]`, Euler↔matrix conversions, the NDT Gaussian fitting constants, and the f32 `Matrix4` cloud pipeline mirroring the C++ `Matrix4f` path |
| [`tpe`](src/tpe.rs) | The Tree-Structured Parzen Estimator behind the align-service pose search: prior startup sampling, then expected-improvement selection over above/below Gaussian KDEs |

## Design goals

- Operation-for-operation ports of the C++ originals, so they differential-test bit-close.
- `no_std` + `alloc`; the only dependencies are `nalgebra` (with `libm`) and `libm`.
- No hidden panics: fallible operations return `Result`/`Option` and invalid numeric inputs are
  rejected or clamped as documented per function.
- Deterministic behavior: the TPE sampler is seedable and reproducible for a fixed seed.

## Usage

The TPE propose/evaluate loop (the body is the crate's compiled doctest):

```rust
use realtime_localization_util::tpe::{Direction, Error, TreeStructuredParzenEstimator, Trial};

fn pose_search() -> Result<(), Error> {
    // 5 prior dims (x, y, z, roll, pitch); yaw is sampled uniformly.
    let mean = [0.0; 5];
    let stddev = [0.25, 0.25, 0.25, 0.1, 0.1];
    let mut tpe = TreeStructuredParzenEstimator::new(Direction::Maximize, 5, &mean, &stddev)?;

    // Propose a candidate, evaluate it (here a stub score), and feed the result back.
    let candidate = tpe.get_next_input()?;
    tpe.add_trial(Trial { input: candidate, score: 1.5 })?;
    assert_eq!(tpe.trials_len(), 1);
    Ok(())
}
```

Generate and browse the complete Rust API documentation with:

```sh
cargo doc --no-deps --open
```

## Build and validation

The crate has no Cargo features. Run the normal host checks from this directory:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Workspace

Part of the [`realtime_core`](../) workspace. The workspace documentation book is published at
<https://tier4.github.io/realtime_core/>.

Licensed under either of the Apache License, Version 2.0 or the MIT license, at your option
(`MIT OR Apache-2.0`).
