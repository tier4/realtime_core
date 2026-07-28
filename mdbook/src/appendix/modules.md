# Module index

One line per Rust source module in the engine crate (`realtime_ndt_scan_matcher/src/`). Modules
marked *(sibling crate)* were extracted into workspace crates next to the engine crate and are
re-exported under their original module paths, so both path forms resolve. The C ABI
and ROS node modules live in the node crate and are indexed in its book.

## Core algorithm (portable, `no_std` + `alloc`)

- `lib.rs` — crate root, feature gating, `nalgebra` re-export, and the `init_thread_pool` /
  rayon-worker-affinity helpers.
- `engine.rs` — persistent `NdtEngine` handle, `MatchScratch`, config/map/align API.
- `ndt.rs` — `align`, derivative assembly, `NdtParams` / `AlignResult` / `AlignWorkspace`.
- `derivatives.rs` — angular + per-point score/gradient/Hessian kernels.
- `voxel_grid.rs` / `kdtree` — target voxel map + spatial index (`kdtree` is private; *(sibling
  crate)* `realtime_kdtree`).
- `convergence.rs` — the pure convergence verdict.
- `covariance.rs` / `cov_estimate.rs` — pose-covariance math + the four estimation modes.
- `transform` — SE3 transforms, Gauss constants, euler↔matrix *(sibling crate:
  `realtime_localization_util`)*.
- `tpe` — Tree-Structured Parzen Estimator (the align-service pose search) *(sibling crate:
  `realtime_localization_util`)*.
- `pose_buffer` — time-ordered pose interpolation buffer (`SmartPoseBuffer` port) *(sibling crate:
  `realtime_localization_util`)*.
- `helper.rs` — pure C++ helper ports (`rotate_covariance`, `count_oscillation`).

## Ports & orchestration (portable)

- `host.rs` — the `MapSource` / `OutputSink` / `Clock` / `Host` traits + result types.
- `scan_matcher.rs` — `ScanMatcher` over the `Host` ports (+ `apply_map_update`).

## Tooling & fixtures (`std`-only / opt-in)

- `capture.rs` — real-drive input capture (the `NDT_CAPTURE_DIR` sidecar format); `std` feature.
- `fixture.rs` — frozen WCET benchmark fixtures (capture-once, replay-everywhere); `std` feature.
- `wcet` — deterministic algorithmic-cost counters for the WCET analysis
  (`plan/ndt_wcet.md`); `wcet-count` feature *(sibling crate: `realtime_port_instrument`)*.

> Source: the crate `realtime_ndt_scan_matcher/src/` module docs.
