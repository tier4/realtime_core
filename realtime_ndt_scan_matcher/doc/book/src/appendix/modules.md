# Module index

One line per Rust source module in the engine crate (`realtime_ndt_scan_matcher/src/`). The C ABI
and ROS node modules live in the node crate and are indexed in its book.

## Core algorithm (portable, `no_std` + `alloc`)

- `lib.rs` — crate root, feature gating, `nalgebra` re-export, and the `init_thread_pool` /
  rayon-worker-affinity helpers.
- `engine.rs` — persistent `NdtEngine` handle, `MatchScratch`, config/map/align API.
- `ndt.rs` — `align`, derivative assembly, `NdtParams` / `AlignResult` / `AlignWorkspace`.
- `derivatives.rs` — angular + per-point score/gradient/Hessian kernels.
- `voxel_grid.rs` / `kdtree.rs` — target voxel map + spatial index (`kdtree` is private).
- `convergence.rs` — the pure convergence verdict.
- `covariance.rs` / `cov_estimate.rs` — pose-covariance math + the four estimation modes.
- `transform.rs` — SE3 transforms, Gauss constants, euler↔matrix.
- `tpe.rs` — Tree-Structured Parzen Estimator (the align-service pose search).
- `pose_buffer.rs` — time-ordered pose interpolation buffer (`SmartPoseBuffer` port).
- `helper.rs` — pure C++ helper ports (`rotate_covariance`, `count_oscillation`).

## Ports & orchestration (portable)

- `host.rs` — the `MapSource` / `OutputSink` / `Clock` / `Host` traits + result types.
- `scan_matcher.rs` — `ScanMatcher` over the `Host` ports (+ `apply_map_update`).

## Tooling & fixtures (`std`-only / opt-in)

- `capture.rs` — real-drive input capture (the `NDT_CAPTURE_DIR` sidecar format); `std` feature.
- `fixture.rs` — frozen WCET benchmark fixtures (capture-once, replay-everywhere); `std` feature.
- `wcet.rs` — deterministic algorithmic-cost counters for the WCET analysis
  (`plan/ndt_wcet.md`); `wcet-count` feature.

> Source: the crate `src/` module docs.
