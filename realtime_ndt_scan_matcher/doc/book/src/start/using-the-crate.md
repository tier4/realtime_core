# Using the engine crate

`realtime_ndt_scan_matcher` is usable as a plain Rust library, independently of ROS. This
chapter tours the public API. The examples are the crate's own rustdoc doctests (verified with
`cargo test --doc`); browse the full reference with `cargo doc --no-deps --open`.

## The `nalgebra` re-export

Pose guesses and results are `nalgebra` matrices (`Matrix4<f32>` for poses, `Matrix6<f64>` for
Hessians/covariances). The crate **re-exports the exact `nalgebra` version it is built against**,
so construct those matrices through `realtime_ndt_scan_matcher::nalgebra` rather than pinning
your own (a different version is a distinct, incompatible type).

## Primary API: `NdtEngine`

The persistent engine ([`engine::NdtEngine`](../arch/engine.md)) is the main entry point. Load a
target map, build the kd-tree, then align sensor clouds:

```rust
use realtime_ndt_scan_matcher::engine::NdtEngine;
use realtime_ndt_scan_matcher::nalgebra::Matrix4;

// Empty engine: 2.0 m voxels; MultiVoxelGridCovariance defaults (min 6 points / eig 0.01).
let engine = NdtEngine::new(2.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");

// Register a target map tile (id 0) and build the kd-tree over the voxel centroids.
let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
engine.add_target(&target, b"0");
engine.create_kdtree().expect("map fits the leaf limit");
assert!(engine.has_target());

// Align with caller-owned scratch and read the result from that same scratch.
let source = target.clone();
let mut scratch = realtime_ndt_scan_matcher::engine::MatchScratch::try_for_limits(engine.limits())
    .expect("reserve scratch");
engine.align(&Matrix4::identity(), &source, &mut scratch).expect("align");
let result = scratch.result_ref();
assert!(result.iteration_num >= 0);
```

### Caller-owned scratch

The align scratch (workspace + last result) is a [`engine::MatchScratch`](../arch/scratch.md).
Every alignment method takes scratch owned by the caller. Reuse it across frames to keep the serial
path allocation-free from the first call:

```rust
use realtime_ndt_scan_matcher::engine::{MatchScratch, NdtEngine};
use realtime_ndt_scan_matcher::nalgebra::Matrix4;

let engine = NdtEngine::new(2.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
engine.add_target(&target, b"0");
engine.create_kdtree().expect("map fits the leaf limit");

let mut scratch = MatchScratch::try_for_limits(engine.limits()).expect("reserve scratch");
engine.align(&Matrix4::identity(), &target, &mut scratch).expect("align");
assert!(scratch.result_ref().iteration_num >= 0);
```

## Portable orchestration: `ScanMatcher`

[`scan_matcher::ScanMatcher`](../arch/portability.md) wraps the engine and drives it over the
host ports (`MapSource` / `OutputSink` / `Clock`). Map loading is
`async`; the match is the synchronous WCET hot path. It runs unchanged under ROS, a kernel, or an
async runtime — `examples/tokio_ndt.rs` is the reference host implementation.

## The lower-level kernel: `ndt::align`

[`ndt::align`](../arch/align.md) is the RT-critical alignment function the engine drives. Use it
directly when you hold a `VoxelGridMap` yourself:

```rust
use realtime_ndt_scan_matcher::ndt::{align, AlignResult, AlignWorkspace, NdtParams};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;
use realtime_ndt_scan_matcher::nalgebra::Matrix4;

let mut map = VoxelGridMap::new([2.0; 3], 6, 0.01);
let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
map.add_target(&target, b"0");
map.try_create_kdtree(418_000).expect("map fits the leaf limit");

let params = NdtParams { resolution: 2.0, ..NdtParams::default() };
let mut ws = AlignWorkspace::try_with_capacity(target.len()).expect("reserve workspace");
let mut out = AlignResult::try_with_capacity(30).expect("reserve result");
align(&map, &target, &Matrix4::identity(), &params, &mut ws, &mut out).expect("align");
assert!(out.iteration_num >= 0);
```

## Parallelism

Under the `parallel` feature, set `num_threads > 1` (in `NdtParams` / `set_params`) to use the
rayon backend. The rayon *worker count* is the process-global pool size — set it once with
`init_thread_pool(n)` or `RAYON_NUM_THREADS`; see
[Feature flags and build configurations](features.md#parallelism-and-worker-threads).

## Other public modules

- [`tpe::TreeStructuredParzenEstimator`](../arch/tpe.md) — the align-service pose-search sampler.
- [`convergence`](../concepts/scores.md), `covariance`, `cov_estimate` — pure decision/estimation
  kernels reused across the ROS and kernel builds.
- `pose_buffer` — a time-ordered pose interpolation buffer.
