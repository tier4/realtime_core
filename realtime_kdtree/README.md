# realtime_kdtree

`realtime_kdtree` is a static 3-D kd-tree with bounded radius search over `[f32; 3]` points — the
`pcl::KdTreeFLANN` replacement used by the `realtime_ndt_scan_matcher` voxel-grid map. It is
dependency-free, `no_std` + `alloc`, and shared through the `realtime_core` workspace.

The crate holds only the spatial index. The voxel grid it indexes, and everything above it, live
in the consuming engine crate.

## Design goals

- **Build is control-plane work**: `KdTree::try_build` copies the points, may allocate and
  recurse, and rejects a tree whose query depth would exceed the fixed stack before publishing it.
- **Search is real-time work**: `radius_search` runs on a fixed-size iterative stack (no
  recursion, no allocation when the caller pre-reserves the output vector) and visits nodes in
  exactly the recursive near-then-far order, so the neighbor order — and every downstream float
  summation — is deterministic.
- **Checked failures**: allocation, arithmetic, and stack-capacity violations surface as
  `KdSearchError` values instead of panics.
- `max_nn` caps the retained results; `KdSearchOutcome::result_limit_exceeded` reports whether
  another in-radius point existed beyond the cap.

## Usage

Build a tree and collect the indices of all points within a radius (from the crate's tests):

```rust
use realtime_kdtree::KdTree;

let pts: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
let tree = KdTree::try_build(&pts).expect("build kd-tree");

let mut out = Vec::new();
let outcome = tree
    .radius_search(&[0.0, 0.0, 0.0], 1.0, 0, &mut out) // max_nn == 0: unlimited
    .expect("radius search");
out.sort_unstable(); // `out` arrives in deterministic near-then-far traversal order
assert_eq!(out, vec![0, 1]);
assert!(!outcome.result_limit_exceeded);
```

## Features

| Feature | Purpose |
|---|---|
| `count` | `radius_search_counted`: the same search, also reporting the number of tree nodes visited — the deterministic traversal-cost counter for the WCET analysis (forwarded from the engine crate's `wcet-count`) |
| `bench-kd-force-count`, `bench-kd-stop-at-cap` | Benchmark-only ablations reconstructing pre-optimization hot-path behavior for matched-binary timing diagnosis; off in normal builds |

## Validation

```sh
cargo fmt --check
cargo clippy --all-targets --features count -- -D warnings
cargo test --features count
```

The test suite includes brute-force equivalence checks and an exact visit-order oracle against the
reference recursive search.

## Workspace

Part of the [`realtime_core`](../) workspace. The workspace documentation book is published at
<https://tier4.github.io/realtime_core/>.

Licensed under either of the Apache License, Version 2.0 or the MIT license, at your option
(`MIT OR Apache-2.0`).
