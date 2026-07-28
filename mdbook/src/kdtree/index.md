# The kd-tree crate

`realtime_kdtree` (`realtime_kdtree/src/lib.rs`) is a dependency-free, `no_std` + `alloc` static
3-D kd-tree with bounded radius search — the `pcl::KdTreeFLANN` replacement for the NDT voxel-grid
map. Construction is control-plane work (may allocate and recurse); queries run on a fixed
iterative stack in exactly the recursive near-then-far visit order, so downstream float summation
stays bit-identical. Its `count` feature (forwarded from the engine crate's `wcet-count`) adds the
counted-search API for the WCET analysis; the `bench-kd-*` features are benchmark-only ablations.

How the engine uses it — and the voxel grid it indexes — is described in
[Voxel grid and kd-tree](../arch/voxel-grid.md).
