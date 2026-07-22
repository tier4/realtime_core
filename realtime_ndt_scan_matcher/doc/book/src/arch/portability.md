# Portability and the Host ports

The same algorithm core runs under ROS, a bare-metal kernel, or an async runtime because it is
written against the Host ports, not against ROS.

## `ScanMatcher` and the `Host` ports

`scan_matcher::ScanMatcher` is a concrete struct wrapping an [`NdtEngine`](engine.md). It is not
generic over `Host`; instead its map-update methods are generic over the `MapSource` port, so a
caller supplies map I/O without the matcher depending on ROS:

- `match_scan(guess, source, &mut scratch) -> Result<MatchResult, AlignError>` and
  `match_scan_with_covariance(...) -> Result<(MatchResult, CovarianceResult), AlignError>` —
  **synchronous**, the WCET hot path.
- `update_map<S: MapSource>(&source, center, radius).await` — **async**, over the `MapSource` port.

It holds no per-align mutable state, so a shared `&ScanMatcher` is sound across concurrent tasks in
the std and `mt` configs (map updates publish atomically).

## The `no_std` gating discipline

`lib.rs` is `#![cfg_attr(not(any(test, feature = "std")), no_std)]`. The algorithmic modules
(`scan_matcher`, `host`, `engine`, `convergence`, `covariance`, `cov_estimate`, `ndt`, `voxel_grid`,
`tpe`, …) are `no_std`. The only `#[cfg(feature = "std")]`-gated modules are `capture` and `fixture`
(both file-IO tooling); `wcet` is gated on `wcet-count`. New algorithmic code goes in the portable
core so the kernel build keeps it; only host-only tooling may be `std`-gated.

## Reference adapters

- `examples/tokio_ndt.rs` — the async reference host over synthetic data (proves the `Host` seam).
- `examples/threads_ndt.rs` — multi-threaded shared-engine usage.

The kernel (Track-B) target is the reason for this split: it supplies its own async runtime and map
source (flash/DMA) and builds the core with `--no-default-features` (single-core) or `--features mt`
(multi-core); see [The `mt` multi-core engine](../rt/mt.md).

> Source: `src/scan_matcher.rs`, `src/host.rs`, `examples/{tokio_ndt,threads_ndt}.rs`, `src/lib.rs`.
