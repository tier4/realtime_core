# realtime_ndt_scan_matcher

`realtime_ndt_scan_matcher` is the ROS-free Rust implementation of the Autoware Normal
Distributions Transform (NDT) localization engine. It provides the reusable algorithmic core for
hosted Rust applications, the Autoware ROS 2 integration, and `no_std` kernel targets.

The crate contains no ROS message types, `rclcpp`, C ABI, or `extern "C"` interface. Those
integration layers belong to the consuming node crate.

## Design goals

- A bounded-work alignment API with explicit source-point, active-leaf, and iteration limits.
- Caller-owned scratch storage so the serial alignment path allocates no heap memory after setup.
- Checked failures for capacity, arithmetic, numeric, and kd-tree stack violations.
- Deterministic work counters and traces for offline timing and conformance analysis.
- One algorithmic core for `std`, single-core `no_std`, and multi-core `no_std` deployments.

The structural work envelope is conditional on the configured limits and accepted inputs. It does
not by itself provide a hard platform WCET bound.

## Primary API

[`NdtEngine`](src/engine.rs) is the main persistent engine. A caller:

1. Constructs an engine with limits for source points (`Pmax`), active leaves (`Lmax`), and Newton
   iterations (`Imax`, at most 30).
2. Adds or removes target-map tiles and publishes a kd-tree.
3. Allocates one [`MatchScratch`](src/engine.rs) for each concurrent alignment session.
4. Reuses that scratch while calling `NdtEngine::align` for successive scans.

[`ScanMatcher`](src/scan_matcher.rs) adds portable orchestration over host-provided map, clock, and
output ports. [`ndt::align`](src/ndt.rs) exposes the lower-level alignment kernel directly.

Generate and browse the complete Rust API documentation with:

```sh
cargo doc --no-deps --open
```

## Build configurations

| Configuration | Command | Behavior |
|---|---|---|
| Host, default | `cargo build --release` | `std`, lock-free map publication, rayon support |
| Host, serial | `cargo build --release --no-default-features --features std` | `std` without rayon |
| Single-core `no_std` | `cargo check --lib --no-default-features` | `RefCell` state, intentionally not `Sync` |
| Multi-core `no_std` | consumer enables `mt` and an `awkernel_sync` architecture backend | mutex-backed state, `Sync` |

Important optional features:

| Feature | Purpose |
|---|---|
| `parallel` | Rayon derivative reduction; enabled by default and implies `std` |
| `mt` | Multi-core `no_std` state synchronization |
| `wcet-count` | Deterministic iteration, neighbor, and Rust kd-node counters |
| `wcet-trace` | Analysis-only per-pass work-trace records; implies `wcet-count` |
| `bench-kd-*` | Benchmark-only kd-tree ablations |

The final binary that consumes this crate must keep `overflow-checks = true`. A multi-core
`no_std` binary must also select exactly one `awkernel_sync` architecture backend.

## Validation

Run the normal host checks from this directory:

```sh
cargo fmt --check
cargo clippy --all-targets --features wcet-count -- -D warnings
cargo test --features wcet-count
cargo check --lib --no-default-features
```

The test suite includes numeric-kernel tests, map-update ordering tests, serial/parallel
equivalence checks, bounded-work property tests, concurrency tests, and a zero-allocation check.

## Frozen paper fixtures

The repository includes generators for the synthetic fixtures used by the accompanying paper.
Generated `.ndtfix` files are written below `bench/fixtures/` and are not source files.

Reproduce all frozen fixtures and verify their SHA-256 manifest:

```sh
./gen_fixtures.sh
```

Run the short Rust-only evaluation corresponding to simplified versions of the paper's frozen
counter and frame-time tables:

```sh
./eval_paper_fixtures.sh
```

The evaluation regenerates the fixtures, performs 10 warmups and 5 measured serial alignments per
fixture, and reports deterministic counters plus Rust p50 and maximum frame time. It is a local
diagnostic, not a reproduction of the paper's isolated, multi-session timing protocol.

To run the search against the current engine without replacing the frozen paper fixtures:

```sh
./gen_fixtures.sh --fresh-search
```

Current-engine search results are written under `bench/fixtures/fresh-search/`.

## Further documentation

The documentation book starts at [`doc/book/src/SUMMARY.md`](doc/book/src/SUMMARY.md). It covers
the engine architecture, alignment kernel, map updates, numeric parity, trace verification,
real-time constraints, allocation behavior, and the `no_std` configurations.
