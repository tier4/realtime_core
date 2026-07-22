# Feature flags and build configurations

The crate has one code base and several build configurations, selected by Cargo features. The
`std` feature is default-on so a plain `cargo build`/`test`/`clippy` has `std` (test harness +
panic handler); the `no_std` build opts out with `--no-default-features`.

## Features

| Feature | Default | Effect |
|---|---|---|
| `std` | ✅ | Host build. Pulls `arc-swap` for lock-free engine-state publication. Alignment uses caller-owned `MatchScratch`; the engine is `Sync`. |
| `parallel` | ✅ | rayon-backed `compute_derivatives` reduction. Implies `std`. Bit-identical to serial (per-point contributions reduced in index order), so it is a pure throughput option — the serial backend stays the predictable WCET baseline. |
| `mt` | ❌ | Multi-core `no_std` (kernel). Replaces the single-core `RefCell` state cells with `awkernel_sync` mutexes. Callers pass a `&mut MatchScratch`, as in every build. Ignored when `std` is also on. |
| `wcet-count` | ❌ | Deterministic algorithmic-cost counters on the align path (iterations, neighbours, kd-tree nodes visited) — the worst-input search fitness. Compiled out entirely when off, so the shipping hot path is untouched. |
| `wcet-trace` | ❌ | Analysis-only per-pass records for cross-language work-shape conformance (SHA-256 leaf ids + kernel payloads). Serial backend only; implies `wcet-count`. Absent from shipping builds. |
| `bench-kd-force-count` / `bench-kd-stop-at-cap` | ❌ | Benchmark-only kd-tree ablations that reconstruct pre-optimization hot-path components for matched-binary timing diagnosis. Disabled in normal and `wcet-count` builds. |

## Parallelism and worker threads

The `parallel` backend runs the derivative reduction on rayon's **process-global thread pool**.
There are two independent knobs:

- **Enable parallel** — set the `num_threads` param `> 1` (`NdtParams.num_threads`, the ROS node's
  `num_threads` parameter, or `ScanMatcher::set_params`). This is a *switch*: `> 1` selects the rayon
  backend, `≤ 1` stays serial. It does **not** by itself decide how many workers rayon uses.
- **Set the worker count** — size the process-global pool, in one of two ways (equivalent,
  process-wide):

  1. **Explicit API.** Call `init_thread_pool(n)` once, early. Best-effort and idempotent — a
     consuming application (e.g. a ROS node) typically calls it once from its own `num_threads`
     configuration before any align.
  2. **Environment.** `RAYON_NUM_THREADS=n` (rayon's built-in), read on first use.

  If neither is set, the pool defaults to the number of logical CPUs.

Because the pool is process-global, `n` is the total worker count for the whole process (not
per-engine). The reduction is bit-identical regardless of `n`, so this only trades throughput for
WCET predictability — the serial backend (`num_threads ≤ 1`) stays the predictable RT baseline.

## The three engine configurations

The interior mutability of the engine (its target map + params) is chosen at compile time. This
is the single most important thing to understand about the build matrix; it is detailed in
[Concurrency and interior mutability](../arch/concurrency.md).

| Configuration | How to build | Cells | Align scratch | `Sync`? |
|---|---|---|---|---|
| **std** (default) | `cargo build` | `ArcSwap<EngineState>` (lock-free) | caller-owned | yes |
| **`no_std` single-core** | `--no-default-features` | `RefCell<Arc<…>>` | caller-owned | **no** (rejected at compile time) |
| **`no_std` multi-core** | `--no-default-features --features mt` | `awkernel_sync::Mutex<Arc<…>>` | **caller-owned** `MatchScratch` | yes |

Notes:

- `parallel` implies `std`, so `mt` + `parallel` resolves to the `std` backend (`mt` is ignored).
- Every configuration exposes only alignment methods that take a `&mut MatchScratch`.
- The `no_std` builds are *libraries* linked into a final binary that supplies the
  `#[panic_handler]`; building the `no_std` rlib standalone will report a missing panic
  handler. The kernel gate used in CI is a targeted `cargo rustc --no-default-features --lib
  --target {x86_64,aarch64}-unknown-none --crate-type rlib`.

## Source of truth

The feature definitions and their rationale live in the crate's `Cargo.toml`; the engine's
interior-mutability matrix is documented at the top of `src/engine.rs`; the crate-level overview
is the `//!` doc in `src/lib.rs`.
