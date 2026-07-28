# realtime_port_instrument

`realtime_port_instrument` is the instrumentation ABI used to analyze and verify C++→Rust
real-time ports in the `realtime_core` workspace: deterministic algorithmic-cost counters and
per-pass trace records. The engine crate (`realtime_ndt_scan_matcher`) forwards its `wcet-count` /
`wcet-trace` features here and re-exports this crate as its `wcet` module.

The crate **counts work, not time**: derivative passes, points processed, neighbors collected, and
kd-tree nodes visited are platform-independent and bit-reproducible, which makes them a stable
worst-input search fitness and the machine-checked link to an analytic work bound. Wall-clock
measurement belongs to the consumer's benchmarks, not to this crate.

It is `no_std`, allocation-free, and has no runtime dependencies (`sha2` is a dev-dependency for
the shared test vectors only).

## Features

| Feature | Purpose |
|---|---|
| `count` | `WcetCounters`: the per-align algorithmic-cost counters (forwarded from the engine crate's `wcet-count`) |
| `trace` | `PassTrace` / `AlignTrace`: analysis-only per-pass records for cross-language work-shape conformance — FNV-1a hashes for pass-final score/gradient/Hessian diagnostics and SHA-256 chain digests for canonical neighbor shapes and kernel payloads (implies `count`; forwarded from `wcet-trace`) |

The FNV-1a fold and the digest layout are bit-identical to the traced C++ analysis build
(`ndt_trace.hpp` on the C++ side); both implementations are covered by the same test vectors, so a
Rust trace and a C++ trace of the same work compare byte-for-byte.

## Usage

Counters start at zero and are reset by the instrumented align path (`--features count`):

```rust
use realtime_port_instrument::WcetCounters;

let counters = WcetCounters::new();
assert_eq!(counters.derivative_passes, 0);
```

The shared cross-language FNV-1a test vector (`--features trace`, from the crate's tests):

```rust
use realtime_port_instrument::{FNV_OFFSET, fnv1a_u64};

// 1.0_f64's bit pattern folded from the offset basis; the C++ side must produce the same value.
assert_eq!(fnv1a_u64(FNV_OFFSET, 1.0_f64.to_bits()), 0xaab1_6932_29ba_1db8);
```

## Validation

```sh
cargo fmt --check
cargo clippy --all-targets --features trace -- -D warnings
cargo test --features trace
```

The tests pin the FNV-1a and SHA-256 vectors shared with the C++ analysis build and the trace
storage-cap behavior.

## Workspace

Part of the [`realtime_core`](../) workspace. The workspace documentation book is published at
<https://tier4.github.io/realtime_core/>.

Licensed under either of the Apache License, Version 2.0 or the MIT license, at your option
(`MIT OR Apache-2.0`).
