# How to read this book

This book covers the **engine crate** (`realtime_ndt_scan_matcher`). For the FFI boundary, the
`Host` C vtable, the ROS 2 node, and the C++→Rust symbol map, see the **node crate** book
(`autoware_ndt_scan_matcher_rs`).

| You are… | Start with | Then |
|---|---|---|
| **A Rust crate user** | [Getting Started](start/using-the-crate.md) (using the crate, feature flags) | [The NDT engine](arch/engine.md), [The align hot path](arch/align.md), and the generated rustdoc |
| **A maintainer / contributor** | [Architecture](arch/engine.md) (engine → align → covariance) | [Numeric parity](port/numeric-parity.md), [Divergences from upstream](port/divergences.md) |
| **An RT / `no_std` (kernel) engineer** | [Real-Time and no_std](rt/wcet.md) (WCET, zero-alloc, `mt`, panic-free) | [Concurrency and interior mutability](arch/concurrency.md) |
| **An upstream / PR reviewer** | [Numeric parity](port/numeric-parity.md) and [Divergences](port/divergences.md) | [Differential testing](port/differential.md), [Trace verification](port/trace-verification.md) |

## Conventions

- **Code that is meant to compile** is shown as tested `rust` doctests copied from the crate's
  rustdoc. **Illustrative snippets and excerpts** use `rust,ignore` or `text` and are not compiled.
- **Paths** like `src/engine.rs` are relative to **this crate's root**
  (`autoware_ndt_scan_matcher/autoware_ndt_scan_matcher_rs/realtime_ndt_scan_matcher/`). A path
  under the sibling node crate is written `../src/…`.
- **`Ndt*` / `ScanMatcher`** are the engine's Rust types; **`Aw*`** C-ABI names belong to the node
  crate.
- **TP** = transform probability, **NVTL** = nearest-voxel transformation likelihood.
- Each chapter ends with a **Source** note listing the in-tree files it distills, so a reader can go
  from the prose to the authoritative code.
