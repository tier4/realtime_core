# Summary

[Introduction](introduction.md)
[How to read this book](reader-map.md)

# Part I — Concepts

- [NDT scan matching primer](concepts/ndt-primer.md)
- [Scores: TP and NVTL](concepts/scores.md)

# Part II — Getting Started

- [Using the engine crate](start/using-the-crate.md)
- [Feature flags and build configurations](start/features.md)

# Part III — Architecture

- [The NDT engine](arch/engine.md)
    - [Engine state and the config API](arch/engine-state.md)
    - [Concurrency and interior mutability](arch/concurrency.md)
    - [MatchScratch and the align entry points](arch/scratch.md)
- [The align hot path](arch/align.md)
    - [Voxel grid and kd-tree](arch/voxel-grid.md)
    - [Serial and parallel derivatives](arch/derivatives.md)
- [Covariance estimation](arch/covariance.md)
    - [The TPE pose search](arch/tpe.md)
- [Map update](arch/map-update.md)
- [Portability and the Host ports](arch/portability.md)

# Part IV — The C++ to Rust Port

- [Numeric parity](port/numeric-parity.md)
- [Differential testing](port/differential.md)
- [Trace-based state-machine verification](port/trace-verification.md)
- [Divergences from upstream](port/divergences.md)

# Part V — Real-Time and no_std

- [The WCET contract](rt/wcet.md)
- [Zero-allocation guarantees](rt/zero-alloc.md)
- [The `mt` multi-core engine](rt/mt.md)
- [Panic-free, bounded execution](rt/panic-free.md)

# Appendices

- [Module index](appendix/modules.md)
