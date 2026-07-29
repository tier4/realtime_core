# Summary

[Introduction](introduction.md)
[How to read this book](reader-map.md)

# Part I — Concepts

- [NDT scan matching primer](concepts/ndt-primer.md)
- [Scores: TP and NVTL](concepts/scores.md)

# Part II — realtime_localization_util

- [The localization utility crate](localization_util/index.md)
    - [The TPE pose search](localization_util/tpe.md)

# Part III — realtime_kdtree

- [The kd-tree crate](kdtree/index.md)

# Part IV — realtime_port_instrument

- [The port-instrumentation crate](port_instrument/index.md)
    - [Trace-based state-machine verification](port_instrument/trace-verification.md)

# Part V — realtime_ndt_scan_matcher

- [Using the engine crate](start/using-the-crate.md)
- [Feature flags and build configurations](start/features.md)
- [The NDT engine](arch/engine.md)
    - [Engine state and the config API](arch/engine-state.md)
    - [Concurrency and interior mutability](arch/concurrency.md)
    - [MatchScratch and the align entry points](arch/scratch.md)
- [The align hot path](arch/align.md)
    - [Voxel grid and kd-tree](arch/voxel-grid.md)
    - [Serial and parallel derivatives](arch/derivatives.md)
- [Covariance estimation](arch/covariance.md)
- [Map update](arch/map-update.md)
- [Portability and the Host ports](arch/portability.md)
- [Numeric parity](port/numeric-parity.md)
- [Differential testing](port/differential.md)
- [Divergences from upstream](port/divergences.md)
- [The WCET contract](rt/wcet.md)
- [Zero-allocation guarantees](rt/zero-alloc.md)
- [The `mt` multi-core engine](rt/mt.md)
- [Panic-free, bounded execution](rt/panic-free.md)

# Part VI — The EKF localizer port

- [The Kalman filter crate](ekf/kalman-filter.md)
- [The EKF localizer crate](ekf/ekf-localizer.md)
- [FFI integration and real-time results](ekf/conformance.md)

# Appendices

- [Module index](appendix/modules.md)
