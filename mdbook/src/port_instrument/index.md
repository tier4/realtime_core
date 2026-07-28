# The port-instrumentation crate

`realtime_port_instrument` (`realtime_port_instrument/src/lib.rs`) holds the deterministic,
platform-independent instrumentation used to analyze and verify C++→Rust real-time ports: the
algorithmic-cost counters (`WcetCounters` — derivative passes, points, neighbors, kd nodes) behind
the `count` feature, and the cross-language FNV/SHA-256 trace ABI (`PassTrace` / `AlignTrace`)
behind the `trace` feature. The engine crate forwards its `wcet-count` / `wcet-trace` features
here and re-exports the crate as its `wcet` module; the counters are the worst-input search
fitness of [The WCET contract](../rt/wcet.md), and the trace ABI drives the next chapter.
