# MatchScratch and alignment

`MatchScratch` owns the mutable state for one alignment: a preallocated `AlignWorkspace`, the last
`AlignResult`, and the per-iteration TP/NVTL traces. Construct it from the engine limits and reuse it
across frames. `result_ref()` and `score_slices()` provide borrowed access to the completed result
without allocating.

Every engine alignment entry point requires a caller-owned `&mut MatchScratch`. This makes mutable
workspace ownership explicit, permits independent alignment on multiple threads, and keeps the
real-time path allocation-free from the first call when construction succeeds. A scratch instance
must not be shared between concurrent calls; use one instance per task or worker.

The scratch capacities are derived from `Pmax` and `Imax`. Alignment rejects a scratch whose
capacities do not cover the engine's configured limits, rather than growing it in the real-time
path. The same explicit contract applies in `std`, single-core `no_std`, and `mt` builds.

> Source: `src/engine.rs` (`MatchScratch`, `align`, and `align_outcome`).
