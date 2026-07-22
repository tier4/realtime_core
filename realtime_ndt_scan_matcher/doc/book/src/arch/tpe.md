# The TPE pose search

The align-service (initial-pose) search uses a Tree-Structured Parzen Estimator over a 6-D pose
`[x, y, z, roll, pitch, yaw]` — `tpe::TreeStructuredParzenEstimator`.

## The propose/evaluate loop

The search (driven by the consuming application) runs: draw a candidate with `get_next_input`,
build a pose, align the engine from it, then feed the outcome back with `add_trial`. It keeps the
best-scoring particle and reports the per-particle initial/result poses, scores, and iteration
counts.

## The sampler

Until `n_startup_trials` trials accumulate, TPE samples the prior: the five prior dims
(`x, y, z, roll, pitch`) from Gaussians around the initial pose (standard deviations from the request
covariance diagonal), plus `yaw` uniform over `-π, π)`. After startup it partitions trials into an
"above" (better) and "below" set and picks candidates by expected improvement from above/below
Gaussian KDEs.

## Determinism

The RNG is a Rust-owned `SplitMix64` + Box-Muller, seeded per request with a fixed seed. This
deliberately does **not** reproduce libstdc++'s `std::normal_distribution` sample sequence — that
sequence is implementation-defined and not portable — which is why exact candidate-trace equivalence
with the C++ search is out of scope (see [Divergences](../port/divergences.md)). It is deterministic
for a fixed seed, so the search is reproducible and unit-testable.

## Public API

The sampler is plain Rust — no FFI. `TreeStructuredParzenEstimator` (`get_next_input` /
`add_trial` / `new` / `with_seed`) plus the `Direction { Maximize, Minimize }`, `Error`, and `Trial`
types. The propose/evaluate search loop that drives it lives in the consuming ROS node, not this
crate.

> Source: `src/tpe.rs`.
