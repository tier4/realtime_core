# Trace-based state-machine verification

For control-flow-heavy paths — notably the align service — a byte-for-byte output diff is too brittle
and too coarse. Instead the port proves equivalence with an **abstract semantic trace**: a spec-level
state machine emitting only the fields that matter, instrumented on both sides and diffed.

## The workflow

1. Establish the C++ baseline behavior.
2. Model it as a spec-level state machine (the decision points, not the ROS plumbing).
3. Emit an abstract trace of semantic events from both C++ and Rust.
4. Differentially test the traces; triage divergences; add regression tests.

## Where the trace lives

The concrete align-service trace buffer — a semantic-event record covering the gate decision, the
diagnostic level, the search summary, and the response payload (never raw ROS message bytes) — is
part of the consuming ROS node crate, since the align service itself is. This engine crate
contributes the piece that makes those traces reproducible: the deterministic, seedable TPE sampler
(`src/tpe.rs`), so a fixed seed yields a fixed candidate sequence for the checks below.

## Split by determinism

Deterministic control flow (request validation, availability gates, initial-pose branch selection,
response packaging, and the host side-effect summary: topic/type/count) gets **exact** trace checks.
The TPE/NDT search — where libstdc++'s sampler makes exact candidate traces unstable — gets
tolerance checks on the align outcome and property/statistical checks on search quality instead (see
Verification). Tolerances are established from measured C++ baseline self-variance,
not invented up front.

> Source: `src/tpe.rs` (the deterministic, seedable sampler that makes the search reproducible).
