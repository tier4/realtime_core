# Divergences from upstream

Porting diffs the C++ and Rust implementations directly, which makes it the best opportunity to
find **upstream bugs**. The port's policy is deliberate and may be surprising: when the C++
reference diverges from expected-correct behavior, the Rust port **reproduces the C++ behavior**
rather than fixing it locally.

## Why reproduce, not fix

The differential test against C++ is the oracle (see *Behavior equivalence and verification* in the node crate book). If the Rust
port silently "corrected" a C++ bug, the differential test would go red and we would lose the
ability to prove equivalence. So the correct fix belongs **upstream** (in pcl / Autoware), and the
port keeps the C++ behavior verbatim until that upstream fix lands.

## The process for a discovered divergence

1. **Notify the user immediately** — what / where / evidence / impact. Never silently absorb it.
2. **Record it** in this chapter's divergence list — one entry per finding: location · type ·
   evidence · correct value · impact · decision · revisit trigger · upstream link · verification.
3. **Reproduce, don't fix locally.** Mark the site in code (`PORT-QUIRK`) and pin it with a test,
   so an accidental "fix" fails loudly.
4. **Fix upstream.** File/draft an upstream issue and link it.

## Follow the fix when upstream lands

Reproduce-not-fix is temporary per finding. Once the upstream fix lands, the port **follows it**:
update the Rust to match the corrected C++, widen or re-point the pinning test to the corrected
value, and move the entry to *Resolved* below. Following the fix restores `ON == OFF` (both now
compute the corrected value).

## Resolved

- **NDT `h_ang` "d1" sign bug — fixed upstream (PR #1217), port followed.**
  `computeAngleDerivatives()` built the angular Hessian table `h_ang` with the pitch² x-coefficient
  as `+sy`, where the exact second derivative is `-sy` (row 6, "d1"). It affected only the Newton
  search direction, not the optimum. The port originally reproduced `+sy` to stay bit-identical with
  the then-buggy C++; when the upstream fix landed the port followed it (`src/derivatives.rs`, now
  `-sy`). With the sign corrected the pcl NDT Hessian is the **exact** analytic Hessian, so the full
  6×6 Hessian now finite-difference-validates — the FD tests
  (`update_derivatives_gradient_and_hessian_match_finite_difference` in `src/derivatives.rs`,
  `compute_derivatives_matches_finite_difference` in `src/ndt.rs`) cover all six rows and pin the
  corrected sign directly (they fail on `+sy`).

- **Step-length clamp panic — a port bug (not upstream), fixed 2026-07-10.**
  The default-path step length port used `f64::clamp(step_min, step_size)`, which **panics** when a
  misconfigured `trans_epsilon / 2 > step_size` makes `min > max`. The C++ (`computeStepLengthMT`,
  `multigrid_ndt_omp_impl.hpp:953-955`) applies `std::min(a_t, step_max)` **then**
  `std::max(a_t, step_min)` — never panicking, yielding `step_min` in that case. The port now
  mirrors the exact min-then-max order (`src/ndt.rs`; Rust `f64::min`/`max` also match
  `std::min`/`max` on single-NaN operands). Found by the 2026-07-10 numeric-hazard scan; pinned by
  `align_degenerate_step_bounds_do_not_panic`.

## Degenerate-domain guards (Rust-only)

Per the roadmap's Rule 3 (`plan/ndt_pr.md`), the port adds **branch-only guards** at sites the C++
leaves unguarded. Each fires **only on the degenerate domain** (where C++ produces NaN/Inf/garbage),
so the valid-domain differential tests are unaffected. Added by the 2026-07-10 hazard scan:

- **Non-finite result pose → non-converged** (`src/engine.rs`, `run_align`): the
  convergence verdict is forced to `is_converged = false` when the align result pose is not finite,
  so no consumer publishes NaN/Inf downstream (the node gates every pose/TF publish on
  `is_converged`). Pinned by `run_align_gates_non_finite_pose_to_not_converged`.
- **`asin` domain clamp** (`src/transform.rs`, `matrix_to_euler`): the sine argument is
  clamped to `[-1, 1]` — FP error near gimbal can push `|m02|` past 1, where `asin` returns NaN
  (C++ uses the atan2-based Eigen `eulerAngles`, which cannot NaN). Pinned by
  `matrix_to_euler_clamps_asin_overshoot`.
- **Gauss-constants config clamp** (`src/transform.rs`, `gauss_constants`): degenerate
  configs (`resolution <= 0`, `outlier_ratio` outside `(0, 1)`, non-finite) are clamped into the
  valid domain; the C++ (`computeTransformation` lines 229-233) yields `inf`/`log(0)`/NaN there,
  poisoning every score. Pinned by `gauss_constants_degenerate_configs_stay_finite` (+ a
  bit-identical valid-domain test).
- **Softmax temperature guard** (`src/covariance.rs`, `calc_weight_vec`): non-positive or
  non-finite temperature falls back to uniform `1/n` weights; C++
  (`estimate_covariance.cpp:135`) divides unguarded. Pinned by
  `calc_weight_vec_degenerate_temperature_is_uniform`.

(Further degenerate-input guards on the sensor-input and align-service paths live in the consuming
ROS node crate and are documented there, not here.)

## Intentional differences

**Transcendental ULPs: `libm` kept over platform parity.** The engine's `sinf`/`cosf`/`exp` come
from the pure-Rust `libm` crate for build/ISA determinism; glibc rounds ~1.25 % of f32 trig
arguments differently (both ≤ 1 ULP correct). Together with the nalgebra-vs-Eigen eigensolver's
≤ 2e-15 inverse-covariance noise, this leaves an irreducible ±1-iteration divergence on 0.12 % of
real-map frames (28 of 22,416 measured; knife-edge convergence tests). Adopting platform trig was
measured to recover exactly one frame and was rejected — the per-frame `iteration_num`
certification in the comparison harness handles the residual. See
[Numeric parity](numeric-parity.md) for the fixed (non-intentional) rotated-pose findings.

The align-service TPE uses a deterministic Rust-owned sampler instead of libstdc++'s
implementation-defined `std::normal_distribution` sequence, because that sequence is not portable
(this is why exact candidate-trace equivalence for the TPE search is out of scope; see
Verification).

**Gimbal-lock RPY interpolation is kept C++-identical (an intentional non-divergence).** The pose
buffer's quaternion↔RPY conversions (`src/pose_buffer.rs`) share tf2 `getRPY`'s singularity
at pitch ±90°: the roll/yaw split is ill-conditioned there, giving a possibly-degenerate (but
always finite, never panicking) interpolated orientation. Guarding it would change behavior where
the C++ is equally degenerate, so parity wins.

> Source: `src/derivatives.rs` (the `h_ang` table + FD tests); upstream issue/PR links.
