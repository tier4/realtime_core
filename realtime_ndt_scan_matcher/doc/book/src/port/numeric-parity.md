# Numeric parity

Keeping the Rust math bit-comparable to the C++/pcl reference, so the [differential
tests](differential.md) can hold tight tolerances.

## The f32 `Matrix4f` pipeline

The per-iteration cloud transform runs in **f32**, mirroring the C++ `Matrix4f` pipeline exactly —
including the deliberate `f32`/`usize`↔int casts that the crate's lint allowlist documents. Matching
the quantization (not "improving" it to f64) is what keeps the poses bit-comparable.

Real-map replay (22,416 İstanbul frames) exposed three f32 subtleties that **translation-only
fixtures are structurally blind to** (rotated poses are where f32 op order matters; products with
the 0/1 entries of a pure translation are exact in any order). All three are now mirrored:

- **Initial pass = guess-matrix transform.** C++ transforms the first derivative pass by the guess
  `Matrix4f` directly (`pcl::transformPointCloud(output, output, guess)`) and records the guess in
  `transformation_array[0]`; only subsequent passes rebuild from the Euler 6-vector.
  `R_rebuilt(euler(R)) != R` in f32 — rebuilding on the first pass displaced points by
  ~ULP(coordinate), millimetres at 60 km map coordinates.
- **pcl's transform association.** pcl 1.12's SSE `Transformer<float>::se3` computes
  `x·c0 + (y·c1 + (z·c2 + t))` — right-associated, translation innermost.
  `transform_cloud_by_matrix` keeps that association (a left-associated `r·v + t` rounds
  differently for rotated poses).
- **Eigen's `AngleAxisf` diagonal.** `toRotationMatrix` computes the on-axis diagonal as
  `(1 − c) + c`, which is not always exactly `1.0f`; `se3_matrix_f32` reproduces it.

## Scalar math: `libm`, and where it can differ from glibc

`libm` provides `sqrt`/trig/`exp` in both std and `no_std`. IEEE `sqrt` is correctly rounded, so it
matches `std::sqrt` bit-for-bit; the Gauss constants (`gauss_constants` from `outlier_ratio` /
`resolution`) are computed identically. The transcendentals are **not** correctly rounded in either
library: `libm`'s `sinf`/`cosf` disagree with glibc's on ~1.25 % of arguments (both ≤ 1 ULP,
differently rounded; measured), and f64 `exp` differs at ULP level. `libm` is kept deliberately —
build/ISA determinism of the engine outweighs the last ULP of host parity. Measured consequence on
22,416 real frames: **99.9 % identical iteration counts**; the 0.12 % residual is ±1-iteration
flips on knife-edge frames driven by sub-f32-ULP f64-trajectory differences (`libm`-vs-glibc `exp`
plus the nalgebra-vs-Eigen eigensolver at ≤ 2e-15 relative on inverse covariances). The
comparison harness certifies equal work **per frame** (`iteration_num` assertion), so residual
frames are flagged, never silently compared.

## Serial/parallel bit-identity

The parallel derivative reduction is order-preserving, so it is bit-for-bit identical to serial (see
[Serial and parallel derivatives](../arch/derivatives.md)) — parallelism never perturbs parity.

## The Hessian

The NDT angle Hessian (`h_ang`) is the exact analytic second derivative since the PR #1217 "d1"
pitch² sign fix; the full 6×6 Hessian now finite-difference-validates. Historically this was the one
place the port intentionally reproduced a C++ quirk — see [Divergences from upstream](divergences.md).

## Tolerances

The differential checks use: pose translation ≤ 1e-3 m, rotation ≤ 1e-3 rad, TP / NVTL ≤ 1e-4,
iteration count **exact**. On the real-map replay, iteration equality holds on 99.9 % of frames
(see above); synthetic fixtures hold it on 100 % — and because they are translation-only, they
would not have caught the rotated-pose subtleties. Differential fixtures for parity work should
include **rotated** guesses at **large map coordinates**.

> Source: `src/transform.rs`, `src/ndt.rs`, `src/derivatives.rs`.
