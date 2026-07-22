//! SE3 transforms and the NDT Gaussian fitting constants, ported from
//! `multigrid_ndt_omp_impl.hpp`. The 6-vector parameterization is `p = [tx, ty, tz, roll, pitch,
//! yaw]` and the rotation is `Rx(roll) * Ry(pitch) * Rz(yaw)` (the C++ `AngleAxis` order at
//! `computeTransformation` lines 332-339; the same intrinsic XYZ order Eigen's
//! `eulerAngles(0, 1, 2)` produces at line 283). All math is `f64`; `no_std` via `libm`.

// Numeric kernel. Suppressions are scoped per-function (no module-wide `#![allow]`): nalgebra f64
// matrix/scalar operators are float math that cannot integer-overflow (arithmetic_side_effects);
// indexing is into fixed-size vectors/matrices with constant indices (indexing_slicing); the f32
// cloud transform mirrors the C++ `Matrix4f` pipeline, so the f64->f32 narrowing is intentional
// (as_conversions/cast_*); x/y/z are geometry names (many_single_char_names).

use alloc::vec::Vec;

use nalgebra::{Matrix3, Matrix4, Vector3, Vector6};

/// The Gaussian fitting constants `d1`, `d2`, `d3` (Magnusson 2009 eq. 6.8), recomputed per align
/// from the outlier ratio and voxel resolution (`computeTransformation` lines 256-260).
#[derive(Clone, Copy, Debug)]
pub struct GaussConstants {
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

/// Compute the Gaussian fitting constants. `outlier_ratio` defaults to 0.55 in Autoware;
/// `resolution` is the voxel size (also the neighbor search radius).
///
/// Degenerate configs (`resolution <= 0`, `outlier_ratio` outside `(0, 1)`, or non-finite inputs)
/// are clamped into the open valid domain before the C++-mirrored math runs, so `d1`/`d2`/`d3` are
/// always finite (they feed every per-point score `exp`). The C++ (`computeTransformation` lines
/// 229-233) is unguarded there: `resolution == 0` → `inf`, `outlier_ratio ∈ {0, 1}` → `log(0)`/NaN.
/// Documented divergence — see `doc/book/src/port/divergences.md`; valid configs are untouched.
#[must_use]
pub fn gauss_constants(outlier_ratio: f64, resolution: f64) -> GaussConstants {
    // Branch-only degenerate-config clamps (no effect on the valid domain; RT-safe).
    const RATIO_EPS: f64 = 1e-6;
    let outlier_ratio = if outlier_ratio.is_finite() {
        outlier_ratio.clamp(RATIO_EPS, 1.0 - RATIO_EPS)
    } else {
        0.55 // Autoware default when the config is NaN/Inf.
    };
    let resolution = if resolution.is_finite() && resolution > 0.0 {
        resolution
    } else {
        1.0 // safe positive fallback for a zero/negative/non-finite voxel size
    };
    let gauss_c1 = 10.0 * (1.0 - outlier_ratio);
    let gauss_c2 = outlier_ratio / (resolution * resolution * resolution);
    let d3 = -libm::log(gauss_c2);
    let d1 = -libm::log(gauss_c1 + gauss_c2) - d3;
    let d2 = -2.0 * libm::log((-libm::log(gauss_c1 * libm::exp(-0.5) + gauss_c2) - d3) / d1);
    GaussConstants { d1, d2, d3 }
}

/// `Rx(roll) * Ry(pitch) * Rz(yaw)` — the 3x3 rotation for the angle triple `(roll, pitch, yaw)`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix product"
)]
fn rotation(roll: f64, pitch: f64, yaw: f64) -> Matrix3<f64> {
    let (sx, cx) = (libm::sin(roll), libm::cos(roll));
    let (sy, cy) = (libm::sin(pitch), libm::cos(pitch));
    let (sz, cz) = (libm::sin(yaw), libm::cos(yaw));

    let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx);
    let ry = Matrix3::new(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
    let rz = Matrix3::new(cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0);
    rx * ry * rz
}

/// Build the 4x4 affine transform `Translation(tx,ty,tz) * Rx(roll) * Ry(pitch) * Rz(yaw)` from the
/// 6-vector `p` (matches `computeTransformation` lines 332-339).
#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "constant indices into fixed-size nalgebra Vector6/Matrix4"
)]
#[must_use]
pub fn euler_to_matrix(p: &Vector6<f64>) -> Matrix4<f64> {
    let r = rotation(p[3], p[4], p[5]);
    let mut m = Matrix4::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    m[(0, 3)] = p[0];
    m[(1, 3)] = p[1];
    m[(2, 3)] = p[2];
    m
}

/// Extract `p = [tx, ty, tz, roll, pitch, yaw]` from a 4x4 affine transform — the inverse of
/// [`euler_to_matrix`] for `pitch` in `(-pi/2, pi/2)`.
///
/// NOTE: this is *a* valid inverse (the intrinsic-XYZ extraction), sufficient for round-trip tests.
/// Initial-guess extraction must reconcile the range convention with Eigen's
/// `eulerAngles(0, 1, 2)` (line 283), which folds angles differently for large rotations.
#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "constant indices into a fixed-size nalgebra Matrix4"
)]
#[must_use]
pub fn matrix_to_euler(m: &Matrix4<f64>) -> Vector6<f64> {
    // For R = Rx(a)*Ry(b)*Rz(c): R[0][2] = sin(b); R[1][2] = -sin(a)cos(b); R[2][2] = cos(a)cos(b);
    // R[0][1] = -cos(b)sin(c); R[0][0] = cos(b)cos(c).
    // Guard: FP error can push |m02| of a near-gimbal rotation just past 1, where asin returns NaN
    // (C++ uses the atan2-based Eigen eulerAngles, which cannot NaN). Clamping to the asin domain
    // only fires there — a documented divergence (doc/book/src/port/divergences.md).
    let pitch = libm::asin(m[(0, 2)].clamp(-1.0, 1.0));
    let roll = libm::atan2(-m[(1, 2)], m[(2, 2)]);
    let yaw = libm::atan2(-m[(0, 1)], m[(0, 0)]);
    Vector6::new(m[(0, 3)], m[(1, 3)], m[(2, 3)], roll, pitch, yaw)
}

/// Apply the transform of `p` to a point: `R(p) * x + t(p)`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix math; constant indices into fixed-size Vector6"
)]
#[must_use]
pub fn transform_point(p: &Vector6<f64>, x: &Vector3<f64>) -> Vector3<f64> {
    let r = rotation(p[3], p[4], p[5]);
    (r * x) + Vector3::new(p[0], p[1], p[2])
}

/// The 4x4 affine of `p` built in **f32** — the C++ `final_transformation_` (`Matrix4f`):
/// `Translation(f32) * AngleAxis<float>` for roll/pitch/yaw. The angles are narrowed to f32 and the
/// trig is f32, matching the C++ float pipeline so the transformed cloud agrees bit-closely.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::allow_attributes,
    reason = "C++ Matrix4f parity: deliberate f64->f32 narrowing; nalgebra math; fixed-size indexing"
)]
#[must_use]
pub fn se3_matrix_f32(p: &Vector6<f64>) -> Matrix4<f32> {
    // NOTE on trig choice: Eigen's AngleAxisf uses the PLATFORM sinf/cosf (glibc here), which
    // disagrees with the pure-Rust `libm` crate on ~1.25 % of arguments (both ≤1-ULP-correct,
    // differently rounded; measured on this toolchain). We deliberately KEEP `libm` — build/ISA
    // determinism of the engine outweighs the last ULP of host C++ parity (measured impact:
    // 1 extra ±1-iteration flip in 22,416 real frames; see the porting notes' divergence
    // finding). The wcet-count counters confirm the compared per-frame work counts.
    let (sx, cx) = (libm::sinf(p[3] as f32), libm::cosf(p[3] as f32));
    let (sy, cy) = (libm::sinf(p[4] as f32), libm::cosf(p[4] as f32));
    let (sz, cz) = (libm::sinf(p[5] as f32), libm::cosf(p[5] as f32));
    // C++ BIT-PARITY: `convertTransform` composes Eigen `AngleAxisf` rotations, whose
    // `toRotationMatrix` computes the on-axis diagonal entry as `(1 - c) + c` — NOT exactly 1.0
    // in f32 for every angle (Eigen/src/Geometry/AngleAxis.h: `diagonal = cos1_axis∘axis + c`).
    // Reproduce that value exactly; everything else of the axis matrices is identical to the
    // sparse textbook form (products with the 0/1 axis components are exact).
    let dx = (1.0_f32 - cx) + cx;
    let dy = (1.0_f32 - cy) + cy;
    let dz = (1.0_f32 - cz) + cz;
    let rx = Matrix3::<f32>::new(dx, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx);
    let ry = Matrix3::<f32>::new(cy, 0.0, sy, 0.0, dy, 0.0, -sy, 0.0, cy);
    let rz = Matrix3::<f32>::new(cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, dz);
    let r = rx * ry * rz;
    let mut m = Matrix4::<f32>::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    m[(0, 3)] = p[0] as f32;
    m[(1, 3)] = p[1] as f32;
    m[(2, 3)] = p[2] as f32;
    m
}

/// Transform `source` by a 4x4 affine `m` into the reused buffer `out`, in **f32** — mirrors the C++
/// `pcl::transformPointCloud(source, out, m)` (`R·x + t` from the matrix directly). Clears + reserves
/// `out` (capacity is retained across calls), so after the first call this performs no allocation.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::allow_attributes,
    reason = "nalgebra f32 matrix math; constant indices into a fixed-size Matrix4; m/r/t/v locals"
)]
pub fn transform_cloud_by_matrix(m: &Matrix4<f32>, source: &[[f32; 3]], out: &mut Vec<[f32; 3]>) {
    out.clear();
    out.reserve(source.len()); // len == 0 after clear, so this reserves the full size (no-op once warm)
    // C++ BIT-PARITY: pcl 1.12's SSE `Transformer<float>::se3` computes, per component,
    // `x·c0 + (y·c1 + (z·c2 + t))` — RIGHT-associated with the translation innermost
    // (pcl/common/impl/transforms.hpp:124-129). A left-associated `r·v + t` rounds differently
    // for ROTATED poses (translation-only poses are exact either way, which is why the
    // synthetic differential suites never exposed the difference; see the porting notes'
    // divergence finding). Keep this association exactly.
    for &[sx, sy, sz] in source {
        let v = [
            sx * m[(0, 0)] + (sy * m[(0, 1)] + (sz * m[(0, 2)] + m[(0, 3)])),
            sx * m[(1, 0)] + (sy * m[(1, 1)] + (sz * m[(1, 2)] + m[(1, 3)])),
            sx * m[(2, 0)] + (sy * m[(2, 1)] + (sz * m[(2, 2)] + m[(2, 3)])),
        ];
        out.push(v);
    }
}

/// Transform `source` by the pose of `p` into the reused buffer `out`, in **f32** — mirrors the C++
/// `pcl::transformPointCloud(source, trans_cloud, final_transformation_)`. Builds the `Matrix4f`
/// (C++ parity) and delegates to [`transform_cloud_by_matrix`].
pub fn transform_cloud_f32(p: &Vector6<f64>, source: &[[f32; 3]], out: &mut Vec<[f32; 3]>) {
    transform_cloud_by_matrix(&se3_matrix_f32(p), source, out);
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::unreadable_literal,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn gauss_constants_degenerate_configs_stay_finite() {
        // Degenerate configs (C++-unguarded): each must yield finite d1/d2/d3.
        for (outlier_ratio, resolution) in [
            (0.0, 2.0),
            (1.0, 2.0),
            (0.55, 0.0),
            (0.55, -1.0),
            (f64::NAN, 2.0),
            (0.55, f64::INFINITY),
        ] {
            let g = gauss_constants(outlier_ratio, resolution);
            assert!(
                g.d1.is_finite() && g.d2.is_finite() && g.d3.is_finite(),
                "({outlier_ratio}, {resolution}) -> {g:?}"
            );
        }
    }

    #[test]
    fn gauss_constants_valid_config_matches_raw_formula() {
        // The guard must not perturb the valid domain: bit-identical to the raw C++ math.
        let g = gauss_constants(0.55, 2.0);
        let c1 = 10.0 * (1.0 - 0.55);
        let c2 = 0.55 / (2.0 * 2.0 * 2.0);
        let d3 = -libm::log(c2);
        let d1 = -libm::log(c1 + c2) - d3;
        let d2 = -2.0 * libm::log((-libm::log(c1 * libm::exp(-0.5) + c2) - d3) / d1);
        assert_eq!(g.d1, d1);
        assert_eq!(g.d2, d2);
        assert_eq!(g.d3, d3);
    }

    #[test]
    fn matrix_to_euler_clamps_asin_overshoot() {
        // FP error can push |m02| past 1; the clamp keeps pitch finite (±π/2 at the boundary).
        let mut m = Matrix4::<f64>::identity();
        m[(0, 2)] = 1.0 + 1e-9;
        let p = matrix_to_euler(&m);
        assert!(p[4].is_finite());
        assert_eq!(p[4], core::f64::consts::FRAC_PI_2);
        m[(0, 2)] = -1.0 - 1e-9;
        let p = matrix_to_euler(&m);
        assert_eq!(p[4], -core::f64::consts::FRAC_PI_2);
    }

    // Oracle: an independent recomputation of eq. 6.8 using std transcendentals (the impl uses
    // libm), plus a hand-anchored magnitude for the default (0.55, 1.0) case.
    #[test]
    fn gauss_constants_matches_reference_formula() {
        let reference = |outlier: f64, res: f64| {
            let c1 = 10.0 * (1.0 - outlier);
            let c2 = outlier / res.powi(3);
            let d3 = -c2.ln();
            let d1 = -(c1 + c2).ln() - d3;
            let d2 = -2.0 * ((-(c1 * (-0.5f64).exp() + c2).ln() - d3) / d1).ln();
            (d1, d2, d3)
        };
        for (o, r) in [(0.55, 1.0), (0.55, 2.0), (0.4, 1.5)] {
            let g = gauss_constants(o, r);
            let (d1, d2, d3) = reference(o, r);
            assert!(close(g.d1, d1, 1e-12), "d1={} ref={d1}", g.d1);
            assert!(close(g.d2, d2, 1e-12), "d2={} ref={d2}", g.d2);
            assert!(close(g.d3, d3, 1e-12), "d3={} ref={d3}", g.d3);
        }
        // Hand-anchored magnitude (catches an identical-transcription error in both impl + ref).
        let g = gauss_constants(0.55, 1.0);
        assert!(close(g.d1, -2.2172252, 1e-6), "d1={}", g.d1);
        assert!(close(g.d3, -(0.55_f64.ln()), 1e-12), "d3={}", g.d3);
    }

    // Oracle: a hand-built Rx*Ry*Rz product and a manual R*x + t.
    #[test]
    fn euler_matrix_and_transform_point() {
        let p = Vector6::new(1.0, 2.0, 3.0, 0.2, -0.3, 0.4);
        let (sx, cx) = (0.2_f64.sin(), 0.2_f64.cos());
        let (sy, cy) = ((-0.3_f64).sin(), (-0.3_f64).cos());
        let (sz, cz) = (0.4_f64.sin(), 0.4_f64.cos());
        let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx);
        let ry = Matrix3::new(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
        let rz = Matrix3::new(cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0);
        let r = rx * ry * rz;

        let m = euler_to_matrix(&p);
        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    close(m[(row, col)], r[(row, col)], 1e-12),
                    "R[{row}][{col}]"
                );
            }
        }
        assert!(
            close(m[(0, 3)], 1.0, 0.0) && close(m[(1, 3)], 2.0, 0.0) && close(m[(2, 3)], 3.0, 0.0)
        );

        let x = Vector3::new(0.5, -1.5, 2.0);
        let manual = (r * x) + Vector3::new(1.0, 2.0, 3.0);
        let got = transform_point(&p, &x);
        for d in 0..3 {
            assert!(close(got[d], manual[d], 1e-12), "tp[{d}]");
        }
    }

    // matrix_to_euler must invert euler_to_matrix for non-gimbal angles (round-trip oracle).
    #[test]
    fn euler_round_trips_for_small_angles() {
        let p = Vector6::new(-4.0, 0.5, 7.0, 0.3, -0.6, 1.1);
        let back = matrix_to_euler(&euler_to_matrix(&p));
        for d in 0..6 {
            assert!(close(back[d], p[d], 1e-9), "p[{d}] {} vs {}", back[d], p[d]);
        }
    }
}
