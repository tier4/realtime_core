//! Expression-exact ports of the tf2 quaternion/Euler helpers the C++ EKF calls
//! (`tf2::getYaw`, `tf2::Matrix3x3::getRPY`, `tf2::Quaternion::setRPY`/`setRotation`,
//! quaternion multiply/normalize). These deliberately reproduce tf2's formulas — including the
//! `sarg` gimbal thresholds — rather than reusing a generic ZYX conversion, so the port's yaw
//! extraction agrees with the C++ bit-for-bit up to libm rounding.
//!
//! Quaternions are `[x, y, z, w]` (the `geometry_msgs` field order).

use crate::msg::Quaternion;

/// `tf2::getYaw` (`tf2/impl/utils.hpp`): yaw of a (not necessarily normalized) quaternion,
/// with the urdfdom-derived normalization and the `sarg` gimbal branches.
#[expect(
    clippy::many_single_char_names,
    reason = "tf2 component naming (x, y, z, w)"
)]
#[must_use]
pub fn get_yaw(q: &Quaternion) -> f64 {
    let [x, y, z, w] = *q;
    let sqx = x * x;
    let sqy = y * y;
    let sqz = z * z;
    let sqw = w * w;

    let sarg = -2.0 * (x * z - w * y) / (sqx + sqy + sqz + sqw);

    if sarg <= -0.99999 {
        -2.0 * libm::atan2(y, x)
    } else if sarg >= 0.99999 {
        2.0 * libm::atan2(y, x)
    } else {
        libm::atan2(2.0 * (x * y + w * z), sqw + sqx - sqy - sqz)
    }
}

/// `tf2::Matrix3x3(q).getRPY(roll, pitch, yaw)` solution 1: builds the rotation matrix with
/// `Matrix3x3::setRotation` (which normalizes by `2 / |q|²`) and extracts Euler angles with
/// `getEulerYPR`. Returns `(roll, pitch, yaw)`.
#[expect(
    clippy::many_single_char_names,
    reason = "tf2 component naming (x, y, z, w, s, d)"
)]
#[must_use]
pub fn get_rpy(q: &Quaternion) -> (f64, f64, f64) {
    let [x, y, z, w] = *q;

    // Matrix3x3::setRotation.
    let d = x * x + y * y + z * z + w * w;
    let s = 2.0 / d;
    let (xs, ys, zs) = (x * s, y * s, z * s);
    let (wx, wy, wz) = (w * xs, w * ys, w * zs);
    let (xx, xy, xz) = (x * xs, x * ys, x * zs);
    let (yy, yz, zz) = (y * ys, y * zs, z * zs);

    let m00 = 1.0 - (yy + zz);
    let m10 = xy + wz;
    let m20 = xz - wy;
    let m21 = yz + wx;
    let m22 = 1.0 - (xx + yy);

    // Matrix3x3::getEulerYPR, solution 1 (m_el[2].x() == m20, .y() == m21, .z() == m22).
    if libm::fabs(m20) >= 1.0 {
        let yaw = 0.0;
        let delta = libm::atan2(m21, m22);
        if m20 < 0.0 {
            // gimbal locked down
            (delta, core::f64::consts::FRAC_PI_2, yaw)
        } else {
            // gimbal locked up
            (delta, -core::f64::consts::FRAC_PI_2, yaw)
        }
    } else {
        let pitch = -libm::asin(m20);
        let cp = libm::cos(pitch);
        let roll = libm::atan2(m21 / cp, m22 / cp);
        let yaw = libm::atan2(m10 / cp, m00 / cp);
        (roll, pitch, yaw)
    }
}

/// `tf2::Quaternion::setRPY` (used by `create_quaternion_from_rpy`).
#[must_use]
pub fn quaternion_from_rpy(roll: f64, pitch: f64, yaw: f64) -> Quaternion {
    let half_yaw = yaw * 0.5;
    let half_pitch = pitch * 0.5;
    let half_roll = roll * 0.5;
    let cos_yaw = libm::cos(half_yaw);
    let sin_yaw = libm::sin(half_yaw);
    let cos_pitch = libm::cos(half_pitch);
    let sin_pitch = libm::sin(half_pitch);
    let cos_roll = libm::cos(half_roll);
    let sin_roll = libm::sin(half_roll);
    [
        sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
        cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
        cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
        cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
    ]
}

/// `tf2::Quaternion::setRotation(axis, angle)`: `s = sin(angle/2) / |axis|`.
#[must_use]
pub fn quaternion_from_axis_angle(axis: &[f64; 3], angle: f64) -> Quaternion {
    let d = vector3_length(axis);
    let s = libm::sin(angle * 0.5) / d;
    let [ax, ay, az] = *axis;
    [ax * s, ay * s, az * s, libm::cos(angle * 0.5)]
}

/// `tf2::Quaternion::operator*` — Hamilton product, tf2 component order.
#[must_use]
pub fn quaternion_multiply(a: &Quaternion, b: &Quaternion) -> Quaternion {
    let [x1, y1, z1, w1] = *a;
    let [x2, y2, z2, w2] = *b;
    [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
}

/// `tf2::Quaternion::normalize` — divide by `sqrt(dot)`.
#[expect(
    clippy::many_single_char_names,
    reason = "tf2 component naming (x, y, z, w)"
)]
#[must_use]
pub fn quaternion_normalize(q: &Quaternion) -> Quaternion {
    let [x, y, z, w] = *q;
    let len = libm::sqrt(x * x + y * y + z * z + w * w);
    [x / len, y / len, z / len, w / len]
}

/// `tf2::Vector3::length`.
#[must_use]
pub fn vector3_length(v: &[f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// `tf2::Vector3::normalized` — `v / |v|`.
#[must_use]
pub fn vector3_normalized(v: &[f64; 3]) -> [f64; 3] {
    let len = vector3_length(v);
    [v[0] / len, v[1] / len, v[2] / len]
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn yaw_roundtrip() {
        for &yaw in &[0.0, 0.3, -0.3, PI / 2.0, -PI / 2.0, 3.0, -3.0] {
            let q = quaternion_from_rpy(0.0, 0.0, yaw);
            assert!((get_yaw(&q) - yaw).abs() < 1e-12, "yaw {yaw}");
        }
    }

    #[test]
    fn rpy_roundtrip() {
        let (roll, pitch, yaw) = (0.1, -0.2, 0.7);
        let q = quaternion_from_rpy(roll, pitch, yaw);
        let (r, p, y) = get_rpy(&q);
        assert!((r - roll).abs() < 1e-12);
        assert!((p - pitch).abs() < 1e-12);
        assert!((y - yaw).abs() < 1e-12);
    }

    #[test]
    fn multiply_and_normalize() {
        let a = quaternion_from_rpy(0.0, 0.0, 0.3);
        let b = quaternion_from_rpy(0.0, 0.0, 0.4);
        let c = quaternion_normalize(&quaternion_multiply(&a, &b));
        assert!((get_yaw(&c) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn axis_angle_about_z() {
        let q = quaternion_from_axis_angle(&[0.0, 0.0, 1.0], 0.5);
        assert!((get_yaw(&q) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gimbal_branches() {
        // pitch = +pi/2 -> sarg >= 0.99999 branch in get_yaw, |m20| >= 1 branch in get_rpy.
        let q = quaternion_from_rpy(0.0, PI / 2.0, 0.0);
        let (_, p, _) = get_rpy(&q);
        assert!((p.abs() - PI / 2.0).abs() < 1e-6);
        let yaw_at_gimbal = get_yaw(&q); // value defined by the tf2 branch
        assert!(yaw_at_gimbal.is_finite());
    }
}
