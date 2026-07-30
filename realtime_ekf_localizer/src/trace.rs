//! Trace records for the conformance harness — the 20-column `AUTOWARE_EKF_POSE_TRACE` CSV
//! format as a prefix, extended with the event kind, the full latest state block, the `P`
//! diagonal, and the z/roll/pitch scalar-filter state (see the port-equivalence contract §4).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Which filter event a trace row describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// `predict_with_delay` (one per tick).
    Predict,
    /// `measurement_update_pose` (one per drained pose measurement).
    Pose,
    /// `measurement_update_twist` (one per drained twist measurement).
    Twist,
}

impl EventKind {
    /// CSV token.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Predict => "predict",
            Self::Pose => "pose",
            Self::Twist => "twist",
        }
    }
}

/// One trace row (all comparison fields of the contract's observability matrix).
#[derive(Clone, Debug)]
pub struct TraceEvent {
    /// Event kind.
    pub kind: EventKind,
    /// Fake-clock time of the tick processing this event \[ns\].
    pub current_ns: i64,
    /// Measurement header stamp \[ns\] (0 for predict).
    pub measurement_ns: i64,
    /// Clamped measurement delay \[s\] (dt for predict).
    pub delay_s: f64,
    /// Chosen delay-table index (0 for predict).
    pub delay_step: usize,
    /// Observation: pose `[x, y, yaw]`, twist `[vx, wz, nan]`, predict all-NaN.
    pub obs: [f64; 3],
    /// Prediction at the delayed block (same layout as `obs`).
    pub pred: [f64; 3],
    /// `obs - pred` (yaw component wrapped for pose; NaN where undefined).
    pub innovation: [f64; 3],
    /// Mahalanobis distance (NaN on pre-gate rejections and predict).
    pub mahalanobis: f64,
    /// Delay gate passed.
    pub delay_gate: bool,
    /// Mahalanobis gate passed.
    pub mahalanobis_gate: bool,
    /// Update applied.
    pub accepted: bool,
    /// Latest state block after the event `[x, y, yaw, yaw_bias, vx, wz]`.
    pub state: [f64; 6],
    /// Latest `P` diagonal after the event.
    pub p_diag: [f64; 6],
    /// Scalar-filter states `[z, roll, pitch]` after the event.
    pub filters: [f64; 3],
    /// Scalar-filter variances `[z, roll, pitch]` after the event.
    pub filter_vars: [f64; 3],
}

/// The CSV header: 20-column pose-trace prefix + extension columns.
#[must_use]
pub fn csv_header() -> &'static str {
    "current_ns,measurement_ns,delay_s,delay_step,obs_x,obs_y,obs_yaw,\
     pred_x,pred_y,pred_yaw,innovation_x,innovation_y,innovation_yaw,\
     mahalanobis,delay_gate,mahalanobis_gate,accepted,post_x,post_y,post_yaw,\
     event,x0,x1,x2,x3,x4,x5,p0,p1,p2,p3,p4,p5,z,roll,pitch,z_var,roll_var,pitch_var"
}

/// Format one float like the C++ `setprecision(17)` stream (17 significant digits; `nan`/
/// `inf` tokens are parseable by the comparator on both sides).
fn fmt_float(v: f64) -> String {
    if v.is_nan() {
        String::from("nan")
    } else if v.is_infinite() {
        String::from(if v > 0.0 { "inf" } else { "-inf" })
    } else {
        // {:.16e} is 17 significant digits — exact for round-tripping f64.
        format!("{v:.16e}")
    }
}

impl TraceEvent {
    /// Serialize as one CSV row matching [`csv_header`].
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        let mut cols: Vec<String> = Vec::with_capacity(39);
        cols.push(format!("{}", self.current_ns));
        cols.push(format!("{}", self.measurement_ns));
        cols.push(fmt_float(self.delay_s));
        cols.push(format!("{}", self.delay_step));
        for v in self.obs.iter().chain(&self.pred).chain(&self.innovation) {
            cols.push(fmt_float(*v));
        }
        cols.push(fmt_float(self.mahalanobis));
        cols.push(format!("{}", i32::from(self.delay_gate)));
        cols.push(format!("{}", i32::from(self.mahalanobis_gate)));
        cols.push(format!("{}", i32::from(self.accepted)));
        // post_x, post_y, post_yaw == latest state block components.
        for v in self.state.iter().take(3) {
            cols.push(fmt_float(*v));
        }
        cols.push(String::from(self.kind.as_str()));
        for v in self
            .state
            .iter()
            .chain(&self.p_diag)
            .chain(&self.filters)
            .chain(&self.filter_vars)
        {
            cols.push(fmt_float(*v));
        }
        cols.join(",")
    }
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

    #[test]
    fn header_and_row_have_same_column_count() {
        let header_cols = csv_header().split(',').count();
        let ev = TraceEvent {
            kind: EventKind::Pose,
            current_ns: 1,
            measurement_ns: 2,
            delay_s: 0.5,
            delay_step: 3,
            obs: [1.0, 2.0, 3.0],
            pred: [1.0, 2.0, 3.0],
            innovation: [0.0, 0.0, 0.0],
            mahalanobis: 0.1,
            delay_gate: true,
            mahalanobis_gate: false,
            accepted: false,
            state: [0.0; 6],
            p_diag: [0.0; 6],
            filters: [0.0; 3],
            filter_vars: [0.0; 3],
        };
        assert_eq!(ev.to_csv_row().split(',').count(), header_cols);
        assert_eq!(header_cols, 39);
    }

    #[test]
    fn nan_and_bool_tokens() {
        let ev = TraceEvent {
            kind: EventKind::Predict,
            current_ns: 0,
            measurement_ns: 0,
            delay_s: 0.02,
            delay_step: 0,
            obs: [f64::NAN; 3],
            pred: [f64::NAN; 3],
            innovation: [f64::NAN; 3],
            mahalanobis: f64::NAN,
            delay_gate: true,
            mahalanobis_gate: true,
            accepted: true,
            state: [0.0; 6],
            p_diag: [0.0; 6],
            filters: [0.0; 3],
            filter_vars: [0.0; 3],
        };
        let row = ev.to_csv_row();
        assert!(row.contains(",nan,"));
        assert!(row.contains(",predict,"));
        assert!(row.contains(",1,1,1,"));
    }
}
