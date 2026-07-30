//! Rust side of the EKF differential-conformance harness: reads the same line-oriented
//! scenario grammar as the C++ `ekf_replay` tool (fork `tools/ekf_replay.cpp`) and writes the
//! same extended trace CSV (contract §4).
//!
//! Usage: `ekf_replay <scenario_file> <out.csv>`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    reason = "test code (conformance harness example)"
)]

use std::env;
use std::fs;
use std::io::Write as _;
use std::process::ExitCode;

use realtime_ekf_localizer::hyper_parameters::HyperParameters;
use realtime_ekf_localizer::msg::{PoseWithCovariance, Transform, TwistWithCovariance};
use realtime_ekf_localizer::tick::EkfLocalizerCore;
use realtime_ekf_localizer::trace::{TraceEvent, csv_header};

fn apply_param(p: &mut HyperParameters, key: &str, value: f64) -> Result<(), String> {
    match key {
        "predict_frequency" => {
            p.ekf_rate = value;
            p.ekf_dt = 1.0 / p.ekf_rate.max(0.1);
        }
        "enable_yaw_bias_estimation" => p.enable_yaw_bias_estimation = value != 0.0,
        "extend_state_step" => p.extend_state_step = value as usize,
        "pose_additional_delay" => p.pose_additional_delay = value,
        "pose_gate_dist" => p.pose_gate_dist = value,
        "pose_smoothing_steps" => p.pose_smoothing_steps = value as usize,
        "max_pose_queue_size" => p.max_pose_queue_size = value as usize,
        "twist_additional_delay" => p.twist_additional_delay = value,
        "twist_gate_dist" => p.twist_gate_dist = value,
        "twist_smoothing_steps" => p.twist_smoothing_steps = value as usize,
        "max_twist_queue_size" => p.max_twist_queue_size = value as usize,
        "proc_stddev_vx_c" => p.proc_stddev_vx_c = value,
        "proc_stddev_wz_c" => p.proc_stddev_wz_c = value,
        "proc_stddev_yaw_c" => p.proc_stddev_yaw_c = value,
        "z_filter_proc_dev" => p.z_filter_proc_dev = value,
        "roll_filter_proc_dev" => p.roll_filter_proc_dev = value,
        "pitch_filter_proc_dev" => p.pitch_filter_proc_dev = value,
        "threshold_observable_velocity_mps" => p.threshold_observable_velocity_mps = value,
        _ => return Err(format!("unknown param '{key}'")),
    }
    Ok(())
}

fn parse_floats(fields: &[&str]) -> Vec<f64> {
    fields.iter().map(|f| f.parse::<f64>().unwrap()).collect()
}

fn parse_pose(fields: &[&str]) -> PoseWithCovariance {
    let stamp_ns: i64 = fields[0].parse().unwrap();
    let v = parse_floats(&fields[1..]);
    assert_eq!(v.len(), 7 + 36, "pose needs x y z qx qy qz qw + 36 cov");
    let mut covariance = [0.0_f64; 36];
    covariance.copy_from_slice(&v[7..]);
    PoseWithCovariance {
        stamp_ns,
        position: [v[0], v[1], v[2]],
        orientation: [v[3], v[4], v[5], v[6]],
        covariance,
    }
}

fn parse_twist(fields: &[&str]) -> TwistWithCovariance {
    let stamp_ns: i64 = fields[0].parse().unwrap();
    let v = parse_floats(&fields[1..]);
    assert_eq!(v.len(), 6 + 36, "twist needs vx vy vz wx wy wz + 36 cov");
    let mut covariance = [0.0_f64; 36];
    covariance.copy_from_slice(&v[6..]);
    TwistWithCovariance {
        stamp_ns,
        linear: [v[0], v[1], v[2]],
        angular: [v[3], v[4], v[5]],
        covariance,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: ekf_replay <scenario_file> <out.csv>");
        return ExitCode::from(2);
    }
    let scenario = fs::read_to_string(&args[1]).expect("cannot read scenario");
    let mut out = fs::File::create(&args[2]).expect("cannot create output");

    let mut params = HyperParameters::default();
    let mut core: Option<EkfLocalizerCore> = None;
    let mut trace: Vec<TraceEvent> = Vec::new();

    for (line_no, line) in scenario.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields[0] {
            "param" => {
                assert!(core.is_none(), "param after init (line {})", line_no + 1);
                apply_param(&mut params, fields[1], fields[2].parse().unwrap()).unwrap();
            }
            "init" => {
                let pose = parse_pose(&fields[1..]);
                let c = core.get_or_insert_with(|| EkfLocalizerCore::new(params.clone()).unwrap());
                c.initialize(&pose, &Transform::identity()).unwrap();
            }
            "pose" => {
                let pose = parse_pose(&fields[1..]);
                if let Some(c) = core.as_mut() {
                    c.enqueue_pose(pose);
                }
            }
            "twist" => {
                let twist = parse_twist(&fields[1..]);
                if let Some(c) = core.as_mut() {
                    c.enqueue_twist(twist);
                }
            }
            "tick" => {
                let now_ns: i64 = fields[1].parse().unwrap();
                if let Some(c) = core.as_mut() {
                    let _output = c.tick(now_ns, &mut trace).unwrap();
                }
            }
            other => panic!("unknown statement '{other}' (line {})", line_no + 1),
        }
    }

    writeln!(out, "{}", csv_header()).unwrap();
    for ev in &trace {
        writeln!(out, "{}", ev.to_csv_row()).unwrap();
    }
    ExitCode::SUCCESS
}
