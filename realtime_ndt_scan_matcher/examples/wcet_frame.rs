//! Frame-time (WCET) benchmark for `align` — a manual diagnostic, NOT a CI gate (timing is
//! environment-dependent). Reports min / mean / p50 / p99 / p99.9 / max over many frames; for
//! WCET, the **max + tail** matter (not the mean).
//!
//! Two modes:
//! - `cargo run --release --example wcet_frame` — fixed synthetic fixture (regression watch).
//! - `cargo run --release --example wcet_frame -- FIXTURE.ndtfix [...]` — replay frozen fixtures
//!   (`bench/fixtures/*.ndtfix` from `wcet_fixtures` / `wcet_search`), one HWM row each. With
//!   `--features wcet-count` each row also reports the deterministic cost counters.
//!
//! `WCET_FRAMES` overrides the per-fixture frame count (default 20000 synthetic / 2000 fixture).
//! `WCET_JSON=path` (fixture mode) additionally writes a machine-readable JSON — per fixture:
//! sizes, `iteration_num`, timing samples (µs), and (under `wcet-count`) the cost counters —
//! consumed by `bench/wcet_report.py` for the unit-cost regression.
//! Record baselines in `porting_notes/ndt_wcet_audit.md`. (Single core, warm cache — a relative
//! regression watch, not a hardware WCET proof.)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::arithmetic_side_effects,
    clippy::let_underscore_must_use,
    clippy::too_many_lines,
    clippy::allow_attributes,
    reason = "benchmark example (diagnostic, not a gate)"
)]

use std::time::Instant;

use nalgebra::Matrix4;
use realtime_ndt_scan_matcher::fixture::Fixture;
use realtime_ndt_scan_matcher::ndt::{AlignResult, AlignWorkspace, NdtParams, align};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;

fn dense_cluster(cx: f32, cy: f32, cz: f32) -> Vec<[f32; 3]> {
    (0..8)
        .map(|i| {
            let f = i as f32 * 0.02;
            [cx + f, cy - f, cz + 0.5 * f]
        })
        .collect()
}

struct Stats {
    nanos: Vec<u128>,
}

impl Stats {
    fn measure(
        frames: usize,
        map: &VoxelGridMap,
        source: &[[f32; 3]],
        guess: &Matrix4<f32>,
        params: &NdtParams,
        ws: &mut AlignWorkspace,
        out: &mut AlignResult,
    ) -> Self {
        // Warmup (JIT-free, but page/cache/branch warmup + buffer growth).
        for _ in 0..(frames / 20).clamp(10, 100) {
            align(map, source, guess, params, ws, out).expect("warmup align");
        }
        let mut nanos: Vec<u128> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let t = Instant::now();
            align(map, source, guess, params, ws, out).expect("timed align");
            nanos.push(t.elapsed().as_nanos());
        }
        nanos.sort_unstable();
        Self { nanos }
    }

    fn pct(&self, p: f64) -> u128 {
        let n = self.nanos.len();
        self.nanos[((p * (n as f64 - 1.0)) as usize).min(n - 1)]
    }

    fn mean(&self) -> u128 {
        self.nanos.iter().sum::<u128>() / (self.nanos.len() as u128)
    }
}

fn us(n: u128) -> f64 {
    (n as f64) / 1000.0
}

fn frames_from_env(default: usize) -> usize {
    std::env::var("WCET_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Synthetic default fixture (the original regression-watch numbers).
fn run_synthetic() {
    let frames = frames_from_env(20_000);
    let mut map_points: Vec<[f32; 3]> = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            map_points.extend(dense_cluster((i as f32) * 3.0, (j as f32) * 3.0, 0.0));
        }
    }
    let source: Vec<[f32; 3]> = map_points
        .iter()
        .map(|p| [p[0] + 0.3, p[1] - 0.2, p[2] + 0.1])
        .collect();

    let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
    map.add_target(&map_points, b"0");
    map.try_create_kdtree(418_000).expect("build synthetic map");

    let params = NdtParams {
        trans_epsilon: 0.01,
        step_size: 0.1,
        resolution: 2.0,
        max_iterations: 30,
        outlier_ratio: 0.55,
        regularization: None,
        num_threads: 1,
    };
    let guess = Matrix4::identity();
    let mut ws =
        AlignWorkspace::try_with_capacity(source.len()).expect("reserve synthetic workspace");
    let mut out = AlignResult::try_with_capacity(30).expect("reserve synthetic result");

    let s = Stats::measure(frames, &map, &source, &guess, &params, &mut ws, &mut out);
    let n = s.nanos.len();
    println!(
        "align frame time over {frames} frames ({} src pts, {} iters):",
        source.len(),
        out.iteration_num
    );
    println!("  min  {:.2} us", us(s.nanos[0]));
    println!("  mean {:.2} us", us(s.mean()));
    println!("  p50  {:.2} us", us(s.pct(0.50)));
    println!("  p99  {:.2} us", us(s.pct(0.99)));
    println!("  p99.9 {:.2} us", us(s.pct(0.999)));
    println!("  max  {:.2} us", us(s.nanos[n - 1]));
}

/// Replay frozen fixtures: one HWM row per file (+ cost counters under `wcet-count`).
fn run_fixtures(paths: &[String]) {
    let frames = frames_from_env(2_000);
    let mut json = String::new();
    json.push_str("{\n  \"fixtures\": {\n");
    let mut first = true;
    println!("align frame time over {frames} frames/fixture (serial, pre-reserved workspace):");
    println!(
        "  {:16} {:>6} {:>6} {:>5} | {:>9} {:>9} {:>9} {:>9} | extra",
        "fixture", "map", "src", "iter", "p50 us", "p99 us", "p99.9 us", "max us"
    );
    for path in paths {
        let fx = Fixture::read(std::path::Path::new(path)).expect("read fixture");
        let mut map = VoxelGridMap::new([fx.params.resolution; 3], 6, 0.01);
        for (id, tile) in fx.tiles.iter().enumerate() {
            map.add_target(tile, &(id as u64).to_be_bytes());
        }
        map.try_create_kdtree(418_000).expect("build fixture map");
        let mut ws =
            AlignWorkspace::try_with_capacity(fx.source.len()).expect("reserve fixture workspace");
        let mut out = AlignResult::try_with_capacity(fx.params.max_iterations as usize)
            .expect("reserve fixture result");
        let s = Stats::measure(
            frames, &map, &fx.source, &fx.guess, &fx.params, &mut ws, &mut out,
        );
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let n = s.nanos.len();
        print!(
            "  {name:16} {:>6} {:>6} {:>5} | {:>9.1} {:>9.1} {:>9.1} {:>9.1} |",
            fx.map_len(),
            fx.source.len(),
            out.iteration_num,
            us(s.pct(0.50)),
            us(s.pct(0.99)),
            us(s.pct(0.999)),
            us(s.nanos[n - 1]),
        );
        #[cfg(feature = "wcet-count")]
        {
            let c = out.counters;
            let kbar = if c.points_processed == 0 {
                0.0
            } else {
                c.sum_neighbors as f64 / c.points_processed as f64
            };
            print!(
                " passes={} K̄={kbar:.1} kd/pt={:.1}",
                c.derivative_passes,
                c.kd_nodes_visited as f64 / c.points_processed.max(1) as f64
            );
        }
        println!();

        // Machine-readable entry for bench/wcet_report.py (WCET_JSON mode).
        {
            use std::fmt::Write as _;
            if !first {
                json.push_str(",\n");
            }
            first = false;
            let _ = write!(
                json,
                "    \"{name}\": {{\n      \"n_map\": {},\n      \"n_tiles\": {},\n      \
                 \"n_source\": {},\n      \"iteration_num\": {}",
                fx.map_len(),
                fx.tiles.len(),
                fx.source.len(),
                out.iteration_num
            );
            #[cfg(feature = "wcet-count")]
            {
                let c = out.counters;
                let _ = write!(
                    json,
                    ",\n      \"counters\": {{ \"derivative_passes\": {}, \"points_processed\": \
                     {}, \"sum_neighbors\": {}, \"kd_nodes_visited\": {} }}",
                    c.derivative_passes, c.points_processed, c.sum_neighbors, c.kd_nodes_visited
                );
            }
            json.push_str(",\n      \"samples_us\": [");
            for (i, ns) in s.nanos.iter().enumerate() {
                let _ = write!(json, "{}{:.3}", if i == 0 { "" } else { "," }, us(*ns));
            }
            json.push_str("]\n    }");
        }
    }
    json.push_str("\n  }\n}\n");
    if let Ok(path) = std::env::var("WCET_JSON") {
        std::fs::write(&path, json).expect("write WCET_JSON");
        eprintln!("wcet_frame: wrote {path}");
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.first().map(String::as_str) == Some("--capture") {
        let dir = paths.get(1).expect("--capture needs a directory");
        run_capture(std::path::Path::new(dir));
    } else if paths.first().map(String::as_str) == Some("--freeze") {
        run_freeze(&paths[1..]);
    } else if paths.is_empty() {
        run_synthetic();
    } else {
        run_fixtures(&paths);
    }
}

/// Freeze one captured frame as a self-contained `.ndtfix` fixture:
/// `--freeze <capture_dir> <seq> <guess_track.bin> <out.ndtfix>`.
///
/// The map tiles are embedded in the captured sorted-id order (the same order both capture
/// replayers use per epoch), and the guess is taken from the frozen open-loop guess track
/// (`WCET_GUESS_OUT` dump, 64 bytes/frame row-major f32) — i.e. the exact per-frame input of
/// the realdata open-loop protocol, now replayable by every fixture consumer.
fn run_freeze(args: &[String]) {
    use realtime_ndt_scan_matcher::capture;

    let usage = "--freeze <capture_dir> <seq> <guess_track.bin> <out.ndtfix>";
    let dir = std::path::Path::new(args.first().expect(usage));
    let seq: usize = args
        .get(1)
        .expect(usage)
        .parse()
        .expect("seq must be a number");
    let track_path = args.get(2).expect(usage);
    let out_path = args.get(3).expect(usage);

    let params = capture::read_params(dir).expect("read params.bin");
    let frame_paths = capture::list_frames(dir).expect("list frames");
    let fp = frame_paths.get(seq).expect("seq out of range");
    let fr = capture::read_frame(fp).expect("read frame");

    let track = std::fs::read(track_path).expect("read guess track");
    let rec = track
        .get(seq * 64..(seq + 1) * 64)
        .expect("track too short for seq");
    let mut guess: Matrix4<f32> = Matrix4::identity();
    for r in 0..4 {
        for c in 0..4 {
            let off = (r * 4 + c) * 4;
            let bytes: [u8; 4] = rec[off..off + 4].try_into().expect("chunk");
            guess[(r, c)] = f32::from_le_bytes(bytes);
        }
    }

    let tiles: Vec<Vec<[f32; 3]>> = fr
        .ids
        .iter()
        .map(|id| capture::read_tile(dir, &capture::hex_id(id)).expect("read tile"))
        .collect();
    let fx = Fixture {
        tiles,
        source: fr.source.clone(),
        guess,
        params,
    };
    println!(
        "froze frame {} -> {} (tiles={} map_pts={} src={} eps={} max_iter={})",
        seq,
        out_path,
        fx.tiles.len(),
        fx.map_len(),
        fx.source.len(),
        fx.params.trans_epsilon,
        fx.params.max_iterations
    );
    fx.write(std::path::Path::new(out_path))
        .expect("write fixture");
}

/// Replay a real-drive capture directory (`NDT_CAPTURE_DIR` sidecar format): frames grouped into
/// epochs by their active tile-id set, one map build per epoch (tiles added in the captured
/// sorted-id order → the same order the C++ replayer uses), each frame timed `WCET_FRAMES`
/// times (default 5, median reported) + counters under `wcet-count`. `WCET_JSON` writes the
/// per-frame records for `bench/wcet_realdata.py`.
fn run_capture(dir: &std::path::Path) {
    use std::fmt::Write as _;

    use realtime_ndt_scan_matcher::capture;

    let params = capture::read_params(dir).expect("read params.bin");
    let mut frame_paths = capture::list_frames(dir).expect("list frames");
    let repeats = frames_from_env(5);
    if let Ok(limit) = std::env::var("WCET_CAPTURE_LIMIT")
        && let Ok(limit) = limit.parse::<usize>()
    {
        frame_paths.truncate(limit);
    }
    // Chain mode (WCET_CHAIN=1): guess = the previous frame's result pose (frame 0 from
    // WCET_CHAIN_SEED="x,y,z,qx,qy,qz,qw") -- deterministic offline re-localization replay.
    let chain = std::env::var_os("WCET_CHAIN").is_some();
    let mut chain_guess: Matrix4<f32> = Matrix4::identity();
    let mut chain_prev: Option<Matrix4<f32>> = None; // for constant-velocity extrapolation
    let mut guess_dump = std::env::var("WCET_GUESS_OUT")
        .ok()
        .map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("guess dump file")));
    if let Ok(seed) = std::env::var("WCET_CHAIN_SEED") {
        let v: Vec<f32> = seed
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        if v.len() == 7 {
            let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                v[6], v[3], v[4], v[5],
            ));
            chain_guess = q.to_homogeneous();
            chain_guess[(0, 3)] = v[0];
            chain_guess[(1, 3)] = v[1];
            chain_guess[(2, 3)] = v[2];
        }
    }
    println!(
        "capture replay: {} frames, {} timing repeats/frame, eps={} max_iter={}",
        frame_paths.len(),
        repeats,
        params.trans_epsilon,
        params.max_iterations
    );

    let mut json = String::new();
    json.push_str("{\n  \"frames\": [\n");
    let mut first = true;

    let mut cur_ids: Option<Vec<Vec<u8>>> = None;
    let mut map = VoxelGridMap::new([params.resolution; 3], 6, 0.01);
    let mut out = AlignResult::try_with_capacity(params.max_iterations as usize)
        .expect("reserve capture result");
    let mut epochs = 0_u32;
    let mut neighbor_limit_frames = 0_usize;
    let mut neighbor_limit_queries = 0_u64;

    for (fi, fp) in frame_paths.iter().enumerate() {
        let fr = capture::read_frame(fp).expect("read frame");
        if cur_ids.as_ref() != Some(&fr.ids) {
            // New epoch: rebuild the map from the tile library in the captured (sorted) order.
            map = VoxelGridMap::new([params.resolution; 3], 6, 0.01);
            for (i, id) in fr.ids.iter().enumerate() {
                let tile = capture::read_tile(dir, &capture::hex_id(id)).expect("read tile");
                map.add_target(&tile, &(i as u64).to_be_bytes());
            }
            map.try_create_kdtree(418_000).expect("build capture map");
            cur_ids = Some(fr.ids.clone());
            epochs += 1;
        }
        let guess = if chain { chain_guess } else { fr.guess };
        // Timing: median of `repeats` runs (first run also yields the counters).
        let mut ws =
            AlignWorkspace::try_with_capacity(fr.source.len()).expect("reserve capture workspace");
        let mut times: Vec<u128> = Vec::with_capacity(repeats);
        let mut diagnostics = realtime_ndt_scan_matcher::ndt::AlignDiagnostics::default();
        for _ in 0..repeats {
            let t = Instant::now();
            diagnostics =
                align(&map, &fr.source, &guess, &params, &mut ws, &mut out).expect("capture align");
            times.push(t.elapsed().as_nanos());
        }
        if diagnostics.neighbor_limit_exceeded {
            neighbor_limit_frames += 1;
            neighbor_limit_queries += diagnostics.neighbor_limit_exceeded_queries;
        }
        if chain {
            // Constant-velocity extrapolation (the EKF prediction step's stand-in):
            // guess_{k+1} = P_k * (P_{k-1}^-1 * P_k); falls back to P_k on the first frame.
            chain_guess = match chain_prev {
                Some(prev) => prev
                    .try_inverse()
                    .map_or(out.pose, |pi| out.pose * (pi * out.pose)),
                None => out.pose,
            };
            chain_prev = Some(out.pose);
            // WCET_GUESS_OUT: record the guess USED for this frame (row-major f32) so the
            // open-loop pass can freeze the guess track and feed both engines identical inputs.
            if let Some(gp) = &mut guess_dump {
                use std::io::Write as _;
                for r in 0..4 {
                    for c in 0..4 {
                        gp.write_all(&guess[(r, c)].to_le_bytes())
                            .expect("guess dump");
                    }
                }
            }
        }
        times.sort_unstable();
        let med_ms = us(times[times.len() / 2]) / 1000.0;

        if !first {
            json.push_str(",\n");
        }
        first = false;
        let _ = write!(
            json,
            "    {{ \"seq\": {fi}, \"n_source\": {}, \"n_tiles\": {}, \"n_leaves\": {}, \
             \"iteration_num\": {}, \
             \"ms\": {med_ms:.4}",
            fr.source.len(),
            fr.ids.len(),
            map.num_leaves(),
            out.iteration_num
        );
        #[cfg(feature = "wcet-count")]
        {
            let c = out.counters;
            let _ = write!(
                json,
                ", \"counters\": {{ \"derivative_passes\": {}, \"points_processed\": {}, \
                 \"sum_neighbors\": {}, \"kd_nodes_visited\": {}, \"max_neighbors\": {} }}",
                c.derivative_passes,
                c.points_processed,
                c.sum_neighbors,
                c.kd_nodes_visited,
                c.max_neighbors
            );
        }
        json.push_str(" }");
    }
    json.push_str("\n  ]\n}\n");
    println!("epochs: {epochs}");
    println!(
        "neighbor-limit diagnostics: {neighbor_limit_frames} frames, \
         {neighbor_limit_queries} queries"
    );
    if let Ok(path) = std::env::var("WCET_JSON") {
        std::fs::write(&path, json).expect("write WCET_JSON");
        eprintln!("wcet_frame: wrote {path}");
    }
}
