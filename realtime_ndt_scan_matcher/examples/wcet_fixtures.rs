//! Hand-built adversarial WCET fixtures — near-worst inputs by
//! construction, written in the frozen fixture format both the Rust harness and the C++ replay
//! consume:
//!
//! - `dense_neighbors`: 8 overlapping map tiles whose voxel centroids all hug shared block
//!   corners, so each query collects 8 tiles × 8 corner voxels = 64 leaves — pins `K` at
//!   `MAX_NEIGHBORS` (a single tile geometrically caps `K` at ~8; production dynamic map loading
//!   is multi-tile).
//! - `max_iterations`: rough random surface + tiny `trans_epsilon` — the align never converges,
//!   pinning `N_iter = max_iterations`.
//! - `cache_hostile`: a wide map with the source visiting voxels in a shuffled order — defeats
//!   spatial locality in the leaf/kd-tree accesses.
//! - `subnormal`: tight clusters (icov ≈ 360) with the source on a ~1.99 m shell so the kernel's
//!   `exp` lands in the f64 subnormal range (exponent ≈ −710..−745) — the FP-assist timing hazard.
//!
//! Usage: `cargo run --release --example wcet_fixtures [OUT_DIR]`
//! (default `OUT_DIR`: `../../bench/fixtures` — the package-level `bench/` — relative to the engine crate).
//!
//! Note: map voxel `leaf_size == params.resolution` in every fixture so the inputs stay
//! representable in the C++ engine (pcl couples the two).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::cast_possible_wrap,
    reason = "example/benchmark tooling"
)]

use std::path::{Path, PathBuf};

use nalgebra::Matrix4;
use realtime_ndt_scan_matcher::fixture::Fixture;
use realtime_ndt_scan_matcher::ndt::{AlignResult, AlignWorkspace, NdtParams, align};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;

/// Deterministic LCG (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1_u64 << 24) as f32)
    }
}

fn params(max_iterations: i32, trans_epsilon: f64) -> NdtParams {
    NdtParams {
        trans_epsilon,
        step_size: 0.1,
        resolution: 2.0,
        max_iterations,
        outlier_ratio: 0.55,
        regularization: None,
        num_threads: 1,
    }
}

fn guess_xy(dx: f32, dy: f32) -> Matrix4<f32> {
    let mut g = Matrix4::<f32>::identity();
    g[(0, 3)] = dx;
    g[(1, 3)] = dy;
    g
}

/// `K = MAX_NEIGHBORS` by construction: 2×2×2 voxel blocks whose 8 voxels each cram their 8
/// points against the block's shared central corner (centroid → corner from inside each voxel),
/// replicated over 8 overlapping tiles with distinct jitter → 8 tiles × 8 voxels = 64 leaf
/// centroids within ~0.03 m of every block corner. Source: `n_points` cycling the block corners
/// (each collects the full 64 leaves). With `trans_epsilon = 1e-10` the align never converges,
/// so every fixture built here does `31 × n_points × 64` kernel evaluations — the analytic
/// maximum for its `(P, blocks)`.
fn dense_corner(blocks: i32, n_points: usize, guess_dx: f32, guess_dy: f32) -> Fixture {
    let res = 2.0_f32;
    let n_tiles = 8_usize;
    let mut tiles = Vec::with_capacity(n_tiles);
    for t in 0..n_tiles {
        let jt = 0.004 + t as f32 * 0.0015; // per-tile jitter: distinct centroids, same corner
        let mut tile = Vec::new();
        for bx in 0..blocks {
            for by in 0..blocks {
                // Block central corner (shared by its 8 voxels); one block layer in z.
                let kx = (2 * bx + 1) as f32 * res;
                let ky = (2 * by + 1) as f32 * res;
                let kz = res;
                for dx in 0..2_i32 {
                    for dy in 0..2_i32 {
                        for dz in 0..2_i32 {
                            // Inward direction: voxel below the corner (d=0) hugs from -,
                            // voxel above (d=1) hugs from +.
                            let sx = if dx == 0 { -1.0 } else { 1.0 };
                            let sy = if dy == 0 { -1.0 } else { 1.0 };
                            let sz = if dz == 0 { -1.0 } else { 1.0 };
                            for i in 0..8 {
                                let e = jt + i as f32 * 0.002; // spread: min_points + covariance
                                tile.push([
                                    kx + sx * e,
                                    ky + sy * (e * 0.8 + 0.003),
                                    kz + sz * (e * 0.6 + 0.006),
                                ]);
                            }
                        }
                    }
                }
            }
        }
        tiles.push(tile);
    }
    let mut src = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let k = (i as i32) % (blocks * blocks);
        let kx = (2 * (k % blocks) + 1) as f32 * res;
        let ky = (2 * (k / blocks) + 1) as f32 * res;
        src.push([kx, ky, res]);
    }
    Fixture {
        tiles,
        source: src,
        guess: guess_xy(guess_dx, guess_dy),
        params: params(30, 1e-10),
    }
}

/// (a) The hand-built max-`K` fixture (see [`dense_corner`]).
fn dense_neighbors() -> Fixture {
    dense_corner(6, 1500, 0.08, -0.06)
}

/// (e) **Deployment-legal worst case**: the hardest input that can still reach the engine
/// through Autoware's production preprocessing pipeline (contract values from
/// `autoware_launch/config/localization/ndt_scan_matcher/pointcloud_preprocessor/*.param.yaml`):
///
/// - crop box ±60 m (x/y); `voxel_grid_downsample` leaf **3.0 m** → one source point per 3 m
///   cell, no duplicates; `random_downsample` `sample_num` = **1500** → `P ≤ 1500`;
/// - map tiles are disjoint and voxel-aligned (pcd_divider) → leaf multiplicity 1 → `K ≤ 8`
///   (the corner-sharing voxels; centroids at √3 ≈ 1.73 m < the 2.0 m radius);
/// - map PCD survives a ≥ 0.2 m map downsample (per-voxel points spaced ≥ 0.7 m here).
///
/// Construction: single tile, 2 m voxels over ±62 m × 3 z-layers, 8 points per voxel on a
/// 2×2×2 sub-grid (0.8 m pitch, ±0.05 m deterministic jitter). Source: `P = 1500` points on
/// distinct interior voxel-lattice corners at 4 m x/y pitch (distinct 3 m cells) on the z = 2
/// and z = 4 planes (distinct 3 m z-cells), all inside the crop; every corner collects exactly
/// `K = 8`. eps = 1e-10 pins iter = 30. Expected counters: K̄ = 8.0, Σnbr = 1500·8·31 = 372000.
fn legal_worst() -> Fixture {
    let res = 2.0_f32;
    let mut rng = Lcg(0x1E9A_1000);
    let mut map = Vec::new();
    // Voxels i,j ∈ [-31, 30] cover [-62, 62); z-layers k ∈ 0..3 cover [0, 6].
    for i in -31_i32..31 {
        for j in -31_i32..31 {
            for k in 0..3_i32 {
                for sx in 0..2_i32 {
                    for sy in 0..2_i32 {
                        for sz in 0..2_i32 {
                            let jx = (rng.next_f32() - 0.5) * 0.1;
                            let jy = (rng.next_f32() - 0.5) * 0.1;
                            let jz = (rng.next_f32() - 0.5) * 0.1;
                            map.push([
                                i as f32 * res + 0.6 + sx as f32 * 0.8 + jx,
                                j as f32 * res + 0.6 + sy as f32 * 0.8 + jy,
                                k as f32 * res + 0.6 + sz as f32 * 0.8 + jz,
                            ]);
                        }
                    }
                }
            }
        }
    }
    // Source: interior lattice corners, 4 m x/y pitch, z ∈ {2, 4}; first 1500 in scan order.
    let mut src = Vec::with_capacity(1500);
    'outer: for z in [2.0_f32, 4.0] {
        for a in -15_i32..=15 {
            for b in -15_i32..=15 {
                if src.len() == 1500 {
                    break 'outer;
                }
                src.push([a as f32 * 4.0, b as f32 * 4.0, z]);
            }
        }
    }
    Fixture {
        tiles: vec![map],
        source: src,
        guess: guess_xy(0.08, -0.06),
        params: params(30, 1e-10),
    }
}

/// (b) Rough random surface + tiny trans_epsilon → never converges → N_iter = max_iterations.
fn max_iterations() -> Fixture {
    let mut rng = Lcg(0x57C3_7001);
    let mut map = Vec::new();
    for vx in 0..20_i32 {
        for vy in 0..20_i32 {
            for _ in 0..8 {
                map.push([
                    vx as f32 * 2.0 + rng.next_f32() * 2.0,
                    vy as f32 * 2.0 + rng.next_f32() * 2.0,
                    rng.next_f32() * 2.0,
                ]);
            }
        }
    }
    let src: Vec<[f32; 3]> = (0..1200)
        .map(|_| {
            [
                rng.next_f32() * 40.0,
                rng.next_f32() * 40.0,
                rng.next_f32() * 2.0,
            ]
        })
        .collect();
    Fixture {
        tiles: vec![map],
        source: src,
        guess: guess_xy(0.7, 0.5),
        params: params(30, 1e-10),
    }
}

/// (c) Wide map, source visiting voxels in a shuffled order — cache-hostile leaf/kd accesses.
fn cache_hostile() -> Fixture {
    let mut rng = Lcg(0xCAC4_E057);
    let n = 60_i32;
    let mut map = Vec::new();
    for vx in 0..n {
        for vy in 0..n {
            for i in 0..8 {
                let f = i as f32 * 0.05;
                map.push([
                    vx as f32 * 2.0 + 0.5 + f,
                    vy as f32 * 2.0 + 0.5 - f,
                    0.5 + f,
                ]);
            }
        }
    }
    // 2000 source points at pseudo-random voxel centers across the whole 120 m extent.
    let src: Vec<[f32; 3]> = (0..2000)
        .map(|_| {
            let vx = (rng.next_u64() % n as u64) as f32;
            let vy = (rng.next_u64() % n as u64) as f32;
            [vx * 2.0 + 0.55, vy * 2.0 + 0.45, 0.55]
        })
        .collect();
    Fixture {
        tiles: vec![map],
        source: src,
        guess: guess_xy(0.2, 0.1),
        params: params(30, 0.01),
    }
}

/// (d) Tight clusters (σ ≈ 0.05 → icov ≈ 360 per axis) with the source on a ~1.99 m shell:
/// `exp(-d²·icov/2)` lands around e^{-710..-745} — the f64 subnormal band (FP-assist hazard).
fn subnormal() -> Fixture {
    subnormal_shell(1.99)
}

/// A/B control for `subnormal`: the identical geometry with the source
/// on a `shell`-metre ring. At 1.99 m the kernel `exp` lands in the f64 subnormal band; at
/// 1.69 m every result leaves that band, all else equal — the controlled null test that
/// isolates whether subnormal operands (vs. the per-point search overhead) drive the timing.
fn subnormal_shell(shell: f32) -> Fixture {
    let mut rng = Lcg(0x5B40_2417);
    let mut map = Vec::new();
    let n = 10_i32;
    for vx in 0..n {
        for vy in 0..n {
            let cx = vx as f32 * 4.0 + 1.0; // cluster at the voxel center, spacing 4 m
            let cy = vy as f32 * 4.0 + 1.0;
            for _ in 0..12 {
                map.push([
                    cx + (rng.next_f32() - 0.5) * 0.18, // σ ≈ 0.052
                    cy + (rng.next_f32() - 0.5) * 0.18,
                    1.0 + (rng.next_f32() - 0.5) * 0.18,
                ]);
            }
        }
    }
    // Source on a `shell` m ring around each cluster (still inside the 2.0 m search radius).
    let mut src = Vec::new();
    for i in 0..1000_i32 {
        let k = i % (n * n);
        let cx = (k % n) as f32 * 4.0 + 1.0;
        let cy = (k / n) as f32 * 4.0 + 1.0;
        let ang = i as f32 * 0.618;
        src.push([cx + shell * ang.cos(), cy + shell * ang.sin(), 1.0]);
    }
    Fixture {
        tiles: vec![map],
        source: src,
        guess: Matrix4::identity(),
        params: params(30, 0.01),
    }
}

/// Run one align over the fixture and report iteration count (+ counters under wcet-count).
fn probe(name: &str, fx: &Fixture) {
    let mut map = VoxelGridMap::new([fx.params.resolution; 3], 6, 0.01);
    for (id, tile) in fx.tiles.iter().enumerate() {
        map.add_target(tile, &(id as u64).to_be_bytes());
    }
    map.try_create_kdtree(418_000).expect("build kd-tree");
    let mut ws = AlignWorkspace::try_with_capacity(fx.source.len()).expect("reserve workspace");
    let mut out = AlignResult::try_with_capacity(30).expect("reserve result");
    align(&map, &fx.source, &fx.guess, &fx.params, &mut ws, &mut out).expect("align fixture");
    print!(
        "  {name:16} map={:6} src={:5} leaves={:5} iter={:2}",
        fx.map_len(),
        fx.source.len(),
        map.num_leaves(),
        out.iteration_num
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
            "  passes={:2} K̄={kbar:5.1} kd_nodes/pt={:6.1}",
            c.derivative_passes,
            c.kd_nodes_visited as f64 / c.points_processed.max(1) as f64
        );
    }
    println!();
}

/// `--psweep`: latency-vs-`P` sweep set. The `search_00` champion geometry (8 tiles,
/// blocks = 9, eps = 1e-10, guess (0.08, −0.35)) is regenerated at each source cardinality.
/// The replay configures `P_max = P`; iteration count and common kernel evaluations per point are
/// invariant, while implementation-specific traversal is measured rather than assumed invariant.
fn run_psweep(out_dir: &Path) {
    std::fs::create_dir_all(out_dir).expect("create fixture dir");
    println!("P-sweep WCET fixtures -> {}", out_dir.display());
    for p in [250_usize, 500, 1000, 2000, 4000, 8000] {
        let fx = dense_corner(9, p, 0.08, -0.35);
        let name = format!("psweep_p{p:05}");
        fx.write(&out_dir.join(format!("{name}.ndtfix")))
            .expect("write fixture");
        probe(&name, &fx);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--psweep") {
        let out_dir: PathBuf = args.get(1).map_or_else(
            || Path::new("../../bench/fixtures/psweep").to_path_buf(),
            Into::into,
        );
        run_psweep(&out_dir);
        return;
    }

    let out_dir: PathBuf = args.first().map_or_else(
        || Path::new("../../bench/fixtures").to_path_buf(),
        Into::into,
    );
    std::fs::create_dir_all(&out_dir).expect("create fixture dir");

    let fixtures: [(&str, Fixture); 6] = [
        ("dense_neighbors", dense_neighbors()),
        ("max_iterations", max_iterations()),
        ("cache_hostile", cache_hostile()),
        ("subnormal", subnormal()),
        // A/B control for `subnormal`: identical geometry, source shell out of the f64
        // subnormal band (1.69 m vs 1.99 m).
        ("subnormal_ctrl", subnormal_shell(1.69)),
        ("legal_worst", legal_worst()),
    ];
    println!("adversarial WCET fixtures -> {}", out_dir.display());
    for (name, fx) in &fixtures {
        let path = out_dir.join(format!("{name}.ndtfix"));
        fx.write(&path).expect("write fixture");
        probe(name, fx);
    }
}
