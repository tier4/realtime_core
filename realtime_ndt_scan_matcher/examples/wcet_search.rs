//! Counter-guided worst-input search (`plan/ndt_wcet.md`, M3 / Layer 2).
//!
//! Seeded hill-climb over a parameterized input generator, with the **deterministic cost
//! counters** as fitness — platform-independent, so a worst input found here is a worst input
//! everywhere (and, by bit-exactness, for the C++ engine too). Requires `--features wcet-count`.
//!
//! Fitness is the lexicographic counter vector `(iterations, sum_neighbors, kd_nodes_visited)`:
//! the outer-loop count dominates, then the kernel-evaluation count, then the tree-traversal
//! count. Wall time is *not* the fitness (noisy, machine-specific); the frozen outputs are
//! measured separately by `wcet_frame` / the C++ replay.
//!
//! The search mutates: tile count (leaf overlap → `K`), block extent (map size → kd depth),
//! corner-hug tightness vs surface roughness, source composition (corner-sitters vs random
//! rovers), guess offset, and `trans_epsilon` (convergence). The source point count is **fixed**
//! at `P = 2000` (the node caps the downsampled scan; `P` is a caller-side bound, not a search
//! freedom).
//!
//! Usage: `cargo run --release --features wcet-count --example wcet_search [OUT_DIR]`
//! (default `OUT_DIR`: `../../bench/fixtures`). Env: `WCET_SEARCH_GENS` (default 20),
//! `WCET_SEARCH_POP` (default 6), `WCET_SEARCH_TOPK` (default 2), `WCET_SEARCH_SEED`.
//! Deterministic for a fixed seed. Emits `search_00.ndtfix`, `search_01.ndtfix`, ….
//!
//! Ablation controls (plan/paper_fix.md B4; the defaults leave the original behavior
//! byte-identical):
//! - `WCET_SEARCH_MODE=hill|random` — `random` evaluates the same budget (pop + gens×pop)
//!   of freshly sampled genomes (budget-matched baseline).
//! - `WCET_SEARCH_FITNESS=counters|time` — `time` uses the measured align wall time as the
//!   fitness (demonstrates noise/irreproducibility; counter runs are bit-reproducible).
//! - `WCET_SEARCH_JSON=path` — machine-readable run summary (best, saturation evaluation,
//!   best-so-far trajectory, Pareto archive).
//! - `WCET_SEARCH_PARETO_DIR=path` — freeze the non-dominated (iter, Σnbr, kd) archive as
//!   `pareto_NN.ndtfix` there (never written by default).

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
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::needless_range_loop,
    reason = "example/benchmark tooling"
)]

use std::path::{Path, PathBuf};

use nalgebra::Matrix4;
use realtime_ndt_scan_matcher::fixture::Fixture;
use realtime_ndt_scan_matcher::ndt::{AlignResult, AlignWorkspace, NdtParams, align};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;

/// Fixed source point count: `P` is the node's responsibility (downsample cap), not a search
/// freedom — searching it would trivially saturate at the cap and hide the interesting terms.
const SRC_POINTS: usize = 2000;
const RES: f32 = 2.0;
const MAX_ITER: i32 = 30;

/// Deterministic LCG (no rand dependency).
#[derive(Clone)]
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
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
    fn range_u(&mut self, lo: u64, hi_incl: u64) -> u64 {
        lo + self.next_u64() % (hi_incl - lo + 1)
    }
}

/// Search genome: the input-generator parameters the hill-climb mutates.
#[derive(Clone, Debug)]
struct Genome {
    /// Overlapping map tiles (1..=8): leaf multiplicity → per-point `K`.
    n_tiles: u64,
    /// 2×2×2-voxel blocks per axis (3..=10): map extent → kd-tree size/depth.
    blocks: u64,
    /// Corner-hug tightness (0.002..0.02 m): how hard centroids crowd the shared corners.
    hug: f32,
    /// Roughness blend (0..1): 0 = pure corner-hug lattice, 1 = fully random surface.
    rough: f32,
    /// Fraction of source points sitting on block corners (max-K queries); rest rove randomly.
    corner_frac: f32,
    /// Initial-guess translation offset (m).
    guess_dx: f32,
    guess_dy: f32,
    /// `trans_epsilon = 10^eps_log` (-10..-2): convergence threshold.
    eps_log: f32,
    /// Point-stream seed (orderings / jitter).
    seed: u64,
}

impl Genome {
    fn seed_population(rng: &mut Lcg, pop: usize) -> Vec<Genome> {
        let mut v = Vec::with_capacity(pop);
        // Seed 0: the hand-built compound worst case (dense_neighbors-like).
        v.push(Genome {
            n_tiles: 8,
            blocks: 6,
            hug: 0.006,
            rough: 0.0,
            corner_frac: 1.0,
            guess_dx: 0.08,
            guess_dy: -0.06,
            eps_log: -10.0,
            seed: 1,
        });
        // Seed 1: the rough non-converging surface (max_iterations-like).
        v.push(Genome {
            n_tiles: 1,
            blocks: 10,
            hug: 0.01,
            rough: 1.0,
            corner_frac: 0.0,
            guess_dx: 0.7,
            guess_dy: 0.5,
            eps_log: -10.0,
            seed: 2,
        });
        // Rest: random.
        while v.len() < pop {
            v.push(Genome::random(rng));
        }
        v
    }

    /// A uniformly random genome (used by seeding and by the random-search baseline).
    fn random(rng: &mut Lcg) -> Genome {
        Genome {
            n_tiles: rng.range_u(1, 8),
            blocks: rng.range_u(3, 10),
            hug: rng.range_f32(0.002, 0.02),
            rough: rng.next_f32(),
            corner_frac: rng.next_f32(),
            guess_dx: rng.range_f32(-1.0, 1.0),
            guess_dy: rng.range_f32(-1.0, 1.0),
            eps_log: rng.range_f32(-10.0, -2.0),
            seed: rng.next_u64(),
        }
    }

    /// Mutate 1–2 fields (hill-climb neighborhood).
    fn mutate(&self, rng: &mut Lcg) -> Genome {
        let mut g = self.clone();
        for _ in 0..rng.range_u(1, 2) {
            match rng.range_u(0, 8) {
                0 => g.n_tiles = rng.range_u(1, 8),
                1 => g.blocks = rng.range_u(3, 10),
                2 => g.hug = (g.hug * rng.range_f32(0.5, 2.0)).clamp(0.002, 0.02),
                3 => g.rough = (g.rough + rng.range_f32(-0.3, 0.3)).clamp(0.0, 1.0),
                4 => g.corner_frac = (g.corner_frac + rng.range_f32(-0.3, 0.3)).clamp(0.0, 1.0),
                5 => g.guess_dx = (g.guess_dx + rng.range_f32(-0.3, 0.3)).clamp(-1.0, 1.0),
                6 => g.guess_dy = (g.guess_dy + rng.range_f32(-0.3, 0.3)).clamp(-1.0, 1.0),
                7 => g.eps_log = (g.eps_log + rng.range_f32(-2.0, 2.0)).clamp(-10.0, -2.0),
                _ => g.seed = rng.next_u64(),
            }
        }
        g
    }

    /// Materialize the genome into a frozen fixture (deterministic for a fixed genome).
    fn build(&self) -> Fixture {
        let mut rng = Lcg(self.seed | 1);
        let blocks = self.blocks as i32;
        let mut tiles = Vec::with_capacity(self.n_tiles as usize);
        for t in 0..self.n_tiles {
            let jt = self.hug * (0.6 + t as f32 * 0.25);
            let mut tile = Vec::new();
            for bx in 0..blocks {
                for by in 0..blocks {
                    let kx = (2 * bx + 1) as f32 * RES;
                    let ky = (2 * by + 1) as f32 * RES;
                    let kz = RES;
                    for dx in 0..2_i32 {
                        for dy in 0..2_i32 {
                            for dz in 0..2_i32 {
                                let sx = if dx == 0 { -1.0 } else { 1.0 };
                                let sy = if dy == 0 { -1.0 } else { 1.0 };
                                let sz = if dz == 0 { -1.0 } else { 1.0 };
                                for i in 0..8 {
                                    let e = jt + i as f32 * 0.002;
                                    // Corner-hug position, blended toward a random in-voxel
                                    // position by `rough`.
                                    let hug_p = [
                                        kx + sx * e,
                                        ky + sy * (e * 0.8 + 0.003),
                                        kz + sz * (e * 0.6 + 0.006),
                                    ];
                                    let rnd_p = [
                                        kx + sx * rng.range_f32(0.05, 1.95),
                                        ky + sy * rng.range_f32(0.05, 1.95),
                                        kz + sz * rng.range_f32(0.05, 1.95),
                                    ];
                                    tile.push([
                                        hug_p[0] + (rnd_p[0] - hug_p[0]) * self.rough,
                                        hug_p[1] + (rnd_p[1] - hug_p[1]) * self.rough,
                                        hug_p[2] + (rnd_p[2] - hug_p[2]) * self.rough,
                                    ]);
                                }
                            }
                        }
                    }
                }
            }
            tiles.push(tile);
        }
        // Source: corner-sitters (max-K) + random rovers (kd traversal), fixed P.
        let extent = (2 * blocks) as f32 * RES;
        let n_corner = ((SRC_POINTS as f32) * self.corner_frac) as usize;
        let mut src = Vec::with_capacity(SRC_POINTS);
        for i in 0..SRC_POINTS {
            if i < n_corner {
                let k = (i as i32) % (blocks * blocks);
                let kx = (2 * (k % blocks) + 1) as f32 * RES;
                let ky = (2 * (k / blocks) + 1) as f32 * RES;
                src.push([kx, ky, RES]);
            } else {
                src.push([
                    rng.next_f32() * extent,
                    rng.next_f32() * extent,
                    rng.next_f32() * 2.0 * RES,
                ]);
            }
        }
        let mut guess = Matrix4::<f32>::identity();
        guess[(0, 3)] = self.guess_dx;
        guess[(1, 3)] = self.guess_dy;
        Fixture {
            tiles,
            source: src,
            guess,
            params: NdtParams {
                trans_epsilon: f64::from(10.0_f32.powf(self.eps_log)),
                step_size: 0.1,
                resolution: f64::from(RES),
                max_iterations: MAX_ITER,
                outlier_ratio: 0.55,
                regularization: None,
                num_threads: 1,
            },
        }
    }
}

/// Lexicographic fitness: `(iterations, sum_neighbors, kd_nodes_visited)`.
type Fitness = (u64, u64, u64);

/// Counted align; also returns the align wall time (used only by the `time` fitness
/// ablation — the counter fitness never reads it, so determinism is unaffected).
fn evaluate(fx: &Fixture) -> (Fitness, u128) {
    let mut map = VoxelGridMap::new([fx.params.resolution; 3], 6, 0.01);
    for (id, tile) in fx.tiles.iter().enumerate() {
        map.add_target(tile, &(id as u64).to_be_bytes());
    }
    map.try_create_kdtree(418_000).expect("build kd-tree");
    let mut ws = AlignWorkspace::with_capacity(fx.source.len());
    let mut out = AlignResult::default();
    let t0 = std::time::Instant::now();
    align(&map, &fx.source, &fx.guess, &fx.params, &mut ws, &mut out);
    let ns = t0.elapsed().as_nanos();
    let c = out.counters;
    (
        (
            u64::try_from(out.iteration_num.max(0)).unwrap_or(0),
            c.sum_neighbors,
            c.kd_nodes_visited,
        ),
        ns,
    )
}

/// Non-dominated archive over (iter, Σnbr, kd) maximization + best-so-far trace.
struct Recorder {
    archive: Vec<(Genome, Fitness)>,
    trace: Vec<(usize, Fitness)>,
    best: Option<Fitness>,
    evals: usize,
}

impl Recorder {
    fn new() -> Recorder {
        Recorder {
            archive: Vec::new(),
            trace: Vec::new(),
            best: None,
            evals: 0,
        }
    }

    fn record(&mut self, g: &Genome, f: Fitness) {
        self.evals += 1;
        if self.best.is_none_or(|b| f > b) {
            self.best = Some(f);
            self.trace.push((self.evals, f));
        }
        let dominated_by_existing = self
            .archive
            .iter()
            .any(|(_, a)| a.0 >= f.0 && a.1 >= f.1 && a.2 >= f.2);
        if !dominated_by_existing {
            self.archive
                .retain(|(_, a)| !(f.0 >= a.0 && f.1 >= a.1 && f.2 >= a.2));
            self.archive.push((g.clone(), f));
        }
    }

    /// First evaluation whose best-so-far (iter, Σnbr) equals the final best pair.
    fn saturation_eval(&self) -> usize {
        let Some(best) = self.best else { return 0 };
        self.trace
            .iter()
            .find(|(_, f)| (f.0, f.1) == (best.0, best.1))
            .map_or(0, |(e, _)| *e)
    }
}

fn genome_json(g: &Genome) -> String {
    format!(
        "{{\"n_tiles\":{},\"blocks\":{},\"hug\":{:.4},\"rough\":{:.3},\"corner_frac\":{:.3},\
         \"guess_dx\":{:.3},\"guess_dy\":{:.3},\"eps_log\":{:.2},\"seed\":{}}}",
        g.n_tiles,
        g.blocks,
        g.hug,
        g.rough,
        g.corner_frac,
        g.guess_dx,
        g.guess_dy,
        g.eps_log,
        g.seed
    )
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn main() {
    let out_dir: PathBuf = std::env::args().nth(1).map_or_else(
        || Path::new("../../bench/fixtures").to_path_buf(),
        Into::into,
    );
    std::fs::create_dir_all(&out_dir).expect("create fixture dir");
    let gens = env_usize("WCET_SEARCH_GENS", 20);
    let pop_n = env_usize("WCET_SEARCH_POP", 6);
    let top_k = env_usize("WCET_SEARCH_TOPK", 2);
    let seed = std::env::var("WCET_SEARCH_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5EED_5EED_u64);

    let mode = std::env::var("WCET_SEARCH_MODE").unwrap_or_else(|_| "hill".into());
    let fit_time = std::env::var("WCET_SEARCH_FITNESS").is_ok_and(|v| v == "time");
    let json_path = std::env::var("WCET_SEARCH_JSON").ok();
    let pareto_dir = std::env::var("WCET_SEARCH_PARETO_DIR").ok();
    let mut rec = Recorder::new();

    let mut rng = Lcg(seed);
    // The random-search baseline must not inherit the hand-built domain-knowledge seeds --
    // its initial population is fully random (budget stays identical).
    let initial = if mode == "random" {
        (0..pop_n)
            .map(|_| Genome::random(&mut rng))
            .collect::<Vec<_>>()
    } else {
        Genome::seed_population(&mut rng, pop_n)
    };
    let mut pop: Vec<(Genome, Fitness, u128)> = initial
        .into_iter()
        .map(|g| {
            let (f, ns) = evaluate(&g.build());
            rec.record(&g, f);
            (g, f, ns)
        })
        .collect();
    println!(
        "wcet_search: mode={mode} pop={pop_n} gens={gens} seed={seed:#x} (fitness: {})",
        if fit_time {
            "align wall time"
        } else {
            "iter, Σnbr, kd"
        }
    );

    if mode == "random" {
        // Budget-matched random-search baseline: gens x pop fresh genomes.
        for generation in 0..gens {
            for i in 0..pop.len() {
                let cand = Genome::random(&mut rng);
                let (f, ns) = evaluate(&cand.build());
                rec.record(&cand, f);
                let better = if fit_time {
                    ns > pop[i].2
                } else {
                    f > pop[i].1
                };
                if better {
                    pop[i] = (cand, f, ns);
                }
            }
            let best = pop.iter().map(|(_, f, _)| *f).max().unwrap();
            println!(
                "  gen {generation:>3}: best iter={} Σnbr={} kd={}",
                best.0, best.1, best.2
            );
        }
    } else {
        for generation in 0..gens {
            for i in 0..pop.len() {
                let cand = pop[i].0.mutate(&mut rng);
                let (f, ns) = evaluate(&cand.build());
                rec.record(&cand, f);
                let better = if fit_time {
                    ns > pop[i].2
                } else {
                    f > pop[i].1
                };
                if better {
                    pop[i] = (cand, f, ns);
                }
            }
            let best = pop.iter().map(|(_, f, _)| *f).max().unwrap();
            println!(
                "  gen {generation:>3}: best iter={} Σnbr={} kd={}",
                best.0, best.1, best.2
            );
        }
    }

    if let Some(dir) = &pareto_dir {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create pareto dir");
        let mut frontier = rec.archive.clone();
        frontier.sort_by_key(|(_, f)| std::cmp::Reverse(*f));
        println!(
            "pareto archive ({} non-dominated) -> {}",
            frontier.len(),
            dir.display()
        );
        for (i, (g, f)) in frontier.iter().enumerate() {
            let path = dir.join(format!("pareto_{i:02}.ndtfix"));
            g.build().write(&path).expect("write pareto fixture");
            println!("  pareto_{i:02}: iter={} Σnbr={} kd={}", f.0, f.1, f.2);
        }
    }

    if let Some(path) = &json_path {
        let best = rec.best.unwrap_or((0, 0, 0));
        let trace: Vec<String> = rec
            .trace
            .iter()
            .map(|(e, f)| format!("[{},{},{},{}]", e, f.0, f.1, f.2))
            .collect();
        let archive: Vec<String> = rec
            .archive
            .iter()
            .map(|(g, f)| {
                format!(
                    "{{\"iter\":{},\"nbr\":{},\"kd\":{},\"genome\":{}}}",
                    f.0,
                    f.1,
                    f.2,
                    genome_json(g)
                )
            })
            .collect();
        let champ_ns = pop.iter().map(|(_, _, ns)| *ns).max().unwrap_or(0);
        let json = format!(
            "{{\n \"mode\": \"{mode}\",\n \"fitness\": \"{}\",\n \"seed\": {seed},\n \
             \"budget\": {},\n \"best\": {{\"iter\": {}, \"nbr\": {}, \"kd\": {}}},\n \
             \"saturation_eval\": {},\n \"champion_align_ns\": {champ_ns},\n \
             \"trace\": [{}],\n \"pareto\": [{}]\n}}\n",
            if fit_time { "time" } else { "counters" },
            rec.evals,
            best.0,
            best.1,
            best.2,
            rec.saturation_eval(),
            trace.join(","),
            archive.join(",")
        );
        std::fs::write(path, json).expect("write WCET_SEARCH_JSON");
        println!("summary -> {path}");
    }

    pop.sort_by_key(|(_, f, _)| std::cmp::Reverse(*f));
    println!("top-{top_k} frozen -> {}", out_dir.display());
    for (rank, (g, f, _)) in pop.iter().take(top_k).enumerate() {
        let fx = g.build();
        let path = out_dir.join(format!("search_{rank:02}.ndtfix"));
        fx.write(&path).expect("write fixture");
        println!(
            "  search_{rank:02}: iter={} Σnbr={} kd={}  (tiles={} blocks={} hug={:.3} rough={:.2} \
             cf={:.2} guess=({:.2},{:.2}) eps=1e{:.1})",
            f.0,
            f.1,
            f.2,
            g.n_tiles,
            g.blocks,
            g.hug,
            g.rough,
            g.corner_frac,
            g.guess_dx,
            g.guess_dy,
            g.eps_log,
        );
    }
}
