// Copyright 2024 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Deployment-tier worst-input search at **production parameters** (`plan/ndt_wcet.md` follow-up).
//!
//! The engine-level search (`wcet_search.rs`) pins the iteration cap via `trans_epsilon = 1e-10`;
//! this search freezes the params to Autoware's shipped values (eps = 0.01, step 0.1,
//! resolution 2.0, max_iterations 30) and searches **only over preprocessing-legal geometry**:
//!
//! - source: one point per 3 m downsample cell (leaf 3.0), no duplicates, `P = 1500`
//!   (`random_downsample` cap), inside the ±60 m crop — checked by [`verify_legal`];
//! - map: a single tile (disjoint-tile world); per-voxel points on a 0.4 m sub-grid
//!   (pairwise ≥ 0.4 m ⇒ survives a 0.2 m map downsample);
//! - the only remaining freedoms are the *shape* of the map surface (per-column height,
//!   clustering), the source composition (corner-snapped vs roving), and the initial guess.
//!
//! Fitness is the lexicographic counter vector `(iterations, Σneighbors, kd_nodes)`: the goal is
//! a **legal geometry that oscillates to the 30-iteration cap at eps = 0.01**, then maximal work
//! at that cap. The champion is frozen to `bench/fixtures/legal_osc.ndtfix` (only if it passes
//! [`verify_legal`]).
//!
//! Usage: `cargo run --release --features wcet-count --example wcet_search_legal [OUT_DIR]`
//! Env: `WCET_SEARCH_GENS` (default 25), `WCET_SEARCH_POP` (default 8), `WCET_SEARCH_SEED`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::needless_range_loop,
    reason = "example/benchmark tooling"
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use nalgebra::Matrix4;
use realtime_ndt_scan_matcher::fixture::Fixture;
use realtime_ndt_scan_matcher::ndt::{AlignResult, AlignWorkspace, NdtParams, align};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;

const RES: f32 = 2.0;
/// Production preprocessing contract (autoware_launch localization defaults).
const SCAN_LEAF: f32 = 3.0;
const CROP: f32 = 60.0;
const P_CAP: usize = 1500;
/// Map sub-grid pitch: pairwise ≥ 0.4 m survives a 0.2 m map downsample.
const SUBGRID: [f32; 5] = [0.2, 0.6, 1.0, 1.4, 1.8];

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

fn production_params() -> NdtParams {
    NdtParams {
        trans_epsilon: 0.01,
        step_size: 0.1,
        resolution: f64::from(RES),
        max_iterations: 30,
        outlier_ratio: 0.55,
        regularization: None,
        num_threads: 1,
    }
}

/// Search genome — geometry/guess freedoms only; params are frozen to production.
#[derive(Clone, Debug)]
struct Genome {
    /// Map half-extent in voxels (map covers ±2·half m).
    half: u64,
    /// Surface height range as a fraction of the voxel (0..1 → z ∈ sub-grid span).
    amp: f32,
    /// Vertical clustering of per-voxel points around the surface (0 = tight, 1 = full spread).
    cluster: f32,
    /// Fraction of source points snapped to 4 m lattice corners (K-adversarial) vs rovers.
    corner_frac: f32,
    /// Source z placement (within the first 3 m z band).
    src_z: f32,
    /// Initial-guess offset (m) and yaw (rad).
    gdx: f32,
    gdy: f32,
    gyaw: f32,
    /// Point-stream seed.
    seed: u64,
}

impl Genome {
    fn seeds(rng: &mut Lcg, pop: usize) -> Vec<Genome> {
        let mut v = Vec::with_capacity(pop);
        // Seed 0: rough far-guess surface (the 22/30 mechanism, legalized).
        v.push(Genome {
            half: 25,
            amp: 1.0,
            cluster: 0.9,
            corner_frac: 0.0,
            src_z: 1.0,
            gdx: 0.7,
            gdy: 0.5,
            gyaw: 0.0,
            seed: 11,
        });
        // Seed 1: corner lattice (the K = 8 mechanism).
        v.push(Genome {
            half: 30,
            amp: 0.0,
            cluster: 0.2,
            corner_frac: 1.0,
            src_z: 1.0,
            gdx: 0.08,
            gdy: -0.06,
            gyaw: 0.0,
            seed: 12,
        });
        while v.len() < pop {
            v.push(Genome {
                half: rng.range_u(15, 31),
                amp: rng.next_f32(),
                cluster: rng.next_f32(),
                corner_frac: rng.next_f32(),
                src_z: rng.range_f32(0.2, 2.5),
                gdx: rng.range_f32(-1.0, 1.0),
                gdy: rng.range_f32(-1.0, 1.0),
                gyaw: rng.range_f32(-0.15, 0.15),
                seed: rng.next_u64(),
            });
        }
        v
    }

    fn mutate(&self, rng: &mut Lcg) -> Genome {
        let mut g = self.clone();
        for _ in 0..rng.range_u(1, 2) {
            match rng.range_u(0, 8) {
                0 => g.half = rng.range_u(15, 31),
                1 => g.amp = (g.amp + rng.range_f32(-0.3, 0.3)).clamp(0.0, 1.0),
                2 => g.cluster = (g.cluster + rng.range_f32(-0.3, 0.3)).clamp(0.0, 1.0),
                3 => g.corner_frac = (g.corner_frac + rng.range_f32(-0.3, 0.3)).clamp(0.0, 1.0),
                4 => g.src_z = (g.src_z + rng.range_f32(-0.5, 0.5)).clamp(0.2, 2.5),
                5 => g.gdx = (g.gdx + rng.range_f32(-0.3, 0.3)).clamp(-1.0, 1.0),
                6 => g.gdy = (g.gdy + rng.range_f32(-0.3, 0.3)).clamp(-1.0, 1.0),
                7 => g.gyaw = (g.gyaw + rng.range_f32(-0.05, 0.05)).clamp(-0.15, 0.15),
                _ => g.seed = rng.next_u64(),
            }
        }
        g
    }

    /// Materialize into a fixture (legal by construction; re-checked by `verify_legal`).
    fn build(&self) -> Fixture {
        let mut rng = Lcg(self.seed | 1);
        let half = self.half as i32;
        // --- map: single tile, one z voxel layer [0, 2), rough surface via sub-grid choice ---
        let mut map = Vec::new();
        for i in -half..half {
            for j in -half..half {
                // Per-column surface height index into the z sub-grid (0..5), range set by amp.
                let h = (self.amp * rng.next_f32() * 4.999) as usize;
                let mut used = BTreeSet::new();
                let mut placed = 0;
                while placed < 7 {
                    let sx = rng.range_u(0, 4) as usize;
                    let sy = rng.range_u(0, 4) as usize;
                    // z sub-cell: near the surface height, spread by `cluster`.
                    let spread = (self.cluster * 4.999) as i64;
                    let dz = if spread == 0 {
                        0
                    } else {
                        rng.range_u(0, (2 * spread) as u64) as i64 - spread
                    };
                    let sz = (h as i64 + dz).clamp(0, 4) as usize;
                    if used.insert((sx, sy, sz)) {
                        map.push([
                            i as f32 * RES + SUBGRID[sx],
                            j as f32 * RES + SUBGRID[sy],
                            SUBGRID[sz],
                        ]);
                        placed += 1;
                    }
                }
            }
        }
        // --- source: one point per 3 m cell, P = 1500, inside the crop ---
        // Iterate the 3 m xy cells in a seeded shuffled order; per cell, either snap to a 4 m
        // lattice corner inside the cell (K-adversarial) or place a jittered rover.
        let n_cells = (2.0 * CROP / SCAN_LEAF) as i32; // 40
        let mut cells: Vec<(i32, i32)> = (0..n_cells)
            .flat_map(|u| (0..n_cells).map(move |v| (u, v)))
            .collect();
        // Fisher-Yates with the LCG.
        for k in (1..cells.len()).rev() {
            let r = (rng.next_u64() % (k as u64 + 1)) as usize;
            cells.swap(k, r);
        }
        let mut src = Vec::with_capacity(P_CAP);
        for (u, v) in cells {
            if src.len() == P_CAP {
                break;
            }
            let x0 = -CROP + u as f32 * SCAN_LEAF; // cell [x0, x0+3)
            let y0 = -CROP + v as f32 * SCAN_LEAF;
            // 4 m lattice corner inside this cell, if any.
            let cx = (x0 / 4.0).ceil() * 4.0;
            let cy = (y0 / 4.0).ceil() * 4.0;
            let corner_in = cx < x0 + SCAN_LEAF && cy < y0 + SCAN_LEAF;
            if corner_in && rng.next_f32() < self.corner_frac {
                src.push([cx, cy, self.src_z]);
            } else {
                src.push([
                    x0 + rng.range_f32(0.2, SCAN_LEAF - 0.2),
                    y0 + rng.range_f32(0.2, SCAN_LEAF - 0.2),
                    self.src_z + rng.range_f32(-0.4, 0.4),
                ]);
            }
        }
        // --- guess: translation + yaw ---
        let (s, c) = (self.gyaw.sin(), self.gyaw.cos());
        let mut guess = Matrix4::<f32>::identity();
        guess[(0, 0)] = c;
        guess[(0, 1)] = -s;
        guess[(1, 0)] = s;
        guess[(1, 1)] = c;
        guess[(0, 3)] = self.gdx;
        guess[(1, 3)] = self.gdy;
        Fixture {
            tiles: vec![map],
            source: src,
            guess,
            params: production_params(),
        }
    }
}

/// Independent legality check of the deployment contract (not trusted to the generator).
fn verify_legal(fx: &Fixture) -> Result<(), String> {
    let p = &fx.params;
    if p.trans_epsilon != 0.01 || p.step_size != 0.1 || p.max_iterations != 30 {
        return Err("params are not the production set".into());
    }
    if fx.tiles.len() != 1 {
        return Err("multi-tile map (disjoint-tile world requires 1)".into());
    }
    if fx.source.len() > P_CAP {
        return Err(format!("P = {} > {P_CAP}", fx.source.len()));
    }
    // Source: inside the crop + one point per 3 m cell.
    let mut cells = BTreeSet::new();
    for q in &fx.source {
        if q[0].abs() > CROP || q[1].abs() > CROP {
            return Err(format!("source point outside the ±{CROP} m crop: {q:?}"));
        }
        let cell = (
            (q[0] / SCAN_LEAF).floor() as i64,
            (q[1] / SCAN_LEAF).floor() as i64,
            (q[2] / SCAN_LEAF).floor() as i64,
        );
        if !cells.insert(cell) {
            return Err(format!("two source points share the 3 m cell {cell:?}"));
        }
    }
    // Map: per-voxel pairwise spacing ≥ 0.35 m (survives a 0.2 m map downsample).
    let mut by_voxel: std::collections::BTreeMap<(i64, i64, i64), Vec<[f32; 3]>> =
        std::collections::BTreeMap::new();
    for q in &fx.tiles[0] {
        let v = (
            (q[0] / RES).floor() as i64,
            (q[1] / RES).floor() as i64,
            (q[2] / RES).floor() as i64,
        );
        by_voxel.entry(v).or_default().push(*q);
    }
    for pts in by_voxel.values() {
        for (a, qa) in pts.iter().enumerate() {
            for qb in pts.iter().skip(a + 1) {
                let d2 =
                    (qa[0] - qb[0]).powi(2) + (qa[1] - qb[1]).powi(2) + (qa[2] - qb[2]).powi(2);
                if d2 < 0.35_f32.powi(2) {
                    return Err(format!("map points closer than 0.35 m: {qa:?} vs {qb:?}"));
                }
            }
        }
    }
    Ok(())
}

type Fitness = (u64, u64, u64);

/// Fitness mode (`WCET_SEARCH_FITNESS`): `osc` (default) maximizes the lexicographic
/// `(iterations, Σneighbors, kd_nodes)` vector; `maxk` (plan/paper_fix.md B3) maximizes
/// `(max per-point K, iterations, Σneighbors)` to hunt a legal witness above the K = 8
/// voxel-center heuristic (the provable ceiling is 27).
fn fitness_is_maxk() -> bool {
    std::env::var("WCET_SEARCH_FITNESS").is_ok_and(|v| v == "maxk")
}

fn evaluate(fx: &Fixture) -> Fitness {
    let mut map = VoxelGridMap::new([fx.params.resolution; 3], 6, 0.01);
    for (id, tile) in fx.tiles.iter().enumerate() {
        map.add_target(tile, &(id as u64).to_be_bytes());
    }
    map.try_create_kdtree(418_000).expect("build kd-tree");
    let mut ws = AlignWorkspace::with_capacity(fx.source.len());
    let mut out = AlignResult::default();
    align(&map, &fx.source, &fx.guess, &fx.params, &mut ws, &mut out);
    let c = out.counters;
    if fitness_is_maxk() {
        (
            c.max_neighbors,
            u64::try_from(out.iteration_num.max(0)).unwrap_or(0),
            c.sum_neighbors,
        )
    } else {
        (
            u64::try_from(out.iteration_num.max(0)).unwrap_or(0),
            c.sum_neighbors,
            c.kd_nodes_visited,
        )
    }
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
    let gens = env_usize("WCET_SEARCH_GENS", 25);
    let pop_n = env_usize("WCET_SEARCH_POP", 8);
    let seed = std::env::var("WCET_SEARCH_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x1E9A_05C1_u64);

    let mut rng = Lcg(seed);
    let mut pop: Vec<(Genome, Fitness)> = Genome::seeds(&mut rng, pop_n)
        .into_iter()
        .map(|g| {
            let fx = g.build();
            verify_legal(&fx).expect("seed genome must be legal");
            let f = evaluate(&fx);
            (g, f)
        })
        .collect();
    println!(
        "wcet_search_legal: pop={pop_n} gens={gens} seed={seed:#x} \
         (production params frozen; fitness: {})",
        if fitness_is_maxk() {
            "maxK, iter, Σnbr"
        } else {
            "iter, Σnbr, kd"
        }
    );

    for generation in 0..gens {
        for i in 0..pop.len() {
            let cand = pop[i].0.mutate(&mut rng);
            let fx = cand.build();
            if verify_legal(&fx).is_err() {
                continue; // never accept an illegal candidate
            }
            let f = evaluate(&fx);
            if f > pop[i].1 {
                pop[i] = (cand, f);
            }
        }
        let best = pop.iter().map(|(_, f)| *f).max().unwrap();
        if fitness_is_maxk() {
            println!(
                "  gen {generation:>3}: best maxK={} iter={} Σnbr={}",
                best.0, best.1, best.2
            );
        } else {
            println!(
                "  gen {generation:>3}: best iter={} Σnbr={} kd={}",
                best.0, best.1, best.2
            );
        }
    }

    pop.sort_by_key(|(_, f)| std::cmp::Reverse(*f));
    let (g, f) = &pop[0];
    let fx = g.build();
    verify_legal(&fx).expect("champion must be legal");
    println!(
        "champion: {}={} {}={} {}={}  genome: half={} amp={:.2} cluster={:.2} cf={:.2} \
         src_z={:.2} guess=({:.2},{:.2},yaw {:.3}) seed={:#x}",
        if fitness_is_maxk() { "maxK" } else { "iter" },
        f.0,
        if fitness_is_maxk() { "iter" } else { "Σnbr" },
        f.1,
        if fitness_is_maxk() { "Σnbr" } else { "kd" },
        f.2,
        g.half,
        g.amp,
        g.cluster,
        g.corner_frac,
        g.src_z,
        g.gdx,
        g.gdy,
        g.gyaw,
        g.seed,
    );
    println!(
        "LEGALITY: OK (production params, 1 tile, P={} ≤ {P_CAP}, one point per 3 m cell, \
              map sub-grid ≥ 0.35 m)",
        fx.source.len()
    );
    if fitness_is_maxk() {
        // B3: freeze only a witness that beats the K = 8 voxel-center heuristic.
        if f.0 >= 9 {
            let path = out_dir.join("legal_k.ndtfix");
            fx.write(&path).expect("write fixture");
            println!(
                "max-K witness ABOVE the voxel-center heuristic (K={}) -> frozen: {}",
                f.0,
                path.display()
            );
        } else {
            println!(
                "max-K witness NOT above the heuristic (best K={} ≤ 8) -- nothing frozen",
                f.0
            );
        }
    } else if f.0 == 30 {
        let path = out_dir.join("legal_osc.ndtfix");
        fx.write(&path).expect("write fixture");
        println!(
            "iteration cap REACHED at eps=0.01 -> frozen: {}",
            path.display()
        );
    } else {
        println!(
            "iteration cap NOT reached (best {}/30) -- nothing frozen",
            f.0
        );
    }
}
