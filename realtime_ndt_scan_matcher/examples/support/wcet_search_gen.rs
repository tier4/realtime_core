use nalgebra::Matrix4;
use realtime_ndt_scan_matcher::fixture::Fixture;
use realtime_ndt_scan_matcher::ndt::NdtParams;

const SRC_POINTS: usize = 2000;
const RES: f32 = 2.0;
const MAX_ITER: i32 = 30;

/// Deterministic LCG used by the synthetic search-fixture generator.
#[derive(Clone)]
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn next_u64(&mut self) -> u64 {
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

/// Input-generator parameters mutated by the counter-guided search.
#[derive(Clone, Debug)]
pub(crate) struct Genome {
    pub(crate) n_tiles: u64,
    pub(crate) blocks: u64,
    pub(crate) hug: f32,
    pub(crate) rough: f32,
    pub(crate) corner_frac: f32,
    pub(crate) guess_dx: f32,
    pub(crate) guess_dy: f32,
    pub(crate) eps_log: f32,
    pub(crate) seed: u64,
}

impl Genome {
    pub(crate) fn seed_population(rng: &mut Lcg, pop: usize) -> Vec<Genome> {
        let mut v = Vec::with_capacity(pop);
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
        while v.len() < pop {
            v.push(Genome::random(rng));
        }
        v
    }

    pub(crate) fn random(rng: &mut Lcg) -> Genome {
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

    pub(crate) fn mutate(&self, rng: &mut Lcg) -> Genome {
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

    pub(crate) fn build(&self) -> Fixture {
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

        let extent = (2 * blocks) as f32 * RES;
        let n_corner = ((SRC_POINTS as f32) * self.corner_frac) as usize;
        let mut source = Vec::with_capacity(SRC_POINTS);
        for i in 0..SRC_POINTS {
            if i < n_corner {
                let k = (i as i32) % (blocks * blocks);
                let kx = (2 * (k % blocks) + 1) as f32 * RES;
                let ky = (2 * (k / blocks) + 1) as f32 * RES;
                source.push([kx, ky, RES]);
            } else {
                source.push([
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
            source,
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
