//! Materialize the four search-derived fixtures used by the paper timing campaign.
//!
//! These genomes were selected by the historical counter-guided search at Autoware commit
//! `5fad5a17`. Re-running the optimizer against a changed engine can select different inputs even
//! with the same RNG seed, so this tool stores the selected genomes as exact `f32` bit patterns.

#![allow(
    dead_code,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::cast_possible_wrap,
    reason = "example fixture tooling shares generator code with the search target"
)]

#[path = "support/wcet_search_gen.rs"]
mod wcet_search_gen;

use std::path::{Path, PathBuf};

use wcet_search_gen::Genome;

fn genome(
    blocks: u64,
    hug: u32,
    rough: u32,
    corner_frac: u32,
    translation_bits: [u32; 2],
    seed: u64,
) -> Genome {
    Genome {
        n_tiles: 8,
        blocks,
        hug: f32::from_bits(hug),
        rough: f32::from_bits(rough),
        corner_frac: f32::from_bits(corner_frac),
        guess_dx: f32::from_bits(translation_bits[0]),
        guess_dy: f32::from_bits(translation_bits[1]),
        eps_log: f32::from_bits(0xc120_0000),
        seed,
    }
}

fn main() {
    let out_dir: PathBuf = std::env::args().nth(1).map_or_else(
        || Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/bench/fixtures")).to_path_buf(),
        Into::into,
    );
    let pareto_dir = out_dir.join("pareto");
    std::fs::create_dir_all(&pareto_dir).expect("create paper fixture directories");

    let fixtures = [
        (
            out_dir.join("search_00.ndtfix"),
            genome(
                9,
                0x3bc4_9ba6,
                0x0000_0000,
                0x3f80_0000,
                [0x3da3_d70a, 0xbeb1_a1e1],
                1,
            ),
        ),
        (
            out_dir.join("search_01.ndtfix"),
            genome(
                10,
                0x3ba7_a836,
                0x3f47_2c19,
                0x3e7f_57d8,
                [0x3f44_1cc4, 0x3f31_1c54],
                2,
            ),
        ),
        (
            pareto_dir.join("pareto_01.ndtfix"),
            genome(
                9,
                0x3bc4_9ba6,
                0x0000_0000,
                0x3f68_b691,
                [0x3da3_d70a, 0xbe34_54f2],
                1,
            ),
        ),
        (
            pareto_dir.join("pareto_02.ndtfix"),
            genome(
                9,
                0x3bc4_9ba6,
                0x0000_0000,
                0x3f4e_323a,
                [0x3da3_d70a, 0xbe34_54f2],
                1,
            ),
        ),
    ];

    for (path, genome) in fixtures {
        genome.build().write(&path).expect("write paper fixture");
        println!("wrote {}", path.display());
    }
}
