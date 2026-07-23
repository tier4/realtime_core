//! WCET bound as a machine-checked property.
//!
//! The engine's frame cost decomposes as `T ≤ N_iter × Σ_p (T_search + K·T_kernel) + …`; the
//! `wcet-count` counters measure exactly those data-dependent quantities. This suite drives
//! `align` over randomized maps/clouds/params (proptest) and asserts the counters never exceed
//! the analytic bound — a violated bound is an engine bug, not a measurement artifact.
//!
//! Run with: `cargo test --features wcet-count --test wcet_bounds`

#![cfg(feature = "wcet-count")]
// Test code may relax the production lint gates (see the project allowlist).
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::allow_attributes,
    reason = "test code"
)]

use proptest::prelude::*;
use realtime_ndt_scan_matcher::nalgebra::Matrix4;
use realtime_ndt_scan_matcher::ndt::{
    AlignResult, AlignWorkspace, MAX_NEIGHBORS, NdtParams, align,
};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;

/// Build a voxel-grid map from clustered points: `clusters` cluster centers on a coarse grid, each
/// with 8 points so the voxel passes the min-points filter.
fn make_map(clusters: usize, spacing: f32) -> (VoxelGridMap, Vec<[f32; 3]>) {
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let side = (clusters as f32).sqrt().ceil() as usize;
    for c in 0..clusters {
        let cx = (c % side) as f32 * spacing;
        let cy = (c / side) as f32 * spacing;
        for i in 0..8 {
            let f = i as f32 * 0.02;
            pts.push([cx + f, cy - f, 0.5 + 0.5 * f]);
        }
    }
    let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
    map.add_target(&pts, b"0");
    map.try_create_kdtree(418_000).expect("build kd-tree");
    (map, pts)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    /// Counters never exceed the analytic bound, for randomized inputs.
    #[test]
    fn counters_respect_analytic_bound(
        clusters in 1_usize..24,
        n_source in 1_usize..160,
        max_iterations in 1_i32..12,
        guess_dx in -1.5_f64..1.5,
        guess_dy in -1.5_f64..1.5,
        spacing in 2.0_f32..4.0,
    ) {
        let (map, map_pts) = make_map(clusters, spacing);
        // Source = a subset of the map points (cycled), so searches actually find neighbors.
        let source: Vec<[f32; 3]> = (0..n_source)
            .map(|i| map_pts[i % map_pts.len()])
            .collect();
        let mut guess = Matrix4::<f32>::identity();
        guess[(0, 3)] = guess_dx as f32;
        guess[(1, 3)] = guess_dy as f32;

        let params = NdtParams {
            trans_epsilon: 0.01,
            step_size: 0.1,
            resolution: 2.0,
            max_iterations,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1, // serial WCET baseline
        };
        let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
        let iteration_capacity = usize::try_from(max_iterations).expect("positive iteration limit");
        let mut out = AlignResult::try_with_capacity(iteration_capacity).expect("reserve result");
        align(&map, &source, &guess, &params, &mut ws, &mut out).expect("align test input");
        let c = out.counters;

        // N_iter ≤ max_iterations, and one derivative pass per iteration plus the initial pass.
        prop_assert!(out.iteration_num <= max_iterations);
        let iter_u64 = u64::try_from(out.iteration_num.max(0)).unwrap();
        prop_assert!(c.derivative_passes <= iter_u64 + 1,
            "passes {} > iter {} + 1", c.derivative_passes, out.iteration_num);

        // P per pass is the source size.
        let p = u64::try_from(source.len()).unwrap();
        prop_assert!(c.points_processed == c.derivative_passes * p,
            "points {} != passes {} × P {}", c.points_processed, c.derivative_passes, p);

        // K ≤ MAX_NEIGHBORS per point per pass.
        let k = u64::try_from(MAX_NEIGHBORS).unwrap();
        prop_assert!(c.sum_neighbors <= c.points_processed * k,
            "neighbors {} > points {} × K {}", c.sum_neighbors, c.points_processed, k);

        // The per-point max witness: capped by MAX_NEIGHBORS, and the sum cannot exceed
        // points × max (the max dominates the mean).
        prop_assert!(c.max_neighbors <= k,
            "max K {} > MAX_NEIGHBORS {}", c.max_neighbors, k);
        prop_assert!(c.sum_neighbors <= c.points_processed * c.max_neighbors.max(1),
            "neighbors {} > points {} × max K {}", c.sum_neighbors, c.points_processed,
            c.max_neighbors);

        // Traversal ≤ tree nodes per query (nodes == searchable leaves).
        let nodes = u64::try_from(map.num_leaves()).unwrap();
        prop_assert!(c.kd_nodes_visited <= c.points_processed * nodes,
            "kd nodes {} > points {} × tree {}", c.kd_nodes_visited, c.points_processed, nodes);
    }
}

/// The serial and parallel backends must report identical counters (both fold the same
/// per-point contributions in point-index order).
#[cfg(feature = "parallel")]
#[test]
fn parallel_counters_match_serial() {
    let (map, map_pts) = make_map(12, 3.0);
    let source: Vec<[f32; 3]> = (0..96).map(|i| map_pts[i % map_pts.len()]).collect();
    let mut guess = Matrix4::<f32>::identity();
    guess[(0, 3)] = 0.4;

    let mk = |threads: usize| NdtParams {
        trans_epsilon: 0.01,
        step_size: 0.1,
        resolution: 2.0,
        max_iterations: 8,
        outlier_ratio: 0.55,
        regularization: None,
        num_threads: threads,
    };
    let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
    let mut serial = AlignResult::try_with_capacity(8).expect("reserve serial result");
    align(&map, &source, &guess, &mk(1), &mut ws, &mut serial).expect("serial align");
    let mut parallel = AlignResult::try_with_capacity(8).expect("reserve parallel result");
    align(&map, &source, &guess, &mk(4), &mut ws, &mut parallel).expect("parallel align");

    assert_eq!(serial.counters, parallel.counters);
    assert_eq!(serial.iteration_num, parallel.iteration_num);
}
