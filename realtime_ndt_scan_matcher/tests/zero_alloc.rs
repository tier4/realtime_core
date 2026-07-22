//! Zero-allocation gate for the per-frame derivative pass (the runtime allocation policy).
//! A counting global allocator asserts that `compute_derivatives` performs
//! **no heap allocation after warmup**: the first call may grow the workspace's neighbor buffer,
//! but a second call with the same inputs must reuse it (`clear()` keeps capacity).
//!
//! This lives in its own integration-test binary so the global allocator does not perturb the unit
//! tests.

#![allow(
    unsafe_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::allow_attributes,
    reason = "test code"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use nalgebra::{Matrix4, Vector3, Vector6};
use realtime_ndt_scan_matcher::ndt::{
    AlignResult, AlignWorkspace, NdtParams, ScoreConfig, align, compute_derivatives,
};
use realtime_ndt_scan_matcher::transform::{gauss_constants, transform_point};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;

/// A pass-through allocator that counts allocations while `ENABLED` is set. The default
/// `GlobalAlloc::realloc` routes through `alloc`, so `Vec` growth is counted too.
struct Counting;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);

// SAFETY: delegates every call to the System allocator unchanged; only adds an atomic counter.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::SeqCst) {
            ALLOCS.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: same layout contract as the System allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr/layout came from System.alloc above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `f` with allocation counting enabled and return the number of allocations it performed.
/// The `ENABLED` flag is process-global, so the two measurements live in ONE `#[test]` (sequential)
/// — separate parallel tests would have one test's setup counted in the other's window.
fn count_allocs<R>(f: impl FnOnce() -> R) -> u64 {
    let before = ALLOCS.load(Ordering::SeqCst);
    ENABLED.store(true, Ordering::SeqCst);
    let _r = f();
    ENABLED.store(false, Ordering::SeqCst);
    ALLOCS.load(Ordering::SeqCst).saturating_sub(before)
}

fn dense_cluster(cx: f32, cy: f32, cz: f32) -> Vec<[f32; 3]> {
    (0..8)
        .map(|i| {
            let f = i as f32 * 0.02;
            [cx + f, cy - f, cz + 0.5 * f]
        })
        .collect()
}

fn transform_cloud(source: &[[f32; 3]], p: &Vector6<f64>) -> Vec<[f32; 3]> {
    source
        .iter()
        .map(|s| {
            let v = Vector3::new(f64::from(s[0]), f64::from(s[1]), f64::from(s[2]));
            let t = transform_point(p, &v);
            [t[0] as f32, t[1] as f32, t[2] as f32]
        })
        .collect()
}

// Both measurements run sequentially in ONE test (the counting flag is process-global; see
// count_allocs). compute_derivatives must be allocation-free after warmup; align's per-frame
// allocation must be a small O(1) constant (SVD-internal), recorded in
// porting_notes/ndt_wcet_audit.md.
#[test]
fn engine_allocations_after_warmup() {
    // --- compute_derivatives: 0 allocations after warmup ---
    {
        let mut map = VoxelGridMap::new([1.0, 1.0, 1.0], 6, 0.01);
        map.add_target(&dense_cluster(0.5, 0.5, 0.5), b"0");
        map.add_target(&dense_cluster(2.5, 0.5, 0.5), b"1");
        map.try_create_kdtree(418_000).expect("build kd-tree");

        let source: Vec<[f32; 3]> = vec![[0.55, 0.5, 0.5], [0.5, 0.45, 0.52], [2.55, 0.5, 0.5]];
        let p = Vector6::new(0.05, -0.03, 0.02, 0.04, -0.02, 0.03);
        let trans = transform_cloud(&source, &p);
        let gauss = gauss_constants(0.55, 1.0);
        let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");

        let warm = compute_derivatives(
            &map,
            &source,
            &trans,
            &p,
            &ScoreConfig {
                resolution: 1.0,
                gauss: &gauss,
                reg: None,
            },
            &mut ws,
        )
        .expect("warm derivative computation");
        assert!(
            warm.score > 0.0,
            "fixture should produce a non-trivial score"
        );

        let allocs = count_allocs(|| {
            compute_derivatives(
                &map,
                &source,
                &trans,
                &p,
                &ScoreConfig {
                    resolution: 1.0,
                    gauss: &gauss,
                    reg: None,
                },
                &mut ws,
            )
        });
        assert_eq!(
            allocs, 0,
            "compute_derivatives allocated {allocs} time(s) after warmup"
        );
    }

    // --- align: small O(1) per-frame allocation after warmup ---
    let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
    for (id, &(cx, cy, cz)) in [
        (0.0_f32, 0.0_f32, 0.0_f32),
        (8.0, 0.0, 0.0),
        (0.0, 8.0, 0.0),
        (0.0, 0.0, 8.0),
        (8.0, 8.0, 8.0),
    ]
    .iter()
    .enumerate()
    {
        map.add_target(&dense_cluster(cx, cy, cz), &(id as u64).to_be_bytes());
    }
    map.try_create_kdtree(418_000).expect("build kd-tree");

    // ~40-point source (8 per cluster) translated by a known offset — large enough that an O(P)
    // per-frame allocation would show up clearly.
    let centers = [
        (0.0_f32, 0.0_f32, 0.0_f32),
        (8.0, 0.0, 0.0),
        (0.0, 8.0, 0.0),
        (0.0, 0.0, 8.0),
        (8.0, 8.0, 8.0),
    ];
    let mut source: Vec<[f32; 3]> = Vec::new();
    for &(cx, cy, cz) in &centers {
        for q in dense_cluster(cx, cy, cz) {
            source.push([q[0] + 0.2, q[1] - 0.1, q[2]]);
        }
    }
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

    let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
    let mut out = AlignResult::try_with_capacity(30).expect("reserve result");

    // Warm up (grows neighbor_idx + pre-reserves the result/cloud buffers).
    align(&map, &source, &guess, &params, &mut ws, &mut out).expect("warm align");
    // Second align with the same shapes: buffers reused, no growth.
    let allocs = count_allocs(|| align(&map, &source, &guess, &params, &mut ws, &mut out));

    eprintln!(
        "align allocations after warmup: {allocs} (points: {}, iterations: {})",
        source.len(),
        out.iteration_num
    );
    // The hot path is zero-alloc after warmup: all buffers (result Vecs, trans_cloud via
    // transform_cloud_f32, neighbor_idx bounded by MAX_NEIGHBORS) are pre-reserved + reused, and the
    // fixed-size 6x6 SVD is stack-only. (The earlier 1 alloc/frame was a trans_cloud over-reserve,
    // now fixed — see porting_notes/ndt_wcet_audit.md.)
    assert_eq!(
        allocs,
        0,
        "align allocated {allocs} times after warmup with {} points / {} iterations — the hot path \
         must be zero-alloc",
        source.len(),
        out.iteration_num
    );

    // --- pre-reserved workspace: zero allocations INCLUDING the first frame (WCET "hard zero") ---
    // A growth event is a WCET spike, so the amortized warmup above is not enough for a bound:
    // AlignWorkspace::with_capacity + a pre-reserved result must make even the first align
    // allocation-free (plan/ndt_wcet.md, M2).
    let mut ws2 = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
    let iter_cap = usize::try_from(params.max_iterations).unwrap();
    let mut out2 = AlignResult::try_with_capacity(iter_cap).expect("reserve result");

    let first_frame_allocs =
        count_allocs(|| align(&map, &source, &guess, &params, &mut ws2, &mut out2));
    assert_eq!(
        first_frame_allocs, 0,
        "pre-reserved align allocated {first_frame_allocs} time(s) on the FIRST frame — \
         with_capacity must make every frame allocation-free"
    );
    assert_eq!(
        out2.iteration_num, out.iteration_num,
        "same work as the warm run"
    );

    // --- engine path: MatchScratch::with_capacity makes the FIRST engine align allocation-free ---
    // (Same process-global counter, so this stays in the single sequential test.)
    {
        use realtime_ndt_scan_matcher::engine::{
            ConvergenceParams, MatchScratch, NdtEngine, run_align,
        };
        let engine = NdtEngine::new(2.0, 6, 0.01, 2_000, 418_000, 30).expect("valid limits");
        for (id, &(cx, cy, cz)) in centers.iter().enumerate() {
            engine.add_target(&dense_cluster(cx, cy, cz), &(id as u64).to_be_bytes());
        }
        engine.create_kdtree().expect("build kd-tree");
        let max_iter = usize::try_from(engine.max_iterations()).unwrap();
        let mut scratch =
            MatchScratch::try_with_capacity(source.len(), max_iter).expect("reserve scratch");
        let convergence = ConvergenceParams {
            converged_param_type: 0,
            converged_param_transform_probability: 0.0,
            converged_param_nearest_voxel_transformation_likelihood: 0.0,
        };
        let engine_first_allocs =
            count_allocs(|| run_align(&engine, &guess, &source, &convergence, &mut scratch));
        assert_eq!(
            engine_first_allocs, 0,
            "engine run_align allocated {engine_first_allocs} time(s) on the first frame with a \
             pre-reserved MatchScratch"
        );
    }
}
