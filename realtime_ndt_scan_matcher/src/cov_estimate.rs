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

//! Multi-NDT covariance estimation, ported from `src/ndt_omp/estimate_covariance.cpp`. These
//! are the engine-driving estimators the node uses to produce the output XY pose covariance; the
//! pure helpers they build on live in [`crate::covariance`]. Control-plane (runs once per
//! localization frame, not the RT align hot loop), so `Vec` allocation is fine. `no_std` + `alloc`.

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::{Matrix2, Matrix4, Vector2};

use crate::covariance::{calc_weight_vec, calculate_weighted_mean_and_cov};
use crate::ndt::{
    AlignError, AlignResult, AlignWorkspace, NdtParams, align,
    nearest_voxel_transformation_likelihood,
};
use crate::transform::{gauss_constants, transform_cloud_by_matrix};
use crate::voxel_grid::VoxelGridMap;

/// Result of a multi-NDT covariance estimation: the weighted XY `mean` and the row-major 2x2
/// `covariance`, both in the map frame (the C++ `ResultOfMultiNdtCovarianceEstimation`). For the
/// `MULTI_NDT` variant, `candidate_result_poses` holds each candidate's RE-ALIGNED result pose (in
/// search order) — the node publishes these as `multi_ndt_pose`. The `_score` variant does not
/// re-align, so it leaves `candidate_result_poses` empty (its publish uses the proposed poses).
#[derive(Clone, Debug)]
pub struct MultiNdtCovResult {
    pub mean: [f64; 2],
    pub covariance: [f64; 4],
    pub candidate_result_poses: Vec<Matrix4<f32>>,
}

/// Candidate poses around `center` for the covariance search (`propose_poses_to_search`): each 2D
/// `(offset_x[i], offset_y[i])` is rotated by the center pose's 2x2 rotation and added to its
/// translation. `offset_x`/`offset_y` are paired; extra entries of the longer one are ignored.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "nalgebra f64 matrix product; constant indices into a fixed-size Matrix4; deliberate f64->f32 translation narrowing"
)]
#[must_use]
pub fn propose_poses_to_search(
    center: &Matrix4<f32>,
    offset_x: &[f64],
    offset_y: &[f64],
) -> Vec<Matrix4<f32>> {
    let rot = Matrix2::new(
        f64::from(center[(0, 0)]),
        f64::from(center[(0, 1)]),
        f64::from(center[(1, 0)]),
        f64::from(center[(1, 1)]),
    );
    let mut out = Vec::with_capacity(offset_x.len().min(offset_y.len()));
    for (&ox, &oy) in offset_x.iter().zip(offset_y.iter()) {
        let rotated = rot * Vector2::new(ox, oy);
        let mut curr = *center;
        curr[(0, 3)] += rotated.x as f32;
        curr[(1, 3)] += rotated.y as f32;
        out.push(curr);
    }
    out
}

/// `MULTI_NDT` covariance: re-run full `align` from each candidate pose, collect the converged XY
/// positions (plus the main result's), weight them uniformly (`1/n`), and take the weighted mean +
/// covariance with the unbiased `(n-1)/n` correction (`estimate_xy_covariance_by_multi_ndt`).
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "constant indices into a fixed-size Matrix4; n is a small count, the n as f64 average is deliberate"
)]
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn estimate_xy_covariance_by_multi_ndt(
    main_ndt: &AlignResult,
    poses_to_search: &[Matrix4<f32>],
    map: &VoxelGridMap,
    source: &[[f32; 3]],
    params: &NdtParams,
    ws: &mut AlignWorkspace,
) -> Result<MultiNdtCovResult, AlignError> {
    let n = poses_to_search
        .len()
        .checked_add(1)
        .ok_or(AlignError::ArithmeticOverflow)?;
    let mut poses2d: Vec<f64> = Vec::with_capacity(n.saturating_mul(2));
    poses2d.push(f64::from(main_ndt.pose[(0, 3)]));
    poses2d.push(f64::from(main_ndt.pose[(1, 3)]));

    let mut candidate_result_poses: Vec<Matrix4<f32>> = Vec::with_capacity(poses_to_search.len());
    let mut out = AlignResult::default();
    let iterations =
        usize::try_from(params.max_iterations).map_err(|_| AlignError::IterationLimitExceeded)?;
    let result_capacity = iterations
        .checked_add(1)
        .ok_or(AlignError::ArithmeticOverflow)?;
    out.transformation_array
        .try_reserve_exact(result_capacity)
        .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
    out.transform_probability_array
        .try_reserve_exact(result_capacity)
        .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
    out.nearest_voxel_likelihood_array
        .try_reserve_exact(result_capacity)
        .map_err(|_| AlignError::WorkspaceCapacityExceeded)?;
    for cand in poses_to_search {
        align(map, source, cand, params, ws, &mut out)?;
        poses2d.push(f64::from(out.pose[(0, 3)]));
        poses2d.push(f64::from(out.pose[(1, 3)]));
        candidate_result_poses.push(out.pose);
    }

    let weights = vec![1.0_f64 / n as f64; n];
    let (mean, cov) = calculate_weighted_mean_and_cov(&poses2d, &weights);
    // Unbiased covariance (C++ `covariance *= (n - 1) / n`).
    let factor = (n as f64 - 1.0) / n as f64;
    let covariance = [
        cov[0] * factor,
        cov[1] * factor,
        cov[2] * factor,
        cov[3] * factor,
    ];
    Ok(MultiNdtCovResult {
        mean,
        covariance,
        candidate_result_poses,
    })
}

/// `MULTI_NDT_SCORE` covariance: for each candidate pose, transform the source by it and score it
/// with the nearest-voxel likelihood (no re-align), then weight the candidates' own XY positions by a
/// temperature-scaled softmax of the scores (`estimate_xy_covariance_by_multi_ndt_score`). The
/// Gaussian constants are derived from `params.outlier_ratio`/`params.resolution`, exactly as `align`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "constant indices into a fixed-size Matrix4; the main result's nvtl is f32 by contract"
)]
/// # Errors
/// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
pub fn estimate_xy_covariance_by_multi_ndt_score(
    main_ndt: &AlignResult,
    poses_to_search: &[Matrix4<f32>],
    map: &VoxelGridMap,
    source: &[[f32; 3]],
    params: &NdtParams,
    temperature: f64,
    ws: &mut AlignWorkspace,
) -> Result<MultiNdtCovResult, AlignError> {
    let gauss = gauss_constants(params.outlier_ratio, params.resolution);
    let n = poses_to_search
        .len()
        .checked_add(1)
        .ok_or(AlignError::ArithmeticOverflow)?;
    let mut poses2d: Vec<f64> = Vec::with_capacity(n.saturating_mul(2));
    let mut scores: Vec<f64> = Vec::with_capacity(n);
    poses2d.push(f64::from(main_ndt.pose[(0, 3)]));
    poses2d.push(f64::from(main_ndt.pose[(1, 3)]));
    scores.push(f64::from(main_ndt.nearest_voxel_likelihood));

    let mut trans: Vec<[f32; 3]> = Vec::new();
    for cand in poses_to_search {
        // The candidate's own translation (no re-align), matching C++.
        poses2d.push(f64::from(cand[(0, 3)]));
        poses2d.push(f64::from(cand[(1, 3)]));
        transform_cloud_by_matrix(cand, source, &mut trans);
        scores.push(nearest_voxel_transformation_likelihood(
            map,
            &trans,
            params.resolution,
            &gauss,
            ws,
        )?);
    }

    let mut weights = vec![0.0_f64; n];
    calc_weight_vec(&scores, temperature, &mut weights);
    let (mean, covariance) = calculate_weighted_mean_and_cov(&poses2d, &weights);
    // The score variant does not re-align, so it has no per-candidate result poses.
    Ok(MultiNdtCovResult {
        mean,
        covariance,
        candidate_result_poses: Vec::new(),
    })
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    unsafe_code,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::borrow_as_ptr,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    fn spread_map() -> (VoxelGridMap, Vec<[f32; 3]>) {
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for &(cx, cy, cz) in &[
            (0.5_f32, 0.5_f32, 0.5_f32),
            (4.5, 0.5, 0.5),
            (0.5, 4.5, 0.5),
            (0.5, 0.5, 4.5),
            (4.5, 4.5, 4.5),
        ] {
            pts.extend(dense_cluster(cx, cy, cz));
        }
        let mut map = VoxelGridMap::new([1.0, 1.0, 1.0], 6, 0.01);
        map.add_target(&pts, b"0");
        map.try_create_kdtree(418_000).expect("build kd-tree");
        (map, pts)
    }

    fn params() -> NdtParams {
        NdtParams {
            trans_epsilon: 1e-3,
            step_size: 0.2,
            resolution: 1.0,
            max_iterations: 30,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1,
        }
    }

    // propose_poses_to_search: with identity rotation the offsets land on the translation directly.
    #[test]
    fn propose_poses_identity_rotation_adds_offsets() {
        let mut center = Matrix4::<f32>::identity();
        center[(0, 3)] = 1.0;
        center[(1, 3)] = 2.0;
        let poses = propose_poses_to_search(&center, &[0.3, -0.4], &[0.1, 0.2]);
        assert_eq!(poses.len(), 2);
        assert!((poses[0][(0, 3)] - 1.3).abs() < 1e-6);
        assert!((poses[0][(1, 3)] - 2.1).abs() < 1e-6);
        assert!((poses[1][(0, 3)] - 0.6).abs() < 1e-6);
        assert!((poses[1][(1, 3)] - 2.2).abs() < 1e-6);
    }

    // propose_poses_to_search: a 90° yaw rotation maps (ox, oy) -> (-oy, ox).
    #[test]
    fn propose_poses_rotates_offset_by_pose() {
        // R(90°): [[0,-1],[1,0]] in the top-left block.
        let mut center = Matrix4::<f32>::identity();
        center[(0, 0)] = 0.0;
        center[(0, 1)] = -1.0;
        center[(1, 0)] = 1.0;
        center[(1, 1)] = 0.0;
        let poses = propose_poses_to_search(&center, &[1.0], &[0.0]);
        // rot * (1,0) = (0,1)
        assert!(poses[0][(0, 3)].abs() < 1e-6);
        assert!((poses[0][(1, 3)] - 1.0).abs() < 1e-6);
    }

    // multi_ndt: the covariance is symmetric and PSD, and the mean is near the main pose for a
    // well-conditioned map (source == target, so every candidate re-aligns back to the center).
    #[test]
    fn multi_ndt_covariance_is_symmetric_psd() {
        let (map, source) = spread_map();
        let main_pose = Matrix4::<f32>::identity();
        let main_ndt = AlignResult {
            pose: main_pose,
            ..AlignResult::default()
        };
        let poses =
            propose_poses_to_search(&main_pose, &[0.2, -0.2, 0.0, 0.0], &[0.0, 0.0, 0.2, -0.2]);
        let mut ws =
            AlignWorkspace::try_with_capacity(source.len()).expect("reserve covariance workspace");
        let r = estimate_xy_covariance_by_multi_ndt(
            &main_ndt,
            &poses,
            &map,
            &source,
            &params(),
            &mut ws,
        )
        .expect("covariance estimate");

        // symmetric
        assert!((r.covariance[1] - r.covariance[2]).abs() < 1e-12);
        // PSD: diagonal >= 0 and determinant >= 0
        assert!(r.covariance[0] >= 0.0 && r.covariance[3] >= 0.0);
        let det = r.covariance[0] * r.covariance[3] - r.covariance[1] * r.covariance[2];
        assert!(det >= -1e-12, "det = {det}");
        // mean near the origin (the aligned poses cluster around the main pose)
        assert!(r.mean[0].abs() < 1.0 && r.mean[1].abs() < 1.0);
        // multi_ndt re-aligns each candidate, so one result pose per candidate.
        assert_eq!(r.candidate_result_poses.len(), poses.len());
    }

    // multi_ndt_score: symmetric PSD covariance; with a peaked softmax the mean stays near center.
    #[test]
    fn multi_ndt_score_covariance_is_symmetric_psd() {
        let (map, source) = spread_map();
        let main_pose = Matrix4::<f32>::identity();
        let main_ndt = AlignResult {
            pose: main_pose,
            nearest_voxel_likelihood: 1.0,
            ..AlignResult::default()
        };
        let poses =
            propose_poses_to_search(&main_pose, &[0.2, -0.2, 0.0, 0.0], &[0.0, 0.0, 0.2, -0.2]);
        let mut ws = AlignWorkspace::try_with_capacity(2_000).expect("reserve workspace");
        let r = estimate_xy_covariance_by_multi_ndt_score(
            &main_ndt,
            &poses,
            &map,
            &source,
            &params(),
            0.1,
            &mut ws,
        )
        .expect("covariance estimate");
        assert!((r.covariance[1] - r.covariance[2]).abs() < 1e-12);
        assert!(r.covariance[0] >= 0.0 && r.covariance[3] >= 0.0);
        let det = r.covariance[0] * r.covariance[3] - r.covariance[1] * r.covariance[2];
        assert!(det >= -1e-12, "det = {det}");
        // the score variant does not re-align → no per-candidate result poses.
        assert!(r.candidate_result_poses.is_empty());
    }
}
