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

//! Static 3-D kd-tree with radius search over voxel centroids. Replaces `pcl::KdTreeFLANN` for the
//! NDT voxel-grid map. Tree construction is control-plane work and may allocate and recurse;
//! queries use a fixed iterative stack. `no_std` + `alloc`; no external dependency.

// Numeric/index kernel: bounded recursion depth (~log2 N) and indexing into fixed `[f32; 3]` /
// the internal node array; distances are computed in f64 via `f64::from` (no lossy casts).
// Suppressions are scoped per-function (no module-wide `#![allow]`); rationale per the comment above.

use alloc::vec::Vec;

#[derive(Clone)]
struct Node {
    point_idx: usize,
    axis: usize,
    left: Option<usize>,
    right: Option<usize>,
}

/// Immutable 3-D kd-tree built once from a set of points.
#[derive(Clone)]
pub struct KdTree {
    points: Vec<[f32; 3]>,
    nodes: Vec<Node>,
    root: Option<usize>,
}

/// Failure while building or querying the kd-tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KdSearchError {
    /// A control-plane vector reservation failed while building the tree.
    AllocationFailed,
    /// A checked node/depth/visit counter operation overflowed.
    ArithmeticOverflow,
    /// The balanced tree depth or iterative query exceeded [`KdTree::MAX_STACK`].
    StackCapacityExceeded,
}

/// Metadata returned by a bounded radius search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KdSearchOutcome {
    /// At least one additional in-radius point existed after `max_nn` results were retained.
    pub result_limit_exceeded: bool,
    /// Number of tree points distance-tested by this query. This is populated by
    /// [`KdTree::radius_search_counted`] and by the benchmark-only force-count ablation; normal
    /// production queries leave it at zero.
    pub nodes_visited: u64,
}

fn dist_sq(a: &[f32; 3], b: &[f32; 3]) -> f64 {
    let dx = f64::from(a[0]) - f64::from(b[0]);
    let dy = f64::from(a[1]) - f64::from(b[1]);
    let dz = f64::from(a[2]) - f64::from(b[2]);
    (dx * dx) + (dy * dy) + (dz * dz)
}

impl KdTree {
    /// Build a balanced kd-tree from `points` (copied in), rejecting a tree whose required query
    /// depth would exceed the fixed stack before allocating or publishing it.
    /// # Errors
    /// Returns [`KdSearchError::StackCapacityExceeded`] for excessive depth,
    /// [`KdSearchError::AllocationFailed`] when a reservation fails, or
    /// [`KdSearchError::ArithmeticOverflow`] when depth arithmetic cannot be represented.
    pub fn try_build(points: &[[f32; 3]]) -> Result<Self, KdSearchError> {
        let required_depth = usize::BITS
            .checked_sub(points.len().leading_zeros())
            .ok_or(KdSearchError::ArithmeticOverflow)?;
        let stack_depth =
            u32::try_from(Self::MAX_STACK).map_err(|_| KdSearchError::ArithmeticOverflow)?;
        if required_depth > stack_depth {
            return Err(KdSearchError::StackCapacityExceeded);
        }
        let mut pts = Vec::new();
        pts.try_reserve_exact(points.len())
            .map_err(|_| KdSearchError::AllocationFailed)?;
        pts.extend_from_slice(points);
        let mut idx = Vec::new();
        idx.try_reserve_exact(pts.len())
            .map_err(|_| KdSearchError::AllocationFailed)?;
        idx.extend(0..pts.len());
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(pts.len())
            .map_err(|_| KdSearchError::AllocationFailed)?;
        let root = build_rec(&pts, &mut idx, 0, &mut nodes);
        Ok(Self {
            points: pts,
            nodes,
            root,
        })
    }

    /// Indices of points within Euclidean `radius` of `query`, appended to `out` in deterministic
    /// near/far traversal order. `max_nn == 0` means unlimited. Otherwise the first `max_nn` points
    /// are retained and the search continues until either traversal completes or one additional
    /// in-radius point proves that the result limit was exceeded. Callers that require an
    /// allocation-free query must clear and pre-reserve `out` before calling.
    /// # Errors
    /// Returns [`KdSearchError::StackCapacityExceeded`] on fixed-stack exhaustion or
    /// [`KdSearchError::ArithmeticOverflow`] if the visited-node counter overflows.
    pub fn radius_search(
        &self,
        query: &[f32; 3],
        radius: f64,
        max_nn: usize,
        out: &mut Vec<usize>,
    ) -> Result<KdSearchOutcome, KdSearchError> {
        let r2 = radius * radius;
        let mut visited = 0_u64;
        let result_limit_exceeded = self
            .search_iter::<{ cfg!(feature = "bench-kd-force-count") }>(
                query,
                r2,
                max_nn,
                out,
                &mut visited,
            )?;
        Ok(KdSearchOutcome {
            result_limit_exceeded,
            nodes_visited: visited,
        })
    }

    /// [`Self::radius_search`] that also returns the number of tree nodes visited — the
    /// deterministic traversal-cost counter for the WCET analysis.
    #[cfg(feature = "wcet-count")]
    /// # Errors
    /// Returns an explicit error when allocation, arithmetic, numeric input, or a declared runtime bound fails.
    pub fn radius_search_counted(
        &self,
        query: &[f32; 3],
        radius: f64,
        max_nn: usize,
        out: &mut Vec<usize>,
    ) -> Result<KdSearchOutcome, KdSearchError> {
        let r2 = radius * radius;
        let mut visited = 0_u64;
        let result_limit_exceeded =
            self.search_iter::<true>(query, r2, max_nn, out, &mut visited)?;
        Ok(KdSearchOutcome {
            result_limit_exceeded,
            nodes_visited: visited,
        })
    }

    /// Depth cap for the explicit search stack. Only deferred *far* children are stacked — at most
    /// one per descent level — so the stack need is the tree depth, ≤ ⌈log₂N⌉ + 1 by the median
    /// build. A `Node` occupies ≥ 48 bytes, so any in-memory tree has N < 2^59 ⇒ depth < 60 < 64;
    /// the build-time guard rejects deeper trees before publication.
    pub(crate) const MAX_STACK: usize = 64;

    /// Iterative radius search — an explicit fixed-size stack instead of recursion, so the
    /// RT-critical path uses O(1) stack independent of tree size (WCET audit item; see
    /// `porting_notes/ndt_wcet_audit.md`). Visits nodes in **exactly** the recursive near-then-far
    /// order (far children are deferred LIFO), so the neighbor order — and thus every downstream
    /// float summation — is bit-identical to the recursive implementation (oracle-tested below).
    #[allow(
        clippy::indexing_slicing,
        clippy::allow_attributes,
        reason = "axis is depth % 3 ∈ 0..3 indexing a fixed-size [f32; 3]; the stack index is \
                  guarded < MAX_STACK"
    )]
    fn search_iter<const COUNT_NODES: bool>(
        &self,
        query: &[f32; 3],
        r2: f64,
        max_nn: usize,
        out: &mut Vec<usize>,
        visited: &mut u64,
    ) -> Result<bool, KdSearchError> {
        let mut stack = [0_usize; Self::MAX_STACK];
        let mut sp = 0_usize; // stack length
        let mut cur = self.root;
        loop {
            while let Some(ni) = cur {
                #[cfg(feature = "bench-kd-stop-at-cap")]
                if max_nn != 0 && out.len() >= max_nn {
                    return Ok(false);
                }
                let Some(n) = self.nodes.get(ni) else { break };
                let Some(p) = self.points.get(n.point_idx) else {
                    break;
                };
                if COUNT_NODES {
                    *visited = visited
                        .checked_add(1)
                        .ok_or(KdSearchError::ArithmeticOverflow)?;
                }

                if dist_sq(p, query) <= r2 {
                    if max_nn != 0 && out.len() >= max_nn {
                        return Ok(true);
                    }
                    out.push(n.point_idx);
                }

                let diff = f64::from(query[n.axis]) - f64::from(p[n.axis]);
                let (near, far) = if diff < 0.0 {
                    (n.left, n.right)
                } else {
                    (n.right, n.left)
                };

                // Defer the far side (visited after the whole near subtree) if the splitting plane
                // is within the radius. The depth bound makes overflow unreachable (see MAX_STACK).
                if (diff * diff) <= r2
                    && let Some(fi) = far
                {
                    let Some(slot) = stack.get_mut(sp) else {
                        return Err(KdSearchError::StackCapacityExceeded);
                    };
                    *slot = fi;
                    sp = sp.checked_add(1).ok_or(KdSearchError::ArithmeticOverflow)?;
                }
                cur = near;
            }
            if sp == 0 {
                return Ok(false);
            }
            sp = sp.checked_sub(1).ok_or(KdSearchError::ArithmeticOverflow)?;
            cur = stack.get(sp).copied();
        }
    }

    /// Reference recursive search — kept as the test oracle for [`Self::search_iter`]'s exact
    /// visit-order equivalence (the neighbor order feeds float summation order = bit-exactness).
    #[cfg(test)]
    #[allow(
        clippy::indexing_slicing,
        clippy::allow_attributes,
        reason = "test-only oracle; axis is depth % 3 ∈ 0..3 indexing a fixed-size [f32; 3]"
    )]
    fn search_rec(
        &self,
        node: Option<usize>,
        query: &[f32; 3],
        r2: f64,
        max_nn: usize,
        out: &mut Vec<usize>,
    ) {
        let Some(ni) = node else { return };
        if max_nn != 0 && out.len() >= max_nn {
            return;
        }
        let Some(n) = self.nodes.get(ni) else { return };
        let Some(p) = self.points.get(n.point_idx) else {
            return;
        };

        if dist_sq(p, query) <= r2 {
            out.push(n.point_idx);
        }

        let diff = f64::from(query[n.axis]) - f64::from(p[n.axis]);
        let (near, far) = if diff < 0.0 {
            (n.left, n.right)
        } else {
            (n.right, n.left)
        };

        self.search_rec(near, query, r2, max_nn, out);
        if (diff * diff) <= r2 {
            self.search_rec(far, query, r2, max_nn, out);
        }
    }
}

#[allow(
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "axis ∈ 0..3; idx holds valid point indices and mid < idx.len()"
)]
fn build_rec(
    pts: &[[f32; 3]],
    idx: &mut [usize],
    depth: usize,
    nodes: &mut Vec<Node>,
) -> Option<usize> {
    if idx.is_empty() {
        return None;
    }
    let axis = depth % 3;
    idx.sort_unstable_by(|&a, &b| pts[a][axis].total_cmp(&pts[b][axis]));
    let mid = idx.len() / 2;
    let point_idx = idx[mid];

    let (left_slice, rest) = idx.split_at_mut(mid);
    // `rest[0]` is the median; recurse on the elements after it.
    let right_slice = &mut rest[1..];
    let left = build_rec(pts, left_slice, depth.saturating_add(1), nodes);
    let right = build_rec(pts, right_slice, depth.saturating_add(1), nodes);

    let id = nodes.len();
    nodes.push(Node {
        point_idx,
        axis,
        left,
        right,
    });
    Some(id)
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::unreadable_literal,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    // Tiny deterministic LCG so the test needs no rand dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // top 24 bits -> [0,1), then scale to [-5, 5)
            let bits = (self.0 >> 40) as u32;
            ((bits as f32 / (1u32 << 24) as f32) * 10.0) - 5.0
        }
    }

    fn brute(points: &[[f32; 3]], q: &[f32; 3], radius: f64) -> Vec<usize> {
        let r2 = radius * radius;
        let mut v: Vec<usize> = (0..points.len())
            .filter(|&i| dist_sq(&points[i], q) <= r2)
            .collect();
        v.sort_unstable();
        v
    }

    // Heavy LCG property test (500 points) — skipped under Miri to keep the unsafe-UB run fast.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn radius_search_matches_brute_force() {
        let mut rng = Lcg(0x1234_5678);
        let pts: Vec<[f32; 3]> = (0..500)
            .map(|_| [rng.next_f32(), rng.next_f32(), rng.next_f32()])
            .collect();
        let tree = KdTree::try_build(&pts).expect("build kd-tree");

        for _ in 0..50 {
            let q = [rng.next_f32(), rng.next_f32(), rng.next_f32()];
            let radius = 1.3_f64;
            let mut got = Vec::new();
            tree.radius_search(&q, radius, 0, &mut got)
                .expect("radius search");
            got.sort_unstable();
            assert_eq!(got, brute(&pts, &q, radius));
        }
    }

    #[test]
    fn empty_tree_returns_nothing() {
        let tree = KdTree::try_build(&[]).expect("build kd-tree");
        let mut out = Vec::new();
        tree.radius_search(&[0.0, 0.0, 0.0], 1.0, 0, &mut out)
            .expect("radius search");
        assert!(out.is_empty());
    }

    #[cfg(not(feature = "bench-kd-stop-at-cap"))]
    #[test]
    fn max_nn_caps_results_and_reports_the_next_neighbor() {
        let pts: Vec<[f32; 3]> = (0..65).map(|i| [i as f32 * 0.001, 0.0, 0.0]).collect();
        let tree = KdTree::try_build(&pts).expect("build kd-tree");
        let mut out = Vec::new();
        let outcome = tree
            .radius_search(&[0.032, 0.0, 0.0], 1.0, 64, &mut out)
            .expect("radius search");
        assert_eq!(out.len(), 64);
        assert!(outcome.result_limit_exceeded);
    }

    #[cfg(not(feature = "bench-kd-stop-at-cap"))]
    #[test]
    fn exactly_max_nn_is_not_reported_as_exceeded() {
        let pts: Vec<[f32; 3]> = (0..64).map(|i| [i as f32 * 0.001, 0.0, 0.0]).collect();
        let tree = KdTree::try_build(&pts).expect("build kd-tree");
        let mut out = Vec::new();
        let outcome = tree
            .radius_search(&[0.032, 0.0, 0.0], 1.0, 64, &mut out)
            .expect("radius search");
        assert_eq!(out.len(), 64);
        assert!(!outcome.result_limit_exceeded);
    }

    #[cfg(not(feature = "bench-kd-force-count"))]
    #[test]
    fn production_search_does_not_count_nodes() {
        let tree = KdTree::try_build(&[[0.0, 0.0, 0.0]]).expect("build kd-tree");
        let mut out = Vec::new();
        let outcome = tree
            .radius_search(&[0.0, 0.0, 0.0], 1.0, 0, &mut out)
            .expect("radius search");
        assert_eq!(outcome.nodes_visited, 0);
    }

    #[cfg(feature = "wcet-count")]
    #[test]
    fn counted_search_reports_distance_tests() {
        let tree = KdTree::try_build(&[[0.0, 0.0, 0.0]]).expect("build kd-tree");
        let mut out = Vec::new();
        let outcome = tree
            .radius_search_counted(&[0.0, 0.0, 0.0], 1.0, 0, &mut out)
            .expect("radius search");
        assert_eq!(outcome.nodes_visited, 1);
    }

    // The iterative search must reproduce the recursive oracle's output in EXACT order (the
    // neighbor order feeds downstream float summation order = bit-exactness), for both unlimited
    // and capped searches, over many random trees/queries/radii.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn iterative_matches_recursive_oracle_exact_order() {
        let mut rng = Lcg(0xBEEF_CAFE);
        for n in [0usize, 1, 2, 3, 7, 64, 257, 800] {
            let pts: Vec<[f32; 3]> = (0..n)
                .map(|_| [rng.next_f32(), rng.next_f32(), rng.next_f32()])
                .collect();
            let tree = KdTree::try_build(&pts).expect("build kd-tree");
            for _ in 0..24 {
                let q = [rng.next_f32(), rng.next_f32(), rng.next_f32()];
                let radius = f64::from(0.4 + (rng.next_f32() + 5.0) * 0.3); // ~[0.4, 3.4]
                for max_nn in [0usize, 1, 3, 64] {
                    let mut iterative = Vec::new();
                    tree.radius_search(&q, radius, max_nn, &mut iterative)
                        .expect("radius search");
                    let mut recursive = Vec::new();
                    tree.search_rec(tree.root, &q, radius * radius, max_nn, &mut recursive);
                    assert_eq!(
                        iterative, recursive,
                        "order mismatch: n={n} max_nn={max_nn} q={q:?} r={radius}"
                    );
                }
            }
        }
    }
}
