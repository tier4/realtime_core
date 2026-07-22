//! Deterministic algorithmic-cost counters for the WCET analysis (`plan/ndt_wcet.md`, Layer 2).
//!
//! The engine's frame time decomposes as
//! `T ≤ N_iter × [Σ_p (T_search(p) + K(p)·T_kernel) + T_solve]`, so these counters capture the only
//! data-dependent cost drivers: derivative passes (≤ iterations + 1), points processed, neighbors
//! collected (K ≤ [`crate::ndt::MAX_NEIGHBORS`] per point per pass), and kd-tree nodes visited.
//! Being platform-independent and reproducible, they are the worst-input **search fitness** (far
//! more stable than wall time) and the machine-checked link to the analytic bound
//! (`tests/wcet_bounds.rs`).
//!
//! Only compiled under the `wcet-count` feature; the shipping hot path carries zero instrumentation
//! cost when the feature is off. The counters cover the **serial** backend (the WCET baseline) —
//! the `parallel` reduction does not accumulate them.

/// Algorithmic-cost counters for one [`crate::ndt::align`] call (reset at the start of each align).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WcetCounters {
    /// Derivative computations performed: 1 (initial guess) + 1 per Newton iteration.
    pub derivative_passes: u64,
    /// Source points processed, summed over all derivative passes.
    pub points_processed: u64,
    /// Neighbor leaves collected, summed over all points and passes
    /// (≤ `points_processed × MAX_NEIGHBORS`).
    pub sum_neighbors: u64,
    /// kd-tree nodes visited by the neighbor searches, summed over all points and passes
    /// (≤ `points_processed × tree node count`).
    pub kd_nodes_visited: u64,
    /// Largest single-point neighbor count seen in this align (≤ `MAX_NEIGHBORS`). The per-point
    /// witness for the `K` ceiling — `sum_neighbors` only exposes the mean.
    pub max_neighbors: u64,
}

impl WcetCounters {
    /// All-zero counters for preallocated alignment workspaces.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            derivative_passes: 0,
            points_processed: 0,
            sum_neighbors: 0,
            kd_nodes_visited: 0,
            max_neighbors: 0,
        }
    }
}

/// 64-bit FNV-1a offset basis for numeric diagnostics. Shared with the C++ analysis
/// build (`bench/traced/include/ndt_trace.hpp`); the two implementations are bit-identical and
/// covered by the same test vectors.
#[cfg(feature = "wcet-trace")]
pub const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
/// 64-bit FNV-1a prime.
#[cfg(feature = "wcet-trace")]
pub const FNV_PRIME: u64 = 1_099_511_628_211;
/// Maximum derivative passes a trace stores (`max_iterations` caps at 30 in production, so 40
/// leaves margin; `AlignTrace::len` still counts every pass even past the storage cap).
#[cfg(feature = "wcet-trace")]
pub const MAX_TRACE_PASSES: usize = 40;

/// SHA-256 digest width used by the cross-language trace ABI.
#[cfg(feature = "wcet-trace")]
pub const TRACE_DIGEST_BYTES: usize = 32;

/// Fold one `u64` (little-endian bytes) into a 64-bit FNV-1a state.
#[cfg(feature = "wcet-trace")]
#[must_use]
pub fn fnv1a_u64(mut h: u64, v: u64) -> u64 {
    let mut x = v;
    for _ in 0..8 {
        h ^= x & 0xff;
        h = h.wrapping_mul(FNV_PRIME);
        x >>= 8;
    }
    h
}

/// One derivative pass of the analysis trace: structural work, canonical shape and payload
/// digests, engine-own kd work, and pass-final numeric diagnostics.
#[cfg(feature = "wcet-trace")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassTrace {
    /// Source points processed in this pass.
    pub points: u64,
    /// Neighbor leaves collected in this pass (Sigma over points of K(p)).
    pub neighbors: u64,
    /// This engine's own kd traversal count for the pass (NOT cross-language comparable).
    pub kd_nodes: u64,
    /// SHA-256 chain over sorted canonical `(grid ordinal, voxel id)` sets, in point order.
    pub shape_digest: [u8; TRACE_DIGEST_BYTES],
    /// SHA-256 chain over the same ids plus mean and inverse-covariance f64 bits.
    pub payload_digest: [u8; TRACE_DIGEST_BYTES],
    /// Bit pattern of the pass-final score (the value handed to the Newton step).
    pub score_bits: u64,
    /// FNV-1a over the pass-final gradient's 6 f64 bit patterns (index order).
    pub grad_hash: u64,
    /// FNV-1a over the pass-final Hessian's 36 f64 bit patterns (row-major (r, c) order).
    pub hess_hash: u64,
}

#[cfg(feature = "wcet-trace")]
impl PassTrace {
    /// Empty pass record (numeric diagnostic hashes start at the FNV offset basis).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            points: 0,
            neighbors: 0,
            kd_nodes: 0,
            shape_digest: [0; TRACE_DIGEST_BYTES],
            payload_digest: [0; TRACE_DIGEST_BYTES],
            score_bits: 0,
            grad_hash: FNV_OFFSET,
            hess_hash: FNV_OFFSET,
        }
    }
}

#[cfg(feature = "wcet-trace")]
impl Default for PassTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-align trace: one [`PassTrace`] per derivative pass, in pass order.
#[cfg(feature = "wcet-trace")]
#[derive(Clone, Copy, Debug)]
pub struct AlignTrace {
    /// Total passes recorded (counts every pass, even beyond [`MAX_TRACE_PASSES`]).
    pub len: usize,
    /// `true` when the trace is not deterministic-order valid (parallel backend ran).
    pub poisoned: bool,
    /// The stored pass records (`passes[..len.min(MAX_TRACE_PASSES)]` are valid).
    pub passes: [PassTrace; MAX_TRACE_PASSES],
}

#[cfg(feature = "wcet-trace")]
impl AlignTrace {
    /// Empty trace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            len: 0,
            poisoned: false,
            passes: [PassTrace::new(); MAX_TRACE_PASSES],
        }
    }

    /// Append one pass record (drops the record body past the storage cap; `len` still counts).
    pub fn push(&mut self, pass: PassTrace) {
        if let Some(slot) = self.passes.get_mut(self.len) {
            *slot = pass;
        }
        self.len = self.len.saturating_add(1);
    }
}

#[cfg(feature = "wcet-trace")]
impl Default for AlignTrace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "wcet-trace"))]
#[allow(
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "test code may relax freely"
)]
mod trace_tests {
    use super::*;

    /// Shared C++/Rust test vectors: `bench/traced/include/ndt_trace.hpp` must produce the same
    /// values (checked by the traced replay's self-test at startup).
    #[test]
    fn fnv1a_shared_vectors() {
        // Canonical FNV-1a of eight 0x00 bytes from the offset basis.
        assert_eq!(fnv1a_u64(FNV_OFFSET, 0), 0xa8c7_f832_281a_39c5);
        // 1.0_f64's bit pattern (0x3FF0000000000000) folded from the offset basis.
        assert_eq!(
            fnv1a_u64(FNV_OFFSET, 1.0_f64.to_bits()),
            0xaab1_6932_29ba_1db8
        );
        // Chained fold is order-sensitive.
        let a = fnv1a_u64(fnv1a_u64(FNV_OFFSET, 1), 2);
        let b = fnv1a_u64(fnv1a_u64(FNV_OFFSET, 2), 1);
        assert_ne!(a, b);
    }

    #[test]
    fn sha256_shared_vector() {
        use sha2::{Digest, Sha256};

        let digest: [u8; TRACE_DIGEST_BYTES] = Sha256::digest(b"abc").into();
        assert_eq!(
            digest,
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
    }

    #[test]
    fn trace_push_saturates() {
        let mut tr = AlignTrace::new();
        for i in 0..(MAX_TRACE_PASSES + 3) {
            let mut p = PassTrace::new();
            p.points = i as u64;
            tr.push(p);
        }
        assert_eq!(tr.len, MAX_TRACE_PASSES + 3);
        assert_eq!(
            tr.passes[MAX_TRACE_PASSES - 1].points,
            (MAX_TRACE_PASSES - 1) as u64
        );
    }
}
