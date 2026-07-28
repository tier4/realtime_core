#!/usr/bin/bash
# Test-coverage runner for the realtime_core workspace (source-based, cargo-llvm-cov).
# Covers all four crates: realtime_localization_util, realtime_kdtree,
# realtime_port_instrument, and realtime_ndt_scan_matcher.
#
# Requires (one-time): rustup component add llvm-tools-preview && cargo install cargo-llvm-cov
#
# Usage:
#   ./coverage.sh             # print a per-file summary table (all crates)
#   ./coverage.sh --html      # also write an HTML report under target/llvm-cov/html
#   ./coverage.sh --lcov      # also write lcov.info (for CI upload)

cd "$(dirname "$(readlink -f "$0")")"

# Dependency sources are excluded from the coverage denominator.
IGNORE='/build/|/registry/'

# wcet-trace (implies wcet-count) forwards `count`/`trace` into realtime_kdtree and
# realtime_port_instrument, so their feature-gated code is exercised by the test run
# instead of showing up as spuriously uncovered.
FEATURES='realtime_ndt_scan_matcher/wcet-trace'

extra=()
case "${1:-}" in
  --html) extra=(--html) ;;
  --lcov) extra=(--lcov --output-path lcov.info) ;;
  "")     extra=(--summary-only) ;;
  *)      extra=("$@") ;;
esac

exec cargo llvm-cov --workspace --features "${FEATURES}" \
  --ignore-filename-regex "${IGNORE}" "${extra[@]}"
