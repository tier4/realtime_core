#!/usr/bin/bash
# Test-coverage runner for the realtime_ndt_scan_matcher crate (source-based, cargo-llvm-cov).
#
# Requires (one-time): rustup component add llvm-tools-preview && cargo install cargo-llvm-cov
#
# Usage:
#   ./coverage.sh             # print a per-file summary table
#   ./coverage.sh --html      # also write an HTML report under target/llvm-cov/html
#   ./coverage.sh --lcov      # also write lcov.info (for CI upload)

cd "$(dirname "$(readlink -f "$0")")"

# Dependency sources are excluded from the coverage denominator.
IGNORE='/build/|/registry/'

extra=()
case "${1:-}" in
  --html) extra=(--html) ;;
  --lcov) extra=(--lcov --output-path lcov.info) ;;
  "")     extra=(--summary-only) ;;
  *)      extra=("$@") ;;
esac

exec cargo llvm-cov --ignore-filename-regex "${IGNORE}" "${extra[@]}"
