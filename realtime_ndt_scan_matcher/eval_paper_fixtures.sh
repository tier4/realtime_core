#!/bin/sh
# Reproduce and run the paper's frozen Rust fixture set as a short diagnostic.
set -eu

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ "$#" -ne 0 ]; then
  echo "Usage: ./eval_paper_fixtures.sh" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to format the evaluation output" >&2
  exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/ndt-paper-eval.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
generation_log="$work/generation.log"
evaluation_log="$work/evaluation.log"
result_json="$work/result.json"

echo ">> regenerating and verifying the frozen paper fixtures"
if ! ./gen_fixtures.sh >"$generation_log" 2>&1; then
  cat "$generation_log" >&2
  echo "ERROR: fixture generation failed" >&2
  exit 1
fi

set -- \
  bench/fixtures/search_00.ndtfix \
  bench/fixtures/pareto/pareto_01.ndtfix \
  bench/fixtures/pareto/pareto_02.ndtfix \
  bench/fixtures/search_01.ndtfix \
  bench/fixtures/dense_neighbors.ndtfix \
  bench/fixtures/max_iterations.ndtfix \
  bench/fixtures/cache_hostile.ndtfix \
  bench/fixtures/subnormal.ndtfix \
  bench/fixtures/legal_worst.ndtfix \
  bench/fixtures/legal_osc.ndtfix

echo ">> measuring the Rust engine (10 warmups + 5 measured aligns per fixture)"
if ! WCET_FRAMES=5 WCET_JSON="$result_json" \
  cargo run --release --features wcet-count --example wcet_frame -- "$@" \
  >"$evaluation_log" 2>&1; then
  cat "$evaluation_log" >&2
  echo "ERROR: fixture evaluation failed" >&2
  exit 1
fi

python3 - "$result_json" <<'PY'
import json
import math
import sys

EXPECTED_SAMPLES = 5
FIXTURES = (
    ("search_00", "search-00"),
    ("pareto_01", "pareto-01"),
    ("pareto_02", "pareto-02"),
    ("search_01", "search-01"),
    ("dense_neighbors", "dense-nbrs"),
    ("max_iterations", "max-iters"),
    ("cache_hostile", "cache-hostile"),
    ("subnormal", "subnormal"),
    ("legal_worst", "geom-stress"),
    ("legal_osc", "shipped-osc"),
)


def fail(message):
    raise SystemExit(f"ERROR: {message}")


with open(sys.argv[1], encoding="utf-8") as stream:
    document = json.load(stream)

fixtures = document.get("fixtures")
if not isinstance(fixtures, dict):
    fail("wcet_frame JSON does not contain a fixtures object")

rows = []
for internal_name, display_name in FIXTURES:
    entry = fixtures.get(internal_name)
    if not isinstance(entry, dict):
        fail(f"missing fixture result: {internal_name}")
    counters = entry.get("counters")
    samples = entry.get("samples_us")
    if not isinstance(counters, dict):
        fail(f"missing counters for {internal_name}")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        fail(f"expected {EXPECTED_SAMPLES} samples for {internal_name}")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in samples):
        fail(f"invalid timing sample for {internal_name}")
    samples = sorted(samples)
    rows.append((display_name, entry, counters, samples))

print()
print("Table 1. Frozen fixtures: deterministic Rust work counters")
print(f"{'fixture':<16} {'N_iter':>6} {'sum_nbr':>12} {'kd_nodes':>12} {'N_leaves':>10} {'tiles':>6}")
for display_name, entry, counters, _samples in rows:
    print(
        f"{display_name:<16} "
        f"{entry['iteration_num']:>6} "
        f"{counters['sum_neighbors']:>12} "
        f"{counters['kd_nodes_visited']:>12} "
        f"{entry['n_map']:>10} "
        f"{entry['n_tiles']:>6}"
    )

print()
print("Table 2. Rust frame time (5 measured aligns per fixture, ms)")
print(f"{'fixture':<16} {'p50':>12} {'max':>12}")
for display_name, _entry, _counters, samples in rows:
    p50_ms = samples[2] / 1000.0
    max_ms = samples[-1] / 1000.0
    print(f"{display_name:<16} {p50_ms:>12.3f} {max_ms:>12.3f}")

print()
print("Diagnostic only: current host, serial Rust engine; not the paper's isolated three-session protocol.")
PY
