#!/bin/sh
# Generate the synthetic fixtures used by the paper, or search the current engine separately.
set -eu

usage() {
  cat <<'EOF'
Usage:
  ./gen_fixtures.sh [OUT_DIR]
      Reproduce the paper's synthetic fixture set. Generated files are published only after all
      17 SHA-256 checks pass. Default OUT_DIR: bench/fixtures.

  ./gen_fixtures.sh --fresh-search [OUT_DIR]
      Run the counter-guided search against the current engine. Results are written below
      OUT_DIR/fresh-search and never replace the paper fixtures.
EOF
}

# Resolve paths relative to the crate, independently of the caller's working directory.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

mode=paper
case "${1:-}" in
  --fresh-search)
    mode=fresh
    shift
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  --*)
    echo "ERROR: unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if [ "$#" -gt 1 ]; then
  echo "ERROR: expected at most one OUT_DIR" >&2
  usage >&2
  exit 2
fi

OUT="${1:-$PWD/bench/fixtures}"
MANIFEST="$PWD/paper_fixtures.sha256"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ndt-fixtures.XXXXXX")
trap 'rm -rf "$WORK"' EXIT HUP INT TERM

publish_file() {
  src=$1
  dst=$2
  tmp="${dst}.tmp.$$"
  cp "$src" "$tmp"
  mv -f "$tmp" "$dst"
}

verify_paper_set() {
  dir=$1
  expected=$(wc -l < "$MANIFEST" | tr -d ' ')
  actual=$(find "$dir" -type f -name '*.ndtfix' | wc -l | tr -d ' ')
  if [ "$actual" -ne "$expected" ]; then
    echo "ERROR: expected $expected paper fixtures, found $actual under $dir" >&2
    return 1
  fi
  verify_paper_hashes "$dir"
}

verify_paper_hashes() {
  dir=$1
  (
    cd "$dir"
    sha256sum -c "$MANIFEST"
  )
}

publish_paper_set() {
  stage=$1
  mkdir -p "$OUT" "$OUT/pareto" "$OUT/psweep"

  while read -r _hash rel; do
    case "$rel" in
      pareto/*|psweep/*)
        ;;
      *)
        publish_file "$stage/$rel" "$OUT/$rel"
        ;;
    esac
  done < "$MANIFEST"

  # These directories are reserved for the paper's generated fixture namespaces. Replacing them
  # removes stale pareto_00 or extra frontier points from an earlier current-engine search.
  find "$OUT/pareto" -maxdepth 1 -type f -name '*.ndtfix' -exec rm -f {} +
  find "$OUT/psweep" -maxdepth 1 -type f -name '*.ndtfix' -exec rm -f {} +
  while read -r _hash rel; do
    case "$rel" in
      pareto/*|psweep/*)
        publish_file "$stage/$rel" "$OUT/$rel"
        ;;
    esac
  done < "$MANIFEST"
}

if [ "$mode" = fresh ]; then
  stage="$WORK/fresh-search"
  mkdir -p "$stage/pareto"
  echo ">> searching the current engine (isolated from paper fixtures)"
  WCET_SEARCH_JSON="$stage/search.json" \
  WCET_SEARCH_PARETO_DIR="$stage/pareto" \
    cargo run --release --features wcet-count --example wcet_search -- "$stage"

  for rel in search_00.ndtfix search_01.ndtfix pareto/pareto_00.ndtfix search.json; do
    if [ ! -f "$stage/$rel" ]; then
      echo "ERROR: current-engine search did not produce $rel" >&2
      exit 1
    fi
  done

  mkdir -p "$OUT"
  fresh_out="$OUT/fresh-search"
  rm -rf "$fresh_out"
  mv "$stage" "$fresh_out"
  trap - EXIT HUP INT TERM
  rm -rf "$WORK"
  echo "OK: current-engine search written under $fresh_out"
  exit 0
fi

stage="$WORK/paper"
mkdir -p "$stage/psweep" "$stage/pareto"

echo ">> [1/4] generating hand-built paper fixtures"
cargo run --release --example wcet_fixtures -- "$stage"

echo ">> [2/4] generating the paper P-sweep"
cargo run --release --example wcet_fixtures -- --psweep "$stage/psweep"

echo ">> [3/4] materializing the paper's frozen search genomes"
cargo run --release --example wcet_paper_search_fixtures -- "$stage"

echo ">> [4/4] reproducing the shipped-configuration legal fixture"
cargo run --release --features wcet-count --example wcet_search_legal -- "$stage"

echo ">> verifying the complete paper fixture manifest"
verify_paper_set "$stage"
publish_paper_set "$stage"
verify_paper_hashes "$OUT"

echo
echo "OK: paper fixtures reproduced under $OUT"
echo "  unified: search_00 search_01 pareto_01 pareto_02 dense_neighbors"
echo "           max_iterations cache_hostile subnormal legal_worst legal_osc"
echo "  control: subnormal_ctrl"
echo "  P-sweep: psweep_p00250 .. psweep_p08000"
echo "  real_median and real_slowest are capture-derived and are not modified."
