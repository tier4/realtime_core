#!/bin/sh
# Render the mdbook documentation for the NDT engine crate.
#
# The book source lives in doc/book/src; book.toml redirects the rendered output to the repo-root
# docs tree (realtime_core/docs/realtime_ndt_scan_matcher), which is tracked in git and served as
# static HTML. Regenerate and commit that tree whenever the sources under doc/book/src change.
set -eu

usage() {
  cat <<'EOF'
Usage:
  ./gen_docs.sh
      Build the mdbook book. Output goes to realtime_core/docs/realtime_ndt_scan_matcher
      (the build-dir configured in doc/book/book.toml). Requires `mdbook` and `mdbook-mermaid`
      on PATH (both are installed in the dev container image).
EOF
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "ERROR: unexpected argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

# Resolve paths relative to the crate, independently of the caller's working directory.
cd "$(CDPATH= cd -- "$(dirname -- "$0")/doc/book" && pwd)"

if ! command -v mdbook >/dev/null 2>&1; then
  echo "ERROR: mdbook not found on PATH" >&2
  exit 1
fi

echo ">> building mdbook book (source: $PWD/src)"
mdbook build

# Report the resolved output directory (book.toml build-dir is relative to this book.toml).
OUT="$(CDPATH= cd -- ../../../docs/realtime_ndt_scan_matcher && pwd)"
echo
echo "OK: book rendered to $OUT"
echo "  commit that tree to publish the updated documentation."
