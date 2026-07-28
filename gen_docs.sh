#!/bin/sh
# Render the mdbook documentation for the realtime_core workspace.
#
# The book source lives in mdbook/src; book.toml redirects the rendered output to the repo-root
# docs tree (realtime_core/docs, book root at docs/index.html), which is tracked in git and served
# by GitHub Pages. Regenerate and commit that tree whenever the sources under mdbook/src change.
set -eu

usage() {
  cat <<'EOF'
Usage:
  ./gen_docs.sh
      Build the mdbook book. Output goes to realtime_core/docs
      (the build-dir configured in mdbook/book.toml). Requires `mdbook` and `mdbook-mermaid`
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

# Resolve paths relative to the repo root, independently of the caller's working directory.
cd "$(CDPATH= cd -- "$(dirname -- "$0")/mdbook" && pwd)"

if ! command -v mdbook >/dev/null 2>&1; then
  echo "ERROR: mdbook not found on PATH" >&2
  exit 1
fi

echo ">> building mdbook book (source: $PWD/src)"
mdbook build

# Report the resolved output directory (book.toml build-dir is relative to this book.toml).
OUT="$(CDPATH= cd -- ../docs && pwd)"
echo
echo "OK: book rendered to $OUT"
echo "  commit that tree to publish the updated documentation."
