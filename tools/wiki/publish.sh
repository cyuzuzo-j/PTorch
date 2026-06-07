#!/usr/bin/env bash
# Sync docs_src/ → PTorch.wiki.git.
#
# One-time setup required on github.com:
#   1. Settings → Features → Wikis must be ON.
#   2. Click the repo's "Wiki" tab and create any page (the default Home is
#      fine). This is what makes the .wiki.git repo cloneable.
#
# After that, every subsequent sync is `./tools/wiki/publish.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WIKI_REMOTE="${WIKI_REMOTE:-git@github.com:cyuzuzo-j/PTorch.wiki.git}"
WIKI_DIR="${WIKI_DIR:-${REPO_ROOT}/../PTorch.wiki}"

if [ ! -d "${WIKI_DIR}/.git" ]; then
  echo "Cloning ${WIKI_REMOTE} → ${WIKI_DIR}"
  echo "(If this fails with 'Repository not found': open the Wiki tab on github.com first and create any page.)"
  git clone "${WIKI_REMOTE}" "${WIKI_DIR}"
fi

(
  cd "${WIKI_DIR}"
  git fetch origin
  git checkout master 2>/dev/null || git checkout main
  git pull --ff-only
)

# Regenerate figures into docs_src/images/. Idempotent and conservative —
# skips experiments whose results/ dirs are empty.
python "${REPO_ROOT}/tools/wiki/build_figures.py" \
  --out "${REPO_ROOT}/docs_src/images"

# Flatten docs_src/ into the wiki working tree.
python "${REPO_ROOT}/tools/wiki/flatten.py" \
  --prune \
  "${REPO_ROOT}/docs_src" \
  "${WIKI_DIR}"

REPO_REV="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"

(
  cd "${WIKI_DIR}"
  git add -A
  if git diff --cached --quiet; then
    echo "Wiki is already up to date with docs_src/ @ ${REPO_REV}"
    exit 0
  fi
  git commit -m "Sync from docs_src @ ${REPO_REV}"
  git push origin HEAD
)

echo
echo "Done. View: https://github.com/cyuzuzo-j/PTorch/wiki"
