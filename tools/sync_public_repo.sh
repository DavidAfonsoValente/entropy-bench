#!/usr/bin/env bash
# Publish a curated snapshot of this repo to the public entropy-bench repository.
#
# Publishing has been manual and undocumented, and it drifted: between 2026-08-22 and 2026-08-30
# the public repo sat 26 commits behind and served a headline framing the project had already
# audited and replaced. This script exists so that never depends on someone remembering the
# exclusion list again.
#
#   bash tools/sync_public_repo.sh            # stage + verify, DO NOT push (default)
#   bash tools/sync_public_repo.sh --push     # stage, verify, commit and push
#
# It refuses to push unless the full gate passes inside the staged public tree -- which is a
# stronger check than running `make check` here, because it catches artifacts that exist in the
# working directory but were never committed. That is not hypothetical: it is exactly how
# results/selection_divergence.json was found missing from the results/ allowlist.
set -euo pipefail

PUBLIC_URL="${PUBLIC_URL:-https://github.com/DavidAfonsoValente/entropy-bench.git}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK_DIR:-${SCRATCH:-/tmp}/entropy-bench-sync}"
PYTHON="${PYTHON:-$ROOT/venv_lm_adapt/bin/python}"
PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

# Internal-only paths. Everything tracked here is published EXCEPT these.
#   - agent instructions and internal state docs (routing, ledger, roadmap, sales positioning)
#   - GCP runners: they hard-code our private GCS bucket and are useless to an outside reader
#   - per-example samples: ~370 MB of samples_*.jsonl(.gz). The public repo has never carried
#     these; they live in gs://gpu-llm-training-gceval/token_gain/adapted_bench/. The scored
#     results_*.json summaries ARE published, so every number in the paper stays greppable.
EXCLUDE_RE='^(CLAUDE|AGENTS|GEMINI)\.md$|^docs/(POSITIONING|PROGRESS|RUN_LEDGER|PLAN)\.md$|^tools/(gcp_[a-z_]*\.sh|gce_run_watchdog\.sh)$|samples_.*\.jsonl(\.gz)?$|^\.claude/'

# Files that exist only in the public repo and must survive a sync rather than being deleted.
KEEP_PUBLIC='^docs/TITLE_OPTIONS\.md$'

cd "$ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to sync: working tree is dirty. Commit first so the snapshot is identifiable." >&2
  exit 1
fi
SRC_REV="$(git rev-parse --short HEAD)"

echo "==> staging public tree in $WORK"
mkdir -p "$(dirname "$WORK")"
# This directory is disposable: it gets hard-reset and cleaned on every run. Only ever touch one
# this script created itself, identified by a marker file. Reusing "any clone of the public repo"
# is not safe enough -- that could be somebody's working checkout with unpushed branches and
# untracked notes in it, and the clean below would destroy them.
MARKER="$WORK/.entropy-bench-sync-workdir"
MARKER_TOKEN="disposable-staging-dir-for-tools/sync_public_repo.sh"
if [[ -e "$WORK" ]]; then
  if ! grep -qxF "$MARKER_TOKEN" "$MARKER" 2>/dev/null; then
    echo "refusing to sync: $WORK exists but was not created by this script." >&2
    echo "It may be a real checkout; this script hard-resets and cleans its work dir." >&2
    echo "Set WORK_DIR to a disposable path, or delete $WORK yourself if it is scratch." >&2
    exit 1
  fi
  if [[ "$(git -C "$WORK" remote get-url origin 2>/dev/null)" != "$PUBLIC_URL" ]]; then
    echo "refusing to sync: $WORK points at a different remote than $PUBLIC_URL." >&2
    exit 1
  fi
  git -C "$WORK" fetch --quiet origin
  # Force onto main explicitly: a reused checkout left on another branch would commit there and
  # then push a stale local main. Then drop every untracked and ignored stray, because anything
  # left behind would be swept up by the `git add -A` below and published.
  git -C "$WORK" checkout -q -B main origin/main
  git -C "$WORK" reset --hard --quiet origin/main
  git -C "$WORK" clean -qxdff -e "$(basename "$MARKER")"
else
  git clone --quiet "$PUBLIC_URL" "$WORK"
  git -C "$WORK" checkout -q -B main origin/main
fi
printf '%s\n' "$MARKER_TOKEN" > "$MARKER"

git ls-files | grep -vE "$EXCLUDE_RE" > "$WORK/.sync-manifest"
# Clear the tracked tree so deletions propagate, then lay the snapshot down over it.
#
# The `rm` MUST run with $WORK as its working directory. `git -C "$WORK" ls-files` prints paths
# relative to $WORK, but a bare `xargs rm` inherits this script's cwd -- which is $ROOT -- and so
# deletes those same relative paths out of the SOURCE repository instead. That is not a
# hypothetical: it happened on this script's first dry run and removed every file the two repos
# have in common from the working tree here. It was recoverable only because everything was
# committed. Keep the subshell.
( cd "$WORK" && git ls-files -z | grep -zvE "$KEEP_PUBLIC" | xargs -0 -r rm -f )
rsync -a --files-from="$WORK/.sync-manifest" "$ROOT/" "$WORK/"
find "$WORK" -type d -empty -not -path "$WORK/.git/*" -delete
rm -f "$WORK/.sync-manifest"

echo "==> leak check"
leaked=0
while IFS= read -r f; do
  [[ -e "$WORK/$f" ]] && { echo "  LEAK: $f"; leaked=1; }
done < <(git ls-files | grep -E "$EXCLUDE_RE")
[[ $leaked -eq 0 ]] || { echo "refusing to sync: internal files present in the public tree" >&2; exit 1; }
echo "  clean"

echo "==> running the full gate inside the staged public tree"
( cd "$WORK" && make check PYTHON="$PYTHON" ) || {
  echo "refusing to sync: the public tree does not pass its own gate." >&2
  echo "A file the gate reads is probably untracked here -- check .gitignore's results/ allowlist." >&2
  exit 1
}

cd "$WORK"
if git diff --quiet && git diff --cached --quiet && [[ -z "$(git status --porcelain)" ]]; then
  echo "==> public repo already matches this snapshot; nothing to do"
  exit 0
fi
# The marker is scaffolding, not content. Deleting it before staging is deterministic, where
# an ignore rule can be overridden by a higher-precedence .gitignore entry.
rm -f "$MARKER"
git add -A
echo "==> staged:"
git diff --cached --stat | tail -3

if [[ $PUSH -eq 0 ]]; then
  echo
  echo "Dry run. Nothing pushed. Inspect $WORK, then re-run with --push."
  printf '%s\n' "$MARKER_TOKEN" > "$MARKER"
  exit 0
fi

git -c user.name="$(git -C "$ROOT" config user.name)" \
    -c user.email="$(git -C "$ROOT" config user.email)" \
    commit -q -m "release: sync public snapshot from $SRC_REV

Curated snapshot of the research repository. Per-example samples_*.jsonl and
internal working documents are excluded; every number cited by the paper
regenerates from the artifacts here via 'make check'."
git push -q origin main
printf '%s\n' "$MARKER_TOKEN" > "$MARKER"
echo "==> pushed $(git rev-parse --short HEAD) to $PUBLIC_URL"
