#!/bin/bash
#SBATCH --job-name=prep_corpora
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=AIFAC_S03_029
#
# Cleans the raw Reddit / Hacker News scrapes into adaptation-ready JSONL, using the
# recipe Filip applied to the news corpus. CPU only.
#
#   sbatch slurm/prepare_corpora.sbatch

set -uo pipefail
mkdir -p logs
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
PY="$REPO/venv_lm_adapt/bin/python"
: "${WORK:?Set WORK to a high-capacity filesystem before submission.}"
RAW="${RAW:-$WORK/scraped/2026-08}"
OUT="${OUT:-$WORK/corpora}"
mkdir -p "$OUT"

# Reddit and HN records are comments, far shorter than news articles. --group-by builds
# longer documents so the corpus is comparable to the news slice and so the
# one-BOS-per-document packing artefact stays proportionally small.
echo "=== reddit (grouped by subreddit) ==="
$PY tools/prepare_scraped_corpus.py \
    --inputs "$RAW"/*/reddit_*.json \
    --source reddit \
    --out "$OUT/reddit_2026-08.jsonl" \
    --min-words 30 --group-by subreddit --group-target-words 600 \
    --fuzzy-threshold 0.60

echo
echo "=== hacker news (grouped) ==="
$PY tools/prepare_scraped_corpus.py \
    --inputs "$RAW"/*/hacker_news_*.json \
    --source hackernews \
    --out "$OUT/hackernews_2026-08.jsonl" \
    --min-words 30 --group-target-words 600 \
    --fuzzy-threshold 0.60

echo
ls -la "$OUT"
echo "DONE"
