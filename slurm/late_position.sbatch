#!/bin/bash
# E10: does discarding each block's cold start make the FREE reading as good as the adapted one?
#
# Account: AIFAC_S03_029 expired 2026-07-13. AIFAC_F02_541 is live to 2026-09-23 (verified with
# `saldo -b` on 2026-09-11: 80,000 h, 17.2% consumed). Check `saldo -b` before assuming either.
#
# This is a falsification run, not a confirmation. If late-position zero-shot BPB reaches adapted
# BPB's in-band selection accuracy (~0.900), the paper's mechanism section is wrong and the finding
# shrinks to "do not block-average cold starts". The job is designed so that outcome is visible.
#
# One array task per model, one A100 each, forward passes only -- no training, no adapters.
#
#SBATCH --job-name=late_pos
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_F02_541
#SBATCH --mem=64G
#SBATCH --array=0-10
#
# Usage:  sbatch slurm/late_position.sbatch

set -euo pipefail
mkdir -p logs

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

: "${SCRATCH:?SCRATCH must be set}"
# Verified 2026-09-11: all eleven checkpoints are under $SCRATCH/hf_cache/hub.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$REPO/venv_lm_adapt/bin/python"
DATASET="$SCRATCH/entropy-bench-release/google_news_from_2026-06-08_to_2026-06-08_cleaned.jsonl"
# Caches and outputs go to SCRATCH: $HOME is at 47G of a 50G quota and tokenizer caches are ~1.3 GB
# per model.
OUTDIR="$SCRATCH/e10_late_position"
mkdir -p "$OUTDIR"

# The eleven-model cohort -- the population that carries the mechanism table the control tests.
# Batch sizes are the ones the published cells used for these checkpoints.
MODELS=(
  "Qwen/Qwen2.5-0.5B|16"
  "meta-llama/Llama-3.2-1B|16"
  "LiquidAI/LFM2.5-1.2B-Base|16"
  "Qwen/Qwen2.5-1.5B|16"
  "Qwen/Qwen3.5-4B-Base|8"
  "Qwen/Qwen2.5-7B|8"
  "Qwen/Qwen3.5-9B-Base|8"
  "google/gemma-4-12B|4"
  "mistralai/Ministral-3-14B-Base-2512|4"
  "google/gemma-4-31B|2"
  "Qwen/Qwen3.5-35B-A3B-Base|2"
)

ENTRY="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MID="${ENTRY%%|*}"
BS="${ENTRY##*|}"
SLUG="$(echo "$MID" | tr '/.' '__')"

echo "=============================================================="
echo "[$(date +%H:%M:%S)] task $SLURM_ARRAY_TASK_ID  $MID  (batch $BS)"
echo "=============================================================="

# The 35B MoE does not fit one 64 GB A100 in bf16; give it every GPU on the node.
DEVMAP="cuda:0"
case "$MID" in *35B*|*31B*) DEVMAP="auto" ;; esac

$PY tools/late_position_bpb.py \
    --model-id "$MID" \
    --device-map "$DEVMAP" \
    --dataset "$DATASET" \
    --out "$OUTDIR/${SLUG}.json" \
    --output-dir "$OUTDIR" \
    --batch-size "$BS" \
    --max-blocks 2500

echo "[$(date +%H:%M:%S)] done -> $OUTDIR/${SLUG}.json"
