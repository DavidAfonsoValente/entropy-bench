#!/bin/bash
#SBATCH --job-name=token_gain
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_S03_029
#SBATCH --mem=0
#
# Decomposes the zero-shot -> adapted BPB gain into document-boundary versus content
# tokens, for the three models that bracket the effect:
#   gemma-4-12B  -- 28.7% gain, injects <bos>
#   LFM2.5-1.2B  -- 36.5% gain, injects <|startoftext|>
#   Qwen2.5-7B   --  3.3% gain, injects nothing  (control)
#
# Usage:  sbatch slurm/token_gain.sbatch

set -euo pipefail
mkdir -p logs

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

: "${WORK:?Set WORK to a high-capacity filesystem before submission.}"
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$REPO/venv_lm_adapt/bin/python"
OUT="${OUT:-$WORK/token_gain}"   # never $HOME -- quota
mkdir -p "$OUT"

# model_id | adapter dir | tokenized-split cache (hash derived from model id + dataset)
run () {
  local mid="$1" slug="$2" cache="$3" bs="$4"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $mid"
  echo "=============================================================="
  $PY tools/token_level_gain.py \
      --model-id "$mid" \
      --adapter "results/$slug/final_checkpoint" \
      --cache "results/cache/$cache" \
      --out "$OUT/${slug}.json" \
      --split test \
      --max-blocks 4000 \
      --batch-size "$bs" \
      --tail 16
}

run "Qwen/Qwen2.5-7B"            "Qwen_Qwen2_5-7B"            "02e9966cc32ec31d.pt" 8
run "LiquidAI/LFM2.5-1.2B-Base"  "LiquidAI_LFM2_5-1_2B-Base"  "b7fddf12c2efdc44.pt" 16
run "google/gemma-4-12B"         "google_gemma-4-12B"         "2de5c1c5f4f7be65.pt" 4

echo "[$(date +%H:%M:%S)] all done -- results in $OUT"
