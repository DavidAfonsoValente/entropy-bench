#!/bin/bash
#SBATCH --job-name=bpb_ctxlen
#SBATCH --output=logs/context-length-%j.out
#SBATCH --error=logs/context-length-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_S03_029
#SBATCH --mem=0

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${WORK:?Set WORK to a high-capacity filesystem before submission.}"
OUT="${OUT:-$WORK/context_length}"
PY="$REPO/venv_lm_adapt/bin/python"
TG="${TOKEN_GAIN_STAGE:-$WORK/token_gain/stage}"

cd "$REPO"
mkdir -p logs "$OUT"
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {
  local model_id="$1" slug="$2" batch_size="$3"
  "$PY" tools/context_length_eval.py \
    --model-id "$model_id" \
    --adapter "results/$slug/final_checkpoint" \
    --cache "$TG/$slug.blocks.pt" \
    --out "$OUT/$slug.json" \
    --contexts 16,32,64,128,256,512,1024,2048 \
    --target-tokens 64 --windows 64 --batch-size "$batch_size"
}

run "Qwen/Qwen2.5-1.5B" "Qwen_Qwen2_5-1_5B" 4
run "google/gemma-4-12B" "google_gemma-4-12B" 1

echo "all context-length evaluations complete"
