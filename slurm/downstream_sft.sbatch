#!/bin/bash
# E11: does the selector's pick actually build the better fine-tuned system?
#
# Account: AIFAC_S03_029 expired 2026-07-13. AIFAC_F02_541 is live to 2026-09-23 (verified with
# `saldo -b` on 2026-09-11: 80,000 h, 17.2% consumed). Check `saldo -b` before assuming either.
#
# Two size-matched pairs on which adapted BPB and the static benchmarks disagree, at opposite ends
# of the cohort. Each task fine-tunes ONE model on the in-domain lede task and writes its held-out
# generations, plus a control pass with the unadapted base weights. The verdict is computed by
# tools/analyze_downstream.py, which was written before this job ran and states its outcomes in
# advance -- BPB PREDICTS / BENCHMARKS PREDICT / EQUIVALENT / INCONCLUSIVE.
#
# One A100 per task: gradient checkpointing keeps the 14B inside 64 GB, so both members of a pair
# get identical hardware and neither can be advantaged by its shape.
#
#SBATCH --job-name=e11_sft
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_F02_541
#SBATCH --mem=64G
#SBATCH --array=0-3
#
# Usage:  sbatch slurm/downstream_sft.sbatch
#         (tools/build_lede_task.py must have run first -- it writes the task and the manifest)

set -euo pipefail
mkdir -p logs

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

: "${SCRATCH:?SCRATCH must be set}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Any Accelerate weight offload goes to SCRATCH; $HOME is at 47G of a 50G quota.
export LM_OFFLOAD_DIR="$SCRATCH/lm_adapt_offload"

PY="$REPO/venv_lm_adapt/bin/python"
TASKDIR="$SCRATCH/e11_downstream"
[ -f "$TASKDIR/train.jsonl" ] || { echo "no task data in $TASKDIR; run tools/build_lede_task.py"; exit 1; }

# label|hf-id|micro-batch|generation-batch. The micro-batch differs with model size only; the
# EFFECTIVE batch is 32 for every model, set inside the tool, so the optimisation is identical.
MODELS=(
  "gemma-4-12B|google/gemma-4-12B|2|8"
  "Ministral-3-14B|mistralai/Ministral-3-14B-Base-2512|2|8"
  "Llama-3.2-1B|meta-llama/Llama-3.2-1B|4|16"
  "Qwen2.5-1.5B|Qwen/Qwen2.5-1.5B|4|16"
)

ENTRY="${MODELS[$SLURM_ARRAY_TASK_ID]}"
LABEL="$(echo "$ENTRY" | cut -d'|' -f1)"
MID="$(echo "$ENTRY" | cut -d'|' -f2)"
MB="$(echo "$ENTRY" | cut -d'|' -f3)"
GB="$(echo "$ENTRY" | cut -d'|' -f4)"

echo "=============================================================="
echo "[$(date +%H:%M:%S)] task $SLURM_ARRAY_TASK_ID  $LABEL  ($MID, micro-batch $MB)"
echo "=============================================================="

$PY tools/downstream_sft.py \
    --model-id "$MID" \
    --label "$LABEL" \
    --task-dir "$TASKDIR" \
    --out "$TASKDIR/sft__${LABEL}.json" \
    --micro-batch "$MB" \
    --gen-batch "$GB"

echo "[$(date +%H:%M:%S)] done -> $TASKDIR/sft__${LABEL}.json"
