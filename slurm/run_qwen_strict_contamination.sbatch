#!/bin/bash
# Account: AIFAC_S03_029 expired 2026-07-13. AIFAC_F02_541 is live to 2026-09-23
# (verified with `saldo -b` on 2026-09-01: 80,000 h, 3.7% consumed). Check `saldo -b`
# before assuming either.
#SBATCH --account=AIFAC_F02_541
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=lmab-qwen-strict-contam
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Strict-Mode Contamination Benchmark
# This script runs a full adaptation benchmark on comparable Qwen models
# while enforcing STRICT contamination hygiene. Min-K outliers are swept 
# into the DCQ perturbation quiz. Risk-flagged examples are conservatively quarantined
# up to the soft-evidence cap.

# How to submit:
# sbatch slurm/run_qwen_strict_contamination.sbatch

# To override defaults:
# DATASET="data/my_other_corpus.zip" sbatch slurm/run_qwen_strict_contamination.sbatch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source slurm/setup_env.sh

if [[ -z "${DATASET:-}" ]]; then
    echo "DATASET must point to an authorized TXT, JSONL, CSV, Hugging Face, or ZIP corpus." >&2
    exit 2
fi
OUTDIR="${OUTDIR:-runs/qwen_strict_contamination_${SLURM_JOB_ID}}"

# Prevent CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Models: We use Qwen2.5 0.5B and 1.5B as they are highly capable but cluster-safe.
# You can easily swap these to Qwen3 models (e.g. Qwen/Qwen3-1.7B) if desired.

python -m lm_adapt_bench.cli \
  --models "Qwen/Qwen2.5-0.5B" "Qwen/Qwen2.5-1.5B" \
  --dataset "$DATASET" \
  --output "$OUTDIR" \
  --flash-attention \
  --check-contamination \
  --contam-check-level strict \
  --contam-min-k-variant minkpp \
  --contam-enable-dcq \
  --contam-cleaning-policy quarantine \
  --contam-min-k-percent 20 \
  --contam-min-k-min-tokens 50 \
  --contam-min-k-outlier-percentile 95 \
  --contam-min-k-high-percentile 99 \
  --contam-min-k-robust-z-suspicious 2.5 \
  --contam-min-k-robust-z-high 3.5 \
  --contam-codec-context-size 4 \
  --contam-codec-samples 500 \
  --contam-dcq-samples 100 \
  --contam-dcq-num-distractors 3 \
  --contam-dcq-min-margin 3.0 \
  --contam-max-soft-eval-quarantine-fraction 0.10 \
  --contam-min-active-test-examples 100 \
  --contam-min-active-test-fraction 0.70 \
  --n-trials 10 \
  --sweep-steps 200 \
  --final-epochs 1

echo "Strict-mode benchmark completed. Reports and quarantine artifacts saved to $OUTDIR"
