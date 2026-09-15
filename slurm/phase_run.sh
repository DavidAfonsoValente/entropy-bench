#!/bin/bash
# Account: AIFAC_S03_029 expired 2026-07-13. AIFAC_F02_541 is live to 2026-09-23
# (verified with `saldo -b` on 2026-09-01: 80,000 h, 3.7%% consumed). Check `saldo -b`
# before assuming either -- this file was unusable for seven weeks because it named a
# dead account and nothing checked.
#SBATCH --job-name=lm_phase
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_F02_541
#SBATCH --mem=0
#
# Single-node phased pipeline with self-chaining training-until-convergence.
#   sbatch slurm/phase_run.sh sweep  <MODEL>
#   sbatch slurm/phase_run.sh train  <MODEL> [CHAIN_INDEX]   # self-resubmits until converged
#   sbatch slurm/phase_run.sh aggregate
# A 'train' link that converges (writes result_json) auto-submits 'aggregate' to refresh the
# report. A non-converged link resubmits the next link (dependency) up to MAX_CHAIN; the last
# link uses --final-link so it always yields a usable result.
MODE="${1:?usage: phase_run.sh <sweep|train|aggregate> [MODEL] [CHAIN_INDEX]}"
MODEL="${2:-}"
CHAIN="${3:-0}"
MAX_CHAIN=5

: "${DATASET:?Set DATASET to the authorized benchmark corpus path before submission.}"
: "${HF_TOKEN:?Set HF_TOKEN in the submission environment; never commit access tokens.}"
ALL_MODELS=( "meta-llama/Llama-3.2-1B" "LiquidAI/LFM2.5-1.2B-Base" "Qwen/Qwen2.5-0.5B" \
  "Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-7B" "Qwen/Qwen3.5-35B-A3B-Base" "Qwen/Qwen3.5-9B-Base" \
  "Qwen/Qwen3.5-4B-Base" "google/gemma-4-31B" "mistralai/Ministral-3-14B-Base-2512" "google/gemma-4-12B" )

mkdir -p logs
export LM_SETUP_PHASE=false
source slurm/setup_env.sh
# setup_env.sh enables `set -e`; disable it (and unbound-var errors) so the post-srun chaining
# logic (resubmit / aggregate) always runs regardless of srun's exit code.
set +eu
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

COMMON=( --reuse-contamination --hf-token "$HF_TOKEN" --contam-check-level strict \
  --contam-cleaning-mode global_unified --dataset "$DATASET" --output ./results \
  --device cuda --flash-attention --world-size 1 --rank 0 )

case "$MODE" in
  sweep)
    srun python -m lm_adapt_bench.cli --phase sweep --no-aggregate "${COMMON[@]}" \
        --wall-time-seconds 75600 --n-trials 20 --models "$MODEL"
    echo "[phase_run] sweep done for $MODEL"
    ;;

  train)
    FINAL_FLAG=()
    if [ "$((CHAIN+1))" -ge "$MAX_CHAIN" ]; then FINAL_FLAG=(--final-link); fi
    srun python -m lm_adapt_bench.cli --phase train --no-aggregate --chain-index "$CHAIN" \
        "${FINAL_FLAG[@]}" "${COMMON[@]}" --wall-time-seconds 79200 --models "$MODEL"

    SLUG=$(python -c "from lm_adapt_bench.utils import slugify; print(slugify('$MODEL'))")
    NO_AGG="${LM_CHAIN_NO_AGGREGATE:-0}"   # 1 => don't auto-aggregate (multi-model batch: aggregate once at the end)
    if [ -f "results/$SLUG/result_$SLUG.json" ]; then
        if [ "$NO_AGG" = "1" ]; then
            echo "[phase_run] $MODEL CONVERGED at link $CHAIN -> aggregation deferred (LM_CHAIN_NO_AGGREGATE=1)"
        else
            echo "[phase_run] $MODEL CONVERGED at link $CHAIN -> submitting aggregation"
            sbatch slurm/phase_run.sh aggregate
        fi
    else
        NEXT=$((CHAIN+1))
        if [ "$NEXT" -lt "$MAX_CHAIN" ]; then
            echo "[phase_run] $MODEL not converged -> chaining link $NEXT (after $SLURM_JOB_ID)"
            LM_CHAIN_NO_AGGREGATE="$NO_AGG" sbatch --export=ALL --dependency=afterany:"$SLURM_JOB_ID" slurm/phase_run.sh train "$MODEL" "$NEXT"
        elif [ "$NO_AGG" = "1" ]; then
            echo "[phase_run] $MODEL hit MAX_CHAIN=$MAX_CHAIN -> aggregation deferred (LM_CHAIN_NO_AGGREGATE=1)"
        else
            echo "[phase_run] $MODEL hit MAX_CHAIN=$MAX_CHAIN -> aggregating best-so-far"
            sbatch slurm/phase_run.sh aggregate
        fi
    fi
    ;;

  finalize)
    # Force-finalize from the best checkpoint without further meaningful training, then aggregate.
    # Tiny wall-time -> train_budget floors at 600s; --final-link -> write result from best_adapter.
    srun python -m lm_adapt_bench.cli --phase train --no-aggregate --chain-index 99 --final-link \
        "${COMMON[@]}" --wall-time-seconds 1800 --models "$MODEL"
    SLUG=$(python -c "from lm_adapt_bench.utils import slugify; print(slugify('$MODEL'))")
    if [ -f "results/$SLUG/result_$SLUG.json" ]; then
        echo "[phase_run] $MODEL finalized -> submitting aggregation"
        sbatch slurm/phase_run.sh aggregate
    else
        echo "[phase_run] WARN: finalize did not produce result_json for $MODEL"
    fi
    ;;

  aggregate)
    # All models have result_json -> all skip -> rank 0 aggregates + re-evals -> report.
    srun python -m lm_adapt_bench.cli "${COMMON[@]}" --wall-time-seconds 79200 --models "${ALL_MODELS[@]}"
    echo "[phase_run] aggregation done -> results/report.pdf"
    ;;

  *) echo "unknown mode: $MODE"; exit 1 ;;
esac
