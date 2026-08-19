#!/bin/bash
#SBATCH --job-name=lm_bench_final3
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_S03_029
#SBATCH --mem=0

# Final run: only Qwen3.5-35B-A3B-Base re-trains (no result_json); the other 10 models
# have cached result_json and are skipped. 3 nodes is enough: with all_models sorted and
# world_size=3, rank0 skips all its models (free aggregator), rank1 runs the 35B, rank2 skips.
# Schedules far faster than 11 nodes while producing the full 11-model unified report.

mkdir -p logs
source slurm/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
: "${DATASET:?Set DATASET to the authorized benchmark corpus path before submission.}"
: "${HF_TOKEN:?Set HF_TOKEN in the submission environment; never commit access tokens.}"

srun python -m lm_adapt_bench.cli \
    --dataset "$DATASET" \
    --output "./results" \
    --device cuda \
    --flash-attention \
    --models \
        "meta-llama/Llama-3.2-1B" \
        "LiquidAI/LFM2.5-1.2B-Base" \
        "Qwen/Qwen2.5-0.5B" \
        "Qwen/Qwen2.5-1.5B" \
        "Qwen/Qwen2.5-7B" \
        "Qwen/Qwen3.5-35B-A3B-Base" \
        "Qwen/Qwen3.5-9B-Base" \
        "Qwen/Qwen3.5-4B-Base" \
        "google/gemma-4-31B" \
        "mistralai/Ministral-3-14B-Base-2512" \
        "google/gemma-4-12B" \
    --hf-token "$HF_TOKEN" \
    --contam-check-level strict \
    --contam-cleaning-mode global_unified \
    --wall-time-seconds 50400

echo "final_3node run completed."
