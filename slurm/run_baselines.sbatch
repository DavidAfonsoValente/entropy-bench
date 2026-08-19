#!/bin/bash
#SBATCH --job-name=baselines_eval
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -p boost_usr_prod
#SBATCH -A AIFAC_S03_029

source slurm/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
: "${DATASET:?Set DATASET to an authorized benchmark corpus before submission.}"

# Run baseline-only evaluations for the three giant models
srun python -m lm_adapt_bench.cli \
    --dataset "$DATASET" \
    --output ./results \
    --device cuda \
    --flash-attention \
    --baseline-only \
    --models \
    "Qwen/Qwen3.5-35B-A3B-Base" \
    "google/gemma-4-31B" \
    "deepseek-ai/DeepSeek-V4-Flash-Base"

echo "Baselines evaluation completed successfully."
