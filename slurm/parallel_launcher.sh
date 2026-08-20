#!/bin/bash
#SBATCH --job-name=lm_bench_parallel
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=0

# Parallel Launcher for Entropy Bench
# This script is launched by run_bench.sh which provides dynamic GRES, node counts, and flags.

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch slurm/parallel_launcher.sh <DATASET_PATH> <MODEL1> <MODEL2> ... [--extra-flags]"
    exit 1
fi

DATASET=$1
shift

# The MODELS and EXTRA_FLAGS are mixed in the arguments. 
# cli.py will handle the parsing.
ARGS=("$@")

mkdir -p logs
source slurm/setup_env.sh

# Prevent CUDA fragmentation and handle large models efficiently
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# Use srun to launch N processes in parallel.
# We pass all remaining arguments directly to the CLI.
srun python -m lm_adapt_bench.cli \
    --dataset "$DATASET" \
    --output "./results" \
    --device cuda \
    --flash-attention \
    "${ARGS[@]}"

echo "Parallel benchmark run completed."
