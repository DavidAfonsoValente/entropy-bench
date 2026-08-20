#!/bin/bash
#SBATCH --job-name=lm_adapt_bench
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# This script runs a SINGLE model adaptation run.
# To run multiple models in parallel across multiple GPUs, 
# use the parallel_launcher.sh or submit multiple jobs.

# Check if model and dataset are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch slurm/submit_batch.sh <MODEL_ID> <DATASET_PATH> [OUTPUT_DIR]"
    exit 1
fi

MODEL=$1
DATASET=$2
OUTPUT=${3:-"./results"}

mkdir -p logs

source slurm/setup_env.sh

echo "Running Entropy Bench for model: $MODEL"
python -m lm_adapt_bench.cli \
    --models "$MODEL" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --device cuda \
    --flash-attention \
    --n-trials 20 \
    --sweep-steps 200 \
    --final-epochs 3

echo "Job completed."
