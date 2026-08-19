#!/bin/bash
#SBATCH --job-name=lm_bench_setup
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# This script runs the environment setup on a compute node to avoid login node resource limits.
# To run this, use: sbatch slurm/setup_srun.sh

mkdir -p logs
chmod +x slurm/setup_env.sh
srun slurm/setup_env.sh
echo "Setup completed successfully."
