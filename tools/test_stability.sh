#!/bin/bash
# Stability Tester for LM Adapt Bench
# Usage: ./test_stability.sh <MODEL_ID> <DATASET>

MODEL=$1
DATASET=$2

echo "Running Stability Test for $MODEL..."

# 1. Low Budget Sweep
echo "--- Phase 1: 100 Step Budget ---"
python -m lm_adapt_bench.cli --models $MODEL --dataset $DATASET --sweep-steps 100 --output results_stability/low --n-trials 10 --baseline-only

# 2. Medium Budget Sweep
echo "--- Phase 2: 500 Step Budget ---"
python -m lm_adapt_bench.cli --models $MODEL --dataset $DATASET --sweep-steps 500 --output results_stability/medium --n-trials 10 --baseline-only

echo "Done. Compare 'results_stability/low/$MODEL/best_config.json' and 'results_stability/medium/$MODEL/best_config.json'"
