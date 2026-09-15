#!/usr/bin/env bash
# Submit E8 with the array range computed from the manifest, so a hand-typed range can never
# silently skip cells, and with each model's MEASURED GPU count -- the cohort splits into
# one-GPU and four-GPU models and that split is not monotonic in parameter count (see the
# header of slurm/e8_cohort.txt).  Submits one array per GPU tier, each restricted to the
# cells that belong to it.
#
# logs/ is created here because SLURM opens the log files before the batch script runs and
# cannot create the directory itself.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MANIFEST="${MANIFEST:-$REPO/slurm/e8_cohort.txt}"
N_SEEDS=3                       # must match SEED_ARR in slurm/seed_sensitivity.sbatch
mkdir -p logs

mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "$MANIFEST")
declare -A CELLS_FOR_TIER
for row_idx in "${!ROWS[@]}"; do
  read -r _ slug gpus <<< "${ROWS[$row_idx]}"
  [ -n "${gpus:-}" ] || { echo "FATAL: manifest row $row_idx ($slug) has no gpus column" >&2; exit 2; }
  for seed_idx in $(seq 0 $(( N_SEEDS - 1 ))); do
    cell=$(( row_idx * N_SEEDS + seed_idx ))
    CELLS_FOR_TIER[$gpus]="${CELLS_FOR_TIER[$gpus]:+${CELLS_FOR_TIER[$gpus]},}$cell"
  done
done

echo "[e8] ${#ROWS[@]} models x $N_SEEDS seeds = $(( ${#ROWS[@]} * N_SEEDS )) cells"
for gpus in $(printf '%s\n' "${!CELLS_FOR_TIER[@]}" | sort -n); do
  list="${CELLS_FOR_TIER[$gpus]}"
  n=$(tr ',' '\n' <<< "$list" | wc -l)
  echo "[e8] ${gpus}-GPU tier: $n cells -> --array=$list"
  sbatch --array="$list%8" --gres="gpu:$gpus" "$@" "$REPO/slurm/seed_sensitivity.sbatch"
done
