#!/bin/bash
# E8 -- seed sensitivity of adapted BPB.  Account AIFAC_F02_541 is live to 2026-09-23
# (`saldo -b`, verified 2026-09-01: 80,000 h, 3.7% consumed).  Check `saldo -b` before
# assuming it -- slurm/ named a dead account for seven weeks and nothing checked.
#SBATCH --job-name=e8_seed
#SBATCH --output=logs/e8-%A_%a.out
#SBATCH --error=logs/e8-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_F02_541
#SBATCH --mem=0
#
# Every published domain-transfer cell ran with the torch RNG UNSEEDED: DataConfig.seed
# fixes the split and the 40,000-document subsample, but LoRA A/B initialisation, dropout
# and DataLoader batch order drew from the global torch RNG.  The paper's bootstrap
# resamples MODELS, so it carries no within-cell training variance at all.  This job
# measures that variance directly by re-running the identical protocol under explicit seeds.
#
# Everything except the seed is pinned to the published E3 protocol
# (tools/gcp_cohort_ext_run.sh): STEPS=250, EVAL_BLOCKS=2500, MAX_DOCS=40000, one shared
# LoRA config for every model.  Those five values are baked in below rather than made
# overridable, precisely so a stale artifact cannot be silently reused across a protocol
# change -- and the fingerprint check below enforces it even if someone edits them.
#
# Usage -- `logs/` must exist before submission, SLURM opens the log files before this
# script runs and cannot create the directory itself:
#   bash slurm/submit_e8.sh                 # computes the array range and submits
#   sbatch --array=0-50 slurm/seed_sensitivity.sbatch     # 17 models x 3 seeds

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

: "${SCRATCH:?SCRATCH must be set -- weights and outputs never go to \$HOME (50G quota).}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1          # compute nodes have no internet; weights are pre-staged
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MANIFEST="${MANIFEST:-$REPO/slurm/e8_cohort.txt}"
DATASET="${DATASET:-$SCRATCH/e4/news_full.jsonl}"
OUT="${OUT:-$SCRATCH/e8_seed}"
DT_CACHE="${DT_CACHE:-$SCRATCH/e8_seed/dt_cache}"
export DT_CACHE

# Pinned to the published E3 protocol.  NOT overridable: a replicate that changes any of
# these measures something other than seed variance.
STEPS=250
EVAL_BLOCKS=2500
MAX_DOCS=40000
SEED_ARR=(1 2 3)

# $HOME has a 50G quota and sits near it; large writes must never land there.
for path_var in HF_HOME OUT DT_CACHE; do
  case "${!path_var}" in
    "$HOME"/*|"$HOME") echo "FATAL: $path_var=${!path_var} is under \$HOME (50G quota)" >&2; exit 2 ;;
  esac
done

[ -f "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST" >&2; exit 2; }
[ -f "$DATASET" ]  || { echo "FATAL: dataset not found: $DATASET" >&2; exit 2; }
mkdir -p "$OUT" "$DT_CACHE"

mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "$MANIFEST")
N_MODELS=${#ROWS[@]}
N_SEEDS=${#SEED_ARR[@]}
TOTAL=$(( N_MODELS * N_SEEDS ))

# No cell may be run twice: reject duplicate manifest rows, slugs, or seeds up front.
dup_row=$(printf '%s\n' "${ROWS[@]}" | sort | uniq -d)
[ -z "$dup_row" ] && dup_row=$(printf '%s\n' "${ROWS[@]}" | awk '{print $2}' | sort | uniq -d)
[ -z "$dup_row" ] || { echo "FATAL: duplicate manifest entry: $dup_row" >&2; exit 2; }
dup_seed=$(printf '%s\n' "${SEED_ARR[@]}" | sort | uniq -d)
[ -z "$dup_seed" ] || { echo "FATAL: duplicate seed: $dup_seed" >&2; exit 2; }

# A non-array submission must NOT silently run cell 0 and report success.
: "${SLURM_ARRAY_TASK_ID:?FATAL: submit as a job array (--array=0-$(( TOTAL - 1 ))); a non-array run would silently cover one cell of $TOTAL}"
IDX="$SLURM_ARRAY_TASK_ID"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "FATAL: non-numeric array index: '$IDX'" >&2; exit 2; }
if [ "$IDX" -ge "$TOTAL" ]; then
  echo "FATAL: array index $IDX >= $TOTAL cells ($N_MODELS models x $N_SEEDS seeds)" >&2
  exit 2
fi

read -r MODEL_ID SLUG WANT_GPUS <<< "${ROWS[$(( IDX / N_SEEDS ))]}"
SEED="${SEED_ARR[$(( IDX % N_SEEDS ))]}"
[ -n "${SLUG:-}" ] || { echo "FATAL: manifest row $(( IDX / N_SEEDS )) has no slug" >&2; exit 2; }
[ -n "${WANT_GPUS:-}" ] || { echo "FATAL: manifest row for $SLUG has no gpus column" >&2; exit 2; }

# Fail loud rather than OOM twenty minutes in: this cell needs its measured GPU count.
GOT_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$GOT_GPUS" -lt "$WANT_GPUS" ]; then
  echo "FATAL: $SLUG needs $WANT_GPUS GPUs (measured, see slurm/e8_cohort.txt) but this" >&2
  echo "       allocation has $GOT_GPUS. Submit via slurm/submit_e8.sh." >&2
  exit 2
fi

# The artifact must be identifiable by the protocol that produced it, not just by model and
# seed, so a protocol change can never silently reuse a stale cell.
FINGERPRINT="steps${STEPS}_eval${EVAL_BLOCKS}_docs${MAX_DOCS}_$(basename "$DATASET" .jsonl)"
RESULT="$OUT/${SLUG}__seed${SEED}__${FINGERPRINT}.json"

# Resume only on an artifact that is complete AND matches this exact configuration.
if [ -s "$RESULT" ] && "$REPO/venv_lm_adapt/bin/python" - "$RESULT" "$MODEL_ID" "$SEED" "$STEPS" <<'PYCHECK'
import json, sys
path, model_id, seed, steps = sys.argv[1:5]
try:
    d = json.load(open(path))
except Exception:
    sys.exit(1)
ok = (d.get("model_id") == model_id and d.get("seed") == int(seed)
      and d.get("train_steps") == int(steps) and d.get("adapted_bpb") is not None)
sys.exit(0 if ok else 1)
PYCHECK
then
  echo "[skip] complete and configuration-matched: $RESULT"; exit 0
fi

echo "[e8] cell $IDX/$TOTAL  model=$MODEL_ID  slug=$SLUG  seed=$SEED"
echo "[e8] steps=$STEPS eval_blocks=$EVAL_BLOCKS max_docs=$MAX_DOCS dataset=$(basename "$DATASET")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Write to a per-task temp path and publish atomically, so an interrupted or failed run can
# never leave a partial artifact that a later retry would mistake for a finished cell.
TMP="$OUT/.tmp_${SLURM_ARRAY_JOB_ID:-$$}_${IDX}.json"
trap 'rm -f "$TMP"' EXIT

"$REPO/venv_lm_adapt/bin/python" tools/domain_transfer_eval.py \
  --model-id "$MODEL_ID" \
  --dataset "$DATASET" \
  --out "$TMP" \
  --seed "$SEED" \
  --max-train-steps "$STEPS" \
  --eval-blocks "$EVAL_BLOCKS" \
  --max-samples "$MAX_DOCS"

# A job is DONE only if it exits 0 AND writes a complete artifact.
[ -s "$TMP" ] || { echo "FATAL: exited 0 but wrote no artifact" >&2; exit 3; }
"$REPO/venv_lm_adapt/bin/python" -c "
import json,sys
d=json.load(open(sys.argv[1]))
assert d['adapted_bpb'] is not None, 'no adapted_bpb'
assert d['seed'] == int(sys.argv[2]), f\"seed {d['seed']} != {sys.argv[2]}\"
" "$TMP" "$SEED" || { echo "FATAL: artifact incomplete or seed not recorded" >&2; exit 3; }

mv -n "$TMP" "$RESULT"
[ -s "$RESULT" ] || { echo "FATAL: publish failed (concurrent writer?): $RESULT" >&2; exit 3; }
echo "[e8] OK -> $RESULT"
