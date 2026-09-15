#!/bin/bash
# E9 Probe C -- the ONLY open question after Probe B: does the largest cell fit in walltime?
#
# Probe B settled correctness (job 56610639, PASS): the uniform protocol reproduces July on
# two models and the realised cap is 256 tokens, measured inside _model_generate. It also
# measured throughput, but only on 0.5B and 1.2B models. HF decode is bandwidth-bound on
# weights, so per-token cost scales roughly with model size, and the 2048-token CoT path on
# mmlu_pro_1k is where that bites. This probe measures the two worst cells directly rather
# than extrapolating 60x from a 0.5B model.
#
# It is throughput-only. --limit 2 is 28 items across the 14 subject subtasks and is NEVER
# collected as a score.
# Account AIFAC_F02_541, live to 2026-09-23.
#SBATCH --job-name=e9_probec
#SBATCH --output=logs/e9probec-%j.out
#SBATCH --error=logs/e9probec-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_F02_541
#SBATCH --mem=0
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
: "${SCRATCH:?}"
export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
VENV="${VENV:-$SCRATCH/venv_e9hf}"
V="$VENV/bin/python"
[ -x "$V" ] || { echo "FATAL: no interpreter at $V" >&2; exit 2; }
OUT="$SCRATCH/e9/probe_c"; rm -rf "$OUT"; mkdir -p "$OUT"
BATCH="${BATCH:-16}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GOT=$("$V" -c "import torch; print(torch.cuda.device_count())")
echo "[probec] GPUs allocated to this job: $GOT"
[ "$GOT" -ge 2 ] || { echo "FATAL: need 2 GPUs, got $GOT" >&2; exit 2; }

# A loader/capacity fault here is an EXCLUSION under the preregistration's rule 1, diagnosed
# before any score for the model exists. Record it verbatim and keep going: knowing which of
# the two big models loads at all is the point of the probe.
run_big () {
  local mid="$1" label="$2"
  local dir="$OUT/$label"; mkdir -p "$dir"
  echo "=== $label x mmlu_pro_1k (--limit 2, throughput only) ==="
  local t0 t1 rc
  t0=$(date +%s)
  set +e
  "$V" eval/lm_eval_uniform.py --gen-audit "$dir/gen_audit.json" -- \
    --model hf \
    --model_args "pretrained=$mid,dtype=bfloat16,trust_remote_code=True,max_length=8192,parallelize=True" \
    --tasks mmlu_pro_1k --num_fewshot 5 --batch_size "$BATCH" \
    --seed 0,1234,1234,1234 --include_path "$SCRATCH/tasks" --limit 2 \
    --log_samples --output_path "$dir" > "$dir/run.log" 2>&1
  rc=$?
  set -e
  t1=$(date +%s)
  echo "$((t1-t0))" > "$dir/elapsed_s"; echo "$rc" > "$dir/rc"
  if [ "$rc" -ne 0 ]; then
    echo "[probec] $label FAILED rc=$rc -- verbatim tail follows (prereg rule 1: exclusion evidence)"
    tail -25 "$dir/run.log"
  else
    echo "[probec] $label ok in $((t1-t0))s"
    grep -E "^\[uniform\]" "$dir/run.log" || true
  fi
}

run_big google/gemma-4-31B            gemma-4-31B
run_big Qwen/Qwen3.5-35B-A3B-Base     Qwen3.5-35B-MoE

echo "=== PROJECTION ==="
"$V" - "$OUT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
# Probe B, job 56610639: the same cell on Qwen2.5-0.5B took 235s over 70 items.
BASELINE = ("Qwen2.5-0.5B", 3.357)
WALLTIME_H = 20.0
print(f"reference: {BASELINE[0]} mmlu_pro_1k {BASELINE[1]:.3f} s/item (probe B, job 56610639)\n")
for label in ("gemma-4-31B", "Qwen3.5-35B-MoE"):
    d = f"{root}/{label}"
    rc = open(f"{d}/rc").read().strip() if os.path.exists(f"{d}/rc") else "?"
    if rc != "0":
        print(f"{label}: DID NOT RUN (rc={rc}) -- exclusion evidence in {d}/run.log")
        continue
    e = int(open(f"{d}/elapsed_s").read().strip())
    hits = sorted(glob.glob(d + "/**/results_*.json", recursive=True))
    if not hits:
        print(f"{label}: exited 0 but wrote no results -- treat as failed")
        continue
    ns = json.load(open(hits[-1]))["n-samples"]
    n = sum(v["effective"] if isinstance(v, dict) else v for v in ns.values())
    # Load time is paid once per cell and does not scale with items, but at --limit 2 it is a
    # large share of elapsed. Report both bounds so the projection is not quietly optimistic.
    per = e / n
    full_h = per * 1000 / 3600
    print(f"{label}: {e}s over {n} items = {per:.3f} s/item "
          f"({per/BASELINE[1]:.1f}x the 0.5B reference)")
    print(f"    upper bound incl. per-item load amortisation: full 1000-item cell "
          f"= {full_h:.2f} h vs {WALLTIME_H:.0f} h walltime "
          f"-- {'FITS' if full_h < WALLTIME_H else 'DOES NOT FIT'}")
PY
