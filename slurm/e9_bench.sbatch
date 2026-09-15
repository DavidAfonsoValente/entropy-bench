#!/bin/bash
# E9 -- full public-benchmark column, one host, one build. 17 models x 3 benchmarks = 51 cells.
# Preregistration: docs/RUN_LEDGER.md, "E9 -- full benchmark-column re-score".
# Account AIFAC_F02_541, live to 2026-09-23.
#SBATCH --job-name=e9_bench
#SBATCH --output=logs/e9-%A_%a.out
#SBATCH --error=logs/e9-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Allocator strategy only -- it changes how PyTorch reuses freed blocks, never what is
# computed, so it cannot move a score. Probe C's gemma-4-31B OOM had 9.54 GiB reserved but
# unallocated, i.e. fragmentation rather than genuine exhaustion. Set for EVERY cell so the
# column stays uniform, rather than only for the cells that would otherwise fail.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Leonardo's driver is 535.x = CUDA 12.2, so the engine must be a cu121 build; a cu130
# wheel imports and reports torch.cuda.is_available() True, then fails at engine init
# with "driver too old (found version 12020)". lm-eval stays pinned at 0.4.12 because
# the July task_hashes are the reproduction target.
VENV="${VENV:-$SCRATCH/venv_e9hf}"
V="$VENV/bin/python"
[ -x "$V" ] || { echo "FATAL: no interpreter at $V" >&2; exit 2; }
# eval/lm_eval_uniform.py patches HFLM.__init__ and HFLM._model_generate and reads the
# private _model/model layout, so stack drift between array cells is a correctness problem,
# not just a reproducibility one. Assert the pinned versions rather than printing them.
"$V" -c "
import lm_eval, torch, transformers, sys
want = {'lm_eval': '0.4.12', 'transformers': '5.11.0'}
got = {'lm_eval': lm_eval.__version__, 'transformers': transformers.__version__}
if got != want:
    sys.exit(f'FATAL: stack drift -- expected {want}, got {got}')
print(f'[e9] stack pinned: lm_eval {got[\"lm_eval\"]}, transformers {got[\"transformers\"]}, torch {torch.__version__}')
"
MANIFEST="${MANIFEST:-$REPO/slurm/e9_cells.txt}"
OUTROOT="${OUTROOT:-$SCRATCH/e9/cells}"

mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "$MANIFEST")
TOTAL=${#ROWS[@]}
: "${SLURM_ARRAY_TASK_ID:?submit as an array (--array=0-$((TOTAL-1)))}"
IDX="$SLURM_ARRAY_TASK_ID"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "FATAL: bad index '$IDX'" >&2; exit 2; }
[ "$IDX" -lt "$TOTAL" ] || { echo "FATAL: index $IDX >= $TOTAL" >&2; exit 2; }

# "<hf-id> <label> <bench> <tp>" -- tp is July's value for the published eleven, 1 for the six
# extension models. The E8 GPU column is a LoRA-TRAINING measurement and must not be used here.
read -r MODEL_ID LABEL BENCH TP <<< "${ROWS[$IDX]}"
# INC: --include_path makes lm_eval apply the mmlu_pro_1k task dir's _default_template_yaml
# to every task loaded through it, and that template sets max_gen_toks: 2048 -- which is the
# intended protocol for mmlu_pro_1k and NOT for gsm8k, whose July cap was lm_eval's 256
# default. Scoping the include to the one task that needs it keeps each task on its own
# July setting. (This is hygiene, not the fix for the 2048 cap -- see the wrapper below.)
case "$BENCH" in
  gsm8k)       TASK=gsm8k;       SHOTS=5;  MML=4096; INC="" ;;
  hellaswag)   TASK=hellaswag;   SHOTS=10; MML=4096; INC="" ;;
  mmlu_pro_1k) TASK=mmlu_pro_1k; SHOTS=5;  MML=8192; INC="--include_path $SCRATCH/tasks" ;;
  *) echo "FATAL: unknown benchmark '$BENCH'" >&2; exit 2 ;;
esac

# Per-cell output dir: run_benchmark.py-style shared tables clobber under a concurrent array.
OUT="$OUTROOT/$LABEL/$BENCH"
# A result counts as a finished cell only if a valid protocol audit came with it. Skipping
# on the result alone accepts anything left in OUTROOT -- including a cell scored before
# this wrapper existed, under the very max_new_tokens=2048 the experiment exists to remove.
if compgen -G "$OUT/**/results_*.json" > /dev/null 2>&1; then
  if "$V" tools/check_gen_audit.py "$OUT/gen_audit.json" "$BENCH"; then
    echo "[skip] $LABEL x $BENCH already scored under the uniform protocol"; exit 0
  fi
  echo "FATAL: $LABEL x $BENCH has a result with no valid gen_audit.json -- it was not" \
       "scored under the uniform protocol. Delete it deliberately before re-running." >&2
  exit 4
fi
mkdir -p "$OUT"

# Count the GPUs allocated to THIS job, not those present on the node: nvidia-smi reports
# the host's physical devices and would pass on a job holding none of them.
GOT=$("$V" -c "import torch; print(torch.cuda.device_count())")
[ "$GOT" -ge "$TP" ] || {
  echo "FATAL: $LABEL needs tp=$TP, this job was allocated $GOT GPUs." \
       "Submit the tp=2 cells with --gres=gpu:2." >&2; exit 2; }

echo "[e9] cell $IDX/$TOTAL  $LABEL x $BENCH  tp=$TP  shots=$SHOTS  max_model_len=$MML"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# HF backend, not vLLM. Every vLLM wheel new enough for transformers 5 (needed by the 2026
# checkpoints) links libcudart.so.13, and Leonardo's driver is 535 = CUDA 12.2, so no vLLM
# build can run here. lm-eval and the task definitions are unchanged, and task_hashes are
# computed from doc/prompt/target text, so they are backend-independent and still validate.
# parallelize=True shards big models over the allocated GPUs via device_map.
# eval/lm_eval_uniform.py replaces each model's shipped generation_config with a
# library-default one carrying only its bos/eos/pad ids. Without it HF generate() honours
# whatever the checkpoint ships and lm_eval's length control is silently outranked: the five
# Qwen2.5 models ship max_new_tokens=2048 against gsm8k's July cap of 256 (seen live in probe
# 56341304), and gemma-4-12B ships suppress_tokens where gemma-4-31B does not. The audit JSON
# records the shipped-vs-effective diff per cell so the column can be checked after the fact.
PARALLEL=$([ "$TP" -gt 1 ] && echo ",parallelize=True" || echo "")
# max_length=$MML is July's max_model_len and MUST be passed: HFLM falls back to the
# checkpoint's own config/tokenizer when it is absent, which is model-dependent and changes
# prompt truncation and generation headroom per model -- exactly the confound this column
# exists to remove. An earlier revision selected MML, echoed it, and never applied it.
"$V" eval/lm_eval_uniform.py --gen-audit "$OUT/gen_audit.json" -- \
  --model hf \
  --model_args "pretrained=$MODEL_ID,dtype=bfloat16,trust_remote_code=True,max_length=$MML${PARALLEL}" \
  --tasks "$TASK" --num_fewshot "$SHOTS" --batch_size "${BATCH:-16}" \
  --seed 0,1234,1234,1234 $INC --log_samples --output_path "$OUT"

# Done only if it exits 0 AND writes the artifact.
compgen -G "$OUT/**/results_*.json" > /dev/null 2>&1 || {
  echo "FATAL: exited 0 but wrote no results_*.json" >&2; exit 3; }
"$V" tools/check_gen_audit.py "$OUT/gen_audit.json" "$BENCH" || {
  echo "FATAL: $LABEL x $BENCH did not run under the uniform protocol" >&2; exit 3; }
echo "[e9] OK -> $OUT"
