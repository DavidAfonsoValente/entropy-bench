#!/bin/bash
# E9 Probe B -- does forcing a uniform decode protocol actually make the hf backend
# reproduce July, and is the resulting throughput survivable for 51 cells?
#
# Probe A (job 56341304) established the blocker: HF generate() honours each model's
# shipped generation_config.json, so Qwen2.5 ran gsm8k under max_new_tokens=2048 against
# July's 256. eval/lm_eval_uniform.py strips that. This probe tests the fix on the two
# models whose shipped configs differ in the two distinct ways the survey found, and
# times the CoT benchmark that decides whether the full column fits in walltime.
# Account AIFAC_F02_541, live to 2026-09-23 (`saldo -b`, verified 2026-09-07).
#SBATCH --job-name=e9_probe
#SBATCH --output=logs/e9probe-%j.out
#SBATCH --error=logs/e9probe-%j.err
#SBATCH --time=04:00:00
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
# Leonardo's driver is 535.x = CUDA 12.2, so the engine must be a cu121-era build; a cu130
# wheel imports and reports torch.cuda.is_available() True, then fails at engine init
# with "driver too old (found version 12020)". lm-eval stays pinned at 0.4.12 because
# the July task_hashes are the reproduction target.
VENV="${VENV:-$SCRATCH/venv_e9hf}"
V="$VENV/bin/python"
[ -x "$V" ] || { echo "FATAL: no interpreter at $V" >&2; exit 2; }
OUT="$SCRATCH/e9/probe_b"; rm -rf "$OUT"; mkdir -p "$OUT"
BATCH="${BATCH:-16}"

echo "=== driver / gpu ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== import chain (loud if the driver is too old for cu130) ==="
"$V" -c "
import torch, transformers, lm_eval
print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())
print('transformers', transformers.__version__, 'lm_eval', lm_eval.__version__)
want = {'lm_eval': '0.4.12', 'transformers': '5.11.0'}
got = {'lm_eval': lm_eval.__version__, 'transformers': transformers.__version__}
assert got == want, f'stack drift -- expected {want}, got {got}'
assert torch.cuda.is_available(), 'CUDA not available on a GPU node -- driver/toolkit mismatch'
# is_available() does not exercise the CUDA runtime -- a cu130 wheel passes it and then dies at
# engine init. Force a real kernel launch so a driver mismatch fails HERE, loudly.
x = torch.ones(8, device='cuda') @ torch.ones(8, 8, device='cuda')
torch.cuda.synchronize()
assert float(x.sum()) == 64.0, 'CUDA compute returned the wrong answer'
print('CUDA compute verified (real kernel launch, not just is_available)')
"

echo "=== custom task must resolve, and a misspelled one must ABORT ==="
"$V" - <<'PY'
from lm_eval.tasks import TaskManager
import os
tm = TaskManager(include_path=os.environ["SCRATCH"] + "/tasks")
names = set(tm.all_tasks)
assert "mmlu_pro_1k" in names, "custom task did NOT register"
assert "mmlu_pro_1k_typo_check" not in names, "a nonexistent task resolved -- name matching is unsafe"
print("task registry OK: mmlu_pro_1k present, bogus name absent")
PY

# ---------------------------------------------------------------------------
# The two cells. Qwen2.5-0.5B ships max_new_tokens=2048 (the confound that is LIVE under
# greedy); Llama-3.2-1B ships do_sample/temperature/top_p (inert, because both tasks set
# do_sample=false explicitly -- so this cell must reproduce July whether or not the
# wrapper runs, and it is here to prove the wrapper does not BREAK an unaffected model).
# ---------------------------------------------------------------------------
# Mirrors slurm/e9_bench.sbatch exactly, max_length included -- a probe that runs a
# different command from the runner it is clearing does not clear it.
run_cell () {
  local mid="$1" label="$2" task="$3" shots="$4" mml="$5" inc="$6" lim="$7"
  local dir="$OUT/$label/$task"
  mkdir -p "$dir"
  echo "=== cell: $label x $task (batch $BATCH, max_length $mml) ==="
  local t0 t1
  t0=$(date +%s)
  "$V" eval/lm_eval_uniform.py --gen-audit "$dir/gen_audit.json" -- \
    --model hf \
    --model_args "pretrained=$mid,dtype=bfloat16,trust_remote_code=True,max_length=$mml" \
    --tasks "$task" --num_fewshot "$shots" --batch_size "$BATCH" \
    --seed 0,1234,1234,1234 $inc $lim --log_samples --output_path "$dir"
  t1=$(date +%s)
  echo "[timing] $label x $task elapsed $((t1-t0))s"
  echo "$((t1-t0))" > "$dir/elapsed_s"
}

run_cell Qwen/Qwen2.5-0.5B       Qwen2.5-0.5B gsm8k 5 4096 "" ""
run_cell meta-llama/Llama-3.2-1B Llama-3.2-1B gsm8k 5 4096 "" ""

# CoT throughput only -- never collected as a score. mmlu_pro_1k is a GROUP of 14 subject
# subtasks and --limit applies per subtask, so --limit 5 is ~70 items, not 5.
run_cell Qwen/Qwen2.5-0.5B Qwen2.5-0.5B mmlu_pro_1k 5 8192 "--include_path $SCRATCH/tasks" "--limit 5"

echo "=== GATE ==="
"$V" - "$OUT" <<'PY'
import glob, json, os, subprocess, sys

root = sys.argv[1]
JULY_HASH = "2330f4ebfcccaf66a892922df2819cdb1f118e448d076d3f42bdde4177678ac7"
# July gsm8k, flexible-extract (results/gsm8k/**/results_*.json).
JULY = {"Qwen2.5-0.5B": 0.34723275208491283, "Llama-3.2-1B": 0.06520090978013647}
# The survey (docs/RUN_LEDGER.md, "E9 -- generation_config survey") says exactly this must
# be stripped. Asserting the CONTENT, not just that stripping happened, is what catches a
# wrapper that silently no-ops or a checkpoint that changed under us.
MUST_STRIP = {
    "Qwen2.5-0.5B": {"max_new_tokens"},
    "Llama-3.2-1B": {"do_sample", "temperature", "top_p"},
}
# July's effective gsm8k cap: the task sets no max_gen_toks, so lm_eval's default applies.
CAP = 256

ok = True
def fail(msg):
    global ok
    print("FAIL " + msg); ok = False

def latest(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[-1] if hits else None

for label, july in JULY.items():
    d = f"{root}/{label}/gsm8k"
    f = latest(d + "/**/results_*.json")
    if f is None:
        fail(f"{label}: no results_*.json"); continue
    r = json.load(open(f))

    h = r["task_hashes"]["gsm8k"]
    if h != JULY_HASH:
        fail(f"{label}: task_hash {h} != July -- a different task/dataset/few-shot draw")
    if r["versions"]["gsm8k"] != 3.0:
        fail(f"{label}: task version {r['versions']['gsm8k']} != 3.0")
    n = r["n-samples"]["gsm8k"]
    n = n["original"] if isinstance(n, dict) else n
    if n != 1319:
        fail(f"{label}: n {n} != 1319")

    s = r["results"]["gsm8k"]["exact_match,flexible-extract"]
    delta = 100 * (s - july)
    print(f"{label}: score {s:.6f}  July {july:.6f}  delta {delta:+.3f} pp")
    # The preregistration's overlap tolerance: 1.0 absolute point.
    if abs(delta) > 1.0:
        fail(f"{label}: |delta| {abs(delta):.3f} pp > 1.0 -- outside the prereg tolerance")

    # The wrapper did what it claims. tools/check_gen_audit.py is the same validator the
    # full runner gates every cell on, so the probe clears the runner's gate rather than a
    # probe-only imitation of it.
    a = f"{d}/gen_audit.json"
    if not os.path.exists(a):
        fail(f"{label}: no gen_audit.json -- the wrapper did not run")
        continue
    if subprocess.call([sys.executable, "tools/check_gen_audit.py", a, "gsm8k"]) != 0:
        fail(f"{label}: gen_audit.json failed the runner's own validator")

    aud = json.load(open(a))
    got = set(aud["stripped"])
    want = MUST_STRIP[label]
    if not want <= got:
        fail(f"{label}: stripped {sorted(got)} is missing {sorted(want - got)}")
    else:
        print(f"{label}: stripped {sorted(got)}")

    # The cap, measured directly. The wrapper records out.shape[1] - context.shape[1] from
    # inside _model_generate -- the true count of generated token ids on the object that
    # actually generated. A decode/re-encode of the logged text is neither count-preserving
    # nor complete, since stop-string handling trims it before it is logged.
    obs = aud.get("max_new_tokens_observed")
    calls = aud.get("generate_calls", 0)
    print(f"{label}: longest generation {obs} tokens over {calls} batched calls (July cap {CAP})")
    if obs is None:
        fail(f"{label}: the wrapper recorded no generation at all")
    elif obs > CAP:
        fail(f"{label}: a generation produced {obs} tokens > {CAP} -- the cap is NOT uniform")
    elif obs < CAP:
        # Not a failure: stop strings end most completions early. Recorded because it means
        # this cell did not actually exercise the cap.
        print(f"{label}: note -- the cap was never reached, so it was not exercised here")

print("=== throughput ===")
# Item counts come from the artifact, never hardcoded: mmlu_pro_1k is a group of 14 subject
# subtasks and --limit applies per subtask, so the probe's item count is not the limit value.
for label, task, full_n in (("Qwen2.5-0.5B", "gsm8k", 1319),
                            ("Llama-3.2-1B", "gsm8k", 1319),
                            ("Qwen2.5-0.5B", "mmlu_pro_1k", 1000)):
    d = f"{root}/{label}/{task}"
    ep = f"{d}/elapsed_s"
    rf = latest(d + "/**/results_*.json")
    if not (os.path.exists(ep) and rf):
        continue
    e = int(open(ep).read().strip())
    ns = json.load(open(rf))["n-samples"]
    n = sum(v["effective"] if isinstance(v, dict) else v for v in ns.values())
    print(f"{label} x {task}: {e}s over {n} items, {e/n:.3f} s/item "
          f"-> the full {full_n}-item benchmark = {e/n*full_n/3600:.2f} h on this model")

print("PROBE B PASS" if ok else "PROBE B FAIL")
sys.exit(0 if ok else 1)
PY
