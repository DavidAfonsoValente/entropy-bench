#!/bin/bash
# Is gemma-4-12B's generative failure caused by OUR wrapper, or by HF-vs-vLLM?
#
# The wrapper replaces model.generation_config with a fresh object carrying only bos/eos/pad.
# The survey that justified that read generation_config.JSON -- it cannot see fields the model
# class sets programmatically. Gemma uses sliding-window/hybrid attention, and losing a cache
# default would look exactly like what we see: loglikelihood correct, generation degraded, and
# worse on longer outputs (mmlu_pro_1k -14.3 vs gsm8k -9.9).
#
# Two parts: dump the as-loaded config with vars() (NOT to_json_string(use_diff=True), which
# hides class-level and private fields -- the whole reason the survey missed this), then run
# gsm8k with PLAIN lm_eval, no wrapper, everything else identical. July: 0.6360879454131918.
#SBATCH --job-name=e9gdiag
#SBATCH --output=logs/e9gdiag-%j.out
#SBATCH --error=logs/e9gdiag-%j.err
#SBATCH --time=03:00:00
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V="${VENV:-$SCRATCH/venv_e9hf}/bin/python"
OUT="$SCRATCH/e9/diag_gemma_plain"; rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== PART 1: what does the wrapper actually discard? ==="
"$V" - <<'PY'
import torch, json
from transformers import AutoModelForCausalLM, GenerationConfig
m = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B", dtype=torch.bfloat16,
                                         trust_remote_code=True, device_map="cpu")
loaded, fresh = vars(m.generation_config), vars(GenerationConfig())
print("as-loaded generation_config (vars):")
for k in sorted(loaded):
    print(f"   {k} = {loaded[k]!r}")
lost = {k: loaded[k] for k in loaded
        if k not in ("bos_token_id", "eos_token_id", "pad_token_id")
        and (k not in fresh or fresh[k] != loaded[k])}
print("\nFIELDS THE WRAPPER WOULD DISCARD (differ from a fresh GenerationConfig):")
for k in sorted(lost):
    print(f"   {k} = {lost[k]!r}   (fresh: {fresh.get(k, '<absent>')!r})")
print("\nalso: model.config.sliding_window =", getattr(m.config, "sliding_window", None),
      "| text_config.sliding_window =",
      getattr(getattr(m.config, "text_config", object()), "sliding_window", None))
PY

echo "=== PART 2: PLAIN lm_eval, no wrapper (July: 0.6360879454131918) ==="
"$V" -m lm_eval --model hf \
  --model_args "pretrained=google/gemma-4-12B,dtype=bfloat16,trust_remote_code=True,max_length=4096" \
  --tasks gsm8k --num_fewshot 5 --batch_size 16 \
  --seed 0,1234,1234,1234 --log_samples --output_path "$OUT"

"$V" - "$OUT" <<'PY'
import glob, json, sys
f = sorted(glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True))[-1]
d = json.load(open(f))
v = d["results"]["gsm8k"]["exact_match,flexible-extract"]
JULY, WRAPPED = 0.6360879454131918, 0.5375284306292646
print(f"plain lm_eval : {v:.6f}")
print(f"July (vLLM)   : {JULY:.6f}   delta {100*(v-JULY):+.2f} pp")
print(f"our wrapper   : {WRAPPED:.6f}   delta vs plain {100*(v-WRAPPED):+.2f} pp")
print("VERDICT:", "THE WRAPPER IS THE CAUSE -- find the discarded field above"
      if abs(v - JULY) <= 0.01 else
      "not the wrapper -- HF vs vLLM for this architecture; exclusion is the path")
PY
