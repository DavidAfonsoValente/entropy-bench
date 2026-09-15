#!/bin/bash
# Materialise the Ministral-3-14B text tower, then re-run its three E9 cells against it.
# Rationale and evidence: docs/RUN_LEDGER.md, "Root cause: the checkpoint and the class use
# disjoint key spaces". CPU-only for the conversion; the eval cells are submitted separately.
#SBATCH --job-name=e9_minfix
#SBATCH --output=logs/e9minfix-%j.out
#SBATCH --error=logs/e9minfix-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
V="${VENV:-$SCRATCH/venv_e9hf}/bin/python"
DEST="$SCRATCH/e9/ministral3_text"

"$V" tools/materialise_prefixed_text_model.py \
  mistralai/Ministral-3-14B-Base-2512 "$DEST" --prefix language_model

# Load it for real and check it is coherent before spending eval time. A random model has
# near-uniform next-token entropy; a trained one is far below that on ordinary English.
"$V" - "$DEST" <<'PY'
import sys, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).cuda().eval()
ids = tok("The capital of France is Paris. The capital of Italy is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = m(**ids)
lp = torch.log_softmax(out.logits[0, -1].float(), -1)
ent = float(-(lp.exp() * lp).sum())
top = tok.decode(int(lp.argmax()))
uniform = math.log(out.logits.shape[-1])
print(f"vocab {out.logits.shape[-1]}  uniform-entropy {uniform:.2f}  observed {ent:.2f}")
print(f"top continuation: {top!r}")
assert ent < uniform * 0.5, f"entropy {ent:.2f} is near-uniform ({uniform:.2f}) -- weights look RANDOM"
print("COHERENCE OK -- this is a trained model, not a randomly initialised one")
PY
echo "[minfix] done -> $DEST"
