#!/bin/bash
# Set up an isolated vLLM + lm-evaluation-harness environment for the base-model
# benchmarks. Tested on a GCP Deep Learning VM image (pytorch-2-9-cu129,
# Ubuntu 22.04, Python 3.10) with A100-80GB GPUs.
#
# Usage:  bash setup_env.sh
set -euo pipefail

# 1. System deps
#   - python3.x-venv : DL images ship without ensurepip
#   - ninja-build    : vLLM's engine JIT-compiles kernels and needs `ninja` on PATH
#   - python3-dev    : needed by Triton to build its Python C-extension
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv ninja-build python3-dev

# 2. Isolated venv (keeps the base image's torch untouched)
python3 -m venv ~/vllm_env
~/vllm_env/bin/pip install --upgrade pip -q
# Install the eval-critical packages first and on their own. A single unsatisfiable pin anywhere
# in requirements.txt makes pip fail atomically and leaves the venv without vLLM at all, which
# previously turned a plotting-only version conflict into "no benchmark can run".
~/vllm_env/bin/pip install -q vllm==0.25.1 lm-eval==0.4.12 "transformers>=5.0" hf_transfer
~/vllm_env/bin/pip install -q -r "$(dirname "$0")/requirements.txt" \
  || echo "[warn] optional (figure-generation) deps failed to install; benchmarks are unaffected"

# 3. torchaudio MUST match the torch version vLLM pulled in: transformers>=5
#    hard-imports torchaudio (loss_rnnt/Parakeet), and a mismatched ABI aborts
#    every run with "Could not load _torchaudio.abi3.so".
if ! ~/vllm_env/bin/python -c "import torch,torchaudio,transformers,vllm,lm_eval" 2>/dev/null; then
  TV=$(~/vllm_env/bin/python -c "import torch;print(torch.__version__.split('+')[0])")
  CU=$(~/vllm_env/bin/python -c "import torch;print(torch.version.cuda.replace('.',''))")
  ~/vllm_env/bin/pip install -q "torchaudio==${TV}" --index-url "https://download.pytorch.org/whl/cu${CU}" \
    || ~/vllm_env/bin/pip install -q "torchaudio==${TV}"
fi

~/vllm_env/bin/python - <<'PY'
import torch, torchaudio, transformers, vllm, lm_eval
print("OK | torch", torch.__version__, "| torchaudio", torchaudio.__version__,
      "| transformers", transformers.__version__, "| vllm", vllm.__version__,
      "| lm_eval", lm_eval.__version__, "| gpus", torch.cuda.device_count())
PY

# 4. Build the 1000-random MMLU-Pro task (HellaSwag is a built-in task)
~/vllm_env/bin/python "$(dirname "$0")/build_mmlu_pro_1k.py"

echo "Environment ready. For gated models (Llama-3.2), place a token with an"
echo "approved license at ~/.cache/huggingface/token."
