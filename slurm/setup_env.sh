#!/bin/bash -l
# Robust Environment Setup for Entropy Bench
set -e

# 1. Module Discovery
if ! declare -f module > /dev/null; then
    for f in /usr/share/modules/init/bash /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh /etc/profile; do
        if [ -f "$f" ]; then source "$f" 2>/dev/null && break; fi
    done
fi

# 2. Load Modules
if declare -f module > /dev/null; then
    module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
    # We use cuda/11.8 or 12.1 for maximum compatibility with SoC A100/H100 drivers
    module load cuda/11.8 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
fi

VENV_DIR="./venv_lm_adapt"
PYTHON_CMD=$(which python3 2>/dev/null || echo "python3")

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Bypass user home quota on Cineca's Leonardo by routing Pip Cache & TMP to work directory
if [[ $(hostname) == *"leonardo"* || $(hostname) == *"login"* || $(hostname) == *"lrdn"* ]]; then
    WORK_DIR="/leonardo_work/AIFAC_S03_029/$(whoami)"
    mkdir -p "$WORK_DIR/.cache/pip" "$WORK_DIR/tmp" "$WORK_DIR/.cache/huggingface"
    export PIP_CACHE_DIR="$WORK_DIR/.cache/pip"
    export TMPDIR="$WORK_DIR/tmp"
    export HF_HOME="$WORK_DIR/.cache/huggingface"
    # Force 100% Offline Mode on Compute Nodes to prevent online calls for adapter configs
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_EVALUATE_OFFLINE=1
    echo "🛡️ Routing Pip Cache and Temp to: $WORK_DIR"
    echo "🛡️ Routing HuggingFace Cache to: $HF_HOME"
else
    export TMPDIR="/tmp/pip-install-$USER"
    mkdir -p "$TMPDIR"
fi

# 3. Optimized Installation
if [ "$LM_SETUP_PHASE" = "true" ] || [ ! -f "$VENV_DIR/updated.tag" ]; then
    echo "--- Installing Hardware-Compatible PyTorch (CUDA 11.8) ---"
    
    pip install --upgrade pip
    # Using cu118 ensures it runs on A100/H100 even with older drivers
    pip install --no-cache-dir torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install --no-cache-dir -r requirements.txt
    pip install --no-cache-dir weasyprint
    
    if [ -d "${TMPDIR:-}" ] && [[ $TMPDIR == *"/tmp/pip-install"* ]]; then
        rm -rf "$TMPDIR"
    fi
    touch "$VENV_DIR/updated.tag"
fi

echo "--- Environment Ready (PyTorch $(python -c 'import torch; print(torch.__version__)')) ---"
