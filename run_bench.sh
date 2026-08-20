#!/bin/bash
# Master Launcher (Heterogeneous GPU Resource-Aware Discovery)
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 DATASET [MODEL[:GPU_COUNT] ...] [-- EXTRA_FLAGS]" >&2
    echo "Example: $0 /path/to/corpus.jsonl Qwen/Qwen2.5-0.5B:1 -- --contam-check-level strict" >&2
    exit 2
fi
DATASET=$1
shift
if [ ! -e "$DATASET" ]; then
    echo "Dataset not found: $DATASET" >&2
    exit 2
fi

# Separate raw args into models and flags
RAW_ARGS=()
EXTRA_FLAGS=()
while [ $# -gt 0 ]; do
    if [[ $1 == "--" ]]; then
        shift
        EXTRA_FLAGS=("$@")
        break
    elif [[ $1 == --* ]]; then
        EXTRA_FLAGS=("$@")
        break
    else
        RAW_ARGS+=("$1")
        shift
    fi
done

if [ ${#RAW_ARGS[@]} -eq 0 ]; then
    RAW_ARGS=(
        "Qwen/Qwen3.5-35B-A3B-Base:2"
        "Qwen/Qwen3.5-9B-Base:1"
        "Qwen/Qwen3.5-4B-Base:1"
        "google/gemma-4-31B:2"
        "mistralai/Ministral-3-14B-Base-2512:1"
        "google/gemma-4-12B:1"
    )
fi

VENV_DIR="./venv_lm_adapt"

echo "===================================================="
echo "   Entropy Bench: Automated Comparison Pipeline"
echo "===================================================="

# Discover the fallback GPU for models with no explicit mapping
FALLBACK_GPU="a100-40"
IS_LEONARDO=false
if [[ $(hostname) == *"leonardo"* || $(hostname) == *"login"* ]]; then
    IS_LEONARDO=true
    FALLBACK_GPU="gpu"
fi

if [ "$IS_LEONARDO" = "false" ] && command -v sinfo >/dev/null 2>&1; then
    get_node_count() {
        sinfo -o "%G %D" --noheader | grep "$1" | awk '{sum+=$2} END {print sum}' || echo 0
    }
    if [ "$(get_node_count "h100-96")" -gt 0 ]; then
        FALLBACK_GPU="h100-96"
    fi
fi

# Parse models, their explicit GPU mappings, and GPU counts
MODELS=()
GPUS=()
for arg in "${RAW_ARGS[@]}"; do
    if [[ $arg == *:* ]]; then
        MODEL_ID="${arg%%:*}"
        GPU_VAL="${arg##*:}"
    else
        MODEL_ID="$arg"
        GPU_VAL="1" # default to 1 GPU
    fi
    MODELS+=("$MODEL_ID")
    GPUS+=("$GPU_VAL")
done

# Group models by GPU requirements to construct the Slurm heterogeneous sbatch call.
UNIQUE_GPUS=($(echo "${GPUS[@]}" | tr ' ' '\n' | sort -u))

SBATCH_COMPONENTS=()
SORTED_MODELS=()

for gpu in "${UNIQUE_GPUS[@]}"; do
    count=0
    for i in "${!GPUS[@]}"; do
        if [ "${GPUS[$i]}" == "$gpu" ]; then
            SORTED_MODELS+=("${MODELS[$i]}")
            count=$((count+1))
        fi
    done
    
    # Determine the GRES resource string based on environment
    if [ "$IS_LEONARDO" = "true" ]; then
        # On Leonardo, gpu represents a numeric count of custom A100-64GB GPUs
        if [[ $gpu =~ ^[0-9]+$ ]]; then
            GRES_REQ="gpu:${gpu}"
        else
            GRES_REQ="gpu:1"
        fi
        SBATCH_COMPONENTS+=("--nodes=$count --ntasks=$count --gres=${GRES_REQ} -p boost_usr_prod -A AIFAC_S03_029")
    else
        # Standard cluster: if gpu is numeric, map to FALLBACK_GPU
        if [[ $gpu =~ ^[0-9]+$ ]]; then
            SBATCH_COMPONENTS+=("--nodes=$count --ntasks=$count --gres=gpu:${FALLBACK_GPU}:${gpu}")
        else
            SBATCH_COMPONENTS+=("--nodes=$count --ntasks=$count --gres=gpu:${gpu}:1")
        fi
    fi
done

NUM_MODELS=${#SORTED_MODELS[@]}

# Join SBATCH components with Slurm's ":" separator
SBATCH_GRES_STRING=""
for i in "${!SBATCH_COMPONENTS[@]}"; do
    if [ $i -gt 0 ]; then
        SBATCH_GRES_STRING="${SBATCH_GRES_STRING} : "
    fi
    SBATCH_GRES_STRING="${SBATCH_GRES_STRING}${SBATCH_COMPONENTS[$i]}"
done
# Leonardo Supercomputer Detection & Configuration Override
# To prevent heterogeneous srun crashes on Leonardo (which require complex --het-group syntax),
# we request a homogeneous allocation of 2 GPUs per node globally across all nodes.
# This yields 128GB VRAM per model (ideal for 35B/31B models) and schedules instantly.
if [ "$IS_LEONARDO" = "true" ]; then
    # Leonardo Booster nodes have 4x A100-64GB. We request a FULL node (gpu:4 = 256GB VRAM)
    # per model so large models (>=9B, incl. 31B/35B) fit entirely in GPU memory without the
    # slow disk/CPU offload that crippled the previous run, while small models simply use the
    # 1-2 GPUs they need. Heterogeneous srun is avoided (it crashes on Leonardo), so we keep a
    # single homogeneous allocation: one full node per model, NUM_MODELS nodes total.
    SBATCH_GRES_STRING="--nodes=$NUM_MODELS --ntasks=$NUM_MODELS --gres=gpu:4 -p boost_usr_prod -A AIFAC_S03_029"
    FALLBACK_GPU="gpu"
    echo "🦁 Leonardo Booster Supercomputer detected!"
    echo "Partition: 'boost_usr_prod' | Account: 'AIFAC_S03_029' | Homogeneous GRES: gpu:4 (256GB VRAM per node)"

    # Configure HuggingFace to use high-capacity Lustre storage to bypass tight user home quotas
    export HF_HOME="/leonardo_work/AIFAC_S03_029/$(whoami)/.cache/huggingface"
    mkdir -p "$HF_HOME"
    echo "🛡️ Routing HuggingFace Cache to: $HF_HOME"
fi

# 1. Setup environment if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/2] Building environment..."
    export LM_SETUP_PHASE="true"
    if [ "$IS_LEONARDO" = "true" ]; then
        # On Leonardo, compute nodes have no internet access. We must build the venv
        # locally on the login node, then share it over the Lustre filesystem!
        /bin/bash -l slurm/setup_env.sh
    else
        srun --gres=gpu:1 -p boost_usr_prod -A AIFAC_S03_029 --mem=32G --cpus-per-task=4 --time=00:30:00 /bin/bash -l slurm/setup_env.sh
    fi
    unset LM_SETUP_PHASE
else
    echo "[1/2] Environment found. Proceeding to execution."
fi

# Extract HuggingFace token from EXTRA_FLAGS if present and not already in environment
if [ -z "$HF_TOKEN" ]; then
    for i in "${!EXTRA_FLAGS[@]}"; do
        if [ "${EXTRA_FLAGS[$i]}" == "--hf-token" ]; then
            export HF_TOKEN="${EXTRA_FLAGS[$((i+1))]}"
            echo "🔑 Extracted HF_TOKEN from command line arguments for the pre-download phase."
            break
        fi
    done
fi

# Leonardo Offline Cache Pre-download Phase
if [ "$IS_LEONARDO" = "true" ]; then
    echo "⬇️ Starting Leonardo Pre-download Phase on Login Node (with Internet)..."
    source "$VENV_DIR/bin/activate"
    python -c "
import os, sys
import huggingface_hub
from huggingface_hub import snapshot_download

token = os.getenv('HF_TOKEN')
if token:
    print('Logging in to Hugging Face Hub using HF_TOKEN...')
    try:
        huggingface_hub.login(token=token, add_to_git_credential=False)
    except Exception as login_err:
        print(f'Warning: HF login failed: {login_err}')

models = sys.argv[1:]
for m in models:
    print(f'Caching {m} using snapshot_download...')
    try:
        snapshot_download(repo_id=m, repo_type=\"model\", token=token, max_workers=1)
        print(f'✨ Successfully cached {m}!')
    except Exception as e:
        print(f'❌ ERROR: Failed caching {m}: {e}')
        print(f'   Since Leonardo compute nodes are OFFLINE, running {m} WILL FAIL on compute nodes.')
        if 'gated' in str(e).lower() or 'access' in str(e).lower() or 'unauthorized' in str(e).lower() or '401' in str(e) or '403' in str(e):
            print(f'   💡 This is a gated model. Ensure you have accepted the terms on HF and passed a valid HF_TOKEN via --hf-token.')
        elif 'network' in str(e).lower() or 'connection' in str(e).lower():
            print(f'   💡 Network connection error. Check your internet connection or proxy settings on the login node.')
" "${SORTED_MODELS[@]}"
    echo "✨ All models pre-cached on Lustre! Proceeding to execution."
fi

# 2. Launch Parallel Benchmark

echo "[2/2] Launching Parallel Benchmark across $NUM_MODELS nodes..."
mkdir -p logs

echo "Resource allocation details:"
echo "GRES Allocation: $SBATCH_GRES_STRING"

# Submit heterogeneous job to Slurm
sbatch $SBATCH_GRES_STRING slurm/parallel_launcher.sh "$DATASET" --models "${SORTED_MODELS[@]}" "${EXTRA_FLAGS[@]}"

echo "----------------------------------------------------"
echo "Benchmark submitted for:"
for i in "${!SORTED_MODELS[@]}"; do
    echo " $((i+1)). ${SORTED_MODELS[$i]} -> GPU GRES: ${GPUS[$i]}"
done
if [ ${#EXTRA_FLAGS[@]} -gt 0 ]; then
    echo "Extra Flags: ${EXTRA_FLAGS[*]}"
fi
echo "Check progress with: tail -f logs/slurm-*.out"
echo "----------------------------------------------------"
