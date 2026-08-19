#!/bin/bash
# Run the fixed-target context-length evaluation on the existing GCP A100 worker.
#
# Usage on the worker:
#   bash gcp_context_length_run.sh Qwen_Qwen2_5-1_5B google_gemma-4-12B
set -uo pipefail

GCS=gs://gpu-llm-training-gceval/token_gain
WORK=/opt/tokengain
export HF_HOME=$WORK/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

slug_to_id () {
  case "$1" in
    Qwen_Qwen2_5-1_5B)  echo "Qwen/Qwen2.5-1.5B" ;;
    google_gemma-4-12B) echo "google/gemma-4-12B" ;;
    *) echo "UNKNOWN" ;;
  esac
}
slug_to_hub () {
  local id; id=$(slug_to_id "$1")
  echo "models--${id%%/*}--${id##*/}"
}
slug_to_bs () {
  case "$1" in
    google_gemma-4-12B) echo 1 ;;
    *) echo 4 ;;
  esac
}

sudo mkdir -p "$WORK"
sudo chown -R "$(id -u):$(id -g)" "$WORK"
mkdir -p "$HF_HOME/hub" "$WORK/context_length" "$WORK/stage/adapters"

if [ ! -f "$WORK/.deps_done" ]; then
  python3 -m pip uninstall -y -q torchaudio torchvision
  python3 -m pip install -q peft safetensors
  python3 -c "import transformers, peft, torch; print(transformers.__version__, peft.__version__, torch.__version__)" || exit 1
  touch "$WORK/.deps_done"
fi

gcloud storage cp "$GCS/code/context_length_eval.py" "$WORK/" || exit 1
[ -f "$WORK/hub_manifest.json" ] || gcloud storage cp "$GCS/hub_manifest.json" "$WORK/" || exit 1
[ -f "$WORK/rebuild_hf_cache.py" ] || gcloud storage cp "$GCS/code/rebuild_hf_cache.py" "$WORK/" || exit 1

for slug in "$@"; do
  mid=$(slug_to_id "$slug"); hub=$(slug_to_hub "$slug"); bs=$(slug_to_bs "$slug")
  if [ "$mid" = "UNKNOWN" ]; then
    echo "[skip] unknown slug $slug"
    continue
  fi
  echo "=============================================================="
  echo "[$(date -u +%H:%M:%S)] $slug -> $mid"
  echo "=============================================================="

  if [ ! -d "$HF_HOME/hub/$hub/snapshots" ]; then
    echo "[fetch] weights $hub"
    gcloud storage rsync -r "$GCS/hub/$hub" "$HF_HOME/hub/$hub" || exit 1
    python3 "$WORK/rebuild_hf_cache.py" --manifest "$WORK/hub_manifest.json" \
      --hub "$HF_HOME/hub" --only "$hub" || exit 1
  fi
  [ -f "$WORK/stage/$slug.blocks.pt" ] || \
    gcloud storage cp "$GCS/stage/$slug.blocks.pt" "$WORK/stage/" || exit 1
  [ -d "$WORK/stage/adapters/$slug" ] || \
    gcloud storage rsync -r "$GCS/stage/adapters/$slug" "$WORK/stage/adapters/$slug" || exit 1

  python3 "$WORK/context_length_eval.py" \
    --model-id "$mid" \
    --adapter "$WORK/stage/adapters/$slug" \
    --cache "$WORK/stage/$slug.blocks.pt" \
    --out "$WORK/context_length/$slug.json" \
    --contexts 16,32,64,128,256,512,1024,2048 \
    --target-tokens 64 --windows 64 --batch-size "$bs"
  rc=$?
  echo "[$(date -u +%H:%M:%S)] $slug exit=$rc"
  if [ "$rc" -eq 0 ]; then
    gcloud storage cp "$WORK/context_length/$slug.json" "$GCS/context_length/" || exit 1
  else
    exit "$rc"
  fi
  rm -rf "$HF_HOME/hub/$hub"
done

echo "[$(date -u +%H:%M:%S)] ALL DONE"
