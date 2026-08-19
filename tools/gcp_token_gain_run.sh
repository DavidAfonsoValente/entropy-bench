#!/bin/bash
# Runs the token-level gain decomposition on a GCP GPU VM.
#
# Expects a Deep Learning VM (PyTorch/CUDA preinstalled). Pulls the staged blocks,
# adapters and model weights from GCS, runs one model at a time, pushes results back.
#
#   bash gcp_token_gain_run.sh <model_slug> [<model_slug> ...]
#
# Slugs must match the staged directory names, e.g. LiquidAI_LFM2_5-1_2B-Base.
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
    LiquidAI_LFM2_5-1_2B-Base) echo "LiquidAI/LFM2.5-1.2B-Base" ;;
    Qwen_Qwen2_5-1_5B)         echo "Qwen/Qwen2.5-1.5B" ;;
    Qwen_Qwen2_5-7B)           echo "Qwen/Qwen2.5-7B" ;;
    google_gemma-4-12B)        echo "google/gemma-4-12B" ;;
    *) echo "UNKNOWN" ;;
  esac
}
slug_to_hub () {
  local id; id=$(slug_to_id "$1")
  echo "models--${id%%/*}--${id##*/}"
}
# batch size per model -- keep activations modest on the 12B
slug_to_bs () {
  case "$1" in
    google_gemma-4-12B) echo 4 ;;
    Qwen_Qwen2_5-7B)    echo 8 ;;
    *)                  echo 16 ;;
  esac
}

sudo mkdir -p $WORK && sudo chown -R "$(id -u):$(id -g)" $WORK
mkdir -p "$HF_HOME/hub" $WORK/out $WORK/stage

if [ ! -f $WORK/.deps_done ]; then
  echo "[setup] deps"
  # The DLVM image ships torch/transformers already. Do NOT --upgrade: that pulls a
  # torchaudio built against a different torch and every `import transformers` then dies
  # on `undefined symbol: torch_library_impl`. We only need peft, and neither
  # torchaudio nor torchvision is used here.
  python3 -m pip uninstall -y -q torchaudio torchvision 2>&1 | tail -2
  python3 -m pip install -q peft safetensors 2>&1 | tail -2
  python3 -c "import transformers, peft, torch; print('[setup] transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)" || exit 1
  touch $WORK/.deps_done
fi

if [ ! -f $WORK/token_level_gain.py ]; then
  gcloud storage cp $GCS/code/token_level_gain.py $WORK/ || exit 1
fi

for slug in "$@"; do
  mid=$(slug_to_id "$slug"); hub=$(slug_to_hub "$slug"); bs=$(slug_to_bs "$slug")
  if [ "$mid" = "UNKNOWN" ]; then echo "[skip] unknown slug $slug"; continue; fi
  echo "=============================================================="
  echo "[$(date -u +%H:%M:%S)] $slug -> $mid (batch $bs)"
  echo "=============================================================="

  if [ ! -d "$HF_HOME/hub/$hub/snapshots" ]; then
    echo "[fetch] weights $hub"
    gcloud storage rsync -r "$GCS/hub/$hub" "$HF_HOME/hub/$hub" 2>&1 | tail -1 || exit 1
    # rsync brings blobs/ but drops the snapshots/ symlinks -- rebuild them
    [ -f "$WORK/hub_manifest.json" ] || gcloud storage cp "$GCS/hub_manifest.json" "$WORK/" || exit 1
    [ -f "$WORK/rebuild_hf_cache.py" ] || gcloud storage cp "$GCS/code/rebuild_hf_cache.py" "$WORK/" || exit 1
    python3 "$WORK/rebuild_hf_cache.py" --manifest "$WORK/hub_manifest.json" \
            --hub "$HF_HOME/hub" --only "$hub" || exit 1
  fi
  [ -f "$WORK/stage/$slug.blocks.pt" ] || \
    gcloud storage cp "$GCS/stage/$slug.blocks.pt" "$WORK/stage/" || exit 1
  [ -d "$WORK/stage/adapters/$slug" ] || \
    gcloud storage rsync -r "$GCS/stage/adapters/$slug" "$WORK/stage/adapters/$slug" 2>&1 | tail -1

  python3 $WORK/token_level_gain.py \
      --model-id "$mid" \
      --adapter "$WORK/stage/adapters/$slug" \
      --cache "$WORK/stage/$slug.blocks.pt" \
      --out "$WORK/out/$slug.json" \
      --split test --max-blocks 4000 --batch-size "$bs" --tail 16
  rc=$?
  echo "[$(date -u +%H:%M:%S)] $slug exit=$rc"
  if [ $rc -eq 0 ]; then
    gcloud storage cp "$WORK/out/$slug.json" "$GCS/out/" && echo "[ok] pushed $slug.json"
  fi
  # free the weights again -- the 12B is 23GB and boot disk is finite
  rm -rf "$HF_HOME/hub/$hub"
done

echo "[$(date -u +%H:%M:%S)] ALL DONE"
