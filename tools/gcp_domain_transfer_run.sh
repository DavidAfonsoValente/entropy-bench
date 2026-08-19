#!/bin/bash
# Domain-transfer matrix on a GCP GPU VM: BPB for each model on each corpus, zero-shot and
# after a fixed-budget LoRA adaptation, under the corrected packing.
#
# Results are pushed to GCS after every single cell, so a spot preemption costs at most one
# cell rather than the run.
#
#   bash gcp_domain_transfer_run.sh
set -uo pipefail

# Refuse to run twice on one host. Two concurrent copies previously shared a GPU and
# interleaved into the same log, which cost more time to diagnose than the run itself.
exec 9>/tmp/dt.lock
if ! flock -n 9; then
  echo "another copy of this script is already running (/tmp/dt.lock held); exiting"
  exit 1
fi

GCS=gs://gpu-llm-training-gceval/token_gain
WORK=/opt/dt
export HF_HOME=$WORK/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DT_CACHE=$WORK/dt_cache

# smallest first: cheap cells validate the harness before the expensive ones run
MODELS=(
  "Qwen/Qwen2.5-0.5B|models--Qwen--Qwen2.5-0.5B|16"
  "LiquidAI/LFM2.5-1.2B-Base|models--LiquidAI--LFM2.5-1.2B-Base|16"
  "meta-llama/Llama-3.2-1B|models--meta-llama--Llama-3.2-1B|16"
  "Qwen/Qwen2.5-1.5B|models--Qwen--Qwen2.5-1.5B|16"
  # Qwen-3.5-4B needs a smaller batch than its size suggests: a 248k vocabulary makes the
  # logits tensor (batch x seq x vocab, upcast to fp32 in the loss) about four times larger
  # than a comparable model's, and batch 8 exhausted an 80GB H100 during backward.
  "Qwen/Qwen3.5-4B-Base|models--Qwen--Qwen3.5-4B-Base|4"
  "Qwen/Qwen2.5-7B|models--Qwen--Qwen2.5-7B|8"
  "Qwen/Qwen3.5-9B-Base|models--Qwen--Qwen3.5-9B-Base|4"
  "google/gemma-4-12B|models--google--gemma-4-12B|4"
  "mistralai/Ministral-3-14B-Base-2512|models--mistralai--Ministral-3-14B-Base-2512|4"
  # The two >=31B models need a multi-GPU host: 31.4B and 34.7B in bf16 are 63GB and 69GB
  # of weights before activations. Run them with ONLY set, on a 2-GPU instance, and let
  # device_map="auto" shard.
  "google/gemma-4-31B|models--google--gemma-4-31B|1"
  "Qwen/Qwen3.5-35B-A3B-Base|models--Qwen--Qwen3.5-35B-A3B-Base|1"
)
CORPORA=(
  "news|google_news_from_2026-06-08_to_2026-06-08_cleaned.jsonl"
  "reddit|reddit_2026-08.jsonl"
  "hackernews|hackernews_2026-08.jsonl"
)

STEPS=${STEPS:-250}
EVAL_BLOCKS=${EVAL_BLOCKS:-2500}
# Cap documents read per corpus. Keeps tokenization time and disk bounded, and equalises
# the amount of data each domain contributes rather than letting Reddit dominate.
MAX_DOCS=${MAX_DOCS:-40000}

sudo mkdir -p $WORK && sudo chown -R "$(id -u):$(id -g)" $WORK
mkdir -p "$HF_HOME/hub" $WORK/out $WORK/corpora $WORK/code

if [ ! -f $WORK/.deps ]; then
  echo "[setup] deps"
  sudo python3 -m pip uninstall -y -q torchaudio torchvision 2>&1 | tail -1
  sudo python3 -m pip install -q peft safetensors 2>&1 | tail -1
  sudo python3 -m pip install -q datasets datasketch 2>&1 | tail -1
  python3 -c "import transformers,peft,torch;print('[setup]',transformers.__version__,peft.__version__,torch.__version__)" || exit 1
  touch $WORK/.deps
fi

# domain_transfer_eval.py reuses lm_adapt_bench's DataModule/compute_bpb, so the package has
# to be importable from $WORK (the script inserts its own parent's parent on sys.path).
if [ ! -d $WORK/lm_adapt_bench ]; then
  gcloud storage cp $GCS/code/lm_adapt_bench.tgz $WORK/ || exit 1
  tar -xzf $WORK/lm_adapt_bench.tgz -C $WORK || exit 1
  python3 -c "import sys; sys.path.insert(0,'$WORK'); import lm_adapt_bench.data; print('[setup] lm_adapt_bench importable')" || exit 1
fi

gcloud storage rsync -r $GCS/code $WORK/code 2>&1 | tail -1
for c in "${CORPORA[@]}"; do
  f=${c##*|}
  [ -f "$WORK/corpora/$f" ] || gcloud storage cp "$GCS/corpora/$f" "$WORK/corpora/" || exit 1
done

# ONLY is an extended regex matched against the model id, so one instance can take the
# cheap models while another takes the expensive ones without them colliding.
ONLY=${ONLY:-.}

for m in "${MODELS[@]}"; do
  mid=${m%%|*}; rest=${m#*|}; hub=${rest%%|*}; bs=${rest##*|}
  echo "$mid" | grep -qE "$ONLY" || continue
  if [ ! -d "$HF_HOME/hub/$hub/snapshots" ]; then
    echo "[fetch] $hub"
    gcloud storage rsync -r "$GCS/hub/$hub" "$HF_HOME/hub/$hub" 2>&1 | tail -1 || continue
    [ -f "$WORK/hub_manifest.json" ] || gcloud storage cp "$GCS/hub_manifest_all.json" "$WORK/hub_manifest.json" || exit 1
    python3 "$WORK/code/rebuild_hf_cache.py" --manifest "$WORK/hub_manifest.json" \
            --hub "$HF_HOME/hub" --only "$hub" || continue
  fi
  for c in "${CORPORA[@]}"; do
    tag=${c%%|*}; f=${c##*|}
    slug=$(echo "$mid" | tr '/.' '__')
    out="$WORK/out/${tag}__${slug}.json"
    if gcloud storage ls "$GCS/dt_out/${tag}__${slug}.json" >/dev/null 2>&1; then
      echo "[skip] $tag / $mid already done"; continue
    fi
    echo "=============================================================="
    echo "[$(date -u +%H:%M:%S)] $tag / $mid (batch $bs, $STEPS steps)"
    echo "=============================================================="
    python3 "$WORK/code/domain_transfer_eval.py" \
        --model-id "$mid" --dataset "$WORK/corpora/$f" --out "$out" \
        --eval-blocks "$EVAL_BLOCKS" --eval-batch-size "$bs" \
        --train-batch-size "$bs" --grad-accum $((32 / bs)) \
        --max-train-steps "$STEPS" --max-samples "$MAX_DOCS"
    rc=$?
    echo "[$(date -u +%H:%M:%S)] $tag / $mid exit=$rc"
    [ $rc -eq 0 ] && gcloud storage cp "$out" "$GCS/dt_out/" && echo "[ok] pushed ${tag}__${slug}.json"
    rm -rf "$WORK/dt_cache"      # tokenized-split cache is per model+corpus and large
  done
  rm -rf "$HF_HOME/hub/$hub"     # keep the boot disk from filling
done

echo "[$(date -u +%H:%M:%S)] MATRIX COMPLETE"
