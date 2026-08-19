# Base-Model Static Benchmarks (MMLU-Pro · HellaSwag · GSM8K)

Reproducible evaluation of the paper's base language models on three mainstream
static benchmarks, using [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
with the vLLM backend. This is the exact configuration behind the numbers reported
in the paper.

## Models (all BASE checkpoints)

| Label | HF ID | Notes |
|---|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B-Base` | |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B-Base` | |
| Qwen3.5-35B-MoE | `Qwen/Qwen3.5-35B-A3B-Base` | MoE (3B active); needs 2 GPUs |
| gemma-4-12B | `google/gemma-4-12B` | base = no suffix (`-it` is instruct) |
| gemma-4-31B | `google/gemma-4-31B` | needs 2 GPUs |
| Ministral-3-14B | `mistralai/Ministral-3-14B-Base-2512` | |
| LFM2.5-1.2B | `LiquidAI/LFM2.5-1.2B-Base` | Liquid architecture |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | **gated** — needs an approved HF token |

## Stack

| Component | Value |
|---|---|
| Eval library | EleutherAI **lm-evaluation-harness** `0.4.12` |
| Inference engine | **vLLM** `0.25.1` (`lm_eval --model vllm`) |
| Framework | PyTorch 2.11 (cu13.0), Transformers `5.14.1` |
| Hardware | NVIDIA **A100-80GB**: 1 GPU for models ≤14B; **2 GPUs (`tensor_parallel_size=2`)** for gemma-4-31B and Qwen3.5-35B-MoE |
| Precision | `bfloat16` |
| Decoding (generative tasks) | greedy (`do_sample=false`, `temperature=0`) |
| BOS handling | each model's **native tokenizer default** (`add_bos_token` unset) → Gemma/Llama/Mistral prepend BOS, Qwen does not |
| Determinism | fixed harness seed (1234); the MMLU-Pro subset is seeded (below) |

vLLM was cross-checked against the HF backend (`--model hf`) on a smoke set; scores agreed within stderr.

## Per-benchmark settings

| Benchmark | lm-eval task | Shots | Type | Metric | Eval set |
|---|---|---|---|---|---|
| **MMLU-Pro** | `mmlu_pro` → custom `mmlu_pro_1k` | **5**, CoT | `generate_until`, greedy, `max_gen_toks=2048` | **exact_match** via `custom-extract` regex (`answer is (X)`) | **1,000** random questions, **seed 42**, stratified across the 14 subjects |
| **HellaSwag** | `hellaswag` | **10** | `multiple_choice` (loglikelihood over 4 endings) | **acc_norm** (length-normalized) | full validation split (**10,042**) |
| **GSM8K** | `gsm8k` | **5**, CoT | `generate_until`, greedy | **exact_match**; both `strict-match` (`#### N`) and `flexible-extract` (last number) recorded — **flexible-extract** reported | full test split (**1,319**) |
| **Macro** | — | — | — | arithmetic mean of the three | — |

Notes:
- **MMLU-Pro** uses a fixed 1,000-question subset (built by `build_mmlu_pro_1k.py`) because it is generation-based and the full ~12k set is expensive; the *same* 1,000 questions are used for every model. HellaSwag and GSM8K use their full standard splits (loglikelihood / short generation, so cheap).
- **GSM8K parsing:** for base models `flexible-extract` is the fair headline (they reason correctly but do not always emit the `#### N` format); `strict-match` is the conservative, leaderboard-comparable number. In our runs the two agreed within ~0.01 (clean format adherence).
- **MMLU-Pro** generations terminate primarily on the `"Question:"` stop string (base models continue the few-shot pattern rather than emitting EOS), or the 2048-token cap.
- **BOS** is left to each tokenizer's default — model-appropriate and identical across benchmarks; a guard prevents double-BOS.

## Setup

Tested on a GCP Deep Learning VM (`pytorch-2-9-cu129`, Ubuntu 22.04, Python 3.10) with A100-80GB GPUs.

```bash
bash eval/setup_env.sh          # isolated venv from eval/requirements.txt (+ torchaudio/ninja fixes) + builds mmlu_pro_1k
```

Pinned dependencies live in `eval/requirements.txt` (installed into `~/vllm_env`),
kept separate from the repo's root `requirements.txt`, which targets the LoRA
training pipeline and would conflict with vLLM's pinned torch.

For the gated Llama-3.2-1B, place a token whose account has an **approved** Llama license at `~/.cache/huggingface/token`.

## Running

```bash
# All models, per benchmark (single GPU handles <=14B)
python eval/run_benchmark.py --benchmark hellaswag   --models all
python eval/run_benchmark.py --benchmark mmlu_pro_1k --models all
python eval/run_benchmark.py --benchmark gsm8k       --models all

# The 31B / 35B-MoE need a 2-GPU host (tensor parallelism):
python eval/run_benchmark.py --benchmark mmlu_pro_1k --models gemma-4-31B Qwen3.5-35B-MoE --tp 2
```

Per-model results (aggregate `results_*.json` + per-example `samples_*.jsonl` from
`--log_samples`) are written under `results/<benchmark>/<model>/`, and summarized in
`results/<benchmark>_table.json`. Runs are resumable — re-invoking skips models
already scored.

The committed static scores used by the paper are available in
`results/combined_bpb_vs_static.json`; the corresponding paper figures are under
`figures/`.
