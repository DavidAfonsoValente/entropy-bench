# Base Language Model Leaderboard

Models are ranked by **adapted Bits Per Byte (BPB)** on fresh, decontaminated target-domain text:
each model is adapted to the domain by the same procedure, then scored on held-out text. Lower is
better. There are two tracks; each is its own benchmark, and scores are never mixed across them.

| Track | Corpus | Who can run it |
|---|---|---|
| `arxiv-math-2026-08-fixed-lora-v1` | 3,207 arXiv `math.*` title+abstract documents, 9 Jun–20 Aug 2026 (CC0 metadata) | **anyone** — the corpus is in this repository |
| `primary-news-2026-06-08-fixed-lora-v1` | 119,054 Google News articles, 8 Jun 2026 | the maintainers — the text cannot be redistributed |

Both tracks use the reported fixed recipe: LoRA r=16, alpha=32, dropout 0.05, learning rate 1e-4,
effective batch 32, 250 update steps, 512-token blocks, injected document-start targets masked,
80/10/10 split with seed 42.

## Mathematics track (public)

- **Benchmark ID:** `arxiv-math-2026-08-fixed-lora-v1`
- **Corpus:** [`data/corpora/arxiv-math-2026.jsonl.gz`](data/corpora/arxiv-math-2026.jsonl.gz), identified by the [manifest](benchmarks/arxiv-math-2026-08-fixed-lora-v1.json) (SHA-256 of the decompressed file `78e19669…`)
- **Contamination:** every document postdates every model in the cohort; an n-gram screen also removes overlap with the GSM8K test set
- **Evidence:** [machine-readable board](results/leaderboard_math.json), per-model files `results/domain_transfer/math__*.json`

<!-- leaderboard:start:arxiv-math-2026-08-fixed-lora-v1 -->

| Rank | Model | Parameters | Zero-shot BPB | Adapted BPB | Reduction |
|---:|---|---:|---:|---:|---:|
| 1 | Qwen-3.5-35B-MoE | 35B total | 0.636 | **0.589** | 7.4% |
| 2 | Qwen-3.5-9B | 9B | 0.664 | **0.614** | 7.5% |
| 3 | Ministral-3-14B | 14B | 0.672 | **0.622** | 7.4% |
| 4 | Gemma-4-31B | 31B | 0.798 | **0.631** | 20.9% |
| 5 | Qwen-2.5-7B | 7B | 0.688 | **0.634** | 7.9% |
| 6 | Qwen-3.5-4B | 4B | 0.694 | **0.643** | 7.3% |
| 7 | Gemma-4-12B | 12B | 0.933 | **0.666** | 28.6% |
| 8 | Qwen-2.5-1.5B | 1.5B | 0.748 | **0.701** | 6.3% |
| 9 | Llama-3.2-1B | 1.2B | 0.798 | **0.750** | 6.0% |
| 10 | Qwen-2.5-0.5B | 0.5B | 0.829 | **0.780** | 5.9% |
| 11 | LiquidAI-LFM2.5 | 1.2B | 1.399 | **0.817** | 41.6% |

<!-- leaderboard:end:arxiv-math-2026-08-fixed-lora-v1 -->

## News track (maintainer-run)

- **Benchmark ID:** `primary-news-2026-06-08-fixed-lora-v1`
- **Contamination:** three membership screens per model; the union of every flagged sequence is removed for all
- **Evidence:** [manifest](benchmarks/primary-news-2026-06-08-fixed-lora-v1.json), [machine-readable board](results/leaderboard.json), per-model files `results/domain_transfer/news__*.json`, and [paper](paper_sota.pdf)

<!-- leaderboard:start:primary-news-2026-06-08-fixed-lora-v1 -->

| Rank | Model | Parameters | Zero-shot BPB | Adapted BPB | Reduction |
|---:|---|---:|---:|---:|---:|
| 1 | Gemma-4-31B | 31B | 0.752 | **0.624** | 17.0% |
| 2 | Gemma-4-12B | 12B | 0.866 | **0.653** | 24.5% |
| 3 | Qwen-3.5-35B-MoE | 35B total | 0.707 | **0.678** | 4.1% |
| 4 | Ministral-3-14B | 14B | 0.718 | **0.679** | 5.4% |
| 5 | Qwen-3.5-9B | 9B | 0.740 | **0.708** | 4.4% |
| 6 | Qwen-2.5-7B | 7B | 0.758 | **0.717** | 5.4% |
| 7 | Qwen-3.5-4B | 4B | 0.782 | **0.749** | 4.2% |
| 8 | Llama-3.2-1B | 1.2B | 0.811 | **0.771** | 4.9% |
| 9 | Qwen-2.5-1.5B | 1.5B | 0.833 | **0.797** | 4.3% |
| 10 | LiquidAI-LFM2.5 | 1.2B | 1.340 | **0.852** | 36.4% |
| 11 | Qwen-2.5-0.5B | 0.5B | 0.935 | **0.901** | 3.6% |

<!-- leaderboard:end:primary-news-2026-06-08-fixed-lora-v1 -->

The zero-shot column is the model as released: how much it already resembles the corpus. The
adapted column is the ranking. The reduction shows how far the model's distribution had to move
toward the corpus; do not rank by it.

## Submit a model

**Mathematics track — run it yourself.** On one GPU:

```bash
zcat data/corpora/arxiv-math-2026.jsonl.gz > arxiv-math-2026.jsonl
entropy-dataset verify arxiv-math-2026.jsonl --manifest benchmarks/arxiv-math-2026-08-fixed-lora-v1.json

python tools/domain_transfer_eval.py \
  --model-id ORG/MODEL --dataset arxiv-math-2026.jsonl --out results/my-run/math__MODEL.json \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --learning-rate 1e-4 \
  --train-batch-size 4 --grad-accum 8 --max-train-steps 250

entropy-leaderboard prepare-submission results/my-run/math__MODEL.json \
  --benchmark-id arxiv-math-2026-08-fixed-lora-v1 --output submissions/MODEL.json \
  --model-name Model-Name --model-revision <commit-sha> --tokenizer-revision <commit-sha> --parameters 7B
entropy-leaderboard validate-submission submissions/MODEL.json
```

`prepare-submission` refuses a result whose corpus hash, context length, masking or adaptation
recipe differs from the track's manifest. Open a pull request with the submission JSON and the
result file; maintainers add it with `entropy-leaderboard add submissions/MODEL.json`, and CI checks
every board's schema, arithmetic, ordering and rendered Markdown.

**News track — request it.** Open a **Model evaluation request** issue with the exact model and
tokenizer revisions; the maintainers run the candidate against the same snapshot.

**Your own corpus, or the full method.** Results on another corpus, or with a different adaptation
procedure — including the per-model hyperparameter search trained to convergence
(`lm_adapt_bench.cli --hparams sweep --no-time-limit`) — are a different benchmark ID and a separate
board, so incomparable runs are never mixed into one ranking.
