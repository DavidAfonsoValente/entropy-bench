# Base Language Model Leaderboard

Models are ranked by **bits per byte (BPB) on whole documents** of fresh, decontaminated text from
the target domain: every test document is read from its tokenizer's own beginning-of-document token,
on the same bytes for every model, with no adaptation and no labels. Lower is better. On the paper's
news fine-tuning task this ranking places every one of fifteen models within one rank of where its
fine-tuned system finishes. Each track is its own benchmark, and scores are never mixed across them.

| Track | Corpus | Who can run it |
|---|---|---|
| `arxiv-math-2026-08-documents-v1` | 3,207 arXiv `math.*` title+abstract documents, 9 Jun–20 Aug 2026 (CC0 metadata) | **anyone** — the corpus is in this repository |
| `primary-news-2026-06-08-documents-v1` | 119,054 Google News articles, 8 Jun 2026 | the maintainers — the text cannot be redistributed |

Both document tracks score the leading test documents of the 80/10/10 split (seed 42) up to 6 MB,
with a 2,048-token context and a 1,024-token stride. The older fixed-LoRA tracks, which adapt every
model and score 512-token blocks, are kept below for comparison with the paper's appendix.

## Mathematics track (public)

- **Benchmark ID:** `arxiv-math-2026-08-documents-v1`
- **Corpus:** [`data/corpora/arxiv-math-2026.jsonl.gz`](data/corpora/arxiv-math-2026.jsonl.gz), identified by the [manifest](benchmarks/arxiv-math-2026-08-documents-v1.json) (SHA-256 of the decompressed file `78e19669…`)
- **Contamination:** every document postdates every model in the cohort; an n-gram screen also removes overlap with the GSM8K test set
- **Evidence:** [machine-readable board](results/leaderboard_math_documents.json), per-model files `results/doc_protocol_cells/math_doc__*.json`

<!-- leaderboard:start:arxiv-math-2026-08-documents-v1 -->

| Rank | Model | Parameters | Document BPB |
|---:|---|---:|---:|
| 1 | Qwen-3.5-35B-MoE | 35B total | **0.584** |
| 2 | Gemma-4-31B | 31B | **0.610** |
| 3 | Qwen-3.5-9B | 9B | **0.610** |
| 4 | Qwen-2.5-14B | 14.8B | **0.619** |
| 5 | Ministral-3-14B | 14B | **0.621** |
| 6 | Falcon3-10B | 10.4B | **0.633** |
| 7 | Qwen-2.5-7B | 7B | **0.637** |
| 8 | Qwen-3.5-4B | 4B | **0.640** |
| 9 | Mistral-Nemo-12B | 12.2B | **0.642** |
| 10 | Falcon3-7B | 7.5B | **0.650** |
| 11 | Qwen-2.5-3B | 3.1B | **0.667** |
| 12 | Qwen-2.5-1.5B | 1.5B | **0.697** |
| 13 | Gemma-4-12B | 12B | **0.704** |
| 14 | Llama-3.2-1B | 1.2B | **0.744** |
| 15 | SmolLM2-1.7B | 1.7B | **0.751** |
| 16 | Qwen-2.5-0.5B | 0.5B | **0.776** |
| 17 | LiquidAI-LFM2.5 | 1.2B | **0.842** |

<!-- leaderboard:end:arxiv-math-2026-08-documents-v1 -->

## News track (maintainer-run)

- **Benchmark ID:** `primary-news-2026-06-08-documents-v1`
- **Contamination:** three membership screens per model; the union of every flagged sequence is removed for all
- **Evidence:** [manifest](benchmarks/primary-news-2026-06-08-documents-v1.json), [machine-readable board](results/leaderboard_news_documents.json), per-model files `results/doc_protocol_cells/news_doc__*.json`, and [paper](paper_sota.pdf)

<!-- leaderboard:start:primary-news-2026-06-08-documents-v1 -->

| Rank | Model | Parameters | Document BPB |
|---:|---|---:|---:|
| 1 | Gemma-4-31B | 31B | **0.570** |
| 2 | Mistral-Nemo-12B | 12.2B | **0.619** |
| 3 | Qwen-3.5-35B-MoE | 35B total | **0.626** |
| 4 | Gemma-4-12B | 12B | **0.637** |
| 5 | Qwen-2.5-14B | 14.8B | **0.638** |
| 6 | Ministral-3-14B | 14B | **0.641** |
| 7 | Qwen-3.5-9B | 9B | **0.657** |
| 8 | Qwen-2.5-7B | 7B | **0.678** |
| 9 | Falcon3-10B | 10.4B | **0.690** |
| 10 | Qwen-3.5-4B | 4B | **0.696** |
| 11 | Qwen-2.5-3B | 3.1B | **0.712** |
| 12 | Falcon3-7B | 7.5B | **0.717** |
| 13 | Llama-3.2-1B | 1.2B | **0.733** |
| 14 | SmolLM2-1.7B | 1.7B | **0.738** |
| 15 | Qwen-2.5-1.5B | 1.5B | **0.751** |
| 16 | LiquidAI-LFM2.5 | 1.2B | **0.825** |
| 17 | Qwen-2.5-0.5B | 0.5B | **0.851** |

<!-- leaderboard:end:primary-news-2026-06-08-documents-v1 -->

## Fixed-LoRA tracks (block reading after adaptation)

Every model adapted with one LoRA recipe (r=16, alpha=32, dropout 0.05, learning rate 1e-4,
effective batch 32, 250 update steps), then scored on 512-token blocks cut from the concatenated
test text, injected document-start targets masked. Adaptation repairs most of what the block cut
does to models that expect a document start; the paper reports these readings in its appendix.

### Mathematics, fixed LoRA


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

### News, fixed LoRA

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

Here the zero-shot column is the block reading before adaptation and the reduction is how much
adaptation lowered it; do not rank by the reduction.

## Submit a model

**Mathematics track — run it yourself.** One forward pass per model, on one GPU:

```bash
zcat data/corpora/arxiv-math-2026.jsonl.gz > arxiv-math-2026.jsonl
entropy-dataset verify arxiv-math-2026.jsonl --manifest benchmarks/arxiv-math-2026-08-documents-v1.json

python tools/domain_transfer_eval.py \
  --model-id ORG/MODEL --dataset arxiv-math-2026.jsonl --out results/my-run/math_doc__MODEL.json \
  --zero-shot-only --eval-blocks 0 --max-samples 40000 --doc-context 2048

entropy-leaderboard prepare-submission results/my-run/math_doc__MODEL.json \
  --benchmark-id arxiv-math-2026-08-documents-v1 --output submissions/MODEL.json \
  --model-name Model-Name --model-revision <commit-sha> --tokenizer-revision <commit-sha> --parameters 7B
entropy-leaderboard validate-submission submissions/MODEL.json
```

`prepare-submission` refuses a result whose corpus hash, document context, stride or scored bytes
(or, on a fixed-LoRA track, adaptation recipe) differs from the track's manifest. Open a pull request with the submission JSON and the
result file; maintainers add it with `entropy-leaderboard add submissions/MODEL.json`, and CI checks
every board's schema, arithmetic, ordering and rendered Markdown.

**News track — request it.** Open a **Model evaluation request** issue with the exact model and
tokenizer revisions; the maintainers run the candidate against the same snapshot.

**Your own corpus, or the full method.** Results on another corpus, or with a different adaptation
procedure — including the per-model hyperparameter search trained to convergence
(`lm_adapt_bench.cli --hparams sweep --no-time-limit`) — are a different benchmark ID and a separate
board, so incomparable runs are never mixed into one ranking.
