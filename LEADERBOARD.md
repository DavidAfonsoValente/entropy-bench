# Base Language Model Leaderboard

Models are ranked by **adapted Bits Per Byte (BPB)** on clean target-domain text. Lower is better. BPB supplies the common unit; the complete protocol also requires a recent corpus, contamination filtering, and a controlled adaptation procedure.

## Primary news benchmark

- **Benchmark ID:** `primary-news-2026-06-08-fixed-lora-v1`
- **Corpus:** 119,054 Google News articles captured 8 June 2026
- **Split:** 80/10/10, seed 42, with zero intersection
- **Context:** independent, non-overlapping 512-token blocks with corpus-wide aggregation
- **Adaptation:** fixed LoRA for every candidate (r=16, alpha=32, dropout=0.05, learning rate 1e-4, effective batch 32, 250 steps)
- **Evaluation:** injected document-start marker targets excluded from loss and scored-token count
- **Contamination:** globally unified risk-screened held-out set
- **Evidence:** [benchmark manifest](benchmarks/primary-news-2026-06-08-fixed-lora-v1.json), [machine-readable leaderboard](results/leaderboard.json), per-model files in `results/domain_transfer/`, and [paper](paper_sota.pdf)

<!-- leaderboard:start -->

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

<!-- leaderboard:end -->

Adapted BPB is the selection score **for a model you intend to fine-tune on domain text**. If you
will ship the base model as released, rank by the zero-shot column instead: it costs one forward
pass and, on this cohort, it selects the GSM8K and MMLU-Pro leader outright while the adapted column
selects the HellaSwag leader. Neither is "the" ranking — see
[`results/alignment_matrix.json`](results/alignment_matrix.json) and the paper's Table 6 for what
each one tracks.

The reduction answers a third question: how much did the model's predictive distribution need to
move toward this corpus? Do not rank by it.

## Add a model

If you do not have authorized access to the controlled corpus, open a **Model evaluation request** issue with the exact model and tokenizer revisions. The maintainers will run the candidate against the same snapshot.

If you already have an authorized result, prepare a contribution directly from the pipeline output:

```bash
entropy-leaderboard prepare-submission \
  path/to/result_model.json \
  --output submissions/model.json \
  --model-name Model-Name \
  --model-revision <commit-sha> \
  --tokenizer-revision <commit-sha> \
  --parameters 7B

entropy-leaderboard validate-submission submissions/model.json
```

Submit the JSON and audit summary in a pull request. The canonical board is [results/leaderboard.json](results/leaderboard.json); CI checks its schema, arithmetic, ordering, and rendered Markdown.

Scores produced with another corpus, split, contamination policy, context length, or adaptation budget must use a different benchmark ID and a separate board. This prevents incomparable runs from being mixed into one ranking.
