# Base Language Model Leaderboard

Models are ranked by **adapted Bits Per Byte (BPB)** on clean target-domain text. Lower is better. BPB supplies the common unit; the complete protocol also requires a recent corpus, contamination filtering, and a controlled adaptation procedure.

## Primary news benchmark

- **Benchmark ID:** `primary-news-2026-06-08`
- **Corpus:** 119,054 Google News articles captured 8 June 2026
- **Split:** 80/10/10, seed 42, with zero intersection
- **Context:** independent, non-overlapping 512-token blocks with corpus-wide aggregation
- **Adaptation:** LoRA with the same candidate-specific multi-fidelity search procedure
- **Contamination:** globally unified audited test set
- **Evidence:** [benchmark manifest](benchmarks/primary-news-2026-06-08.json), [full run summary](reports/11-model-run/summary.json), and [paper](paper_sota.pdf)

<!-- leaderboard:start -->

| Rank | Model | Parameters | Zero-shot BPB | Adapted BPB | Reduction |
|---:|---|---:|---:|---:|---:|
| 1 | Gemma-4-31B | 31B | 0.753 | **0.583** | 22.6% |
| 2 | Gemma-4-12B | 12B | 0.867 | **0.618** | 28.7% |
| 3 | Ministral-3-14B | 14B | 0.723 | **0.651** | 9.9% |
| 4 | Qwen-3.5-35B-MoE | 35B total | 0.700 | **0.663** | 5.4% |
| 5 | Qwen-3.5-9B | 9B | 0.733 | **0.679** | 7.4% |
| 6 | Qwen-2.5-7B | 7B | 0.753 | **0.728** | 3.3% |
| 7 | Qwen-3.5-4B | 4B | 0.775 | **0.757** | 2.3% |
| 8 | Llama-3.2-1B | 1.2B | 0.814 | **0.773** | 5.1% |
| 9 | Qwen-2.5-1.5B | 1.5B | 0.829 | **0.802** | 3.2% |
| 10 | LiquidAI-LFM2.5 | 1.2B | 1.346 | **0.854** | 36.5% |
| 11 | Qwen-2.5-0.5B | 0.5B | 0.931 | **0.921** | 1.0% |

<!-- leaderboard:end -->

Adapted BPB is the selection score. The reduction answers a different question: how much did the model's predictive distribution need to move toward this corpus?

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
