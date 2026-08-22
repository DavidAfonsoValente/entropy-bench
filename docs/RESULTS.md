# Understanding results

Every run writes an auditable directory rather than only printing a score:

```text
results/my-corpus/
├── <model>/result_<model>.json   # final metrics and run configuration
├── contamination/               # findings, quarantine, and clean-split metadata
├── summary.json                  # all models in one machine-readable file
├── report.html                   # report for sharing or inspection
└── report.pdf                    # optional; omit with --no-pdf
```

## The numbers that matter

| Field | Meaning | Use |
|---|---|---|
| `zero_shot_bpb` | Predictive cross-entropy before adaptation, normalized by UTF-8 bytes | Diagnose initial fit to the corpus |
| `adapted_bpb` / `best_bpb` | Held-out BPB after controlled adaptation | Primary model-selection score; lower is better |
| `reduction_pct` | Relative BPB change from zero-shot to adapted | Diagnose how much corpus calibration was needed |

Do not rank models by reduction percentage. A weak starting model can improve greatly and still end
with worse adapted BPB than a strong model.

The current primary board uses one fixed LoRA configuration for every model: rank 16, alpha 32,
dropout 0.05, learning rate `1e-4`, effective batch 32, and 250 update steps. Injected
document-start marker targets are excluded from both the loss numerator and scored-token count.

## What the benchmark comparison shows

Across the 11-model cohort, adapted BPB nearly reproduces the HellaSwag order: Pearson
`r = -0.975`, Spearman rank agreement `rho = 0.982`, and 7 of 11 exact rank matches. MMLU-Pro and
GSM8K are complementary rather than interchangeable with entropy: their rank agreements are
`rho = 0.764` and `0.727`.

The disagreements are informative. Qwen models tend to move upward on MMLU-Pro and GSM8K, while the
two Gemma models move downward, consistent with family-specific knowledge/math training and
chain-of-thought behavior. At small scale the distinction is stark: Qwen-2.5-1.5B reaches 61.1% on
GSM8K, while Llama-3.2-1B reaches 6.5%, despite adjacent BPB ranks. This is not evidence that one
metric is wrong: BPB measures broad next-byte predictive fit after controlled adaptation, whereas
GSM8K additionally depends on mathematical reasoning, prompting, generation, and answer extraction.

Point-estimate leaders should not be over-sold. The 0.53 percentage-point GSM8K gap between
Qwen-3.5-35B-MoE and Qwen-3.5-9B is small relative to test-set sampling uncertainty. The committed
analysis reconstructs nearest integer counts from rounded accuracies and reports Wilson intervals;
it does not capture prompt or run-to-run uncertainty. Reproduce it with:

```bash
python tools/analyze_static_benchmarks.py --check
```

## When two results are comparable

They must share all of the following:

- source corpus and exact dataset revision;
- train/validation/test split and seed;
- context length and token-label masking policy;
- contamination audit and cleaning policy;
- adaptation method, search procedure, and budget; and
- BPB aggregation implementation.

If any item changes, create a new benchmark ID. This protects the leaderboard from mixing numbers
that look alike but answer different questions.

## Check a result before submission

```bash
entropy-leaderboard prepare-submission path/to/result.json \
  --output submission.json \
  --model-revision <commit-sha> \
  --tokenizer-revision <commit-sha> \
  --parameters 7B
entropy-leaderboard validate-submission submission.json
```

The generated submission records the SHA-256 of the source result, making later changes detectable.
