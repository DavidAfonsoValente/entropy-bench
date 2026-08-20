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
