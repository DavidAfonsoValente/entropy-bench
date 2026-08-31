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
| `zero_shot_bpb` | Predictive cross-entropy before adaptation, normalized by UTF-8 bytes | Initial fit — and a strong, free selection score in its own right; see below |
| `adapted_bpb` / `best_bpb` | Held-out BPB after controlled adaptation | Primary model-selection score; lower is better |
| `reduction_pct` | Relative BPB change from zero-shot to adapted | Diagnose how much corpus calibration was needed |

Do not rank models by reduction percentage. A weak starting model can improve greatly and still end
with worse adapted BPB than a strong model. This is empirically justified, not just a caution:
across all four corpora the reduction correlates with every benchmark at `|rho| < 0.5`, and with
GSM8K on news at `-0.055`.

The current primary board uses one fixed LoRA configuration for every model: rank 16, alpha 32,
dropout 0.05, learning rate `1e-4`, effective batch 32, and 250 update steps. That budget is equal
by construction but it is 0.21 epochs of the news corpus, and the paper's own token-level diagnostic
shows ~83% of the resulting gain lands on formatting and function words — so adapted BPB here is
*fit after a stated budget*, not attainable fit. `docs/PLAN.md` (E5) specifies the
adapt-to-convergence estimand that would replace it. Injected
document-start marker targets are excluded from both the loss numerator and scored-token count.

## What the benchmark comparison shows

**Read both tiers, not just the adapted one.** The complete picture is
`results/alignment_matrix.json` — four adaptation corpora × {zero-shot, adapted} × three
benchmarks, 24 alignments in total. Regenerate it with:

```bash
python tools/analyze_alignment_matrix.py
```

Across the 11-model cohort, adapted news BPB nearly reproduces the HellaSwag order (Pearson
`r = -0.975`, Spearman `rho = 0.982`, 7 of 11 exact ranks) — which is expected, because HellaSwag
scores continuations by length-normalized conditional log-likelihood, the same object BPB measures.
Against the two generative benchmarks it does *worse* than the zero-shot ranking it replaced:

| News BPB | GSM8K | MMLU-Pro | HellaSwag |
|---|---:|---:|---:|
| zero-shot | **0.836** | **0.855** | 0.718 |
| adapted | 0.727 | 0.764 | **0.982** |

Adaptation rotates the ranking toward continuation quality and away from generative task ability.
The same pattern holds on Reddit and Hacker News. So the practical question is not "adapted or not"
but "adapted **on what**": adapting on mathematical prose puts GSM8K agreement at `rho = 0.936`
with 50 of 55 pairs ordered correctly, the best selector in the study.

**Prefer selection regret to rank correlation when deciding.** `rho` over 11 models has very wide
intervals. Regret — the accuracy you give up by taking the metric's top pick — is the number that
maps onto the decision. Zero-shot news BPB picks the GSM8K and MMLU-Pro leader outright (0.0 pp
regret); adapted news BPB picks the HellaSwag leader instead, at 2.3 and 4.2 pp regret on the other
two.

The disagreements are informative. Qwen models tend to move upward on MMLU-Pro and GSM8K, while the
two Gemma models move downward, consistent with family-specific knowledge/math training and
chain-of-thought behavior. At small scale the distinction is stark: Qwen-2.5-1.5B reaches 61.1% on
GSM8K, while Llama-3.2-1B reaches 6.5%, despite adjacent BPB ranks. This is not evidence that one
metric is wrong: BPB measures broad next-byte predictive fit after controlled adaptation, whereas
GSM8K additionally depends on mathematical reasoning, prompting, generation, and answer extraction.

Point-estimate leaders should not be over-sold. The 0.53 percentage-point GSM8K gap between
Qwen-3.5-35B-MoE and Qwen-3.5-9B is small relative to test-set sampling uncertainty. The committed
analysis derives exact correct/total counts from the raw per-model harness output under
`results/{gsm8k,hellaswag,mmlu_pro_1k}/` and reports Wilson intervals from those; it does not
capture prompt or run-to-run uncertainty. Reproduce it with:

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
