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
| `doc_bpb` | Cross-entropy per UTF-8 byte over whole test documents, each read from its own beginning-of-document token (2,048-token context, 1,024 stride), on the same bytes for every model | **The score to rank candidates by**; lower is better. On the paper's news task it orders 37 of 39 size-matched pairs as the fine-tuned systems do |
| `zero_shot_bpb` | The same loss on fixed 512-token blocks cut from the concatenated test text, before adaptation | Diagnostic only: most blocks start mid-document, which misranks models that expect a document start (30 of 39) |
| `adapted_bpb` / `best_bpb` | The block reading after LoRA adaptation to the domain | Repairs the block reading; on whole documents adaptation barely changes the order, so it is not needed to rank |
| `reduction_pct` | Relative BPB change from zero-shot to adapted | How much a model's block loss depended on the cut; do not rank by it |

Use text from the domain of the task you will fine-tune on. Scored on arXiv mathematics, the same
fifteen models are ordered correctly on only 23 of the 39 news-task pairs.

Injected document-start marker targets are excluded from both the loss numerator and the
scored-token count.

## What the benchmark comparison shows

`results/alignment_matrix.json` holds every corpus × reading × benchmark alignment; regenerate it
with:

```bash
python tools/analyze_alignment_matrix.py
```

On whole documents, which benchmark the eleven-model ranking agrees with follows the text
(`doc_alignment_table.tex`):

| Text scored | GSM8K | MMLU-Pro | HellaSwag |
|---|---:|---:|---:|
| news | 0.755 | 0.827 | **0.991** |
| Reddit | 0.691 | 0.764 | **0.973** |
| Hacker News | 0.782 | 0.873 | **0.982** |
| arXiv maths | 0.918 | **0.945** | 0.836 |

HellaSwag tracks the news fine-tune closely (34 of 39 pairs), but it is not a substitute for the
domain's own text; GSM8K and MMLU-Pro do not track it (18 of 38 and 20 of 39). At small scale:
Qwen-2.5-1.5B reaches 61.1% on GSM8K and Llama-3.2-1B 6.5%, yet their fine-tuned news systems are
not separated.

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
