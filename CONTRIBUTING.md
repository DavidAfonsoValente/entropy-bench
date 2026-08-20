# Contributing

Contributions are welcome as focused pull requests with tests and an evidence trail.

## Report a problem

Open an issue with:

- the command you ran;
- the smallest dataset format that reproduces the problem;
- Python, PyTorch, Transformers, PEFT, and CUDA versions;
- the model ID and immutable revision; and
- the relevant log excerpt, with credentials and private text removed.

## Code changes

```bash
python -m pip install -e '.[test]'
make check
```

GPU behavior should include a small smoke test or a reproducible SLURM script. Do not commit model weights, caches, secrets, private data, or generated Python bytecode.

## Leaderboard contributions

Leaderboard rows are evidence-bearing benchmark results, not manually edited Markdown.

1. Run the exact benchmark ID and protocol.
2. Keep the generated per-model result JSON and contamination audit.
3. Record immutable model and tokenizer revisions.
4. Run `entropy-leaderboard prepare-submission`.
5. Validate the submission locally.
6. Open a pull request containing the submission JSON and audit summary.

The contribution must state:

- benchmark ID;
- model and tokenizer IDs plus immutable revisions;
- parameter count as displayed;
- hardware and software versions;
- zero-shot and adapted BPB;
- contamination configuration and removed-sample counts; and
- SHA-256 of the source result JSON.

Maintainers reject results with a changed corpus, split, seed, context length, contamination policy, or adaptation procedure from the existing board. Such results can start a separate, clearly identified board.

If you cannot access the controlled primary corpus, use the **Model evaluation request** issue template. The maintainers can run the candidate without distributing the source text.

## Dataset contributions

Do not open a pull request containing third-party source text unless its redistribution license is
documented. A new public benchmark must include:

- a stable benchmark ID and snapshot date;
- source URLs or dataset identifier plus the exact upstream revision;
- redistribution license and attribution requirements;
- record count, byte count, text field, and file SHA-256;
- a generated record-hash index; and
- commands that rebuild or download the snapshot deterministically.

Run `entropy-dataset fingerprint DATASET --write-index PATH` to create the identity fields and safe
record index. A new corpus starts a new leaderboard; it does not replace the evidence behind an
existing board.
