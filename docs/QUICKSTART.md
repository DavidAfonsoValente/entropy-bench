# Quickstart

This guide has three paths. Pick the outcome you need.

## 1. Inspect the published results — no GPU required

```bash
git clone https://github.com/DavidAfonsoValente/entropy-bench.git
cd entropy-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

entropy-leaderboard validate
entropy-leaderboard render --check
entropy-dataset validate-manifest
python tools/verify_paper_numbers.py
```

These commands check the leaderboard schema and arithmetic, its rendered Markdown, the dataset
fingerprint artifact, and every number transcribed into the paper. They do not download a model or
run an evaluation.

## 2. Evaluate a base model on your corpus

Prepare `my_corpus.jsonl` with one non-empty text per line:

```json
{"text":"First document."}
{"text":"Second document."}
```

Check it first:

```bash
entropy-dataset describe my_corpus.jsonl
```

Run zero-shot BPB before committing to adaptation:

```bash
entropy-bench \
  --models Qwen/Qwen2.5-0.5B \
  --dataset my_corpus.jsonl \
  --dataset-text-field text \
  --output results/my-corpus \
  --baseline-only \
  --no-pdf
```

Remove `--baseline-only` for the full contamination audit, hyperparameter search, LoRA adaptation,
and held-out evaluation. Use a CUDA machine sized for the selected model. Gated models require that
you accept their terms and export `HF_TOKEN`; never put tokens on the command line or in a file.

The main score is `adapted_bpb`: lower is better. Compare models only when corpus, split, context
length, contamination policy, and adaptation procedure are identical.

## 3. Add a model to the leaderboard

If you do not have the controlled paper corpus, open a [model evaluation
request](https://github.com/DavidAfonsoValente/entropy-bench/issues/new?template=model-evaluation.yml).
Give the exact model and tokenizer revisions; a maintainer runs the model on the canonical data.

If you have an authorized canonical result:

```bash
entropy-leaderboard prepare-submission \
  results/my-run/model/result_model.json \
  --output submission.json \
  --model-name Model-Name \
  --model-revision <commit-sha> \
  --tokenizer-revision <commit-sha> \
  --parameters 7B

entropy-leaderboard validate-submission submission.json
```

Open a pull request containing `submission.json` and the contamination audit summary. CI rejects
wrong benchmark IDs, duplicate models, inconsistent arithmetic, and out-of-order ranks.

## Check a development checkout

```bash
python -m pip install -e '.[test]'
make check
```

See [Understanding results](RESULTS.md) for the output files and how to interpret each metric.
