# Beyond Perplexity: Entropy-Based Evaluation of Base Language Models

[![CI](https://github.com/DavidAfonsoValente/entropy-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/DavidAfonsoValente/entropy-bench/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper_sota.pdf)

**LM Adapt Bench** evaluates base language models using the objective they were trained on: predictive cross-entropy. It makes that signal useful across modern models by combining four pieces:

1. **A common unit:** Bits Per Byte (BPB) removes tokenizer-dependent denominators.
2. **Fresh target-domain data:** the evaluator controls what the model must predict.
3. **Controlled adaptation:** every candidate receives the same kind of domain intervention before selection.
4. **Contamination filtering:** suspected memorized text is removed before adaptation or scoring.

The result is not “just another BPB metric.” It is an auditable protocol for choosing a base model from data that matters to the deployment.

**[Read the paper](paper_sota.pdf) · [View the leaderboard](LEADERBOARD.md) · [Inspect the full report](reports/11-model-run/report.pdf) · [Use the static benchmarks](eval/README.md)**

## Why return to entropy?

Token-level perplexity cannot be compared directly across tokenizers because a token represents a different amount of text for each model. Fixed task benchmarks provide common questions, but each measures only a selected capability; repeated optimization on fixed tests also makes benchmark-specific tuning and contamination difficult to separate from general language-model quality.

Entropy observes the model's complete predictive distribution. LM Adapt Bench returns to that broad signal while fixing the reasons raw perplexity is insufficient for cross-model selection.

## What you get

- one command for zero-shot BPB, hyperparameter search, LoRA adaptation, and held-out BPB;
- TXT, JSONL, CSV, Hugging Face, and ZIP dataset inputs;
- exact and near deduplication, split-leakage checks, Min-K++, CoDeC, and DCQ contamination signals;
- model-specific or globally unified clean test sets;
- resumable multi-fidelity sweeps and adaptation runs;
- JSON, HTML, PDF, and plot outputs;
- a separate reproducible harness for MMLU-Pro, HellaSwag, and GSM8K; and
- a validated, machine-readable leaderboard contribution workflow.

## Install

Python 3.10 or newer is required. Install a PyTorch build appropriate for your CUDA environment first when necessary.

```bash
git clone https://github.com/DavidAfonsoValente/entropy-bench.git
cd entropy-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

entropy-bench --help
entropy-leaderboard validate
```

The original `requirements.txt` remains available for environments that do not use editable installs.

## Evaluate models on your data

Prepare a corpus with one document per line or record; see [data/README.md](data/README.md) for supported formats.

```bash
entropy-bench \
  --models Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B \
  --dataset /path/to/domain-corpus.jsonl \
  --dataset-text-field text \
  --output results/my-domain \
  --contam-check-level strict \
  --contam-cleaning-mode global_unified
```

Add `--baseline-only` to run contamination checks and zero-shot BPB without adaptation. Add `--no-pdf` if WeasyPrint's system libraries are unavailable. Model evaluation and adaptation require accelerator hardware in proportion to model size; do not run full benchmarks on an HPC login node.

### Outputs

Each run writes:

```text
results/my-domain/
├── <model>/result_<model>.json      # auditable per-model metrics and configuration
├── contamination/                   # findings, quarantine, and clean split metadata
├── summary.json                     # consolidated machine-readable results
├── report.html                      # shareable report
└── report.pdf                       # optional PDF report
```

## Leaderboard

The [public leaderboard](LEADERBOARD.md) contains 11 base models from 0.5B to 35B. Its canonical source is [results/leaderboard.json](results/leaderboard.json), and every Markdown row is generated from that file.

Validate that the data and rendered table agree:

```bash
entropy-leaderboard validate
entropy-leaderboard render --check
```

### Add a model

There are two easy paths:

- **You do not have the controlled paper corpus:** open a **Model evaluation request** issue. Provide the immutable model and tokenizer revisions; the maintainers run it against the same snapshot and protocol.
- **You have an authorized result:** convert the generated result JSON into a self-checking submission:

```bash
entropy-leaderboard prepare-submission \
  results/my-run/Qwen_Qwen2_5-7B/result_Qwen_Qwen2_5-7B.json \
  --output submissions/qwen2.5-7b.json \
  --model-name Qwen-2.5-7B \
  --model-revision <commit-sha> \
  --tokenizer-revision <commit-sha> \
  --parameters 7B

entropy-leaderboard validate-submission submissions/qwen2.5-7b.json
```

Open a pull request with the submission JSON and the audit summary. Maintainers merge it with:

```bash
entropy-leaderboard add submissions/qwen2.5-7b.json
```

The command rejects wrong benchmark IDs, duplicate models, malformed metrics, inconsistent reductions, and unsorted ranks. See [CONTRIBUTING.md](CONTRIBUTING.md) for the evidence contract.

## Reproduce the reported evidence

The repository contains:

- the full primary-run [summary](reports/11-model-run/summary.json), [HTML report](reports/11-model-run/report.html), [PDF report](reports/11-model-run/report.pdf), and plots;
- the machine-readable [primary benchmark manifest](benchmarks/primary-news-2026-06-08.json);
- context-length result files under `data/context_length/`;
- all 33 cross-domain result cells under `results/domain_transfer/`;
- token-gain and byte-normalized position analyses under `results/token_gain*/`;
- static benchmark scores in `results/combined_bpb_vs_static.json`;
- analysis scripts under `tools/`; and
- the complete paper and presentation sources.

The news snapshot itself is controlled rather than redistributed because the source text has no repository-compatible redistribution license. This is stated explicitly in the benchmark manifest. It avoids presenting a different or unlicensed archive as the paper corpus while keeping the model-selection protocol reusable on any authorized dataset.

Build the paper and talk locally with:

```bash
make paper
make slides
```

For the MMLU-Pro, HellaSwag, and GSM8K protocol, pinned environment, model roster, and commands, see [eval/README.md](eval/README.md). Those evaluations are intentionally isolated from the adaptation environment because vLLM pins a different PyTorch stack.

## Method in one equation

For corpus cross-entropy in nats,

```text
BPB = (cross_entropy_nats / ln(2)) × (scored_tokens / source_utf8_bytes)
```

Lower BPB means the model assigns more probability to the held-out text. The leaderboard ranks **adapted BPB**; the zero-shot-to-adapted reduction is reported separately as a corpus-calibration diagnostic.

## Development

CPU unit tests cover BPB, data loading, caching, special-token masking, contamination components, and leaderboard invariants. The end-to-end sweep smoke test is skipped when CUDA is unavailable.

```bash
python -m pip install -e '.[test]'
pytest -q
python tools/script_timing.py
make all
```

## Project layout

```text
lm_adapt_bench/       evaluation, adaptation, reporting, and contamination code
eval/                 MMLU-Pro, HellaSwag, and GSM8K harness
benchmarks/           versioned benchmark contracts
results/              canonical machine-readable public results
reports/              complete published primary-run artifacts
tools/                paper analyses and reproducibility checks
slurm/                HPC launch templates
paper_sota.tex        manuscript source
LEADERBOARD.md        rendered public leaderboard
```

## Citation

GitHub exposes citation metadata from [CITATION.cff](CITATION.cff). Until an archival identifier is available, cite:

```bibtex
@misc{valente2026beyondperplexity,
  title        = {Beyond Perplexity: Entropy-Based Evaluation of Base Language Models},
  author       = {Valente, David Afonso},
  year         = {2026},
  howpublished = {\url{https://github.com/DavidAfonsoValente/entropy-bench}},
  note         = {Paper, code, benchmark artifacts, and leaderboard}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). Model weights, external datasets, and generated benchmark samples remain subject to their original licenses and terms.
