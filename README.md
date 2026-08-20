# Back to Entropy: Evaluating Base Language Models Beyond Perplexity and Benchmarks

[![CI](https://github.com/DavidAfonsoValente/entropy-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/DavidAfonsoValente/entropy-bench/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/DavidAfonsoValente/entropy-bench)](https://github.com/DavidAfonsoValente/entropy-bench/releases/latest)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper_sota.pdf)

**Choose the base language model that best predicts the text you actually care about.** Entropy
Bench compares models in one tokenizer-independent unit, on recent target-domain data, before and
after the same controlled adaptation procedure.

**[Read the paper](paper_sota.pdf) · [Start in five minutes](docs/QUICKSTART.md) · [View the
leaderboard](LEADERBOARD.md) · [Understand a result](docs/RESULTS.md) · [Check the
dataset](docs/DATASET.md)**

## Why go back to entropy?

Perplexity was the language-model quality metric for decades because it measures the model's whole
predictive distribution. But token-level perplexity is not comparable across tokenizers. The field
moved to fixed task benchmarks, which are easier to compare but observe selected capabilities and
can become targets for benchmark-specific optimization or training-data contamination.

Entropy Bench keeps the breadth of the language-modeling objective while fixing the practical
problems that made raw perplexity inadequate:

1. **Comparable units:** Bits Per Byte (BPB) divides by UTF-8 bytes, not model-specific tokens.
2. **Relevant evidence:** evaluate on fresh text from the deployment domain, not only permanent test
   questions.
3. **Fair adaptation:** give every candidate the same controlled opportunity to fit the domain.
4. **Contamination control:** detect and remove suspected memorized or leaking samples before they
   influence adaptation or scoring.

BPB is one component, not the entire claim. The contribution is the complete protocol that makes
entropy useful for modern base-model selection.

## What the evidence says

The paper evaluates 11 base models from 0.5B to 35B across dense Transformers, a sparse MoE, and a
Liquid architecture.

- Adapted BPB closely tracks sentence-continuation quality (`r = -0.96` against HellaSwag).
- Adaptation materially reorders candidates within each domain: zero-shot and adapted ranks have
  Spearman `rho = 0.609–0.682`.
- Adapted performance tiers remain highly stable across news, Reddit, and Hacker News
  (`rho = 0.955–0.991`).
- Knowledge and arithmetic benchmarks remain useful complementary diagnostics; they are not treated
  as interchangeable with predictive quality.

The practical rule is simple: **rank models by adapted BPB on recent target-domain data, then use the
target domain and complementary tasks to resolve close calls.**

## Pick your path

| I want to… | Start here | Hardware |
|---|---|---|
| Inspect and verify the published evidence | [Quickstart: path 1](docs/QUICKSTART.md#1-inspect-the-published-results--no-gpu-required) | CPU only |
| Evaluate a model on my own text | [Quickstart: path 2](docs/QUICKSTART.md#2-evaluate-a-base-model-on-your-corpus) | CUDA recommended |
| Add or request a leaderboard model | [Quickstart: path 3](docs/QUICKSTART.md#3-add-a-model-to-the-leaderboard) | None for a request |
| Verify an authorized copy of the paper corpus | [Dataset guide](docs/DATASET.md#verify-an-authorized-copy) | CPU only |
| Understand every result field | [Results guide](docs/RESULTS.md) | None |

## Install

Python 3.10 or newer is required. Install a PyTorch build suitable for your CUDA system first when
necessary.

```bash
git clone https://github.com/DavidAfonsoValente/entropy-bench.git
cd entropy-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm the three public tools are available:

```bash
entropy-bench --help
entropy-dataset --help
entropy-leaderboard --help
```

## Evaluate models on your data

The recommended input is JSONL with one `{"text":"..."}` object per line. TXT, CSV, Hugging Face
datasets saved to disk, and ZIP files are also supported.

Inspect the corpus before allocating a GPU:

```bash
entropy-dataset describe /path/to/domain-corpus.jsonl
```

Start with a zero-shot baseline:

```bash
entropy-bench \
  --models Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B \
  --dataset /path/to/domain-corpus.jsonl \
  --dataset-text-field text \
  --output results/my-domain \
  --contam-check-level strict \
  --contam-cleaning-mode global_unified \
  --baseline-only \
  --no-pdf
```

Remove `--baseline-only` for hyperparameter search and LoRA adaptation. Remove `--no-pdf` when
WeasyPrint's system libraries are installed. Model evaluation and adaptation require accelerator
hardware in proportion to model size; never launch a full run on an HPC login node.

Each run creates:

```text
results/my-domain/
├── <model>/result_<model>.json   # metrics and exact run configuration
├── contamination/               # findings, quarantine, clean split metadata
├── summary.json                 # consolidated machine-readable results
├── report.html                  # shareable report
└── report.pdf                   # optional PDF
```

See [Understanding results](docs/RESULTS.md) before comparing scores.

## Leaderboard and contributions

The [public leaderboard](LEADERBOARD.md) contains 11 base models. Its canonical source is
[`results/leaderboard.json`](results/leaderboard.json); the Markdown table is generated from that
file.

Check it locally:

```bash
entropy-leaderboard validate
entropy-leaderboard render --check
```

To add a model, either open a [model evaluation
request](https://github.com/DavidAfonsoValente/entropy-bench/issues/new?template=model-evaluation.yml)
or turn an authorized run into a self-checking submission:

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

Open a pull request with the submission and contamination audit summary. CI checks the benchmark ID,
model uniqueness, arithmetic, ordering, dataset manifest, and generated leaderboard. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the evidence contract.

## Dataset and reproducibility

The exact paper corpus is identified by
[`benchmarks/primary-news-2026-06-08.json`](benchmarks/primary-news-2026-06-08.json): 119,054
documents, 366,564,042 file bytes, and SHA-256
`279ccc65a7f562e438cbe74827d41e51e41ac8783345250b6f85f607ea1a0162`.

The repository also publishes a complete record-level hash index, all reported metrics, and every
analysis artifact. The third-party news text itself is held pending redistribution permission; its
absence is a rights constraint, not a missing or unidentified experiment file. The [dataset
guide](docs/DATASET.md) explains the exact release condition and verification command.

Verify a permitted local copy with:

```bash
entropy-dataset verify /path/to/google_news_from_2026-06-08_to_2026-06-08_cleaned.jsonl
```

Published evidence includes:

- the primary-run [summary](reports/11-model-run/summary.json), [HTML
  report](reports/11-model-run/report.html), [PDF report](reports/11-model-run/report.pdf), and plots;
- all 33 cross-domain result cells in `results/domain_transfer/`;
- token-gain and byte-normalized position analyses in `results/token_gain*/`;
- static benchmark scores in `results/combined_bpb_vs_static.json`;
- context-length result files in `data/context_length/`; and
- paper, slide, and speaker-script sources.

Run every lightweight release check with:

```bash
python -m pip install -e '.[test]'
make check
```

Build the paper and presentation with `make paper` and `make slides`. The MMLU-Pro, HellaSwag, and
GSM8K environment is documented separately in [eval/README.md](eval/README.md) because vLLM pins a
different PyTorch stack.

## The metric

For corpus cross-entropy in nats,

```text
BPB = (cross_entropy_nats / ln(2)) × (scored_tokens / source_utf8_bytes)
```

Lower BPB means the model assigned more probability to held-out text. The leaderboard ranks
**adapted BPB**. The zero-shot-to-adapted reduction is reported separately as a corpus-calibration
diagnostic; a large reduction is not automatically a better final model.

## Project layout

```text
lm_adapt_bench/       evaluation, adaptation, reporting, and verification code
docs/                 quickstart, dataset policy, and result interpretation
eval/                 MMLU-Pro, HellaSwag, and GSM8K harness
benchmarks/           versioned benchmark contracts
data/manifests/       source-free record fingerprint indexes
results/              canonical machine-readable public results
reports/              complete primary-run artifacts
tools/                paper analyses and reproducibility checks
slurm/                HPC launch templates
paper_sota.tex        manuscript source
LEADERBOARD.md        rendered public leaderboard
```

## Citation

GitHub exposes [citation metadata](CITATION.cff). Until an archival identifier is available:

```bibtex
@misc{valente2026backtoentropy,
  title        = {Back to Entropy: Evaluating Base Language Models Beyond Perplexity and Benchmarks},
  author       = {Valente, David Afonso},
  year         = {2026},
  howpublished = {\url{https://github.com/DavidAfonsoValente/entropy-bench}},
  note         = {Paper, code, benchmark artifacts, and leaderboard}
}
```

## License

Project code is released under [Apache-2.0](LICENSE). Model weights, external datasets, and benchmark
text remain subject to their original licenses and terms.
