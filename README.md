# Back to Entropy: Evaluating Base Language Models Beyond Perplexity and Benchmarks

[![CI](https://github.com/DavidAfonsoValente/entropy-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/DavidAfonsoValente/entropy-bench/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/DavidAfonsoValente/entropy-bench)](https://github.com/DavidAfonsoValente/entropy-bench/releases/latest)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper_sota.pdf)

**Choose the base language model that best predicts the text you actually care about.** Entropy
Bench compares models in one tokenizer-independent unit, on recent target-domain data, before and
after the same controlled adaptation procedure.

The headline finding is that those two measurements answer different questions. Adaptation does not
move the ranking uniformly closer to "better model" — it **rotates** the ranking toward whatever the
adaptation corpus exercises. So the corpus you adapt on is the question you are asking, and the
free, zero-shot ranking is a serious baseline rather than an intermediate step.

**[Read the paper](paper_sota.pdf) · [Start in five minutes](docs/QUICKSTART.md) · [View the
leaderboard](LEADERBOARD.md) · [Understand a result](docs/RESULTS.md) · [Check the
dataset](docs/DATASET.md)**

## Why not just read the benchmark scores?

Because one score cannot separate *this model is stronger* from *this model was trained on more of
this*. Two models in our own cohort make the point without any statistics:

| | Llama-3.2-1B | Qwen-2.5-1.5B | apart by |
|---|---:|---:|---:|
| Adapted news BPB (lower is better) | **0.771** | 0.797 | 3.4% |
| HellaSwag | 0.658 | **0.678** | 2.0 pts |
| GSM8K | 0.065 | **0.611** | **54.6 pts** |

They model fresh English within 3.4% of each other and sit two points apart on commonsense, yet one
scores nine times the other on grade-school arithmetic. Qwen-3.5-4B against Llama-3.2-1B is wider
still: 2.9% apart on BPB, **73.9 points** apart on GSM8K. Both training mixes are legitimate
engineering choices — the score simply cannot tell you which one you are looking at, and
pre-training mixtures are not published.

This is not an accusation. A public benchmark that has existed for years is a development target
whether or not anyone aims at it deliberately, and nobody outside a lab can audit what went into
pre-training. **A corpus dated after every candidate shipped cannot have been studied for.** That is
a weaker guarantee than "a better measure of quality", and it is the one we claim: prediction loss
on your own recent text is a *second, independent axis* to read alongside the scorecard — not a
replacement for it. Reproduce every number here with `make check`.

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
4. **Contamination control:** prefer a corpus that postdates every candidate's release — that is
   protection by construction and needs no detector. Where a snapshot cannot be that fresh (our news
   corpus), quarantine exact/near duplicates and likelihood outliers before they influence adaptation
   or scoring; perturbation tests are risk signals, not proof of memorization.

BPB is the unit; the protocol around it is the contribution — fresh data, fair adaptation, and
contamination control turn a decades-old metric into a practical, auditable base-model selection
tool.

## What the evidence says

The paper evaluates 11 base models from 0.5B to 35B across dense Transformers, a sparse MoE, and a
Liquid architecture.

All 24 (corpus × tier × benchmark) rank agreements are published in
[`results/alignment_matrix.json`](results/alignment_matrix.json). The GSM8K column, which is where
the interesting behaviour lives:

| Adaptation corpus | GSM8K `rho` | Model pairs ordered correctly |
|---|---:|---:|
| *none* (zero-shot news, free) | **0.836** | 46/55 |
| News | 0.727 | 43/55 |
| Reddit | 0.691 | 42/55 |
| Hacker News | 0.791 | 45/55 |
| Math (arXiv abstracts) | **0.936** | **50/55** |

- **Adaptation rotates the ranking; it does not simply improve it.** On news it lifts HellaSwag
  agreement from `rho = 0.718` to `0.982` while lowering GSM8K from `0.836` to `0.727` and MMLU-Pro
  from `0.855` to `0.764`. Reddit and Hacker News show the same three-way movement.
- **In selection terms:** zero-shot news BPB picks Qwen-3.5-35B-MoE, which *is* the GSM8K and
  MMLU-Pro leader — zero regret. Adapted news BPB picks Gemma-4-31B: zero regret on HellaSwag, 2.3
  and 4.2 accuracy points of regret on the other two. HellaSwag scores continuations by
  log-likelihood, the same object BPB measures, so that agreement is convergent validity rather
  than proof the ranking improved.
- **The rotation is steerable, which is what makes it useful.** Adapting on independent,
  contamination-screened mathematical prose — not GSM8K questions — reaches `rho = 0.936` on GSM8K
  with the best pairwise accuracy in the study, and returns the top pick to the oracle. Three
  general-text corpora, run as controls, all sit at or below the do-nothing baseline. HellaSwag
  agreement pays for it, falling to `0.818`.
- **What 11 models cannot settle.** A paired bootstrap puts the math-vs-news difference at
  `Δrho = +0.209` with a 95% interval of `[-0.096, +0.704]`. The evidence is the pattern across four
  corpora, not any single contrast, and the paper says so rather than implying otherwise.
- **Adapted tiers stay stable across domains** (`rho = 0.955–0.991`), and a 512-token adapter keeps
  its gain through 2,048-token inference.

- **Be honest about when this is worth running.** Across our full 0.5B–35B range, parameter count
  orders candidates about as well as we do (`0.855` vs `0.836` of pairs on GSM8K) — a 70x span makes
  "pick the bigger model" a strong baseline, and we say so. The decision that is actually hard is
  between candidates of *comparable* size, and there it reverses: restricted to pairs within 2x in
  parameters, size is a coin flip at `0.545` while math-adapted BPB reaches `0.818`. These bands
  hold 7–20 pairs and are reported as descriptive.

The practical rule is therefore three steps, and the first one is free:

0. **If the candidates differ by more than about 2x in size, take the bigger one and stop.** The
   measurement will not tell you anything size has not already.
1. **Free tier** — rank by zero-shot BPB on recent target-domain text. One forward pass; it picked
   the generative-benchmark leader on three of our four corpora.
2. **Paid tier** — adapt every candidate under one budget and rank by adapted BPB when you will
   actually fine-tune on domain text, or when one specific capability decides the choice. Then
   match the adaptation corpus to that capability, not just to the genre.

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

Remove `--baseline-only` for LoRA adaptation. **Read the zero-shot number too** — on our cohort it
is a strong selector on its own, and it costs one forward pass. For a leaderboard-comparable run,
use the fixed configuration in the benchmark manifest; optional hyperparameter search is exploratory
and creates a different protocol, for the reason documented in
[`results/legacy_sweep/README.md`](results/legacy_sweep/README.md). Remove `--no-pdf` when
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
[`benchmarks/primary-news-2026-06-08-fixed-lora-v1.json`](benchmarks/primary-news-2026-06-08-fixed-lora-v1.json): 119,054
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

- the corrected fixed-protocol leaderboard cells in `results/domain_transfer/news__*.json`;
- the legacy exploratory-run [summary](reports/11-model-run/summary.json), [HTML
  report](reports/11-model-run/report.html), [PDF report](reports/11-model-run/report.pdf), and plots,
  clearly separated from the headline ranking;
- all 33 cross-domain result cells in `results/domain_transfer/`;
- token-gain and byte-normalized position analyses in `results/token_gain*/`;
- static benchmark scores in `results/combined_bpb_vs_static.json` and the reproducible uncertainty
  and rank analysis in `results/static_benchmark_analysis.json`;
- the full corpus × tier × benchmark alignment matrix, selection regret, bootstrap intervals,
  benchmark-redundancy check, and tokenizer-bias bound in `results/alignment_matrix.json`
  (`tools/analyze_alignment_matrix.py`);
- the superseded HPO sweep's search records in `results/legacy_sweep/`, which are the evidence for
  using one fixed adaptation configuration (see that directory's README);
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
