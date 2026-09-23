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
free, zero-shot ranking is worth reporting rather than an intermediate step — though **not the
number to select on**: it reads a model's distance from your corpus's writing conventions as
incapacity, and under-rates exactly the candidates adaptation helps most. See the rule below.

**[Read the paper](paper_sota.pdf) · [Start in five minutes](docs/QUICKSTART.md) · [View the
leaderboard](LEADERBOARD.md) · [Understand a result](docs/RESULTS.md) · [Check the
dataset](docs/DATASET.md)**

## Why not just read the benchmark scores?

**The short version: where the decision is hard, they carry no signal we could resolve.**

We built an outcome measure that is *not* a benchmark — fill-in-the-blank items drawn from our own
held-out split, scored by making the model generate the answer — and asked which selector predicts
it among candidates **within 2× in size**, the only case where anyone needs a tool at all:

| Deciding by | accuracy | interval excludes chance? |
|---|---:|---|
| **One short adaptation run, then rank by BPB** | **0.909** | **yes** |
| HellaSwag | 0.818 | no |
| Pick the bigger model | 0.636 | no |
| GSM8K | 0.182 | no |
| MMLU-Pro | 0.182 | no |

We repeated this on **all four** of our corpora — news, Reddit, arXiv mathematics, Hacker News —
each judged against a criterion built from its own held-out split. Across 4 corpora × 2 scoring
conventions × 7 selectors, **exactly one interval excludes chance, and it is the one above.**
**No public benchmark clears chance on any corpus under either convention** — and neither does the
free zero-shot number, nor parameter count. (`results/cloze_coverage.json`.)

**Read that as a failure to reject, not a demolition.** With 10–11 in-band pairs a selector has to
be near-perfect to separate from a coin flip, so this says the suites do not *resolve* this
decision — not that they are uninformative. It is still the scorecard failing at the job people use
it for.

**And what we do not claim:** that we beat picking the bigger model (across eight contrasts we lead
on five and trail on three, none separating from zero), or that our lead over the benchmarks is
measurable (it isn't, at 11 models). If your candidates differ by more than ~2× in size, use size
and skip this entirely. One more thing a reviewer should know up front: the cloze items come from
the same held-out documents on which BPB is measured, which the benchmarks have no counterpart to.

---

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

The practical rule is therefore three steps:

0. **If the candidates differ by more than about 2x in size, take the bigger one and stop.** The
   measurement will not tell you anything size has not already, and under one of two defensible
   scoring conventions on our own in-domain criterion, size beats us outright.
1. **Otherwise, adapt each finalist under one budget on recent target-domain text and rank by
   adapted BPB.** Minutes on one GPU — 3.7 at 0.5B, 11.6 at 7.7B, 24.7 at 12B. Match the
   adaptation corpus to the capability that decides the choice, not just to the genre.
2. **Read the zero-shot number as a free screen, not as the selector.** One forward pass, and it
   will show you a candidate badly off-distribution for your text.

**This changed on 2026-09-01, and step 1 used to be the zero-shot ranking.** The zero-shot tier
agrees well with the public benchmarks, which is what that recommendation rested on. Measured
against an in-domain criterion that is *not* a benchmark
([`sec:cloze`](paper_sota.pdf), `results/cloze_validity.json`) it reaches chance (0.545) among
candidates within 2x in size — the regime where the decision is actually hard — while the adapted
tier reaches 0.909 there and is the only one of seven selectors whose interval clears chance. We
recommended the free tier as a selector and the independent test did not support it.

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

Remove `--baseline-only` for LoRA adaptation. **Read the zero-shot number too** — it costs one
forward pass and it screens for a candidate badly off-distribution for your text. Select on the
adapted number: see the three-step rule above for why the tiers are not interchangeable.

Adaptation hyperparameters come from one of two sources, and either way the final adaptation trains
until validation BPB stops improving (early stop on a plateau; with `--phase train` it resumes across
chained jobs, so a job's walltime never cuts it short). On a machine with no job time limit,
`--no-time-limit` runs the full method with no shortcuts: every sweep trial, then training until the
validation plateau with no wall-time budget and no epoch cap; the result records `stop_reason`, and
only `"plateau"` means validation BPB converged.

- `--hparams sweep` (default, the method): a per-model Optuna search — TPE sampler, successive
  halving on validation BPB — with the same `--n-trials` for every model and no wall-time cap, so
  no model is better tuned than another. `--sweep-time-fraction` adds a cap when walltime forces
  one; it lets small models finish more trials than large ones, so use it only when necessary.
- `--hparams manual`: skip the search and use `--hparams-file` (YAML/JSON of `TrainingConfig`
  fields). With no file it uses the packaged recipe `lm_adapt_bench/configs/manual_hparams.yaml`,
  the hand-picked configuration behind the paper's reported results.

Remove `--no-pdf` when
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

The [public leaderboard](LEADERBOARD.md) has two tracks over the same eleven base models, each its
own benchmark:

- **Mathematics** (`arxiv-math-2026-08-fixed-lora-v1`) — arXiv `math.*` titles and abstracts from
  9 June to 20 August 2026, CC0 metadata, published in
  [`data/corpora/`](data/corpora/arxiv-math-2026.jsonl.gz). **Anyone can run it** and submit a model.
- **News** (`primary-news-2026-06-08-fixed-lora-v1`) — the paper's primary corpus; its text cannot be
  redistributed, so the maintainers run requested models.

Canonical boards are [`results/leaderboard.json`](results/leaderboard.json) and
[`results/leaderboard_math.json`](results/leaderboard_math.json); the Markdown tables are generated
from them. Check them locally:

```bash
entropy-leaderboard validate
entropy-leaderboard render --check
```

[LEADERBOARD.md](LEADERBOARD.md) gives the exact commands to evaluate a model on the mathematics
track and turn the result into a self-checking submission; `prepare-submission` rejects a result
whose corpus hash, context length or adaptation recipe differs from the track's manifest. For the
news track, open a [model evaluation
request](https://github.com/DavidAfonsoValente/entropy-bench/issues/new?template=model-evaluation.yml).
See [CONTRIBUTING.md](CONTRIBUTING.md) for the evidence contract.

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
- the search records of an earlier, wall-time-capped HPO sweep in `results/legacy_sweep/`, which
  show why the sweep must give every model the same trial budget (see that directory's README);
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
lm_adapt_bench/       the pipeline: config, data, contamination audit, sweep, adapt, evaluate, report
tools/                analysis, figure generators, and the reproducibility gates
docs/                 MAP.md (every file, generated) - STATUS.md (where things stand) - the ledger, plan and positioning
eval/                 standalone MMLU-Pro, HellaSwag and GSM8K harness
results/              committed artifacts and raw result cells; see docs/MAP.md
figures/              paper and slide figures, each produced by a tools/plot_*.py
benchmarks/           versioned benchmark contracts
data/manifests/       source-free record fingerprint indexes
reports/              rendered report of the superseded 11-model sweep
slurm/                Leonardo batch scripts - never run on the login node
slides/               team-talk deck and speaker script
paper_sota.tex        manuscript source; `make` builds paper_sota.pdf
LEADERBOARD.md        rendered public leaderboard
Makefile              every command this repo supports - start here
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
