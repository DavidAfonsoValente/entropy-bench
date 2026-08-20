# What drives the large BPB gain on Gemma and Liquid after adaptation?

Answering the question raised after the 3 August team talk. Measured, not argued: every
number below comes from `tools/token_level_gain.py` scoring the held-out news test split
under each base model and under its own LoRA adapter, on a single spot A100 in
the published evaluation environment. Sanitized raw outputs are under `results/token_gain/` and
`results/token_gain_bpb/`.

## Headline

The gain is **not** a document-boundary artefact, and it is **not** domain knowledge. It is
**cold-start recovery plus surface-form re-calibration** — the models that gain most are the
ones whose base distribution sat furthest from ordinary prose, and much of the gain is
concentrated at the start of each 512-token block where no context is available.

The sharpest single result: **Gemma-4-12B's base model is already better than
Qwen-2.5-1.5B's once it has context** (2.374 vs 2.585 nats/token at block positions 256+).
Its poor zero-shot BPB comes from the first few dozen tokens of every block, where it scores
8.5 nats against Qwen's 5.0. So the puzzle raised in the discussion is a measurement
artefact of fixed-block packing, not a property of the model.

That is a real qualification of the paper's central claim: as currently measured, the
zero-shot→adapted reduction largely isolates **base-model calibration and cold-start
behaviour**, not architectural plasticity.

## Validation

The measurement reproduces the published numbers, so the decomposition can be trusted.

| Model | reported reduction | measured (4,000 test blocks) |
|---|---|---|
| Gemma-4-12B | 28.7% | **29.01%** |
| LFM2.5-1.2B | 36.5% | **36.54%** |
| Qwen-2.5-1.5B | 3.2% | **3.28%** |

## 1. Where the gain sits

Mean nats/token, and the share of the total per-token gain by position class. `marker` is
the injected BOS-like token itself; `next 16` is the post-boundary window; `remainder` is
everything else. These boundaries can occur anywhere inside a packed block.

| Model | base | adapted | reduction | marker | next 16 | remainder |
|---|---|---|---|---|---|---|
| Gemma-4-12B | 2.838 | 2.015 | 29.01% | 3.63% | 5.04% | **91.33%** |
| LFM2.5-1.2B | 4.298 | 2.728 | 36.54% | 1.12% | 1.92% | **96.96%** |
| Qwen-2.5-1.5B | 2.737 | 2.647 | 3.28% | 0% | 0% | **100%** |

**Note the base losses.** A 12B Gemma starts *worse per token* than a 1.5B Qwen
(2.838 vs 2.737) and ends up clearly better (2.015 vs 2.647). The ranking inverts entirely
across adaptation — which is exactly the puzzle raised in the discussion.

## 2. A boundary artefact exists but is small

`lm_adapt_bench/data.py:475` packs each document with `add_special_tokens=True` and then
cuts fixed 512-token blocks, so tokenizers that inject a BOS put it *inside* the scored
stream at every document boundary. The base models are enormously surprised by it —
Gemma's `<bos>` gains **+18.93 nats** and LFM's `<|startoftext|>` **+11.47 nats**, i.e. the
base model assigns it a probability of order 1e-8.

But such tokens are only **0.15% of scored positions** (1 per ~640 tokens on this corpus),
so they contribute 3.6% (Gemma) and 1.1% (LFM) of the total gain; with the post-boundary window,
8.7% and 3.0%. Real, worth fixing, not the explanation.

This was the initial hypothesis, and it is worth recording why it was tempting: across the
whole 11-model cohort the four largest gains are *exactly* the four BOS-injecting models
(LFM2.5 36.5%, Gemma-4-12B 28.7%, Gemma-4-31B 22.6%, Ministral-3-14B 9.9%), while all six
Qwen models — whose tokenizer injects nothing — sit between 1.0% and 7.4%
(Pearson +0.72, Spearman +0.75 on the binary predictor). The correlation is real; the
causal story it suggested is not.

## 3. The gain is on generic English, not news content

Share of the top-200 gaining tokens' mass, by token type:

| Model | whitespace/format | function word | punctuation | digit | content word |
|---|---|---|---|---|---|
| Gemma-4-12B | 23.9% | 37.2% | 22.0% | 11.8% | **5.2%** |
| LFM2.5-1.2B | 21.8% | 50.4% | 10.5% | 3.2% | **14.0%** |
| Qwen-2.5-1.5B | 3.5% | 29.2% | 15.0% | 27.5% | **24.8%** |

Top gainers, in order:

- **Gemma-4-12B**: `<bos>`, `.`, `,`, `\n\n`, ` the`, ` and`, ` a`, ` to`, `6`, ` in`, `’`, `-`
- **LFM2.5-1.2B**: `\n\n`, `<|startoftext|>`, ` `, ` to`, ` the`, ` a`, ` in`, ` and`, ` that`
- **Qwen-2.5-1.5B**: `6`, `.\n\n`, ` the`, `The`, `2`, `5`, `\n\n`, **` June`**, ` with`, ` at`

The contrast is the finding. Gemma and LFM spend ~83% of their gain on formatting,
punctuation and articles — the surface statistics of the corpus. Qwen, already calibrated,
spends its small gain on **digits and ` June`** — dates and figures, i.e. genuine
domain/temporal specifics of a June-2026 news corpus.

No single token carries the gain: for LFM the largest is `\n\n` at 1.55%, so this is a
broad distributional shift rather than a few learned strings.

## 4. Gemma's zero-shot number is a block-start penalty — with context it already wins

This is the direct answer to "without dynamic eval the Gemma models look somewhat poor and
this should not be the case."

Mean base loss (nats/token) by absolute position in the 512-token block:

| position | Gemma-4-12B | Qwen-2.5-1.5B |
|---|---|---|
| 0 | 9.285 | 6.809 |
| 1 | 9.072 | 5.733 |
| 4–7 | 8.004 | 4.279 |
| 16–31 | 4.766 | 3.300 |
| 64–127 | 3.015 | 2.796 |
| 128–255 | 2.663 | 2.669 |
| **256–510** | **2.374** | **2.585** |

**The lines cross.** Over the first 8 positions Gemma averages 8.51 nats against Qwen's
4.97 — far worse. By position 256+ Gemma is at 2.374 and Qwen at 2.585, so the 12B Gemma
base model is already the better language model *once it has context*, without any
adaptation at all. Its whole-block average is dragged down by the opening.

Blocks are cut every 512 tokens irrespective of document boundaries, so almost every block
begins mid-document with no BOS and no preceding context. Gemma pays that opening penalty
on *every* block, and adaptation is largely learning to cope with it: **23.0% of Gemma's
total gain comes from the first 32 positions**, which are 6.3% of the scored tokens, versus
12.9% for Qwen over the same positions.

So the reviewer's instinct was right, and there is a mechanism: the protocol's fixed-block
packing penalises models that depend on a document-start convention, and Gemma depends on it
heavily. This is a fixable measurement issue, not a property of the model — see the
recommendation below.

## 5. LFM cannot use context, and that is most of its gain

Blocks are cut every 512 tokens regardless of document boundaries, so most begin
mid-document with no BOS prefix. Mean base loss and gain by absolute position in the block:

| position in block | Qwen-2.5-1.5B base → gain | LFM2.5-1.2B base → gain |
|---|---|---|
| 0 | 6.809 → +0.043 | 10.576 → +3.382 |
| 1 | 5.733 → +0.370 | 9.374 → +4.007 |
| 2–3 | 5.055 → +0.345 | 8.693 → +3.949 |
| 8–15 | 3.730 → +0.182 | 7.469 → +3.824 |
| 32–63 | 3.003 → +0.120 | 6.111 → +3.097 |
| 128–255 | 2.669 → +0.085 | 4.303 → +1.637 |
| 256–510 | 2.585 → +0.074 | 3.436 → +0.861 |

Early positions are expensive for everyone — Qwen injects no BOS and still shows the shape —
but the magnitude differs enormously, and for LFM it never recovers.

**LFM's base model is close to broken.** At position 0 it scores 10.576 nats against
`ln(64400) = 11.07` for a uniform distribution over its vocabulary — i.e. **96% of the way
to uniform**, effectively no prediction at all. Qwen at position 0 sits at 6.809 against
`ln(151665) = 11.93`, or 57% of uniform. LFM is still worse at position 256–510 (3.436)
than Qwen is at position 0. Its gain stays near +3.9 nats through position 16 and only
decays after ~128 tokens of context.

So LFM's 36.5% is not learning journalism. It is a base model that cannot condition on
context being taught to do so.

## Consequences

1. **The "LiquidAI Paradox" section of the paper is right, and now quantified.** LFM's 36.5%
   is distributional recovery: base loss 4.298 nats/token against 2.737 for a smaller Qwen,
   with 97% of the gain on ordinary tokens and half of it on function words.
2. **The same explanation applies to Gemma**, which the paper currently treats as
   "high-density learning". It is the same mechanism, milder.
3. **"Adaptation velocity" should not be read as architectural plasticity** without
   controlling for how mis-calibrated the base model was. Velocity is largely a measure of
   the starting deficit.
4. **Fix the packing** before extending to new domains. Two separate defects:
   - The injected boundary token was a scored prediction target. Now masked out of the
     labels (kept as context) — `mask_injected_special_tokens`, default on. Worth 3–9% of
     the measured gain for BOS-injecting models, and worse on short-document corpora.
   - **Blocks start with no document-start convention at all**, which is the bigger effect:
     it costs Gemma 23% of its gain over 6.3% of positions and is why its zero-shot BPB
     looks bad. Recommended: prepend each model's BOS to every block, and/or report BPB
     over positions past a fixed warm-up window so the score measures steady-state
     modelling rather than cold-start recovery. Neither is implemented yet — it changes
     every published number, so it needs a decision first.
5. **`calculate_avg_bytes_per_token` vs the packed stream.** The BPB denominator
   (`data.py:446`) counts tokens with `add_special_tokens=False` while the numerator scores
   a stream built with `=True`. On this corpus the mismatch is only ~0.16%, so it does not
   affect conclusions, but it is wrong and should be made consistent.

## Bearing on the proposed framing

The theme suggested in the discussion — *"how to correctly use entropy to evaluate LMs and
bypass ad-hoc benchmarks"* — survives this result, but only one of the paper's two claims
supports it.

**Supported, and strongly.** Byte-normalized entropy on a corpus we control is a better
ruler than a leaderboard. That is the rank-disagreement result: agreement with BPB decays
monotonically as a benchmark moves away from language modelling (6/9 identical ranks on
HellaSwag, 2/9 on MMLU-Pro, 0/9 on GSM8K), and on GSM8K the reshuffle separates perfectly
by model family. The team reached the same reading independently — that the Qwen models look
like benchmark outliers whose scores exceed what their size and BPB imply, and that the
Gemma models look better than their benchmark scores suggest. That is exactly what the
family split says, from the other direction.

**Needs qualification.** "Adaptation velocity measures architectural plasticity" does not
survive as stated. The velocity is dominated by how mis-calibrated the base model was on
the corpus's surface form. Two honest options: report adapted BPB as the headline (a level,
not a delta) and treat the delta as a diagnostic of base-model calibration; or keep the
delta but control for the starting deficit.

## Why this matters for the Reddit / Wikipedia extension

Reddit and HN records are individual comments, with a median of ~81 words each. The news
corpus has a median of 362 words per document (p10 101, p90 929), and because the mean is
pulled up by the long tail it averages ~640 tokens per document — which is the figure that
sets the boundary-token density measured above. Left ungrouped that is roughly **6× the boundary-token
density**, so both packing artefacts would be several times larger on a comment corpus than
on news.

`tools/prepare_scraped_corpus.py` therefore groups records into documents — by subreddit
for Reddit, sequentially for Hacker News, which has no grouping field. Measured over
2026-08-06 to 2026-08-10:

| | Reddit (4 days) | Hacker News (5 days) |
|---|---|---|
| raw records | 3,054,623 | 60,814 |
| grouped documents | 141,527 | 6,257 |
| after exact dedup + Gopher | 141,527 | 6,223 |
| near-duplicate removal | **not completed** | 137 removed (2.2%) |
| final documents | 141,527 | 6,086 |
| text | 533 MB | 46 MB |
| words/document | median 624 (p10 603, p90 732) | median 690 (p10 610, p90 2184) |

Reddit's near-duplicate pass did not finish: a partial run covering 100,000 of the 141,527
documents had removed 469 (**0.47%**) before it was killed. On that rate the step is a
sub-percent correction, so the corpus is usable as-is; it is queued to complete on whatever
compute node runs the adaptation. Two failed attempts before that are the reason
`--fuzzy-shard-size` and `--from-stage` exist — see below.

Both land at or above news-article scale (news median: 362 words), so the domains stay
comparable and the boundary density stays low. Reddit is 533 MB against 361 MB for the news slice, so it is the larger corpus of the two.

Three engineering notes, because this data does not process naively:
the raw dumps are pretty-printed JSON arrays of ~400 MB/day, so they must be streamed with
an index pointer rather than slurped or re-sliced (re-slicing per record is quadratic);
documents must be staged to disk rather than held in a list; and the fuzzy-dedup step
cannot use `Deduper.find_near_duplicates`, which takes every text at once. Even streaming
the texts is not sufficient — a single `MinHashLSH` over 141k documents is itself several
GB, so the index is rebuilt every `--fuzzy-shard-size` documents. That is an approximation:
near-duplicates are topically and temporally clustered and Reddit documents are emitted
subreddit-by-subreddit, so a sharded index catches most of them, but not all. For reference
Hacker News, small enough for a single index, lost only 137 of 6,223 documents (2.2%) to
fuzzy dedup, so the step is a small correction either way.

Two data caveats: **Reddit is missing 2026-08-07** in the bucket (Hacker News only that
day), and there is no Wikipedia yet — which is the domain where the contamination audit
matters most, since Wikipedia is in essentially every pre-training corpus.
