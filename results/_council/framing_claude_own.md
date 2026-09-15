# Claude Opus 5 (adjudicator) — own derivation, written BEFORE reading any other seat

## (A) Positioning

### The hard question, answered plainly

**Can we claim the adaptation step improves selection accuracy? No.** Our own standing rule is that
a claim comparing selectors must hold under both conventions. `adapted − zero-shot` is
`+0.150 [−0.087, +0.476]` lenient and `+0.000 [−0.278, +0.257]` strict. Under strict the point
estimate is exactly zero and **fewer than half the draws are positive (36.2%)**. That is not an
underpowered positive; it is a coin flip. The claim fails our own rule and must not be made.

### But there is a real asymmetry, and it is not the same claim

Three statements are jointly true and must be kept apart:

1. Adapted BPB **clears chance under both** conventions (0.900 / 0.850).
2. Zero-shot BPB **does not** — it fails lenient (0.750, CI `[0.417, 1.000]`).
3. We **cannot** assert adapted > zero-shot.

(1) and (2) are statements about what we can vouch for in each tier separately. (3) is the
head-to-head. **Presenting (1)+(2) as if they established (3) is the "difference between
significant and non-significant is not itself significant" fallacy**, and a hostile reviewer will
name it. So the honest form is: *adapted BPB is the only tier that clears our own both-conventions
bar, and we cannot show the upgrade buys accuracy over the free tier.* Both halves, one sentence.

### What must NOT be rescued

The temptation is item 2 — rank error 2.00 → 0.36, which flatters adaptation enormously. Resist it:
the artifact itself says "a described pattern with named instances, not an estimate; the
correlations place the pattern and are not tested," it is 11 models, and it measures the same
ground truth as the pairwise contest. Preferring the untested framing **because** the tested one
came out flat is textbook forking paths, and this study's whole posture is that we do not do that.
Report it as mechanism/illustration, never as the evidence for the recommendation.

### The recommendation that the evidence actually supports

Restructure the recommendation as two questions, not one ladder:

- **"Which released model is best on my domain right now?"** → **zero-shot BPB on your own recent
  text.** Nearly free. We cannot show adaptation beats it.
- **"Which model will be best after I fine-tune it on this corpus?"** → **adapted BPB**, because
  that is a different question and item 1 (rotation, 23 of 24 settings) is what establishes that
  the two questions have different answers. The adaptation step is justified by *what it measures*,
  not by a demonstrated accuracy edge.

This is a **better** paper, not a retreat: it stops selling a GPU-hour we cannot justify and starts
selling the thing we actually proved.

### Why this beats every other option — the case that does NOT rest on the contested margin

Lead with properties, not accuracy. Each is independently evidenced:

1. **The suites people actually cite are near coin-flips for this decision.** GSM8K 0.359/0.410 and
   MMLU-Pro 0.400/0.450 — neither clears chance under either convention. This is the single most
   sellable number in the study and nothing about our own margin is needed to state it.
2. **Unpreparable** (item 5). Post-release scrapes; no vendor could have trained on the test text.
   No public suite can have this property, ever. It is structural, not empirical.
3. **Host-stable** (item 4). Identical weights move −0.60 to +0.69 GSM8K points across builds,
   larger than most gaps at stake; BPB reproduced to 0.00%/0.01% on a third host. So a BPB number
   is comparable across who computed it; a generative benchmark number is not.
4. **Tokenizer-agnostic by construction** — the byte denominator is the whole point, and it is
   definitional, not an empirical claim.
5. **Seed-robust** (item 3). 0 of 40 pairs reorder; the resolved interval moves 0.001.
6. **Available on day one**, no labels, no waiting for anyone's leaderboard.

And the honest boundary, stated up front because it is the most actionable thing in the paper:
**more than ~2× apart in size, take the bigger model and stop.** We only add value inside 2×.

### Claims to forbid explicitly

- "BPB beats benchmarks." (spans zero, both conventions)
- "Adapting beats not adapting." (spans zero, both conventions)
- "Benchmarks are uninformative." (HellaSwag clears chance under both)
- "Better than size." (lenient only; strict spans zero)

## (B) Exposition

1. **Lead with the benchmark null, not our margin.** The hook is "the two generative suites you
   would reach for are no better than a coin flip for choosing a base model for your domain." It is
   rock solid, it needs none of our contested numbers, and it motivates everything after it.
2. **One table carries the paper**: the 6 selectors × 2 conventions table with the chance line and
   the clears/does-not column. Everything else is support for it.
3. **Put all three contrasts in one block, same format, same sentence shape.** Beats parameter
   count (lenient, resolved) · ties the best benchmark · ties its own free tier. Presenting them
   together is what stops the paper from either burying the flat ones or reading as a null — the
   reader sees one honest ladder rather than one buried admission.
4. **Replace the statistical vocabulary.** "Selection regret" and "Spearman ρ" → "how often it
   picks the better model." Keep the intervals, drop the jargon around them.
5. **State the decision rule in three lines, early**, and make the >2× exclusion the first line.
   Leading with when NOT to use the method buys more credibility than any margin.
6. **Demote to appendix** everything that is not (a) the selector table, (b) rotation, (c) the
   robustness trio (seed / host / tokenizer). The paper currently spends its middle on side
   experiments that a non-statistician reader will not finish.
7. **Say the power story once, in the conclusion, and do not repeat it.** At eleven models nothing
   resolved; at forty pairs two selectors do. Repeating it starts to sound defensive.

## Falsifier

If a larger cohort resolves `adapted − zero-shot` **negative**, the two-question framing collapses
and the honest recommendation becomes zero-shot only. Nothing above depends on that contrast being
positive, which is deliberate.
