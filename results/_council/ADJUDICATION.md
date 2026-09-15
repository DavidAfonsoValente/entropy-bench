# Council adjudication — what is left before this study is done (2026-09-01)

Seats: Codex (gpt-5.6-sol, high) · Gemini 3.1 Pro · Fable 5 · Claude Opus 5 (adjudicator, derived
independently before reading any of the others). Raw answers in this directory. Codex's first run
at `--effort xhigh` with all three state docs TIMED OUT at 360 s and returned nothing; it was re-run
at `--effort high` with `ASK_TIMEOUT=1500` and `docs/PLAN.md` only.

Every engine claim below was re-verified natively against the artifact. Two did not survive.

## The cost reframe — this is the finding that reorders everything

Gemini priced the remaining programme at ~500 + ~200 GPU-h and Fable at ~250-450 A100-h. **Both are
high by roughly an order of magnitude.** Measured cost bases already in the repo:

- `docs/PLAN.md:148` — the existing 44-cell fixed-budget adaptation grid took **23.5 GPU-h total**,
  largest single cell 2.8 h.
- `docs/PROGRESS.md:739` — measured adaptation cost is **~20 min/cell median**, and that line exists
  precisely to correct the 40 GPU-h estimate still sitting in `PLAN.md`.
- `docs/RUN_LEDGER.md:379` — the 11-model cloze run was **02:58:30Z -> 03:18:09Z, 20 minutes wall**,
  loading-dominated.
- `docs/PROGRESS.md:724` — the one genuinely expensive thing is generative benchmarking: **~28 min
  per GSM8K cell** on a 10B model.

`saldo -b` verified live: **77,076 A100-h remaining, expiring 2026-09-23**. The entire remaining
work programme, including the most maximal version any seat proposed, is **under 1% of that budget**.

**Therefore compute is not the constraint and never was.** The constraints are (i) forking-paths
discipline, (ii) engineering time, and (iii) one host-compatibility fact that no amount of A100
hours can buy. Fable put it best and I adopt the framing: *the 77,000 hours are a temptation, not a
resource.* The correct posture is to plan as if the budget were 1,000 hours and let the rest expire.

## Verified open items

**V1 — a preregistered question is committed in PENDING state.** [Fable; VERIFIED]
`results/cohort_extension.json` -> `preregistered_answers.q2_any_benchmark_clears_chance` =
`"PENDING: benchmark column incomplete for the extension cohort"`, and `results/cohort_ext/bench/`
holds **exactly 2 cells** (`gsm8k__Falcon3-10B.json`, `gsm8k__SmolLM2-1.7B.json`) of 18. This is the
single strongest finding of the council and I had missed it. A shipped artifact carrying a
preregistered question marked PENDING, with two stray cells landed, invites exactly the question
"why did you stop, and did you peek?" **PENDING is the one unshippable state.** It must become
either answered or formally closed. Fork F1 below.

**V2 — a preregistered recomputation was never done.** [Fable; VERIFIED]
`docs/RUN_LEDGER.md:714` states the byte denominator and the **tokenizer-bias bound must be
recomputed for the extended cohort**. `tokenizer_bias_bound` appears only in
`tools/analyze_alignment_matrix.py`, `results/alignment_matrix.json`, `verify_paper_numbers.py`,
`test_alignment_matrix.py` and `docs/POSITIONING.md` — all 11-model artifacts. It is absent from
`tools/analyze_cohort_extension.py`. SmolLM2's 49,152 vocab widens the byte/token spread past the
original band, so this is not a formality. **CPU-only, no GPU, no forking-paths exposure** (it was
preregistered before the data). This one is simply owed.

**V3 — every adaptation cell is single-seed and nobody has measured the variance.** [Codex; VERIFIED]
`lm_adapt_bench/config.py:13` `seed: int = 42`; `cli.py:188` calls `set_seed(args.seed)` once; no
seed sweep exists anywhere. The bootstrap resamples **models**, so it captures between-model
variation and none of the within-cell training stochasticity that determines each cell's adapted
BPB. The resolved +0.250 [+0.074, +0.500] interval therefore omits a variance component. No seat but
Codex raised this, and it is the item **best matched to the compute we actually have**: BPB pools
across hosts (the E3 control reproduced a published cell to 0.00%/0.01%), so seed replicates run on
Leonardo A100 are poolable with the published numbers — which is exactly what is NOT true of the
benchmark column.

**V4 — three passages went stale when E3 landed.** [Claude]
`docs/PLAN.md` sequencing still reads "E3 is now the only thing that matters"; `docs/PROGRESS.md:96`
still calls the cohort extension the single highest-value run "and the paper says so in three
places"; `paper_sota.tex:1757` still says a wider cohort "would ... resolve the in-band comparison
the eleven-model bootstrap cannot" — E3 resolved it under lenient. The conclusion is where a
reviewer looks for what is unsettled, and it currently understates the study's own best result.

## Leads that did NOT survive verification

**X1 — "the paper overstates the gemma-4 exclusion." [Fable; FALSE, already fixed.]**
Fable cited `docs/RUN_LEDGER.md:293` ("that argument is weaker than the paper states. Open for
David"). The paper no longer makes that argument: `paper_sota.tex:1608-1615` now says outright
"**The exclusion is therefore a property of this engine version, not of the models**, and we no
longer claim otherwise," and keeps the exclusion on the separate and correct ground that
adapted-versus-base deltas are uninterpretable across an engine change. **The ledger entry is the
stale artifact, not the paper.** Fold into V4.

**X2 — "re-run the benchmark column for 17 models on one A100 stack." [Gemini's item 2; REJECTED.]**
Superficially it dissolves my own objection (re-score everything, so nothing is orphaned). It does
not survive the host fact: the published generative column is an H100 vLLM 0.25.1 build, and the
study measured cross-host GSM8K drift of **-0.60 to +0.69 pp on identical weights**, larger than
most deltas at stake. Re-scoring on Leonardo A100 replaces the published column with a third-host
column and forces every benchmark number in the paper to change on the eve of shipping. Gemini also
priced it at ~500 GPU-h against a measured basis of ~20 min/cell. The scope amendment that refused
this was right, and free A100 hours do not change its reason.

## Where the seats agreed, and where that agreement is worth something

**Unanimous, and I concur: do NOT run full E5** (convergence + size ladder). It replaces the
estimand the paper sells ("a cheap fixed-budget probe as a selection instrument") with a different
question, on a three-week deadline, and would force reconciling a second results universe at
shipping time. Codex additionally caught a real design defect: the ladder specifies 3k/15k/100k
documents while the mathematics corpus holds 3,207 — those cannot both define a common ladder.

**Unanimous, and I concur: the danger is the budget, not the science.** Every seat independently
named the garden of forking paths as the central risk, unprompted by each other. Agreement across
four seats on a *risk* is weaker evidence than agreement on a fact, but the convergence here is on
the same specific mechanism — choosing what to run after seeing which comparisons resolved — and
that mechanism is real and present in this study right now.

## The forks that need David

**F1 — how to close Q2 (V1).** Two honest paths, and they are not equivalent:
 (a) **Finish the 16 benchmark cells on the H100/GCE build** that produced the published column.
     ~5-10 GPU-h, costs GCE money, and is *executing a preregistration rather than forking paths* —
     the question, the scope amendment and the auto-ingest path in `analyze_cohort_extension.py`
     were all committed before any result existed. Risk: a public benchmark may clear chance at 40
     pairs, which weakens the core claim. That result would have to ship.
 (b) **Formally close Q2 at 11 models**, annotate the two stray cells, state in the artifact and the
     paper that the amendment closed it. Zero compute, zero risk, and strictly honest — but it
     leaves the study's most-wanted comparison unanswered while sitting on idle capacity, and a
     reviewer may read the stop as convenient.
 My recommendation: **(a)**. Leaving a question unanswered because the answer might hurt is the one
 reason that cannot be written down, and Gemini named this explicitly as the strongest objection to
 running anything ("leaving it unanswered is highly protective of your current narrative").

**F2 — seed replication (V3), on Leonardo.** ~2 extra seeds over the in-band cells. Using the
measured ~20 min/cell rather than the inflated estimates, this is on the order of **20-60 A100-h**,
not Codex's 70-150. It is the best available match between the compute we have and a gap that is
real, and it attacks the study's own uncertainty accounting rather than trying to widen its claims.
Falsifier, and it is a serious one: **if the resolved +0.250 interval turns out to depend on the
seed, the study's only resolved comparison dissolves.** That is precisely why it is worth knowing
before a reviewer asks, and precisely why the report must be committed as shipping either way.
My recommendation: **run it, second in priority after V2, and preregister it first.**

**F3 — Fable's H1 full-replication package** (~250-450 A100-h claimed; realistically ~60-150 given
the measured cost basis): re-run adaptation + BPB + all four cloze criteria for the whole cohort on
one Leonardo stack, recovering OLMo-2 and Qwen2.5-32B. Genuinely tempting and genuinely dangerous:
the cloze criterion is **greedy generation and does not pool across hosts**, so this either replaces
every cloze number in the paper or produces an incomparable second universe. It needs a
**paper-of-record rule fixed in advance** — which dataset governs each claim if published and
replicated numbers disagree. My recommendation: **not before shipping.** It is the right first
experiment for the *next* paper.

## What I am doing without asking (zero compute, zero forking-paths exposure)

1. V4 + X1 — resync the four stale passages so the docs match what E3 and the paper already say.
2. V2 — the preregistered tokenizer-bias recomputation for the 17-model cohort (CPU only).

## STOP condition

Ship when: Q2 is answered or formally closed (F1); the tokenizer-bias bound exists for the 17-model
cohort and either overturns no rank or its overturning is reported (V2); the single-seed limitation
is either measured (F2) or stated in the paper as an uncovered variance component (V3); and the
stale passages are gone (V4). Rung 3 ships unresolved — that was already decided and remains right.

**Unspent GPU-hours are not unfinished science.** — Codex, and it is the sentence to keep.
