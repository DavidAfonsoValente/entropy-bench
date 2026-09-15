# Codex review of the E8 gate (cross-family, GPT-5 'sol')

Two rounds. Every finding below was verified natively before acting; all were real and all
were fixed. Raw per-engine dumps for the 2026-09-01 council are deliberately untracked --
they are unverified leads. This file is tracked because docs/RUN_LEDGER.md cites it.

## Round 1 -- 2026-09-09, on the first version of the gate

- Makefile · `$E8_DIR` exists and `analyze_seed_sensitivity.py --check` exits nonzero; `|| echo ...` masks the failure, so `make check` succeeds · high
- tools/verify_paper_numbers.py · `results/seed_sensitivity.json` is missing from a clone/release; the verifier skips E8 and succeeds despite claiming the artifact is required · high
- paper_sota.tex · Published BPB intervals use the seed stage disabled, yet the limitations text says BPB intervals cover within-cell stochasticity; only the separately reported seed-resampled interval does · med

## Round 2 -- 2026-09-10, on the fixed gate

- tools/verify_paper_numbers.py · Q3 CI bounds drift from `[0.074, 0.500]` / `[0.073, 0.500]` while point estimates and `separates_from_zero` remain unchanged; verification passes despite stale paper numbers · med
- tools/verify_paper_numbers.py · `q2_pairwise_accuracy.per_seed` omits one or all seeds while top-level summary fields still claim completeness; `all(...)` accepts the remaining entries—and vacuously accepts an empty mapping · med