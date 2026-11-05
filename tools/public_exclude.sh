#!/usr/bin/env bash
# The public snapshot's exclusion list, shared by tools/sync_public_repo.sh (the public repository)
# and tools/build_release_packages.sh (the anonymous TMLR supplementary). Source it; it sets
# EXCLUDE_RE (tracked paths that stay internal) and KEEP_PUBLIC (public-only paths a sync keeps).
# Internal-only paths. Everything tracked here is published EXCEPT these.
#   - agent instructions and internal state docs (routing, ledger, roadmap, sales positioning,
#     docs/STATUS.md, the session-to-session state file added after this list was written, and
#     review triage notes, which must not be public while a paper is under double-blind review)
#   - GCP runners: they hard-code our private GCS bucket and are useless to an outside reader
#   - results/_council/: internal cross-model deliberation notes, which cite the internal docs above
#   - per-example samples: ~370 MB of samples_*.jsonl(.gz). The public repo has never carried
#     these; they live in gs://gpu-llm-training-gceval/token_gain/adapted_bench/. The scored
#     results_*.json summaries ARE published, so every number in the paper stays greppable.
#   - the extension projects (B1/B2 model coverage, P3 cross-tokenizer ensemble): separate work
#     the paper does not use, with their own code, results and GCP cluster runners
EXCLUDE_RE='^(CLAUDE|AGENTS|GEMINI)\.md$|^docs/(POSITIONING|PROGRESS|RUN_LEDGER|PLAN|STATUS|REVIEW_TRIAGE_[0-9-]+)\.md$|^tools/(gcp_[a-z_]*\.sh|gce_run_watchdog\.sh)$|samples_.*\.jsonl(\.gz)?$|^\.claude/|^results/_council/'
EXT_TOOLS='analyze_b1|byte_mix|byte_scores|corpus_affinity|ensemble_bpb|ensemble_gate|ensemble_mc|gsm8k_rerank|gsm8k_sample|mc_segment_scores|mix_corpora|p3_mc_summary|plot_b1|predict_b1|prepare_code_corpus|segment_scores|solution_probe|tau2_dialogues|trace_bpb|trace_gen|trace_length_check'
EXT_TESTS='analyze_b1|b2|corpus_tools|predict_b1|segments|solution_probe'
EXCLUDE_RE="$EXCLUDE_RE|^b1\.mk$|^results/(b1|ensemble)/|^results/(b1_alignment_[0-9-]+|corpus_affinity)\.json$|^figures/fig_b1_|^slurm/gcp/|^lm_adapt_bench/segments\.py$|^tools/($EXT_TOOLS)\.py$|^lm_adapt_bench/tests/test_($EXT_TESTS)\.py$"

# Files that exist only in the public repo and must survive a sync rather than being deleted.
KEEP_PUBLIC='^docs/TITLE_OPTIONS\.md$'

