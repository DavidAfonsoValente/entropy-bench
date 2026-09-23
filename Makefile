# Build the research paper PDFs.
#
# Requires pdflatex; latexmk is optional and used when available. On Debian/Ubuntu:
#   sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
#       texlive-latex-extra texlive-fonts-recommended texlive-pictures latexmk
#
# Usage:
#   make            # build the main paper (paper_sota.pdf)
#   make all        # build the paper and the team-talk slides
#   make slides     # build slides/talk.pdf and slides/script.pdf
#   make clean      # remove LaTeX build artifacts (keeps the PDFs)
#   make distclean  # also remove the generated PDFs
#
# The slides target uses plain pdflatex (run twice) so it works without latexmk.

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error
PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all paper slides check clean distclean

paper: paper_sota.pdf

all: paper_sota.pdf slides

# Interpreter for the release checks. Prefer the project virtualenv when it exists: the bare
# `python` on this machine satisfies the analysis scripts but has no pytest, so the suite at the end
# of `check` silently never ran (make: pytest: Command not found -> Error 127). Override with
# `make check PYTHON=/path/to/python`.
PYTHON ?= $(shell test -x venv_lm_adapt/bin/python && echo venv_lm_adapt/bin/python || echo python)

# CPU-only release checks. Requires `$(PYTHON) -m pip install -e '.[test]'` first.
check:
	$(PYTHON) tools/analyze_static_benchmarks.py --check
	$(PYTHON) tools/analyze_alignment_matrix.py --check
	$(PYTHON) tools/regenerate_leaderboard.py --check
	$(PYTHON) tools/analyze_adapted_benchmarks.py --runs results/adapted_bench --check
	$(PYTHON) tools/audit_e2_confounds.py --check
	$(PYTHON) tools/pairwise_significance.py --check
	$(PYTHON) tools/analyze_selection_divergence.py --check
	$(PYTHON) tools/analyze_cloze_validity.py --check
	$(PYTHON) tools/analyze_tier_mechanism.py --check
	@if [ -d "$${E8_DIR:-$$SCRATCH/e8_seed}" ]; then \
	  $(PYTHON) tools/analyze_seed_sensitivity.py --check; \
	else \
	  echo "seed sensitivity: raw cells absent (SCRATCH is purged); results/seed_sensitivity.json is asserted by verify_paper_numbers.py"; \
	fi
# `test -f X && cmd --check || echo` SWALLOWS A REAL FAILURE: when the artifact is present and the
# check fails, the `||` branch runs and the recipe line still exits 0, so a broken gate reads as
# "artifact absent, skipping". Use if/then/else so only a genuinely missing artifact skips.
	@if [ -f results/cloze_steering.json ]; then \
	  $(PYTHON) tools/analyze_cloze_steering.py --check; \
	else \
	  echo "cloze steering: artifact absent, skipping (needs a steering run per corpus)"; \
	fi
	@if [ -f results/cohort_extension.json ]; then \
	  $(PYTHON) tools/analyze_cohort_extension.py --check; \
	else \
	  echo "cohort extension: artifact absent, skipping"; \
	fi
	@if [ -f results/cloze_coverage.json ]; then \
	  $(PYTHON) tools/analyze_cloze_coverage.py --check; \
	else \
	  echo "cloze coverage: artifact absent, skipping (needs a cloze set per corpus)"; \
	fi
	$(PYTHON) tools/analyze_criterion_uncertainty.py --check
	$(PYTHON) tools/analyze_decision_rule.py --check
# E10's raw cells live in SCRATCH (purged ~40d); the committed analysis is asserted by
# verify_paper_numbers.py, so a purge must not turn this gate red.
	@if [ -d "$${E10_DIR:-$$SCRATCH/e10_late_position}" ]; then \
	  $(PYTHON) tools/analyze_late_position.py --check; \
	else \
	  echo "late position: raw cells absent (SCRATCH purged); results/late_position.json is asserted by verify_paper_numbers.py"; \
	fi
# The metric that decides E11's verdict is hand-rolled (rouge_score is not installed here), so it
# is unit-tested rather than trusted; the test needs no artifacts and always runs.
	$(PYTHON) tools/test_downstream_metric.py
	$(PYTHON) tools/plot_downstream.py --check
# E11's raw generations live in SCRATCH alongside the task data, same purge rule as E10.
	@if [ -f "$${E11_DIR:-$$SCRATCH/e11_downstream}/eval.jsonl" ]; then \
	  $(PYTHON) tools/analyze_downstream.py --check; \
	else \
	  echo "downstream: raw generations absent (SCRATCH purged); results/downstream.json is asserted by verify_paper_numbers.py"; \
	fi
	$(PYTHON) -m lm_adapt_bench.leaderboard validate
	$(PYTHON) -m lm_adapt_bench.leaderboard render --check
	$(PYTHON) -m lm_adapt_bench.dataset_manifest validate-manifest
	$(PYTHON) tools/verify_paper_numbers.py
	$(PYTHON) tools/verify_playbook.py
	$(PYTHON) tools/check_markdown_links.py
	$(PYTHON) tools/plot_selectors.py --check
	$(PYTHON) tools/make_mechanism_table.py --check
	$(PYTHON) tools/make_coverage_table.py --check
	$(PYTHON) tools/make_cohort_table.py --check
	$(PYTHON) tools/repo_map.py --check
	@if [ -f paper_sota.log ]; then \
	  $(PYTHON) tools/check_paper_refs.py; \
	else \
	  echo "paper refs: paper_sota.log absent; run \`make\` to check cross-references"; \
	fi
	$(PYTHON) tools/script_timing.py
	$(PYTHON) -m pytest -q

# 10-minute team talk + speaker script. Figures are pulled from ../figures/.
slides: slides/talk.tex slides/script.tex figures/fig_headline.pdf figures/fig_downstream.pdf figures/fig_block_position_slide.pdf figures/fig_cross_corpus_slide.pdf figures/fig_context_length_slide.pdf
	cd slides && $(PDFLATEX) talk.tex && $(PDFLATEX) talk.tex
	cd slides && $(PDFLATEX) script.tex && $(PDFLATEX) script.tex
	cd slides && rm -f *.aux *.log *.nav *.out *.snm *.toc

paper_sota.pdf: paper_sota.tex dt_table.tex static_benchmark_table.tex figures/fig_benchmark_alignment.tex figures/fig_block_position.pdf figures/fig_cross_corpus.pdf figures/fig_context_length.pdf figures/fig_selectors.pdf figures/fig_headline.pdf
	@if command -v latexmk >/dev/null 2>&1; then \
		$(LATEXMK) $<; \
	else \
		$(PDFLATEX) $< && $(PDFLATEX) $<; \
	fi

figures/fig_selectors.pdf: tools/plot_selectors.py results/cohort_extension.json
	$(PYTHON) tools/plot_selectors.py

figures/fig_headline.pdf: tools/plot_headline.py tools/analyze_downstream.py results/downstream.json results/alignment_matrix.json
	$(PYTHON) tools/plot_headline.py

# Rewrite docs/MAP.md from the tree. `make check` fails if it is stale.
.PHONY: map
map:
	$(PYTHON) tools/repo_map.py

clean:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -c paper_sota.tex; \
	else \
		$(RM) paper_sota.aux paper_sota.log paper_sota.out paper_sota.fdb_latexmk paper_sota.fls; \
	fi

distclean: clean
	$(RM) paper_sota.pdf slides/talk.pdf slides/script.pdf
