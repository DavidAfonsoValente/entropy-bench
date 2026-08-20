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

# CPU-only release checks. Requires `python -m pip install -e '.[test]'` first.
check:
	python -m lm_adapt_bench.leaderboard validate
	python -m lm_adapt_bench.leaderboard render --check
	python -m lm_adapt_bench.dataset_manifest validate-manifest
	python tools/verify_paper_numbers.py
	python tools/check_markdown_links.py
	python tools/script_timing.py
	pytest -q

# 10-minute team talk + speaker script. Figures are pulled from ../figures/.
slides: slides/talk.tex slides/script.tex figures/fig_block_position_slide.pdf figures/fig_cross_corpus_slide.pdf figures/fig_context_length_slide.pdf
	cd slides && $(PDFLATEX) talk.tex && $(PDFLATEX) talk.tex
	cd slides && $(PDFLATEX) script.tex && $(PDFLATEX) script.tex
	cd slides && rm -f *.aux *.log *.nav *.out *.snm *.toc

paper_sota.pdf: paper_sota.tex dt_table.tex figures/fig_block_position.pdf figures/fig_cross_corpus.pdf figures/fig_context_length.pdf
	@if command -v latexmk >/dev/null 2>&1; then \
		$(LATEXMK) $<; \
	else \
		$(PDFLATEX) $< && $(PDFLATEX) $<; \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -c paper_sota.tex; \
	else \
		$(RM) paper_sota.aux paper_sota.log paper_sota.out paper_sota.fdb_latexmk paper_sota.fls; \
	fi

distclean: clean
	$(RM) paper_sota.pdf slides/talk.pdf slides/script.pdf
