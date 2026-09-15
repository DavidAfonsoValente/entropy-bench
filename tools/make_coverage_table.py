"""Emit the four-corpus selector table as LaTeX rows, one block per corpus.

Reads ``results/cloze_coverage.json`` (E4) and renders in-band pairwise selection accuracy for
every selector on every corpus, under both scoring conventions, each corpus judged against a
criterion built from its OWN post-release held-out text. This is the paper's replication evidence:
the comparison does not depend on which of the four yardsticks you pick. A dagger marks an interval
that clears chance. Every number is read from the artifact; none is recomputed here.

Usage:
    python tools/make_coverage_table.py            # -> coverage_table.tex
    python tools/make_coverage_table.py --check    # verify the committed .tex is current
"""
import json
import sys

ARTIFACT = "results/cloze_coverage.json"
OUT = "coverage_table.tex"
BAND = "within_2x"

# Column order: ours first, then the alternatives a practitioner would actually weigh.
SELECTORS = ["matched_adapted_bpb", "zero_shot_bpb", "hellaswag", "gsm8k", "mmlu_pro",
             "parameter_count"]
CORPORA = [("news", "Google News"), ("reddit", "Reddit"), ("hackernews", "Hacker News"),
           ("math", "arXiv math")]


def render():
    cov = json.load(open(ARTIFACT))
    lines = []
    for i, (key, label) in enumerate(CORPORA):
        if key not in cov["per_corpus"]:
            continue
        if i:
            lines.append("\\addlinespace")
        per = cov["per_corpus"][key]
        for j, conv in enumerate(("lenient", "strict")):
            band = per[conv][BAND]
            sel = band["selectors"]
            best = max(sel[s]["pairwise_accuracy"] for s in SELECTORS if s in sel)
            cells = []
            for s in SELECTORS:
                v = sel.get(s)
                if v is None:
                    cells.append("---")
                    continue
                # Bold the leader in the row and dagger anything that clears chance, so the
                # reader can see at a glance that no benchmark ever does in this band.
                txt = "%.3f" % v["pairwise_accuracy"]
                if v["pairwise_accuracy"] >= best - 1e-12:
                    txt = "\\textbf{%s}" % txt
                if v.get("beats_chance"):
                    txt += "$^\\dagger$"
                cells.append(txt)
            # The in-band pair count can differ between conventions (Hacker News is 10 lenient,
            # 11 strict, because a tie under one convention orders nothing), so one parenthetical
            # cannot label both rows: print each row's own count.
            counts = [per[c][BAND]["n_pairs"] for c in ("lenient", "strict")]
            head = ("%s (%d)" % (label, counts[0]) if counts[0] == counts[1]
                    else "%s (%d)" % (label, band["n_pairs"])) if j == 0 else (
                "" if counts[0] == counts[1] else "(%d)" % band["n_pairs"])
            lines.append("%s & %s & %s \\\\" % (head, conv, " & ".join(cells)))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    body = render()
    if "--check" in sys.argv:
        if open(OUT).read() != body:
            print("coverage table: STALE -- regenerate with tools/make_coverage_table.py")
            sys.exit(1)
        print("coverage table: PROBLEMS 0")
    else:
        open(OUT, "w").write(body)
        print("wrote " + OUT)
