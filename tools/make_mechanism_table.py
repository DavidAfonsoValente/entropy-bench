"""Emit the rank-comparison table body as LaTeX rows, ordered by the in-domain criterion.

Reads ``results/tier_mechanism.json`` and renders, for each of the eleven news-cohort models, its
rank under the independent in-domain criterion beside its rank under zero-shot and adapted BPB.
This is the table behind the paper's mechanism claim: the free reading mis-ranks by 2.00 places on
average and the adapted reading by 0.36, and the models it mis-ranks are the ones adaptation helps
most. Every number is read from the artifact; none is recomputed here.

Usage:
    python tools/make_mechanism_table.py                 # -> mechanism_table.tex
    python tools/make_mechanism_table.py --check         # verify the committed .tex is current
"""
import json
import statistics
import sys

ARTIFACT = "results/tier_mechanism.json"
# The criterion's own 95% interval, from the paired item bootstrap. Without it a reader cannot tell
# whether an adjacent pair separated by 0.003 is an ordering at all -- and three of these ten
# adjacent pairs are not.
UNCERTAINTY = "results/criterion_uncertainty.json"
OUT = "mechanism_table.tex"

DISPLAY = {"gemma-4-31B": "Gemma-4-31B", "gemma-4-12B": "Gemma-4-12B",
           "Qwen3.5-35B-MoE": "Qwen-3.5-35B-MoE", "Ministral-3-14B": "Ministral-3-14B",
           "Qwen2.5-7B": "Qwen-2.5-7B", "Qwen3.5-9B": "Qwen-3.5-9B",
           "Llama-3.2-1B": "Llama-3.2-1B", "Qwen3.5-4B": "Qwen-3.5-4B",
           "Qwen2.5-1.5B": "Qwen-2.5-1.5B", "LFM2.5-1.2B": "LiquidAI-LFM2.5",
           "Qwen2.5-0.5B": "Qwen-2.5-0.5B"}


def render():
    tm = json.load(open(ARTIFACT))
    unc = json.load(open(UNCERTAINTY))["lenient"]["per_model"]
    per = tm["per_model"]
    order = sorted(per, key=lambda m: per[m]["rank_cloze"])
    lines = []
    for m in order:
        v = per[m]
        # Bold the rank that matches the criterion exactly: the reader should be able to scan the
        # adapted column and see it agree.
        def cell(rank):
            hit = rank == v["rank_cloze"]
            return ("\\textbf{%d}" if hit else "%d") % rank
        ci = unc[m]
        lines.append("%s & %.3f & [%.3f, %.3f] & %d & %s & %s \\\\" % (
            DISPLAY[m], v["cloze"], ci["ci_lo"], ci["ci_hi"], v["rank_cloze"],
            cell(v["rank_zero_shot"]), cell(v["rank_adapted"])))
    mae = tm["mean_abs_rank_error"]
    exact = tm["models_ranked_exactly_right"]
    lines.append("\\midrule")
    lines.append("\\emph{mean rank error} & & & & %.2f & \\textbf{%.2f} \\\\" % (
        mae["zero_shot"], mae["adapted"]))
    lines.append("\\emph{ranks exactly right} & & & & %d of %d & \\textbf{%d of %d} \\\\" % (
        len(exact["zero_shot"]), len(per), len(exact["adapted"]), len(per)))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    body = render()
    if "--check" in sys.argv:
        cur = open(OUT).read()
        if cur != body:
            print("mechanism table: STALE -- regenerate with tools/make_mechanism_table.py")
            sys.exit(1)
        print("mechanism table: PROBLEMS 0")
    else:
        open(OUT, "w").write(body)
        print("wrote " + OUT)
