"""Figure 1: what adaptation does to the ordering of base models.

A: rank of each of the fifteen fine-tuned models under raw (unadapted) loss, GSM8K, MMLU-Pro and
   adapted BPB, next to where its fine-tuned system finishes on the news task. Two models are
   highlighted because they carry the argument: Gemma-4-12B (mid-table on the benchmarks, near the
   bottom on raw loss, first after fine-tuning) and Llama-3.2-1B (last on both benchmarks).
B: rank correlation between BPB and each benchmark before and after adaptation, eleven models.
   General text pulls the ranking onto HellaSwag; arXiv mathematics pulls it onto GSM8K/MMLU-Pro.

Ranks come from tools/analyze_downstream.py (the same code the paper-number gate uses); the
correlations from results/alignment_matrix.json. Nothing is recomputed differently here.

Usage:
    python tools/plot_headline.py            # writes figures/fig_headline.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import analyze_downstream  # noqa: E402

OUT = ROOT / "figures" / "fig_headline.pdf"
BLUE, ORANGE, GRAY, INK, MUTED = "#2E6DB4", "#D2762A", "#B9BEC5", "#1C2733", "#6B7785"
HIGHLIGHT = {"gemma-4-12B": ("Gemma-4-12B", BLUE), "Llama-3.2-1B": ("Llama-3.2-1B", ORANGE)}
COLUMNS = [("zero_shot_bpb", "Raw loss", True), ("gsm8k", "GSM8K", False),
           ("mmlu_pro", "MMLU-Pro", False), ("matched_adapted_bpb", "Adapted\nBPB", True)]
ROWS = [("news", "hellaswag", "HellaSwag · news"), ("reddit", "hellaswag", "HellaSwag · Reddit"),
        ("hackernews", "hellaswag", "HellaSwag · Hacker News"),
        ("math", "gsm8k", "GSM8K · arXiv maths"), ("math", "mmlu_pro", "MMLU-Pro · arXiv maths")]


def ranks():
    """Rank (1 = best) of every fine-tuned model under each column, plus its fine-tuned finish."""
    sel, _ = analyze_downstream._selectors_17()
    finish = json.load(open(ROOT / "results" / "downstream.json"))["cohort"]["rouge_l"]
    models = sorted(finish)

    def rank(vals, lower):
        order = sorted(models, key=lambda m: vals[m] if lower else -vals[m])
        return {m: i + 1 for i, m in enumerate(order)}

    table = {key: rank(sel[key][0], lower) for key, _, lower in COLUMNS}
    table["finish"] = rank(finish, False)
    return models, table


def panel_ranks(ax):
    models, table = ranks()
    keys = [k for k, _, _ in COLUMNS] + ["finish"]
    labels = [lab for _, lab, _ in COLUMNS] + ["Fine-tuned\nfinish"]
    xs = range(len(keys))
    for m in sorted(models, key=lambda m: m in HIGHLIGHT):  # highlighted lines drawn last, on top
        ys = [table[k][m] for k in keys]
        name, color = HIGHLIGHT.get(m, (None, GRAY))
        ax.plot(xs, ys, color=color, lw=2.2 if name else 1.0, zorder=3 if name else 1,
                marker="o", ms=5 if name else 3, mec="white", mew=0.8)
        if name:
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(list(xs), labels, fontsize=8)
    ax.axvspan(len(keys) - 1.35, len(keys) - 0.65, color="#F1F3F6", zorder=0)
    ax.set_ylim(len(models) + 0.7, 0.3)
    ax.set_yticks([1, 5, 10, 15])
    ax.set_ylabel("Rank among 15 models (1 = best)", fontsize=8, color=MUTED)
    ax.set_xlim(-0.3, len(keys) - 0.3)
    ax.set_title("A.  Where each model stands, and where it finishes", fontsize=9, loc="left", color=INK)


def panel_steering(ax):
    cells = json.load(open(ROOT / "results" / "alignment_matrix.json"))["cells"]
    for i, (corpus, bench, label) in enumerate(ROWS):
        before = cells[f"{corpus}__zero_shot"]["alignment"][bench]["spearman"]
        after = cells[f"{corpus}__adapted"]["alignment"][bench]["spearman"]
        y = len(ROWS) - 1 - i
        ax.annotate("", xy=(after, y), xytext=(before, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, shrinkA=4, shrinkB=4))
        ax.plot(before, y, "o", ms=7, mfc="white", mec=MUTED, mew=1.4, zorder=3)
        ax.plot(after, y, "o", ms=7, color=BLUE, mec="white", mew=0.8, zorder=3)
        ax.text(after + 0.02, y, f"{after:.2f}", va="center", fontsize=7.5, color=INK)
    ax.axhline(1.5, color="#DDE1E6", lw=0.8)  # general text above, mathematics below
    ax.set_yticks(range(len(ROWS)), [lab for _, _, lab in reversed(ROWS)], fontsize=8)
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("Rank correlation of BPB with the benchmark (Spearman, 11 models)",
                  fontsize=8, color=MUTED)
    ax.plot([], [], "o", mfc="white", mec=MUTED, label="before adaptation")
    ax.plot([], [], "o", color=BLUE, label="after adaptation")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title("B.  What the adapted ordering tracks", fontsize=9, loc="left", color=INK)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#C9CED4",
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    panel_ranks(a)
    panel_steering(b)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, labelsize=8)
    a.grid(axis="y", color="#EEF0F3", lw=0.6)
    b.grid(axis="x", color="#EEF0F3", lw=0.6)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
