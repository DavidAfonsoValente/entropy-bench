"""Figures 1 and 3: how well each way of ranking base models anticipates where they finish, and what
the adapted ordering tracks.

fig_headline.pdf (Figure 1): fifteen models, one panel per way of ranking them before any
fine-tuning -- raw (unadapted) loss, GSM8K, MMLU-Pro, HellaSwag, adapted BPB. Each dot is a model:
x is its rank under that measure, y is where its system finishes after fine-tuning on a news task.
A perfect anticipation is the diagonal; the shaded band is within one rank. Gemma-4-12B is marked.

fig_steering.pdf (Figure 3): rank correlation between BPB and each benchmark before and after
adaptation, eleven models. General text pulls the ranking onto HellaSwag; arXiv mathematics pulls
it onto GSM8K and MMLU-Pro.

Ranks come from tools/analyze_downstream.py (the same code the paper-number gate uses); the
correlations from results/alignment_matrix.json. Nothing is recomputed differently here.

Usage:
    python tools/plot_headline.py        # writes figures/fig_headline.{pdf,png} and figures/fig_steering.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import analyze_downstream  # noqa: E402

HEADLINE = ROOT / "figures" / "fig_headline.pdf"
STEERING = ROOT / "figures" / "fig_steering.pdf"
BLUE, ORANGE, GRAY, INK, MUTED, BAND = "#2E6DB4", "#D2762A", "#9AA3AD", "#1C2733", "#6B7785", "#E8EEF6"
FOCUS = "gemma-4-12B"
PANELS = [("zero_shot_bpb", "Raw loss", True), ("gsm8k", "GSM8K", False),
          ("mmlu_pro", "MMLU-Pro", False), ("hellaswag", "HellaSwag", False),
          ("matched_adapted_bpb", "Adapted BPB", True)]
ROWS = [("news", "hellaswag", "HellaSwag · news"), ("reddit", "hellaswag", "HellaSwag · Reddit"),
        ("hackernews", "hellaswag", "HellaSwag · Hacker News"),
        ("math", "gsm8k", "GSM8K · arXiv maths"), ("math", "mmlu_pro", "MMLU-Pro · arXiv maths")]


def ranks():
    """Rank (1 = best) of every fine-tuned model under each measure, plus its fine-tuned finish."""
    sel, _ = analyze_downstream._selectors_17()
    finish = json.load(open(ROOT / "results" / "downstream.json"))["cohort"]["rouge_l"]
    models = sorted(finish)

    def rank(vals, lower):
        order = sorted(models, key=lambda m: vals[m] if lower else -vals[m])
        return {m: i + 1 for i, m in enumerate(order)}

    return models, {key: rank(sel[key][0], lower) for key, _, lower in PANELS}, rank(finish, False)


def headline():
    models, table, finish = ranks()
    n = len(models)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 3.4), sharey=True)
    for ax, (key, label, _) in zip(axes, PANELS):
        ours = key == "matched_adapted_bpb"
        color = BLUE if ours else GRAY
        xs = [table[key][m] for m in models]
        ys = [finish[m] for m in models]
        rho = spearmanr(xs, ys)[0]
        exact = sum(x == y for x, y in zip(xs, ys))
        ax.fill_between([0, n + 1], [-1, n], [1, n + 2], color=BAND, zorder=0, lw=0)  # |x - y| <= 1
        ax.plot([0, n + 1], [0, n + 1], color="#C9D3E0", lw=0.8, zorder=1)
        ax.scatter(xs, ys, s=30 if ours else 22, color=color, edgecolor="white", lw=0.6, zorder=3)
        fx, fy = table[key][FOCUS], finish[FOCUS]
        ax.scatter([fx], [fy], s=64, facecolor="none", edgecolor=ORANGE, lw=1.7, zorder=4)
        if key == "zero_shot_bpb":
            ax.annotate("Gemma-4-12B", (fx, fy), xytext=(-7, -12), textcoords="offset points",
                        fontsize=7, color=ORANGE, fontweight="bold", ha="right")
        ax.text(0.5, 1.30, label, transform=ax.transAxes, ha="center", fontsize=10,
                fontweight="bold", color=BLUE if ours else INK)
        ax.text(0.5, 1.12, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="center", fontsize=13,
                fontweight="bold", color=BLUE if ours else INK)
        ax.text(0.5, 1.02, f"{exact} of {n} in the exact rank", transform=ax.transAxes, ha="center",
                fontsize=7, color=MUTED)
        ax.set_xlim(0.2, n + 0.8)
        ax.set_ylim(n + 0.8, 0.2)
        ax.set_xticks([1, 5, 10, 15])
        ax.set_yticks([1, 5, 10, 15])
        ax.set_aspect("equal")
        ax.tick_params(length=0, labelsize=7.5, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BLUE if ours else "#C9CED4")
            ax.spines[side].set_linewidth(1.4 if ours else 0.8)
    axes[0].set_ylabel("Rank after fine-tuning", fontsize=8.5, color=MUTED)
    fig.supxlabel("Rank before fine-tuning, by each measure (1 = best; shaded: within one rank)",
                  fontsize=8.5, color=MUTED, y=0.02)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.02, 1, 0.82))
    fig.savefig(HEADLINE)
    fig.savefig(HEADLINE.with_suffix(".png"), dpi=200)  # for the README, which cannot show a PDF
    plt.close(fig)


def steering():
    cells = json.load(open(ROOT / "results" / "alignment_matrix.json"))["cells"]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
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
    ax.set_xlabel("Rank correlation of BPB with the benchmark (Spearman, 11 models)", fontsize=8,
                  color=MUTED)
    ax.plot([], [], "o", mfc="white", mec=MUTED, label="before adaptation")
    ax.plot([], [], "o", color=BLUE, label="after adaptation")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color="#EEF0F3", lw=0.6)
    fig.tight_layout()
    fig.savefig(STEERING)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#C9CED4"})
    headline()
    steering()
    print(f"wrote {HEADLINE.relative_to(ROOT)} and {STEERING.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
