"""Figure 1: what adaptation does to the ordering of base models, measured without any referee.

A (slopegraph): the eleven-model cohort ranked three ways -- unadapted news BPB, adapted news BPB,
   HellaSwag. Lines cross between the first two columns (adaptation reorders the models) and run
   nearly parallel between the last two (the adapted order is the language benchmark's).
B (steering): rank correlation between BPB and each benchmark before and after adaptation. General
   text pulls the ranking onto HellaSwag; arXiv mathematics pulls it onto GSM8K and MMLU-Pro.

BPB comes from results/alignment_matrix.json, HellaSwag from results/combined_bpb_vs_static.json;
nothing is recomputed differently here.

Usage:
    python tools/plot_headline.py        # writes figures/fig_headline.pdf and .png (for the README)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = ROOT / "figures" / "fig_headline.pdf"
BLUE, ORANGE, GRAY, INK, MUTED = "#2E6DB4", "#D2762A", "#B3BAC3", "#1C2733", "#6B7785"
HIGHLIGHT = {"gemma-4-12B": BLUE, "LFM2.5-1.2B": ORANGE}
NAMES = {"gemma-4-31B": "Gemma-4-31B", "gemma-4-12B": "Gemma-4-12B", "Qwen3.5-35B-MoE": "Qwen-3.5-35B",
         "Ministral-3-14B": "Ministral-3-14B", "Qwen3.5-9B": "Qwen-3.5-9B", "Qwen2.5-7B": "Qwen-2.5-7B",
         "Qwen3.5-4B": "Qwen-3.5-4B", "Llama-3.2-1B": "Llama-3.2-1B", "Qwen2.5-1.5B": "Qwen-2.5-1.5B",
         "LFM2.5-1.2B": "LFM2.5-1.2B", "Qwen2.5-0.5B": "Qwen-2.5-0.5B"}
ROWS = [("news", "hellaswag", "HellaSwag · news"), ("reddit", "hellaswag", "HellaSwag · Reddit"),
        ("hackernews", "hellaswag", "HellaSwag · Hacker News"),
        ("math", "gsm8k", "GSM8K · arXiv maths"), ("math", "mmlu_pro", "MMLU-Pro · arXiv maths")]


def _cells():
    return json.load(open(ROOT / "results" / "alignment_matrix.json"))["cells"]


def news_ranks():
    """Rank (1 = best) of the eleven models by unadapted BPB, adapted BPB and HellaSwag, on news."""
    cells = _cells()
    zero, adapted = cells["news__zero_shot"]["bpb"], cells["news__adapted"]["bpb"]
    static = json.load(open(ROOT / "results" / "combined_bpb_vs_static.json"))
    hellaswag = {m: v["hellaswag"] for m, v in static.items()}

    def rank(vals, lower):
        order = sorted(vals, key=lambda m: vals[m] if lower else -vals[m])
        return {m: i + 1 for i, m in enumerate(order)}

    return [rank(zero, True), rank(adapted, True), rank(hellaswag, False)]


def panel_slope(ax):
    cols = news_ranks()
    models = sorted(cols[0])
    for m in sorted(models, key=lambda m: m in HIGHLIGHT):  # highlighted lines drawn last, on top
        ys = [c[m] for c in cols]
        color = HIGHLIGHT.get(m, GRAY)
        strong = m in HIGHLIGHT
        ax.plot(range(3), ys, color=color, lw=2.8 if strong else 1.3, zorder=3 if strong else 1,
                marker="o", ms=6.5 if strong else 4.5, mec="white", mew=0.9, solid_capstyle="round")
        for x, y, ha in ((0, ys[0], "right"), (2, ys[2], "left")):
            ax.text(x + (-0.09 if ha == "right" else 0.09), y, NAMES[m], ha=ha, va="center",
                    fontsize=7.2, color=color if strong else MUTED, fontweight="bold" if strong else None)
    ax.set_xticks(range(3), ["Unadapted\nloss", "Adapted\nloss", "HellaSwag"], fontsize=8.5, color=INK)
    ax.set_xlim(-1.05, 3.05)
    ax.set_ylim(len(models) + 0.6, -0.4)
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    for x, text in ((0.5, "reorders"), (1.5, "converges")):
        ax.text(x, -0.25, text, ha="center", va="center", fontsize=8, color=BLUE, style="italic")
    ax.set_title("A.  Eleven models on news, ranked three ways", fontsize=9.5, loc="left", color=INK)


def panel_steering(ax):
    cells = _cells()
    for i, (corpus, bench, label) in enumerate(ROWS):
        before = cells[f"{corpus}__zero_shot"]["alignment"][bench]["spearman"]
        after = cells[f"{corpus}__adapted"]["alignment"][bench]["spearman"]
        y = len(ROWS) - 1 - i
        ax.annotate("", xy=(after, y), xytext=(before, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8, shrinkA=4, shrinkB=4))
        ax.plot(before, y, "o", ms=8, mfc="white", mec=MUTED, mew=1.4, zorder=3)
        ax.plot(after, y, "o", ms=8, color=BLUE, mec="white", mew=0.8, zorder=3)
        ax.text(after + 0.022, y, f"{after:.2f}", va="center", fontsize=8, color=INK, fontweight="bold")
    ax.axhline(1.5, color="#DDE1E6", lw=0.8)  # general text above, mathematics below
    ax.set_yticks(range(len(ROWS)), [lab for _, _, lab in reversed(ROWS)], fontsize=8)
    ax.set_xlim(0.4, 1.07)
    ax.set_xlabel("Rank correlation of BPB with the benchmark (Spearman, 11 models)", fontsize=8,
                  color=MUTED)
    ax.plot([], [], "o", mfc="white", mec=MUTED, label="before adaptation")
    ax.plot([], [], "o", color=BLUE, label="after adaptation")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color="#EEF0F3", lw=0.6)
    ax.set_title("B.  The text aims the ranking", fontsize=9.5, loc="left", color=INK)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#C9CED4"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.05, 1]})
    panel_slope(a)
    panel_steering(b)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(HEADLINE)
    fig.savefig(HEADLINE.with_suffix(".png"), dpi=200)  # for the README, which cannot show a PDF
    plt.close(fig)
    print(f"wrote {HEADLINE.relative_to(ROOT)} and its .png")


if __name__ == "__main__":
    main()
