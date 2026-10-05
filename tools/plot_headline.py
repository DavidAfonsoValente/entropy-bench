"""Figure 1 and the steering figure.

Figure 1 (fig_headline):
A (prediction): the fifteen fine-tuned models, each placed by its adapted-BPB rank (measured before
   any task) against the rank its fine-tuned system finishes; GSM8K's ranks are drawn for contrast.
B (slopegraph): the eleven-model cohort ranked three ways -- unadapted news BPB, adapted news BPB,
   HellaSwag. Lines cross between the first two columns (adaptation reorders the models) and run
   nearly parallel between the last two (the adapted order is the language benchmark's).
Steering figure (fig_steering): rank correlation between BPB and each benchmark before and after
   adaptation. General text pulls the ranking onto HellaSwag; arXiv mathematics onto GSM8K/MMLU-Pro.

Fine-tune ranks come from results/downstream.json and the selectors from analyze_downstream, exactly
as verify_paper_numbers.py reads them; BPB alignment from results/alignment_matrix.json, HellaSwag
from results/combined_bpb_vs_static.json.

Usage:
    python tools/plot_headline.py   # writes figures/fig_headline.{pdf,png}, fig_steering.pdf and
                                    # fig_reorder_slide.pdf (the talk's two-panel version)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = ROOT / "figures" / "fig_headline.pdf"
STEERING = ROOT / "figures" / "fig_steering.pdf"
REORDER_SLIDE = ROOT / "figures" / "fig_reorder_slide.pdf"
BLUE, ORANGE, GRAY, INK, MUTED = "#2E6DB4", "#D2762A", "#B3BAC3", "#1C2733", "#6B7785"
HIGHLIGHT = {"gemma-4-12B": BLUE, "LFM2.5-1.2B": ORANGE}
NAMES = {"gemma-4-31B": "Gemma-4-31B", "gemma-4-12B": "Gemma-4-12B", "Qwen3.5-35B-MoE": "Qwen-3.5-35B-MoE",
         "Ministral-3-14B": "Ministral-3-14B", "Qwen3.5-9B": "Qwen-3.5-9B", "Qwen2.5-7B": "Qwen-2.5-7B",
         "Qwen3.5-4B": "Qwen-3.5-4B", "Llama-3.2-1B": "Llama-3.2-1B", "Qwen2.5-1.5B": "Qwen-2.5-1.5B",
         "LFM2.5-1.2B": "LFM2.5-1.2B", "Qwen2.5-0.5B": "Qwen-2.5-0.5B", "Falcon3-10B": "Falcon3-10B",
         "Falcon3-7B": "Falcon3-7B", "Mistral-Nemo-12B": "Mistral-Nemo-12B", "Qwen2.5-14B": "Qwen-2.5-14B",
         "Qwen2.5-3B": "Qwen-2.5-3B", "SmolLM2-1.7B": "SmolLM2-1.7B"}
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


def finetune_ranks():
    """Rank (1 = best) of the fifteen fine-tuned models by adapted BPB, GSM8K and fine-tuned ROUGE-L."""
    sys.path.insert(0, str(ROOT / "tools"))
    from analyze_downstream import _selectors_17
    sel, _ = _selectors_17()
    ft = json.load(open(ROOT / "results" / "downstream.json"))["cohort"]["rouge_l"]
    models = sorted(ft)

    def rank(vals, lower):
        order = sorted(models, key=lambda m: vals[m] if lower else -vals[m])
        return {m: i + 1 for i, m in enumerate(order)}

    return (rank(sel["matched_adapted_bpb"][0], True), rank(sel["gsm8k"][0], False),
            rank(ft, False))


def _spearman(a, b):
    n = len(a)
    return 1 - 6 * sum((a[m] - b[m]) ** 2 for m in a) / (n * (n * n - 1))


def panel_prediction(ax):
    """One row per model, ordered by where its fine-tuned system finishes. A blue dot on the diagonal
    means the measure, taken before any task, put the model exactly where the fine-tune did."""
    bpb, gsm, ft = finetune_ranks()
    n = len(ft)
    rows = sorted(ft, key=lambda m: ft[m])
    ax.fill_between([0.5, n + 0.5], [-0.5, n - 0.5], [1.5, n + 1.5], color="#E8F0F9", zorder=0, lw=0)
    for m in rows:
        y = ft[m]
        ax.plot([bpb[m], gsm[m]], [y, y], color="#D5DAE0", lw=1.2, zorder=1)
        ax.plot(gsm[m], y, "o", ms=6, mfc="white", mec=GRAY, mew=1.4, zorder=2)
        ax.plot(bpb[m], y, "o", ms=7, color=BLUE, mec="white", mew=0.8, zorder=3)
    ax.set_yticks([ft[m] for m in rows], [f"{ft[m]}. {NAMES[m]}" for m in rows], fontsize=8.2)
    for lab, m in zip(ax.get_yticklabels(), rows):
        if m == "gemma-4-12B":
            lab.set_color(BLUE); lab.set_fontweight("bold")
    ax.plot([], [], "o", color=BLUE, label=f"adapted BPB  ($\\rho$ = {_spearman(bpb, ft):.2f})")
    ax.plot([], [], "o", mfc="white", mec=GRAY, label=f"GSM8K  ($\\rho$ = {_spearman(gsm, ft):.2f})")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.03),
              handletextpad=0.3)
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylim(n + 0.7, 0.3)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_xlabel("Rank the measure gives the model (1 = best); shaded = within one rank", fontsize=8.5,
                  color=MUTED)
    ax.tick_params(length=0, labelsize=8.5, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("A.  Fifteen models, ordered by their fine-tuned finish", fontsize=10.5, loc="left",
                 color=INK)


def panel_slope(ax, title="B.  Eleven models on news, ranked three ways"):
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
                    fontsize=8.2, color=color if strong else MUTED, fontweight="bold" if strong else None)
    ax.set_xticks(range(3), ["Unadapted\nBPB", "Adapted\nBPB", "HellaSwag"], fontsize=9.5, color=INK)
    ax.set_xlim(-1.05, 3.05)
    ax.set_ylim(len(models) + 0.6, -0.4)
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    for x, text in ((0.5, "reorders"), (1.5, "converges")):
        ax.text(x, -0.25, text, ha="center", va="center", fontsize=8, color=BLUE, style="italic")
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)


def panel_steering(ax, title="The text aims the ranking"):
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
    ax.set_xlabel("Rank agreement of BPB with the benchmark (Spearman, 11 models)", fontsize=9,
                  color=MUTED)
    ax.plot([], [], "o", mfc="white", mec=MUTED, label="before adaptation")
    ax.plot([], [], "o", color=BLUE, label="after adaptation")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color="#EEF0F3", lw=0.6)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#C9CED4"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    panel_prediction(a)
    panel_slope(b)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(HEADLINE)
    fig.savefig(HEADLINE.with_suffix(".png"), dpi=200)  # for the README, which cannot show a PDF
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    panel_steering(ax)
    fig.tight_layout()
    fig.savefig(STEERING)
    plt.close(fig)
    # The talk keeps reordering and steering on one slide; the paper splits them across figures.
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.05, 1]})
    panel_slope(a, "A.  Eleven models on news, ranked three ways")
    panel_steering(b, "B.  The text aims the ranking")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(REORDER_SLIDE)
    plt.close(fig)
    print(f"wrote {HEADLINE.relative_to(ROOT)}, its .png, {STEERING.relative_to(ROOT)} and "
          f"{REORDER_SLIDE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
