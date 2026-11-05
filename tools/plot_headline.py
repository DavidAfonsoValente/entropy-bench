"""Figure 1: which signal, read before fine-tuning, predicts how the fifteen models finish?

Three slopegraphs, one per signal (GSM8K, MMLU-Pro, BPB on whole documents). Each line is one
model, from its rank on the signal (left) to its rank after fine-tuning on the news lede task
(right, ROUGE-L). Parallel lines mean the signal predicted the outcome; crossing lines mean it did
not. Signals come from tools/analyze_downstream.py's selector loader and ROUGE-L from
results/downstream.json, so nothing here is computed differently from the tables.

    python tools/plot_headline.py   # writes figures/fig_headline.pdf and .png (for the README)
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
from analyze_downstream import _selectors_17  # noqa: E402
from plot_steering import NAMES  # noqa: E402

HEADLINE = ROOT / "figures" / "fig_headline.pdf"
BLUE, ORANGE, GRAY, INK, MUTED = "#2E6DB4", "#D2762A", "#B3BAC3", "#1C2733", "#6B7785"
HIGHLIGHT = {"gemma-4-12B": BLUE, "Llama-3.2-1B": ORANGE}
PANELS = [("gsm8k", "GSM8K"), ("mmlu_pro", "MMLU-Pro"), ("doc_zero_shot_bpb", "BPB, whole documents")]


def rank(vals, lower):
    order = sorted(vals, key=lambda m: vals[m] if lower else -vals[m])
    return {m: i + 1 for i, m in enumerate(order)}


def panel(ax, signal, label, final, sel, labels_left):
    vals, lower = sel[signal]
    models = sorted(final)
    before = rank({m: vals[m] for m in models}, lower)
    after = rank(final, False)
    rho = spearmanr([before[m] for m in models], [after[m] for m in models])[0]
    for m in sorted(models, key=lambda m: m in HIGHLIGHT):
        color, strong = HIGHLIGHT.get(m, GRAY), m in HIGHLIGHT
        ax.plot([0, 1], [before[m], after[m]], color=color, lw=2.2 if strong else 1.0,
                marker="o", ms=5 if strong else 3.5, mec="white", mew=0.9, zorder=3 if strong else 1)
        if labels_left or strong:
            ax.text(-0.07, before[m], NAMES[m], ha="right", va="center", fontsize=6.3,
                    color=color if strong else MUTED, fontweight="bold" if strong else None)
    ax.set_xticks([0, 1], [label, "After\nfine-tuning"], fontsize=7.5, color=INK)
    ax.set_xlim(-0.9 if labels_left else -0.6, 1.15)
    ax.set_ylim(len(models) + 0.6, 0.4)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Spearman {rho:.2f}", fontsize=9, color=BLUE if signal.startswith("doc") else INK,
                 fontweight="bold")
    return rho


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    sel, _ = _selectors_17()
    final = json.load(open(ROOT / "results" / "downstream.json"))["cohort"]["rouge_l"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.1), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    for i, (ax, (signal, label)) in enumerate(zip(axes, PANELS)):
        panel(ax, signal, label.replace(", ", ",\n"), final, sel, labels_left=i == 0)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(HEADLINE, bbox_inches="tight")
    fig.savefig(HEADLINE.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {HEADLINE.relative_to(ROOT)} and its .png")


if __name__ == "__main__":
    main()
