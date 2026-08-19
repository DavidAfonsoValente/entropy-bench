"""Plot pairwise agreement of adapted BPB across the three evaluation corpora.

Reads the 33 JSON cells written by ``tools/domain_transfer_eval.py`` and produces a
three-panel scatter plot.  Raw BPB shows the domain-difficulty shift; Pearson r and
Spearman rho summarize value and rank agreement respectively.
"""
import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SHORT = {
    "google/gemma-4-31B": "Gemma-4-31B",
    "google/gemma-4-12B": "Gemma-4-12B",
    "mistralai/Ministral-3-14B-Base-2512": "Ministral-3-14B",
    "Qwen/Qwen3.5-35B-A3B-Base": "Qwen-3.5-35B-MoE",
    "Qwen/Qwen3.5-9B-Base": "Qwen-3.5-9B",
    "Qwen/Qwen2.5-7B": "Qwen-2.5-7B",
    "Qwen/Qwen3.5-4B-Base": "Qwen-3.5-4B",
    "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B": "Qwen-2.5-1.5B",
    "LiquidAI/LFM2.5-1.2B-Base": "LiquidAI-LFM2.5",
    "Qwen/Qwen2.5-0.5B": "Qwen-2.5-0.5B",
}
FAMILY = {
    "Gemma": ("#3B6FA0", "o"),
    "Qwen": ("#3F8457", "s"),
    "Mistral": ("#C07C33", "D"),
    "Llama": ("#7965A8", "^"),
    "LiquidAI": ("#B34D5A", "P"),
}
LABELS = {
    "Gemma-4-31B": "G31",
    "Gemma-4-12B": "G12",
    "Ministral-3-14B": "M14",
    "Qwen-3.5-35B-MoE": "Q35",
    "LiquidAI-LFM2.5": "LFM",
}
CORPUS_LABEL = {"news": "News", "reddit": "Reddit", "hackernews": "Hacker News"}
PAIRS = (("news", "reddit"), ("news", "hackernews"), ("reddit", "hackernews"))


def family(name):
    if name.startswith("Gemma"):
        return "Gemma"
    if name.startswith("Qwen"):
        return "Qwen"
    if name.startswith("Ministral"):
        return "Mistral"
    if name.startswith("Llama"):
        return "Llama"
    return "LiquidAI"


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0] * len(values)
    for rank, i in enumerate(order, 1):
        out[i] = rank
    return out


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den


def load_cells(src):
    cells = {}
    for filename in sorted(glob.glob(str(Path(src) / "*.json"))):
        with open(filename) as f:
            row = json.load(f)
        corpus = Path(filename).name.split("__")[0]
        name = SHORT.get(row["model_id"], row["model_id"])
        cells.setdefault(name, {})[corpus] = row["adapted_bpb"]
    complete = {m: v for m, v in cells.items() if all(c in v for c in CORPUS_LABEL)}
    if len(complete) != 11:
        raise SystemExit(f"expected 11 complete models, found {len(complete)}")
    return complete


def draw(cells, out, figsize, font_scale=1.0):
    names = sorted(cells, key=lambda m: cells[m]["news"])
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (cx, cy) in zip(axes, PAIRS):
        xs = [cells[m][cx] for m in names]
        ys = [cells[m][cy] for m in names]
        for fam, (color, marker) in FAMILY.items():
            idx = [i for i, m in enumerate(names) if family(m) == fam]
            ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=38 * font_scale,
                       c=color, marker=marker, edgecolor="white", linewidth=0.6,
                       label=fam, zorder=3)

        fit = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 100)
        ax.plot(xline, fit[0] * xline + fit[1], color="#7B8794", lw=1.1,
                ls="--", zorder=1)

        rho = pearson(ranks(xs), ranks(ys))
        r = pearson(xs, ys)
        ax.text(0.04, 0.95, rf"$r={r:.3f}$" + "\n" + rf"$\rho={rho:.3f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5 * font_scale,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD2D9", alpha=0.92))

        offsets = {
            ("news", "reddit"): {"G31": (4, -11), "G12": (-18, 6), "M14": (4, -13),
                                 "Q35": (5, 6), "LFM": (4, 5)},
            ("news", "hackernews"): {"G31": (4, -11), "G12": (-18, 7),
                                     "M14": (-20, -15), "Q35": (5, -13), "LFM": (4, 5)},
            ("reddit", "hackernews"): {"G31": (4, -11), "G12": (-18, 7),
                                       "M14": (-20, -15), "Q35": (5, -13), "LFM": (4, 5)},
        }[(cx, cy)]
        for i, name in enumerate(names):
            if name in LABELS:
                label = LABELS[name]
                dx, dy = offsets[label]
                ax.annotate(label, (xs[i], ys[i]), xytext=(dx, dy),
                            textcoords="offset points", fontsize=6.7 * font_scale,
                            color="#273746")

        ax.set_xlabel(f"{CORPUS_LABEL[cx]} adapted BPB", fontsize=9 * font_scale)
        ax.set_ylabel(f"{CORPUS_LABEL[cy]} adapted BPB", fontsize=9 * font_scale)
        ax.tick_params(labelsize=7.5 * font_scale)
        ax.grid(alpha=0.22, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(0.08)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=8 * font_scale, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.6)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/domain_transfer")
    p.add_argument("--out", default="figures/fig_cross_corpus.pdf")
    p.add_argument("--slide-out", default="figures/fig_cross_corpus_slide.pdf")
    a = p.parse_args()
    cells = load_cells(a.input)
    draw(cells, a.out, (10.0, 3.25), 1.0)
    draw(cells, a.slide_out, (10.8, 3.15), 1.08)


if __name__ == "__main__":
    main()
