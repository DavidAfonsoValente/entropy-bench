"""Figure: how often each selector picks the better of two similarly-sized models, and which of
those gaps actually resolves.

Two panels, because the paper's claim and the paper's picture were different objects. The left panel
is each selector's MARGINAL accuracy -- intuitive, but its intervals overlap heavily and a reader
cannot see from it what has been established. The right panel is the PAIRED differences against the
three baselines that matter, with zero as the reference: at seventeen bootstrap units exactly one of
them excludes zero, and that is the paper's result. Same hollow/filled encoding in both.

Reads ``results/cohort_extension.json`` (E3) and draws in-band pairwise selection accuracy for
every selector, under both scoring conventions, with the cluster-bootstrap intervals the artifact
already carries. A hollow marker means the interval still contains chance, so that selector has
not resolved -- the distinction the paper turns on, and the reason the figure exists rather than a
table of point estimates.

This regenerates the figure the paper includes at Sec. "The selectors compared"; every number is
read from the artifact, none is recomputed here.

Usage:
    python tools/plot_selectors.py
    python tools/plot_selectors.py --check    # verify the artifact against the paper's caption
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Drawn bottom-to-top, so the strongest selector ends up at the top of the figure.
SELECTORS = [
    ("gsm8k", "GSM8K"),
    ("mmlu_pro", "MMLU-Pro"),
    ("parameter_count", "Parameter count"),
    ("zero_shot_bpb", "Zero-shot BPB"),
    ("hellaswag", "HellaSwag"),
    ("matched_adapted_bpb", "Adapted BPB"),
]
CONVENTIONS = [("lenient", "Lenient", "#3B6FA0"), ("strict", "Strict", "#C07C33")]
CHANCE = 0.5

# Bottom-to-top again. The dagger marks the one difference computed after the fact rather than
# preregistered; the artifact carries that flag and the paper reports it in Limitations.
DIFFERENCES = [
    ("bpb_minus_zero_shot", "Zero-shot BPB$^{\\dagger}$"),
    ("bpb_minus_best_benchmark", "HellaSwag"),
    ("bpb_minus_parameter_count", "Parameter count"),
]


def load(path):
    band = json.load(open(path))["within_2x"]
    rows = []
    for key, label in SELECTORS:
        for conv, conv_label, colour in CONVENTIONS:
            cell = band[conv]["selectors"].get(key)
            if cell is None:
                continue
            rows.append({
                "selector": label, "convention": conv_label, "colour": colour,
                "accuracy": cell["pairwise_accuracy"],
                "lo": cell["ci_lo"], "hi": cell["ci_hi"],
                # `beats_chance` already excludes degenerate [x, x] intervals, which is why it is
                # read rather than recomputed from lo/hi: a perfect score collapses the interval
                # and would otherwise look like it had cleared chance on no evidence.
                "resolved": bool(cell["beats_chance"]),
                "n_pairs": cell["n_pairs"],
            })
    return rows, band["n_band_pairs"]


def load_differences(path):
    """The paired selector differences, which are what the paper actually claims."""
    band = json.load(open(path))["within_2x"]
    rows = []
    for key, label in DIFFERENCES:
        for conv, conv_label, colour in CONVENTIONS:
            cell = band[conv]["paired_differences"].get(key)
            if cell is None:
                continue
            rows.append({
                "difference": label, "convention": conv_label, "colour": colour,
                "observed": cell["observed"], "lo": cell["ci_lo"], "hi": cell["ci_hi"],
                "resolved": bool(cell["ci_excludes_zero"]),
                "preregistered": bool(cell.get("preregistered", False)),
                "key": key,
            })
    return rows


def draw(rows, diffs, out, figsize, scale=1.0):
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.0, 0.80], "wspace": 0.22})
    offset = {"Lenient": +0.17, "Strict": -0.17}
    for row in rows:
        y = [label for _, label in SELECTORS].index(row["selector"]) + offset[row["convention"]]
        ax.plot([row["lo"], row["hi"]], [y, y], color=row["colour"],
                lw=1.6 * scale, solid_capstyle="butt", zorder=2)
        ax.plot([row["accuracy"]], [y], marker="o", markersize=7 * scale,
                markerfacecolor=row["colour"] if row["resolved"] else "white",
                markeredgecolor=row["colour"], markeredgewidth=1.6 * scale, zorder=3)

    ax.axvline(CHANCE, color="#555555", lw=1.0 * scale, ls="--", zorder=1)
    ax.set_yticks(range(len(SELECTORS)))
    ax.set_yticklabels([label for _, label in SELECTORS], fontsize=9 * scale)
    ax.set_ylim(-0.6, len(SELECTORS) - 0.4)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Pairwise selection accuracy", fontsize=9 * scale)
    ax.tick_params(axis="x", labelsize=8.5 * scale)
    ax.grid(axis="x", color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    # Proxy handles, because in this figure a hollow marker *means* "did not resolve" -- letting
    # the legend inherit the first plotted point would draw a hollow key and contradict the
    # encoding it is meant to explain.
    handles = [plt.Line2D([], [], color=colour, marker="o", markersize=7 * scale,
                          markerfacecolor=colour, markeredgecolor=colour,
                          lw=1.6 * scale, label=conv_label)
               for _, conv_label, colour in CONVENTIONS]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5 * scale, frameon=False,
              title="Scoring convention", title_fontsize=8.5 * scale)
    ax.set_title("A.  Where each selector lands", fontsize=9.5 * scale, loc="left", pad=8)

    # Panel B: the paired differences. This is the estimand the paper claims; the left panel is
    # context for it.
    labels2 = [label for _, label in DIFFERENCES]
    for row in diffs:
        y = labels2.index(row["difference"]) + offset[row["convention"]]
        ax2.plot([row["lo"], row["hi"]], [y, y], color=row["colour"],
                 lw=1.6 * scale, solid_capstyle="butt", zorder=2)
        ax2.plot([row["observed"]], [y], marker="o", markersize=7 * scale,
                 markerfacecolor=row["colour"] if row["resolved"] else "white",
                 markeredgecolor=row["colour"], markeredgewidth=1.6 * scale, zorder=3)
    ax2.axvline(0.0, color="#555555", lw=1.0 * scale, ls="--", zorder=1)
    ax2.set_yticks(range(len(labels2)))
    # Labels on the OUTER edge: between the panels they overlap panel A's data region.
    ax2.yaxis.tick_right()
    ax2.set_yticklabels(labels2, fontsize=9 * scale)
    ax2.set_ylim(-0.6, len(labels2) - 0.4)
    ax2.set_xlabel("Paired difference in selection accuracy", fontsize=9 * scale)
    ax2.tick_params(axis="x", labelsize=8.5 * scale)
    ax2.grid(axis="x", color="#DDDDDD", lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("B.  What actually resolves", fontsize=9.5 * scale, loc="left", pad=8)

    # The hollow/filled encoding and the dagger are explained in the LaTeX caption rather than
    # here: an in-figure note fights bbox_inches="tight" and lands on the x-label.
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# The caption the paper prints. Asserted here so the figure and the prose cannot drift apart --
# the same reason every other number in this repository is gated rather than trusted.
CAPTION = {
    ("GSM8K", "Lenient"): (0.359, False),
    ("MMLU-Pro", "Lenient"): (0.400, False),
    ("Adapted BPB", "Lenient"): (0.900, True),
    ("HellaSwag", "Lenient"): (0.825, True),
    ("HellaSwag", "Strict"): (None, True),
}


# Panel B's six values, which the caption also quotes. Exactly one must resolve.
CAPTION_DIFFS = {
    ("Parameter count", "Lenient"): (0.250, True),
    ("Parameter count", "Strict"): (0.100, False),
    ("HellaSwag", "Lenient"): (0.075, False),
    ("HellaSwag", "Strict"): (0.025, False),
    ("Zero-shot BPB$^{\\dagger}$", "Lenient"): (0.150, False),
    ("Zero-shot BPB$^{\\dagger}$", "Strict"): (0.000, False),
}


def check(rows, n_band_pairs, diffs):
    problems = 0
    if n_band_pairs != 40:
        print(f"PROBLEM: expected 40 in-band pairs, artifact says {n_band_pairs}")
        problems += 1
    index = {(r["selector"], r["convention"]): r for r in rows}
    for key, (accuracy, resolved) in CAPTION.items():
        row = index.get(key)
        if row is None:
            print(f"PROBLEM: {key} absent from the artifact")
            problems += 1
            continue
        if accuracy is not None and abs(row["accuracy"] - accuracy) > 5e-4:
            print(f"PROBLEM: {key} accuracy {row['accuracy']:.4f}, caption says {accuracy}")
            problems += 1
        if row["resolved"] is not resolved:
            print(f"PROBLEM: {key} resolved={row['resolved']}, caption says {resolved}")
            problems += 1
    index2 = {(r["difference"], r["convention"]): r for r in diffs}
    for key, (observed, resolved) in CAPTION_DIFFS.items():
        row = index2.get(key)
        if row is None:
            print(f"PROBLEM: paired difference {key} absent from the artifact")
            problems += 1
            continue
        if abs(row["observed"] - observed) > 5e-4:
            print(f"PROBLEM: {key} observed {row['observed']:.4f}, caption says {observed}")
            problems += 1
        if row["resolved"] is not resolved:
            print(f"PROBLEM: {key} resolved={row['resolved']}, caption says {resolved}")
            problems += 1
    n_resolved = sum(1 for r in diffs if r["resolved"])
    if n_resolved != 1:
        print(f"PROBLEM: exactly one paired difference should resolve, artifact has {n_resolved}")
        problems += 1
    if any(r["preregistered"] for r in diffs if r["key"] == "bpb_minus_zero_shot"):
        print("PROBLEM: bpb_minus_zero_shot is flagged preregistered; the dagger would be wrong")
        problems += 1
    print(f"selectors figure: PROBLEMS {problems}")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/cohort_extension.json")
    p.add_argument("--out", default="figures/fig_selectors.pdf")
    p.add_argument("--check", action="store_true", help="verify against the caption, draw nothing")
    a = p.parse_args()
    rows, n_band_pairs = load(a.input)
    diffs = load_differences(a.input)
    if a.check:
        raise SystemExit(1 if check(rows, n_band_pairs, diffs) else 0)
    draw(rows, diffs, a.out, (9.0, 3.05), scale=1.28)


if __name__ == "__main__":
    main()
