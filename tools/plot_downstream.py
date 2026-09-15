#!/usr/bin/env python3
"""Figure: does the selector's pick build the better fine-tuned system? One point per in-band pair.

For every size-matched pair of the fifteen fine-tuned models (E11b), the pair is oriented so the
selector prefers the first model: x is how strongly it prefers it, y is how much better that model's
fine-tuned system actually scored. A point above zero is a pair the selector got right. Adapted BPB is
drawn beside GSM8K, the benchmark a practitioner is most likely to read instead.

Filled markers are pairs the fine-tune itself resolves (paired bootstrap over articles); hollow ones
it cannot separate -- the same encoding as Figure 1.

Reads results/downstream.json; nothing is recomputed. ``--check`` asserts the counts the paper quotes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_downstream import _selectors_17  # noqa: E402

OUT = ROOT / "figures" / "fig_downstream.pdf"
HELLASWAG_EXPECT = (34, 39)  # set from the first render, then asserted on every later one
BLUE, ORANGE, INK, GRID = "#3B6FA0", "#C07C33", "#555555", "#DDDDDD"
GREEN = "#3F8457"
# HellaSwag is the benchmark the paper does NOT separate from, so it belongs in the picture beside
# the one it does: a reader deciding what to trust needs to see both comparisons, not the easy one.
PANELS = [("matched_adapted_bpb", "Adapted BPB", "Adapted-BPB gap (% of better)", BLUE),
          ("hellaswag", "HellaSwag", "HellaSwag gap (points)", GREEN),
          ("gsm8k", "GSM8K", "GSM8K gap (points)", ORANGE)]


def oriented_points(name: str) -> list[tuple[float, float, bool, str]]:
    c = json.loads((ROOT / "results" / "downstream.json").read_text())["cohort"]
    sel, _ = _selectors_17()
    vals, lower = sel[name]
    pts = []
    for p in c["pairs"]:
        a, b = p["pair"].split("|")
        if vals[a] == vals[b]:
            continue                                   # a tie orders nothing
        pick_a = vals[a] < vals[b] if lower else vals[a] > vals[b]
        if name == "matched_adapted_bpb":
            gap = abs(vals[a] - vals[b]) / min(vals[a], vals[b]) * 100
        else:
            gap = abs(vals[a] - vals[b]) * 100
        outcome = p["diff_a_minus_b"] if pick_a else -p["diff_a_minus_b"]
        pts.append((gap, outcome, p["resolved"], p["pair"]))
    return pts


def draw(out: Path) -> dict[str, tuple[int, int]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.2, 2.35), sharey=True)
    counts = {}
    for ax, (name, title, xlabel, colour) in zip(axes, PANELS):
        pts = oriented_points(name)
        right = sum(1 for _, y, _, _ in pts if y > 0)
        counts[name] = (right, len(pts))
        ax.axhline(0, color=INK, lw=1.0, ls="--", zorder=1)
        ax.axhspan(-1, 0, color="#F3F3F3", zorder=0)
        for x, y, res, _ in pts:
            ax.plot(x, y, "o", ms=6.5, mec=colour, mew=1.4,
                    mfc=colour if res else "white", zorder=3)
        ax.set_title(title, fontsize=11, loc="left")
        # A white plate: the count sits in the corner and a mark can land under it.
        ax.text(0.97, 0.95, f"{right} of {len(pts)} right", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=INK, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.tick_params(labelsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("ROUGE-L, pick minus other", fontsize=10)
    lo = min(y for n, *_ in PANELS for _, y, _, _ in oriented_points(n))
    hi = max(y for n, *_ in PANELS for _, y, _, _ in oriented_points(n))
    pad = 0.1 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].text(0.03, 0.05, "below the line:\nthe pick built the worse system",
                 transform=axes[0].transAxes, ha="left", va="bottom", fontsize=8, color=INK)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\\n")[0])
    ap.add_argument("--check", action="store_true", help="assert the counts the paper quotes")
    a = ap.parse_args()
    if a.check:
        problems = []
        for name, expect in (("matched_adapted_bpb", (37, 39)), ("hellaswag", HELLASWAG_EXPECT),
                            ("gsm8k", (18, 38))):
            pts = oriented_points(name)
            got = (sum(1 for _, y, _, _ in pts if y > 0), len(pts))
            if got != expect:
                problems.append(f"{name}: {got} != {expect}")
        if not OUT.exists() or OUT.stat().st_size < 1000:
            problems.append("figures/fig_downstream.pdf missing")
        for p in problems:
            print("  " + p)
        print("downstream figure: PROBLEMS", len(problems))
        sys.exit(1 if problems else 0)
    print("wrote", OUT.relative_to(ROOT), draw(OUT))


if __name__ == "__main__":
    main()
