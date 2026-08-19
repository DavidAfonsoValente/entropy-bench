"""Plot fixed-target BPB as the available inference context increases."""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SHORT = {
    "google/gemma-4-12B": "Gemma-4-12B",
    "Qwen/Qwen2.5-1.5B": "Qwen-2.5-1.5B",
}


def load(src):
    rows = []
    for filename in glob.glob(str(Path(src) / "*.json")):
        with open(filename) as f:
            row = json.load(f)
        if row.get("model_id") in SHORT:
            rows.append(row)
    rows.sort(key=lambda d: 0 if d["model_id"].startswith("google/") else 1)
    if len(rows) != 2:
        raise SystemExit(f"expected Gemma and Qwen results, found {len(rows)}")
    return rows


def draw(rows, out, figsize, scale):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, result in zip(axes, rows):
        contexts = sorted(int(k) for k in result["by_context_length"])
        vals = result["by_context_length"]
        base = [vals[str(c)]["base_bpb"] for c in contexts]
        adapted = [vals[str(c)]["adapted_bpb"] for c in contexts]
        ax.plot(contexts, base, color="#7B8794", marker="o", ms=4 * scale,
                lw=1.6, ls="--", label="Base")
        ax.plot(contexts, adapted, color="#3B6FA0", marker="o", ms=4 * scale,
                lw=1.8, label="Adapted")
        ax.axvline(512, color="#C07C33", lw=1.0, ls=":")
        ax.text(512, 0.98, "512-token\ntraining window", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=7.2 * scale, color="#9B641F")
        reduction = vals[str(max(contexts))]["relative_reduction_pct"]
        ax.text(0.97, 0.07, f"gain at 2,048: {reduction:.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.8 * scale,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD2D9"))
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts)
        ax.set_xticklabels([str(c) if c < 1000 else f"{c // 1024}k" for c in contexts],
                           fontsize=7.5 * scale)
        ax.set_xlabel("Available context tokens", fontsize=8.5 * scale)
        ax.set_ylabel("BPB on the same target windows", fontsize=8.5 * scale)
        ax.set_title(SHORT[result["model_id"]], fontsize=10 * scale)
        ax.tick_params(axis="y", labelsize=7.5 * scale)
        ax.grid(alpha=0.22, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8 * scale, loc="upper right")
    fig.tight_layout(w_pad=1.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/context_length")
    p.add_argument("--out", default="figures/fig_context_length.pdf")
    p.add_argument("--slide-out", default="figures/fig_context_length_slide.pdf")
    a = p.parse_args()
    rows = load(a.input)
    draw(rows, a.out, (7.4, 3.0), 1.0)
    draw(rows, a.slide_out, (8.8, 3.15), 1.12)


if __name__ == "__main__":
    main()
