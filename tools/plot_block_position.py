"""Figure: base loss and adaptation gain versus position in the packed block.

Shows why the zero-shot BPB ranking is partly a cold-start artefact -- Gemma-4-12B is worse
than a 1.5B Qwen for roughly the first half of a block and better once it has context.

  python tools/plot_block_position.py --out figures/fig_block_position.pdf \
      results/token_gain_bpb/*.json

The inputs must be the byte-normalised cells: the default --units bpb reads
mean_base_bits_per_byte, which results/token_gain/ (nats only) does not carry. Verified 2026-09-11
against the caption -- Gemma-4-12B opens a block at 2.83 bits/byte and closes at 0.72, Qwen-2.5-1.5B
at 2.04 and 0.77.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# bucket label -> representative position (geometric midpoint) and width in tokens
BUCKETS = {"0-0": (0.5, 1), "1-1": (1.5, 1), "2-3": (3, 2), "4-7": (6, 4), "8-15": (12, 8),
           "16-31": (24, 16), "32-63": (48, 32), "64-127": (96, 64),
           "128-255": (192, 128), "256-510": (384, 255)}

STYLE = {
    "google/gemma-4-12B":        dict(color="#3B6FA0", marker="o", label="Gemma-4-12B (28.7% gain)"),
    "LiquidAI/LFM2.5-1.2B-Base": dict(color="#C07C33", marker="s", label="LFM2.5-1.2B (36.5% gain)"),
    "Qwen/Qwen2.5-1.5B":         dict(color="#3F8457", marker="^", label="Qwen-2.5-1.5B (3.2% gain)"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--panels", choices=["both", "base"], default="both",
                   help="'base' gives a single larger panel for slide use")
    p.add_argument("--fontsize", type=float, default=None)
    p.add_argument("--units", choices=["bpb", "nats"], default="bpb",
                   help="byte-normalized BPB (default) or original nats/token")
    a = p.parse_args()
    if a.fontsize:
        matplotlib.rcParams.update({"font.size": a.fontsize})

    runs = []
    for f in a.files:
        d = json.load(open(f))
        if d.get("gain_by_block_position"):
            runs.append(d)
    runs.sort(key=lambda d: -d["relative_nats_reduction_pct"])

    if a.panels == "base":
        fig, ax1 = plt.subplots(1, 1, figsize=(6.8, 3.4))
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for d in runs:
        st = STYLE.get(d["model_id"], dict(color="#6B7785", marker="x",
                                           label=d["model_id"].split("/")[-1]))
        bp = d["gain_by_block_position"]
        xs = [BUCKETS[k][0] for k in bp if k in BUCKETS]
        if a.units == "bpb":
            missing = [k for k in bp if k in BUCKETS and "mean_base_bits_per_byte" not in bp[k]]
            if missing:
                raise SystemExit("byte-normalized fields missing; run normalize_block_position.py")
            base = [bp[k]["mean_base_bits_per_byte"] for k in bp if k in BUCKETS]
            gain = [bp[k]["mean_gain_bits_per_byte"] for k in bp if k in BUCKETS]
        else:
            base = [bp[k]["mean_base_loss"] for k in bp if k in BUCKETS]
            gain = [bp[k]["mean_gain"] for k in bp if k in BUCKETS]
        kw = dict(color=st["color"], marker=st["marker"], ms=4.5, lw=1.8)
        ax1.plot(xs, base, label=st["label"], **kw)
        if ax2 is not None:
            ax2.plot(xs, gain, **kw)

    # Mark where Gemma overtakes the far smaller Qwen: the ranking of the *base* models
    # depends on how much context they are given, which is the whole point of the figure.
    g = next((d for d in runs if d["model_id"] == "google/gemma-4-12B"), None)
    q = next((d for d in runs if d["model_id"] == "Qwen/Qwen2.5-1.5B"), None)
    if g and q:
        gb = g["gain_by_block_position"]
        qb = q["gain_by_block_position"]
        metric = "mean_base_bits_per_byte" if a.units == "bpb" else "mean_base_loss"
        cross = next((k for k in BUCKETS
                      if k in gb and k in qb and metric in gb[k] and metric in qb[k]
                      and gb[k][metric] < qb[k][metric]), None)
        if cross:
            x = BUCKETS[cross][0]
            y = gb[cross][metric]
            ax1.axvline(x, color="#6B7785", lw=0.8, ls="--", alpha=0.7)
            ax1.annotate("Gemma-4-12B overtakes\nQwen-2.5-1.5B here",
                         xy=(x, y), xytext=(-50, 88), textcoords="offset points",
                         fontsize=8.5, color="#1C2733", ha="center",
                         arrowprops=dict(arrowstyle="->", lw=0.8, color="#6B7785",
                                         connectionstyle="arc3,rad=-0.25"))

    unit = "bits/byte" if a.units == "bpb" else "nats/token"
    panels = [(ax1, f"Base-model loss ({unit})", "Before adaptation")]
    if ax2 is not None:
        panels.append((ax2, f"Gain from adaptation ({unit})", "Loss reduction from adaptation"))
    for ax, ylab, title in panels:
        ax.set_xscale("log")
        ax.set_xlabel("Position in block (log scale)", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    if a.panels == "base":
        ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
        fig.tight_layout()
    else:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
