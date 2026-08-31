"""Turn the domain-transfer cells into the two results the team asked for.

1. Does the BPB ordering hold across domains? Rank the cohort by adapted BPB within each
   corpus and compare the rankings (Spearman, plus exact-rank agreement).
2. What did the packing fix do? Compare news-with-fix against the published news numbers,
   which were produced with the injected document-start token scored.

  python tools/report_domain_transfer.py --dir $WORK/dt_out [--out-tex results_dt.tex]
"""
import argparse
import json
from pathlib import Path

# published news run (pre-fix): zero-shot -> adapted BPB, from results/legacy_sweep/summary.json
PUBLISHED = {
    "google/gemma-4-31B": (0.7534, 0.5830), "google/gemma-4-12B": (0.8667, 0.6180),
    "mistralai/Ministral-3-14B-Base-2512": (0.7230, 0.6512),
    "Qwen/Qwen3.5-35B-A3B-Base": (0.7002, 0.6625), "Qwen/Qwen3.5-9B-Base": (0.7332, 0.6791),
    "Qwen/Qwen2.5-7B": (0.7533, 0.7284), "Qwen/Qwen3.5-4B-Base": (0.7746, 0.7565),
    "meta-llama/Llama-3.2-1B": (0.8142, 0.7725), "Qwen/Qwen2.5-1.5B": (0.8292, 0.8025),
    "LiquidAI/LFM2.5-1.2B-Base": (1.3461, 0.8543), "Qwen/Qwen2.5-0.5B": (0.9308, 0.9215),
}
SHORT = {
    "google/gemma-4-31B": "Gemma-4-31B", "google/gemma-4-12B": "Gemma-4-12B",
    "mistralai/Ministral-3-14B-Base-2512": "Ministral-3-14B",
    "Qwen/Qwen3.5-35B-A3B-Base": "Qwen-3.5-35B-MoE", "Qwen/Qwen3.5-9B-Base": "Qwen-3.5-9B",
    "Qwen/Qwen2.5-7B": "Qwen-2.5-7B", "Qwen/Qwen3.5-4B-Base": "Qwen-3.5-4B",
    "meta-llama/Llama-3.2-1B": "Llama-3.2-1B", "Qwen/Qwen2.5-1.5B": "Qwen-2.5-1.5B",
    "LiquidAI/LFM2.5-1.2B-Base": "LiquidAI-LFM2.5", "Qwen/Qwen2.5-0.5B": "Qwen-2.5-0.5B",
}
# every model in the cohort, largest first. Must stay complete: an omission here silently
# drops rows from both printed tables rather than erroring.
ORDER = ["Qwen-3.5-35B-MoE", "Gemma-4-31B", "Ministral-3-14B", "Gemma-4-12B", "Qwen-3.5-9B",
         "Qwen-2.5-7B", "Qwen-3.5-4B", "Qwen-2.5-1.5B", "LiquidAI-LFM2.5", "Llama-3.2-1B",
         "Qwen-2.5-0.5B"]
assert set(ORDER) == set(SHORT.values()), sorted(set(SHORT.values()) - set(ORDER))
CORPORA = ["news", "reddit", "hackernews"]


def ranks(pairs):
    """pairs: {name: value}, lower is better -> {name: rank}."""
    o = sorted(pairs, key=lambda k: pairs[k])
    return {k: i + 1 for i, k in enumerate(o)}


def spearman(a, b):
    common = [k for k in a if k in b]
    n = len(common)
    if n < 3:
        return None
    ra, rb = ranks({k: a[k] for k in common}), ranks({k: b[k] for k in common})
    d2 = sum((ra[k] - rb[k]) ** 2 for k in common)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out-tex", default=None)
    a = p.parse_args()

    cells = {}
    for f in sorted(Path(a.dir).glob("*.json")):
        d = json.load(open(f))
        corpus = f.name.split("__")[0]
        cells.setdefault(corpus, {})[SHORT.get(d["model_id"], d["model_id"])] = d

    print("cells loaded:", {k: len(v) for k, v in cells.items()})
    print()

    # ---------------- main table
    hdr = "%-18s" % "Model"
    for c in CORPORA:
        hdr += " | %-22s" % c
    print(hdr)
    print("%-18s" % "" + (" | %-7s %-7s %-6s" % ("zero", "adapt", "red%")) * len(CORPORA))
    print("-" * 100)
    for m in ORDER:
        row = "%-18s" % m
        for c in CORPORA:
            d = cells.get(c, {}).get(m)
            if d is None:
                row += " | %-7s %-7s %-6s" % ("--", "--", "--")
            else:
                row += " | %-7.4f %-7.4f %-6.2f" % (
                    d["zero_shot_bpb"], d.get("adapted_bpb", float("nan")),
                    d.get("reduction_pct", float("nan")))
        print(row)

    # ---------------- 1. ordering stability across domains
    print()
    print("=== Ordering stability (adapted BPB rank) ===")
    adapted = {c: {m: d["adapted_bpb"] for m, d in cells.get(c, {}).items()
                   if "adapted_bpb" in d} for c in CORPORA}
    for i, c1 in enumerate(CORPORA):
        for c2 in CORPORA[i + 1:]:
            s = spearman(adapted.get(c1, {}), adapted.get(c2, {}))
            if s is None:
                continue
            # rank within the common subset only: ranking each corpus over its own model
            # set makes every rank below a missing model shift by one, which reads as
            # disagreement when there is none.
            common = [m for m in adapted[c1] if m in adapted[c2]]
            r1 = ranks({m: adapted[c1][m] for m in common})
            r2 = ranks({m: adapted[c2][m] for m in common})
            same = sum(r1[m] == r2[m] for m in common)
            maxd = max(abs(r1[m] - r2[m]) for m in common)
            print("  %-11s vs %-11s  spearman=%+.3f  identical=%d/%d  max shift=%d"
                  % (c1, c2, s, same, len(common), maxd))
            # name the disagreements: the paper asserts *which* models swap, so derive it
            # here rather than reading it off the order lines by eye.
            moved = sorted((m for m in common if r1[m] != r2[m]), key=lambda m: r1[m])
            for m in moved:
                print("        %-17s rank %d -> %d" % (m, r1[m], r2[m]))
    for c in CORPORA:
        if adapted.get(c):
            r = ranks(adapted[c])
            print("  %-11s order: %s" % (c, " < ".join(sorted(r, key=lambda k: r[k]))))

    # ---------------- 2. effect of the packing fix on news
    print()
    print("=== Packing fix, news corpus (published pre-fix vs re-run post-fix) ===")
    print("  %-18s %-16s %-16s %-16s" % ("Model", "zero pre->post", "adapt pre->post", "red% pre->post"))
    news = cells.get("news", {})
    for m in ORDER:
        d = news.get(m)
        if not d:
            continue
        full = next((k for k, v in SHORT.items() if v == m), None)
        if full not in PUBLISHED:
            continue
        pz, pa = PUBLISHED[full]
        pr = 100 * (pz - pa) / pz
        print("  %-18s %6.4f->%6.4f  %6.4f->%6.4f  %5.1f->%5.1f"
              % (m, pz, d["zero_shot_bpb"], pa, d.get("adapted_bpb", float("nan")),
                 pr, d.get("reduction_pct", float("nan"))))

    if a.out_tex:
        with open(a.out_tex, "w") as f:
            f.write("% generated by tools/report_domain_transfer.py\n")
            for m in ORDER:
                vals = []
                for c in CORPORA:
                    d = cells.get(c, {}).get(m)
                    vals.append("%.3f & %.3f" % (d["zero_shot_bpb"], d["adapted_bpb"])
                                if d and "adapted_bpb" in d else "--- & ---")
                f.write("%s & %s \\\\\n" % (m, " & ".join(vals)))
        print("\nwrote", a.out_tex)


if __name__ == "__main__":
    main()
