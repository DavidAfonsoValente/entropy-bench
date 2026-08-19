"""Compare token-level gain decompositions across models and classify the gain.

Answers, per model: is the adaptation gain concentrated on document-boundary artefacts,
on formatting/function words (i.e. calibration recovery), or on domain-specific content?
"""
import argparse
import json
from pathlib import Path

# Ordinary-English tokens: if the gain piles up here, the base model was mis-calibrated on
# generic prose rather than missing domain knowledge.
FUNCTION = {
    "the", "a", "an", "to", "of", "and", "in", "that", "for", "on", "with", "as", "at",
    "by", "from", "is", "was", "were", "are", "be", "been", "it", "its", "this", "he",
    "she", "they", "we", "you", "his", "her", "their", "but", "or", "not", "have", "has",
    "had", "will", "would", "can", "could", "said", "s", "t",
}


def classify(tokstr: str) -> str:
    s = tokstr
    if s.strip() == "" or s in ("\n", "\n\n", "\t"):
        return "whitespace/format"
    if all(not c.isalnum() for c in s.strip()):
        return "punctuation"
    bare = s.strip().strip(".,!?;:\"'()[]").lower()
    if bare in FUNCTION:
        return "function word"
    if bare.isdigit():
        return "digit"
    return "content word"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--top", type=int, default=200)
    a = p.parse_args()

    print("%-22s %8s %8s %9s | %s" % ("model", "base", "adapt", "reduction", "share of gain by position class"))
    print("%-22s %8s %8s %9s | %8s %8s %8s" % ("", "nats/tok", "nats/tok", "", "marker", "next 16", "remainder"))
    print("-" * 104)
    loaded = []
    for f in a.files:
        d = json.load(open(f))
        loaded.append(d)
        g = d["gain_by_class"]
        def sh(k):
            return g[k]["share_of_total_gain_pct"] if k in g else 0.0
        print("%-22s %8.3f %8.3f %8.2f%% | %7.2f%% %7.2f%% %7.2f%%"
              % (d["model_id"].split("/")[-1], d["mean_nats_per_token_base"],
                 d["mean_nats_per_token_adapted"], d["relative_nats_reduction_pct"],
                 sh("boundary_token"), sh("recovery_tail"), sh("content")))

    print()
    print("Where the gain sits, by token type (top %d gaining tokens per model):" % a.top)
    print("%-22s %18s %13s %13s %8s %13s" % ("model", "whitespace/format", "function word",
                                             "punctuation", "digit", "content word"))
    print("-" * 104)
    for d in loaded:
        tot = sum(r["total_nats_gained"] for r in d["top_tokens_gained"][:a.top])
        buckets = {}
        for r in d["top_tokens_gained"][:a.top]:
            if r["token"] == d.get("boundary_str"):
                c = "whitespace/format"      # the injected BOS is a formatting artefact
            else:
                c = classify(r["token"])
            buckets[c] = buckets.get(c, 0.0) + r["total_nats_gained"]
        row = [100.0 * buckets.get(k, 0.0) / tot if tot else 0.0
               for k in ("whitespace/format", "function word", "punctuation", "digit", "content word")]
        print("%-22s %17.1f%% %12.1f%% %12.1f%% %7.1f%% %12.1f%%"
              % (d["model_id"].split("/")[-1], *row))

    for d in loaded:
        if not d.get("gain_by_block_position"):
            continue
        print()
        print("%s -- gain by absolute position in the 512-token block:" % d["model_id"])
        print("  %-12s %12s %12s" % ("position", "base loss", "mean gain"))
        for k, v in d["gain_by_block_position"].items():
            print("  %-12s %12.3f %+12.4f" % (k, v["mean_base_loss"], v["mean_gain"]))


if __name__ == "__main__":
    main()
