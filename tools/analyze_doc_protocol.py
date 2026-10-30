"""M3: does a clean unadapted reading rank the fine-tuned systems as well as adapted BPB?

Reads the document-protocol cells written by ``slurm/gcp/doc_bpb.sbatch`` (one
``news_doc__<model>.json`` per model, ``doc_protocol.zero_shot.bpb``) and scores them against the
fine-tuned ROUGE-L winners of the 39 size-matched pairs in ``results/downstream.json``, next to the
published readings. The decision rule is the one preregistered in ``docs/RUN_LEDGER.md`` ("M3"):
clean unadapted <= 35 of 39 answers M3; >= 36 calls for the adapted arm (``ADAPT=1`` cells,
``news_docadapt__<model>.json``), and the claim changes if clean unadapted is then within one pair
of clean adapted.

  python tools/analyze_doc_protocol.py --cells /path/to/dt
  python tools/analyze_doc_protocol.py --check   # recompute from --cells and compare
"""
import argparse
import glob
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "doc_protocol.json"
CELLS = ROOT / "results" / "doc_protocol_cells"   # copied from the GCP jobs' dt/ directory
# Published unadapted news BPB the control models must reproduce within 1% (rank_table.tex).
CONTROL = {"gemma-4-12B": 0.866, "Qwen2.5-7B": 0.758}
# Published adapted news BPB of the same two models (results/domain_transfer/news__*.json).
ADAPTED_CONTROL = {"gemma-4-12B": "google_gemma-4-12B", "Qwen2.5-7B": "Qwen_Qwen2_5-7B"}


def short_name(model_id):
    """results/downstream.json's model names from Hugging Face ids."""
    n = model_id.split("/")[-1]
    for suffix in ("-Base-2512", "-Base-2407", "-Base"):
        n = n.removesuffix(suffix)
    return {"Mistral-Nemo": "Mistral-Nemo-12B"}.get(n, n)


def load(cells_dir, prefix):
    return {short_name(d["model_id"]): d
            for d in map(lambda f: json.load(open(f)), glob.glob(f"{cells_dir}/{prefix}__*.json"))}


def pairs_right(pairs, bpb):
    right = 0
    for p in pairs:
        a, b = p["pair"].split("|")
        right += (a if bpb[a] < bpb[b] else b) == p["winner"]
    return right


def published_block_adapted(models):
    """The paper's adapted news BPB (packed blocks), loaded the way analyze_downstream does."""
    sys.path.insert(0, str(ROOT / "tools"))
    from analyze_alignment_matrix import load_bpb_matrix
    from analyze_cohort_extension import _load_ext
    vals = dict(load_bpb_matrix()[0]["news"]["adapted_bpb"])
    vals.update({m: d["adapted_bpb"] for m, d in _load_ext()[0].items()})
    return {m: vals[m] for m in models}


def analyse(cells_dir):
    cells = load(cells_dir, "news_doc")
    down = json.load(open(ROOT / "results" / "downstream.json"))["cohort"]
    models = sorted(down["rouge_l"])
    missing = [m for m in models if m not in cells]
    if missing:
        raise SystemExit(f"missing cells: {missing}")
    byte_counts = {cells[m]["doc_protocol"]["zero_shot"]["bytes"] for m in models}
    assert len(byte_counts) == 1, f"models scored different text: {byte_counts}"
    doc = {m: cells[m]["doc_protocol"]["zero_shot"]["bpb"] for m in models}

    pairs = down["pairs"]
    right = pairs_right(pairs, doc)
    published = {k: sum(p[f"{k}_right"] for p in pairs)
                 for k in ("matched_adapted_bpb", "zero_shot_bpb", "hellaswag")}
    rouge = down["rouge_l"]
    rank = {m: i + 1 for i, m in enumerate(sorted(models, key=doc.get))}
    control = {m: {"published": v, "packed_block": cells[m]["zero_shot_bpb"],
                   "rel_diff": cells[m]["zero_shot_bpb"] / v - 1}
               for m, v in CONTROL.items()}
    result = {
        "question": "Does unadapted BPB on whole documents from their own BOS, 2,048-token "
                    "context, equal bytes, rank the 15 fine-tuned systems as well as adapted BPB?",
        "n_pairs": len(pairs),
        "doc_protocol_unadapted_right": right,
        "published_right": published,
        "spearman_vs_rouge": {
            "doc_protocol_unadapted": spearmanr([-doc[m] for m in models],
                                                [rouge[m] for m in models])[0]},
        "doc_protocol_bpb": doc,
        "doc_protocol_rank": rank,
        "bytes_scored": byte_counts.pop(),
        "control": control,
        "control_passes": all(abs(c["rel_diff"]) <= 0.01 for c in control.values()),
        "verdict": ("M3 answered: the clean unadapted reading does not reach adapted BPB"
                    if right <= 35 else "run the adapted arm (stage 2)"),
    }
    adapted = load(cells_dir, "news_docadapt")
    if adapted:
        missing = [m for m in models if m not in adapted]
        if missing:
            raise SystemExit(f"missing adapted cells: {missing}")
        assert {adapted[m]["doc_protocol"]["adapted"]["bytes"] for m in models} == {result["bytes_scored"]}
        # The adapted job re-reads the unadapted model; the document reading has no seed, so it
        # must reproduce stage 1 exactly.
        for m in models:
            assert adapted[m]["doc_protocol"]["zero_shot"]["bpb"] == doc[m], m
        clean = {m: adapted[m]["doc_protocol"]["adapted"]["bpb"] for m in models}
        clean_right = pairs_right(pairs, clean)
        cells_block_adapted = published_block_adapted(models)
        published_adapted = {m: json.load(open(ROOT / "results" / "domain_transfer"
                                               / f"news__{slug}.json"))["adapted_bpb"]
                             for m, slug in ADAPTED_CONTROL.items()}
        result["stage2"] = {
            "doc_protocol_adapted_right": clean_right,
            "doc_protocol_adapted_bpb": clean,
            "spearman_vs_rouge": spearmanr([-clean[m] for m in models],
                                           [rouge[m] for m in models])[0],
            "control": {m: {"published": v, "packed_block": adapted[m]["adapted_bpb"],
                            "rel_diff": adapted[m]["adapted_bpb"] / v - 1}
                        for m, v in published_adapted.items()},
            "doc_unadapted_vs_block_adapted_spearman": spearmanr(
                [doc[m] for m in models], [cells_block_adapted[m] for m in models])[0],
            "claim_changes": clean_right - right <= 1,
        }
    return result


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cells", default=str(CELLS),
                   help="directory holding news_doc__*.json and news_docadapt__*.json")
    a.add_argument("--check", action="store_true")
    args = a.parse_args()
    result = analyse(args.cells)
    if args.check:
        if json.load(open(OUT)) != json.loads(json.dumps(result)):
            sys.exit(f"{OUT} is stale")
        print(f"{OUT.name} current")
        return
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "doc_protocol_bpb"}, indent=1))


if __name__ == "__main__":
    main()
