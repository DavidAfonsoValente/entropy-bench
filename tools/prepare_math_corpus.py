"""Build the math-domain corpus for the domain-adaptation model-selection experiment
(docs/PLAN.md: "math-domain adaptation as a model-selection signal for GSM8K").

Two modes, run in that order:

  1. Build the GSM8K-test contamination shingle set (cheap, no download beyond the
     already-cached openai/gsm8k, safe on the login node):

       python tools/prepare_math_corpus.py build-shingles \
           --out tools/data/gsm8k_test_shingles.json

  2. Build the math corpus itself. Two sources are implemented; use `arxiv`:

     `arxiv` (default, and the one that matters): pulls math.* paper titles+abstracts from
     arXiv's OAI-PMH feed for a date window strictly AFTER every model in the cohort was
     released (matching the news/Reddit/Hacker News freshness property -- see
     tools/domain_transfer_eval.py's comment on why contamination gating is skipped for
     those corpora: "both new corpora were scraped ... after every model in the cohort was
     released, so the audit has nothing to find by construction"). This lets the math
     corpus use the SAME cheap freshness argument instead of an expensive per-model
     Min-K++/CoDeC/DCQ forensic audit (~3h/model per lm_adapt_bench/cli.py).

       python tools/prepare_math_corpus.py build-corpus --source arxiv \
           --from-date 2026-06-09 --until-date 2026-08-20 \
           --forbidden-shingles tools/data/gsm8k_test_shingles.json \
           --out $WORK/corpora/math_arxiv.jsonl

     `hf` (NOT used for the headline run -- kept for reference/ablation only): streams a
     static HF math-web dataset (default open-web-math/open-web-math). DO NOT use this for
     the real experiment: OpenWebMath was published in 2023 and is a widely-used
     pretraining-corpus component, so most of these 11 models likely already saw it or
     heavily overlapping content during their own pretraining. Using it would make
     "zero-shot BPB" on this corpus partly a memorization measurement, not a fit
     measurement, undermining the whole comparison. Only defensible with a real per-model
     contamination audit, which this script does not run.

Either source gets the same downstream recipe used for Reddit/Hacker News: exact SHA256
dedup, Gopher quality heuristics (lm_adapt_bench.contamination.text_normalize.gopher_ok),
an ADDITIONAL n-gram containment screen against the GSM8K TEST set (not covered by the
freshness argument above -- GSM8K itself long predates the corpus window, so this guards
against a paper/post that happens to quote a GSM8K problem), then fuzzy MinHash/LSH dedup
at 0.60.

Why containment, not document-level Jaccard, for the GSM8K screen: a GSM8K test question is
~50 words; an arXiv abstract or a web math page is hundreds of words. Even a document that
quotes a GSM8K question verbatim has document-level MinHash Jaccard near zero against it, so
ForbiddenChecker-style whole-document similarity would silently pass contaminated documents.
Containment (any shared n-word shingle) is the standard decontamination instrument
(GPT-3/Llama-style) and catches verbatim or near-verbatim copies regardless of document length.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm_adapt_bench.contamination.text_normalize import (  # noqa: E402
    normalize_text, get_word_shingles, gopher_ok,
)

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def cmd_build_shingles(a):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=a.gsm8k_split)
    shingles = set()
    for ex in ds:
        text = normalize_text(ex["question"] + " " + ex["answer"], lowercase=True)
        shingles.update(get_word_shingles(text, a.shingle_n))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"shingle_n": a.shingle_n, "shingles": sorted(shingles)}))
    print(f"[shingles] {len(ds)} GSM8K {a.gsm8k_split} examples -> {len(shingles)} "
          f"{a.shingle_n}-word shingles -> {a.out}")


def iter_hf_records(a):
    from datasets import load_dataset
    ds = load_dataset(a.hf_dataset, split=a.split, streaming=True)
    n = 0
    for rec in ds:
        n += 1
        if a.take and n > a.take:
            return
        text = rec.get(a.text_field)
        if text and isinstance(text, str):
            yield text


def iter_arxiv_records(a):
    """arXiv Atom search API, filtered by ACTUAL submission date (submittedDate), grouped by
    (category, month) into ~600-word documents the same way
    tools/prepare_scraped_corpus.py groups short Reddit/HN comments -- a single ~150-200 word
    abstract is too short on its own (inflates the per-document boundary-marker share of the
    loss, see tools/token_level_gain.py).

    Deliberately NOT OAI-PMH ListRecords with a from/until datestamp filter: datestamp is when
    a record's METADATA was last touched (e.g. a journal-ref added), not when the paper was
    written -- verified against the live feed: a paper created 2008-06-11 had datestamp
    2026-01-01 and matched a "from 2026" query. That would have silently reintroduced the
    exact contamination risk this script exists to avoid. submittedDate on the Atom API is the
    real field.
    """
    # Chunk by DAY and paginate shallowly (start=0, 100, 200...) within each day, rather than
    # one query over the whole window with deep start-based pagination. Verified live: with
    # ~14,500 total results, plain start-based pagination combined with sortBy=submittedDate
    # started returning persistent HTTP 500s past start=10000 -- reproducible across two full
    # runs at the exact same offset, not transient rate-limiting. ~200 records/day means each
    # day needs at most 2-3 pages, which never gets deep enough to hit that failure mode.
    from datetime import datetime, timedelta
    day = datetime.strptime(a.from_date, "%Y-%m-%d")
    end_day = datetime.strptime(a.until_date, "%Y-%m-%d")
    groups, group_words = {}, {}
    n_records = 0
    page = 100

    def fetch(url):
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    return resp.read()
            except Exception as e:  # noqa: BLE001 -- back off and retry within one day's window
                wait = min(15 * (attempt + 1), 90)
                print(f"[arxiv] fetch error ({e}); retrying in {wait}s "
                      f"(attempt {attempt + 1}/6)", flush=True)
                time.sleep(wait)
        return None

    while day <= end_day:
        d = day.strftime("%Y-%m-%d")
        d_compact = d.replace("-", "")
        query = f"cat:math.* AND submittedDate:[{d_compact}0000 TO {d_compact}2359]"
        start = 0
        day_records = 0
        while True:
            url = (f"{ARXIV_API}?search_query={urllib.parse.quote(query)}"
                   f"&start={start}&max_results={page}")
            raw = fetch(url)
            if raw is None:
                print(f"[arxiv] giving up on {d} (start={start}) after 6 attempts; "
                      "moving to next day", flush=True)
                break
            root = ET.fromstring(raw)
            entries = root.findall("a:entry", ATOM_NS)
            if not entries:
                break
            for entry in entries:
                title_el = entry.find("a:title", ATOM_NS)
                summary_el = entry.find("a:summary", ATOM_NS)
                cat_el = entry.find("a:category", ATOM_NS)
                published_el = entry.find("a:published", ATOM_NS)
                if title_el is None or summary_el is None:
                    continue
                title = " ".join((title_el.text or "").split())
                abstract = " ".join((summary_el.text or "").split())
                published = (published_el.text or "") if published_el is not None else ""
                if not title or not abstract:
                    continue
                # Belt-and-suspenders: re-check the actual submission date even though the
                # query already filtered on it -- never trust a filter you haven't verified
                # end to end (see the OAI-PMH datestamp bug this function's docstring documents).
                if published[:10] < a.from_date or published[:10] > a.until_date:
                    continue
                n_records += 1
                day_records += 1
                cat = cat_el.get("term") if cat_el is not None else "misc"
                month = published[:7]
                key = f"{cat}_{month}"
                doc = title + ".\n\n" + abstract
                groups.setdefault(key, []).append(doc)
                group_words[key] = group_words.get(key, 0) + len(doc.split())
                if group_words[key] >= a.group_target_words:
                    yield "\n\n".join(groups.pop(key))
                    group_words[key] = 0
                if a.take and n_records >= a.take:
                    for items in groups.values():
                        if items:
                            yield "\n\n".join(items)
                    return
            if len(entries) < page:
                break
            start += page
            time.sleep(a.oai_delay)
        print(f"[arxiv] {d}: {day_records} records ({n_records} total so far)", flush=True)
        day += timedelta(days=1)
        time.sleep(a.oai_delay)
    for items in groups.values():
        if items:
            yield "\n\n".join(items)


def cmd_build_corpus(a):
    forbidden = json.loads(Path(a.forbidden_shingles).read_text())
    shingle_n = forbidden["shingle_n"]
    forbidden_set = set(forbidden["shingles"])
    print(f"[contam] loaded {len(forbidden_set)} forbidden {shingle_n}-word shingles "
          f"from {a.forbidden_shingles}")
    if a.source == "hf":
        print("[WARNING] --source hf uses a static, long-published dataset with no freshness "
              "guarantee against these models' pretraining data -- see this script's docstring. "
              "Not the headline run.", flush=True)

    stats = {"seen": 0, "drop_empty": 0, "drop_exact_dup": 0, "drop_gopher": 0,
             "drop_gsm8k_contamination": 0, "after_screen": 0}
    seen_hash = set()

    stage = Path(a.out).with_suffix(".stage.jsonl")
    stage.parent.mkdir(parents=True, exist_ok=True)
    records = iter_arxiv_records(a) if a.source == "arxiv" else iter_hf_records(a)

    with stage.open("w", encoding="utf-8") as stage_fh:
        for text in records:
            stats["seen"] += 1
            if not text or not isinstance(text, str):
                stats["drop_empty"] += 1
                continue
            text = text.strip()
            if not text:
                stats["drop_empty"] += 1
                continue

            h = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
            if h in seen_hash:
                stats["drop_exact_dup"] += 1
                continue
            seen_hash.add(h)

            if not gopher_ok(text, a.min_words, a.max_words):
                stats["drop_gopher"] += 1
                continue

            doc_shingles = get_word_shingles(normalize_text(text, lowercase=True), shingle_n)
            if any(s in forbidden_set for s in doc_shingles):
                stats["drop_gsm8k_contamination"] += 1
                continue

            stage_fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats["after_screen"] += 1

            if stats["seen"] % 5000 == 0:
                print(f"  processed {stats['seen']} -> kept {stats['after_screen']} "
                      f"(gsm8k-flagged so far: {stats['drop_gsm8k_contamination']})", flush=True)

    seen_hash.clear()

    # Fuzzy MinHash+LSH dedup at the same 0.60 threshold used for the other corpora,
    # sharded to bound memory (see tools/prepare_scraped_corpus.py for the same pattern).
    drop = set()
    kept_n = stats["after_screen"]
    if not a.no_fuzzy and kept_n:
        from lm_adapt_bench.contamination.fingerprints import Fingerprinter
        fp = Fingerprinter()
        lsh = fp.get_lsh_index(threshold=a.fuzzy_threshold)
        if lsh is None:
            print("[fuzzy] datasketch unavailable; skipping near-duplicate removal")
        else:
            shard = a.fuzzy_shard_size
            print(f"[fuzzy] streaming MinHash+LSH over {kept_n} documents at threshold "
                  f"{a.fuzzy_threshold:.2f} (shard size {shard or 'unbounded'})", flush=True)
            with stage.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if shard and i and i % shard == 0:
                        lsh = fp.get_lsh_index(threshold=a.fuzzy_threshold)
                        print(f"  fuzzy: {i}/{kept_n} scanned, {len(drop)} dropped (new shard)",
                              flush=True)
                    t = json.loads(line)["text"]
                    m = fp.create_minhash(t)
                    if m is None:
                        continue
                    if lsh.query(m):
                        drop.add(i)
                        continue
                    lsh.insert(str(i), m)
            stats["drop_fuzzy_dup"] = len(drop)

    out = Path(a.out)
    total_bytes, n_final, wc = 0, 0, []
    with stage.open("r", encoding="utf-8") as fh, out.open("w", encoding="utf-8") as f:
        for i, line in enumerate(fh):
            if i in drop:
                continue
            t = json.loads(line)["text"]
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            total_bytes += len(t.encode("utf-8"))
            wc.append(len(t.split()))
            n_final += 1
    stage.unlink(missing_ok=True)
    stats["final_documents"] = n_final
    stats["final_text_mb"] = round(total_bytes / 1e6, 1)

    print(f"\n=== math ({a.source}) -> {out} ===")
    for k in ("seen", "drop_empty", "drop_exact_dup", "drop_gopher", "after_screen",
              "drop_gsm8k_contamination", "drop_fuzzy_dup", "final_documents", "final_text_mb"):
        if k in stats:
            print(f"  {k:<26} {stats[k]}")
    if wc:
        wc.sort()
        print(f"  {'words/document':<26} median={wc[len(wc)//2]} p10={wc[len(wc)//10]} "
              f"p90={wc[9*len(wc)//10]}")
    print("\nReport drop_gsm8k_contamination in the paper the same way tab:contam_stats "
          "reports the news/Reddit/HN screen.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("build-shingles")
    s1.add_argument("--gsm8k-split", default="test")
    s1.add_argument("--shingle-n", type=int, default=10)
    s1.add_argument("--out", required=True)
    s1.set_defaults(func=cmd_build_shingles)

    s2 = sub.add_parser("build-corpus")
    s2.add_argument("--source", choices=["arxiv", "hf"], default="arxiv")
    # arxiv source
    s2.add_argument("--from-date", default="2026-06-09",
                     help="day after the news snapshot (2026-06-08) -- after every model's "
                          "release, matching the reddit/hackernews freshness property")
    s2.add_argument("--until-date", default="2026-08-20")
    s2.add_argument("--group-target-words", type=int, default=600)
    s2.add_argument("--oai-delay", type=float, default=3.0)
    # hf source (ablation-only, see docstring warning)
    s2.add_argument("--hf-dataset", default="open-web-math/open-web-math")
    s2.add_argument("--split", default="train")
    s2.add_argument("--text-field", default="text")
    # shared
    s2.add_argument("--take", type=int, default=300000,
                     help="raw records to process before quality/dedup/contamination filtering")
    s2.add_argument("--forbidden-shingles", required=True)
    s2.add_argument("--min-words", type=int, default=50)
    s2.add_argument("--max-words", type=int, default=20000)
    s2.add_argument("--fuzzy-threshold", type=float, default=0.60)
    s2.add_argument("--fuzzy-shard-size", type=int, default=25000)
    s2.add_argument("--no-fuzzy", action="store_true")
    s2.add_argument("--out", required=True)
    s2.set_defaults(func=cmd_build_corpus)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
