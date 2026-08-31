"""Turn the raw scraped Reddit / Hacker News dumps into a corpus the pipeline can adapt on.

Implements the cleanup Filip used for the news corpus:
  1. exact dedup on the SHA256 of normalized text (same normalization the contamination
     auditor uses, so ids stay comparable)
  2. Gopher quality heuristics
  3. MinHash + LSH fuzzy dedup at 0.60

Reads the raw JSON arrays as a stream so a 400MB/day dump does not have to fit in memory
twice, and writes {"text": ...} JSONL -- the format lm_adapt_bench.cli --dataset expects.

CPU only. Example:

  python tools/prepare_scraped_corpus.py \
      --inputs /path/to/reddit_*.json \
      --source reddit --out $WORK/corpora/reddit_2026-08.jsonl

NOTE ON DOCUMENT LENGTH: Reddit/HN records are comments, i.e. far shorter than news
articles. The packing in lm_adapt_bench/data.py injects one BOS-like token per document,
so a corpus of short documents inflates the boundary-token share of the loss (see
tools/token_level_gain.py). Use --group-by to concatenate records into longer documents,
or fix the packing first.
"""
import argparse
import hashlib
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm_adapt_bench.contamination.text_normalize import normalize_text, gopher_ok  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    """HN stores rendered HTML: <p> paragraph breaks, <a> links, escaped entities."""
    text = text.replace("<p>", "\n\n")
    text = TAG_RE.sub("", text)
    return html.unescape(text)


def iter_records(path: Path, chunk_bytes: int = 8 << 20):
    """Stream objects out of a pretty-printed JSON array.

    Reads in chunks rather than slurping the file: these dumps are ~400MB/day and holding
    several of them as one Python str, plus the decoded objects, is what makes this job
    get killed. Keeps at most one chunk plus one partial object in memory.
    """
    dec = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        buf = fh.read(chunk_bytes)
        pos = buf.find("[")
        if pos < 0:
            return
        pos += 1
        while True:
            # raw_decode takes an index, so advance a pointer instead of re-slicing the
            # buffer -- slicing a multi-MB str once per record is quadratic and was enough
            # to make this job look hung.
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, pos = dec.raw_decode(buf, pos)
            except (json.JSONDecodeError, ValueError):
                more = fh.read(chunk_bytes)
                if not more:
                    return
                buf = buf[pos:] + more       # compact only when we actually need more
                pos = 0
                continue
            yield obj
            # keep the tail small so the compaction above stays cheap
            if pos > chunk_bytes:
                buf = buf[pos:]
                pos = 0
                more = fh.read(chunk_bytes)
                if more:
                    buf += more


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--source", choices=["reddit", "hackernews"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-words", type=int, default=50,
                   help="Gopher lower bound; lower it for comment-style corpora")
    p.add_argument("--max-words", type=int, default=100000)
    p.add_argument("--fuzzy-threshold", type=float, default=0.60)
    p.add_argument("--group-by", default=None,
                   help="field to concatenate records under (e.g. subreddit) to build "
                        "longer documents")
    p.add_argument("--group-target-words", type=int, default=600)
    p.add_argument("--group-sequential", action="store_true",
                   help="concatenate consecutive records when there is no grouping field "
                        "(Hacker News); ignored if --group-by is given")
    p.add_argument("--no-fuzzy", action="store_true")
    p.add_argument("--fuzzy-shard-size", type=int, default=25000,
                   help="rebuild the LSH index every N documents to bound memory; "
                        "0 = one index over the whole corpus (needs many GB at 100k+ docs)")
    p.add_argument("--from-stage", metavar="PATH", default=None,
                   help="skip reading the raw dumps and resume from an existing "
                        "*.stage.jsonl (already exact-deduped and Gopher-filtered)")
    a = p.parse_args()

    stats = defaultdict(int)
    # Documents are streamed to a staging file rather than held in a list: 4 days of Reddit
    # is ~1.3M records, and keeping them all resident is what got this job killed.
    stage = Path(a.from_stage) if a.from_stage else Path(a.out).with_suffix(".stage.jsonl")
    stage.parent.mkdir(parents=True, exist_ok=True)
    if a.from_stage:
        kept_n = sum(1 for _ in stage.open("r", encoding="utf-8"))
        stats["after_gopher"] = kept_n
        print("[resume] %d staged documents from %s" % (kept_n, stage), flush=True)
    groups = defaultdict(list)
    group_words = defaultdict(int)
    seen = set()
    if not a.from_stage:
        stage_fh = stage.open("w", encoding="utf-8")
        kept_n = 0

    def emit(text):
        """Exact-dedup on normalized text, then Gopher, then stage to disk."""
        nonlocal kept_n
        h = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
        if h in seen:
            stats["drop_exact_dup"] += 1
            return
        seen.add(h)
        if not gopher_ok(text, a.min_words, a.max_words):
            stats["drop_gopher"] += 1
            return
        stage_fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        kept_n += 1

    import glob as _glob
    paths = []
    if not a.from_stage:
        for pat in a.inputs:
            # Path().glob rejects absolute patterns; glob.glob handles both, and a literal
            # path that exists is passed through unchanged.
            paths.extend(sorted(_glob.glob(pat)) if not Path(pat).exists() else [pat])
        if not paths:
            raise SystemExit("no input files matched: %s" % a.inputs)

    for _p in paths:
        for path in [Path(_p)]:
            print(f"[read] {path}", flush=True)
            for rec in iter_records(path):
                stats["records"] += 1
                t = rec.get("text")
                if not t or not isinstance(t, str):
                    stats["drop_empty"] += 1
                    continue
                title = rec.get("title")
                if isinstance(title, str) and title.strip():
                    t = title.strip() + "\n\n" + t
                if a.source == "hackernews":
                    t = clean_html(t)
                t = t.strip()
                if not t:
                    stats["drop_empty"] += 1
                    continue
                if a.group_target_words and (a.group_by or a.group_sequential):
                    # flush each group as soon as it reaches the target length, so at most
                    # one partial document per key is held. Without a grouping field
                    # (Hacker News has none) fall back to concatenating consecutive records.
                    key = str(rec.get(a.group_by, "_")) if a.group_by else "_seq"
                    groups[key].append(t)
                    group_words[key] += len(t.split())
                    if group_words[key] >= a.group_target_words:
                        emit("\n\n".join(groups.pop(key)))
                        group_words[key] = 0
                        stats["grouped_documents"] += 1
                else:
                    emit(t)
            print("  running: %d records -> %d staged" % (stats["records"], kept_n),
                  flush=True)

    if not a.from_stage:
        for key, items in list(groups.items()):    # trailing partial groups
            if items:
                emit("\n\n".join(items))
                stats["grouped_documents"] += 1
        stage_fh.close()
        stats["after_gopher"] = kept_n
    seen.clear()

    def iter_stage():
        with stage.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)["text"]

    # ---- fuzzy dedup (MinHash + LSH).
    # Deduper.find_near_duplicates() takes every text at once, which OOMs at this scale
    # (141k grouped Reddit documents was enough to get the process killed). We reuse the
    # auditor's Fingerprinter but stream: one document in memory at a time, and only the
    # MinHash signatures are retained.
    drop = set()
    if not a.no_fuzzy and kept_n:
        from lm_adapt_bench.contamination.fingerprints import Fingerprinter
        fp = Fingerprinter()
        lsh = fp.get_lsh_index(threshold=a.fuzzy_threshold)
        if lsh is None:
            print("[fuzzy] datasketch unavailable; skipping near-duplicate removal")
        else:
            # Streaming the texts is not enough on its own: a single MinHashLSH over 141k
            # documents holds ~30 bands x 141k keys and was itself OOM-killed. Shard the
            # index instead. Near-duplicates in these dumps are topically and temporally
            # clustered (and Reddit documents are emitted subreddit-by-subreddit), so a
            # sharded index catches most of them at bounded memory. It is an approximation,
            # and --fuzzy-shard-size 0 restores the exact single-index behaviour.
            shard = a.fuzzy_shard_size
            print("[fuzzy] streaming MinHash+LSH over %d documents at threshold %.2f"
                  " (shard size %s)"
                  % (kept_n, a.fuzzy_threshold, shard or "unbounded"), flush=True)
            for i, t in enumerate(iter_stage()):
                if shard and i and i % shard == 0:
                    lsh = fp.get_lsh_index(threshold=a.fuzzy_threshold)   # release the old index
                    print("  fuzzy: %d/%d scanned, %d dropped (new shard)"
                          % (i, kept_n, len(drop)), flush=True)
                m = fp.create_minhash(t)
                if m is None:                     # too short to shingle
                    continue
                if lsh.query(m):                  # near-duplicate of something already kept
                    drop.add(i)
                    continue
                lsh.insert(str(i), m)
            stats["drop_fuzzy_dup"] = len(drop)
            stats["fuzzy_shard_size"] = shard or kept_n

    out = Path(a.out)
    total_bytes = 0
    n_final = 0
    wc = []
    with out.open("w", encoding="utf-8") as f:
        for i, t in enumerate(iter_stage()):
            if i in drop:
                continue
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            total_bytes += len(t.encode("utf-8"))
            wc.append(len(t.split()))
            n_final += 1
    if not a.from_stage:
        stage.unlink(missing_ok=True)
    stats["final_documents"] = n_final
    stats["final_text_mb"] = round(total_bytes / 1e6, 1)

    print("\n=== %s -> %s ===" % (a.source, out))
    for k in ("records", "drop_empty", "grouped_documents", "drop_exact_dup",
              "drop_gopher", "after_gopher", "drop_fuzzy_dup", "final_documents",
              "final_text_mb"):
        if k in stats:
            print("  %-22s %s" % (k, stats[k]))
    if wc:
        wc.sort()
        print("  %-22s median=%d p10=%d p90=%d"
              % ("words/document", wc[len(wc) // 2], wc[len(wc) // 10], wc[9 * len(wc) // 10]))


if __name__ == "__main__":
    main()
