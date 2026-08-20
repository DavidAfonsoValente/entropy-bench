# Dataset access and provenance

## Exact benchmark identity

| Field | Value |
|---|---|
| Benchmark ID | `primary-news-2026-06-08` |
| Snapshot date | 8 June 2026 |
| Format | JSON Lines, one object per document |
| Text field | `text` |
| Records | 119,054 |
| File bytes | 366,564,042 |
| UTF-8 text bytes | 361,445,123 |
| SHA-256 | `279ccc65a7f562e438cbe74827d41e51e41ac8783345250b6f85f607ea1a0162` |

The file used by the paper has been recovered from the experiment archive and verified against the
reported record and byte counts. It is not the older November 2025 snapshot that appeared in the
private development repository.

## Why the article text is not attached yet

The snapshot contains full third-party news text but only a `text` column. It does not include the
originating URL, publisher, license, or a redistribution grant. The repository's Apache-2.0 license
covers the project code; it cannot relicense article text owned by third parties.

The raw snapshot will be attached to a versioned release when the maintainers can record one of:

1. a license covering redistribution of every included document;
2. written permission from the relevant rights holders or data provider; or
3. a replacement corpus whose source license explicitly permits redistribution, followed by a new
   benchmark ID and rerun.

A gated download is still redistribution and therefore requires the same permission.

## What is already published

The repository publishes the exact file fingerprint and a compressed record index at
[`data/manifests/primary-news-2026-06-08.records.jsonl.gz`](../data/manifests/primary-news-2026-06-08.records.jsonl.gz).
Each row contains:

```json
{"index":0,"text_sha256":"<64 hex characters>","utf8_bytes":1234}
```

It contains no source text and cannot reconstruct an article. It lets reviewers verify membership,
ordering, and byte lengths without trusting a filename.

## Verify an authorized copy

```bash
python -m pip install -e .
entropy-dataset validate-manifest
entropy-dataset verify /path/to/the/snapshot.jsonl
```

The second command reads the file once and fails if its format, text field, record count, byte count,
or SHA-256 differs from the paper benchmark.

## Evaluate a new public corpus

If exact public replication is the priority, choose a corpus with explicit redistribution terms,
store its source/license metadata, create a new benchmark manifest, and run every model under that
new benchmark ID. Do not silently substitute it for `primary-news-2026-06-08`; doing so would make
old and new leaderboard rows incomparable.
