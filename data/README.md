# Data

Entropy Bench accepts any UTF-8 corpus you are allowed to use. The easiest format is JSON Lines:

```json
{"text":"One document per line."}
{"text":"The text field can contain an article, post, report, or other document."}
```

TXT, CSV, Hugging Face datasets saved to disk, and ZIP archives are also supported. Use
`--dataset-text-field FIELD` when the text column has another name.

## Primary paper corpus

The paper's exact benchmark is `primary-news-2026-06-08`: 119,054 third-party news documents
collected through Google News on 8 June 2026. The maintainers retain the verified JSONL snapshot.

What is public:

- the complete [benchmark contract](../benchmarks/primary-news-2026-06-08.json);
- the exact file SHA-256, byte count, record count, and text-field definition;
- a 119,054-row [record fingerprint index](manifests/primary-news-2026-06-08.records.jsonl.gz)
  containing only row numbers, text SHA-256 hashes, and byte lengths; and
- all reported scores, contamination summaries, and analysis artifacts.

What is not yet public is the third-party article text. The collected file contains no source URLs,
publisher metadata, license, or redistribution grant. A project cannot assign Apache-2.0—or any
other open license—to text owned by outside publishers. See [Dataset access and
provenance](../docs/DATASET.md) for the release condition and exact verification procedure.

If you have an authorized copy, prove that it is byte-for-byte identical:

```bash
entropy-dataset verify /path/to/google_news_from_2026-06-08_to_2026-06-08_cleaned.jsonl
```

## Use your own corpus

Inspect a JSONL corpus before spending GPU time:

```bash
entropy-dataset describe my_corpus.jsonl --text-field text
```

Then run a zero-shot baseline or the full controlled-adaptation protocol:

```bash
entropy-bench \
  --models Qwen/Qwen2.5-0.5B \
  --dataset my_corpus.jsonl \
  --dataset-text-field text \
  --output results/my-corpus \
  --baseline-only
```

Do not commit private, copyrighted, access-controlled, or personally identifying text to this
repository.
