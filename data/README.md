# Data

The paper evaluates recent target-domain text. Source-text snapshots are not redistributed unless their licenses explicitly permit it.

The primary leaderboard protocol is recorded in [`benchmarks/primary-news-2026-06-08.json`](../benchmarks/primary-news-2026-06-08.json). The maintainers retain the controlled snapshot used for the reported runs and can evaluate requested models against it.

To evaluate your own domain, pass one of these inputs to `entropy-bench --dataset`:

- UTF-8 text (`.txt`), one document per non-empty line;
- JSON Lines (`.jsonl`) with a `text` field;
- CSV with a `text` column;
- a Hugging Face dataset saved to disk; or
- a ZIP containing one of the supported formats.

Use `--dataset-text-field FIELD` if your JSONL or CSV text column has another name. Do not commit private, copyrighted, or access-controlled corpora to this repository.
