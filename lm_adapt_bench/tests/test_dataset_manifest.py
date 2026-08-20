import gzip
import hashlib
import json
from pathlib import Path

import pytest

from lm_adapt_bench.dataset_manifest import (
    DatasetManifestError,
    fingerprint_dataset,
    validate_manifest,
    verify_dataset,
)


def _dataset(path: Path) -> Path:
    path.write_text(
        '{"text":"first original sentence"}\n'
        '{"text":"second original sentence"}\n',
        encoding="utf-8",
    )
    return path


def _manifest(path: Path, dataset: Path, index_path: Path) -> Path:
    stats = fingerprint_dataset(dataset, index_path=index_path)
    value = {
        "schema_version": 1,
        "benchmark_id": "test-benchmark",
        "distribution": stats,
        "record_index": {
            "path": f"data/{index_path.name}",
            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_fingerprint_and_verify_exact_dataset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset.jsonl")
    index_path = tmp_path / "data" / "index.jsonl.gz"
    manifest_path = _manifest(tmp_path / "benchmarks" / "manifest.json", dataset, index_path)

    assert validate_manifest(manifest_path)["benchmark_id"] == "test-benchmark"
    assert verify_dataset(dataset, manifest_path)["records"] == 2
    with gzip.open(index_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["index"] for row in rows] == [0, 1]
    assert all(len(row["text_sha256"]) == 64 for row in rows)


def test_verify_rejects_changed_text(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset.jsonl")
    index_path = tmp_path / "data" / "index.jsonl.gz"
    manifest_path = _manifest(tmp_path / "benchmarks" / "manifest.json", dataset, index_path)
    dataset.write_text('{"text":"changed"}\n', encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="does not match"):
        verify_dataset(dataset, manifest_path)


def test_fingerprint_rejects_missing_text(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text('{"title":"no text"}\n', encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="no non-empty string field"):
        fingerprint_dataset(dataset)
