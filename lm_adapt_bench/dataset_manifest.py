"""Inspect and verify benchmark JSONL files without loading them into memory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO


class DatasetManifestError(ValueError):
    """Raised when a dataset or its public manifest fails validation."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetManifestError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetManifestError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetManifestError(f"Expected a JSON object in {path}")
    return value


def _open_index(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        if mode == "wt":
            raw = path.open("wb")
            compressed = gzip.GzipFile(
                filename="entropy-bench-record-index.jsonl",
                fileobj=raw,
                mode="wb",
                mtime=0,
            )
            return io.TextIOWrapper(compressed, encoding="utf-8")
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _iter_records(
    dataset: Path, text_field: str, *, file_hasher: Any | None = None
) -> Iterator[tuple[int, str]]:
    try:
        source = dataset.open("rb")
    except FileNotFoundError as exc:
        raise DatasetManifestError(f"Dataset not found: {dataset}") from exc

    with source:
        record_index = 0
        for line_number, raw_line in enumerate(source, start=1):
            if file_hasher is not None:
                file_hasher.update(raw_line)
            if not raw_line.strip():
                raise DatasetManifestError(f"Blank line at {dataset}:{line_number}")
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetManifestError(
                    f"Invalid UTF-8 JSON at {dataset}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise DatasetManifestError(f"Record {line_number} is not a JSON object")
            text = value.get(text_field)
            if not isinstance(text, str) or not text:
                raise DatasetManifestError(
                    f"Record {line_number} has no non-empty string field {text_field!r}"
                )
            yield record_index, text
            record_index += 1


def fingerprint_dataset(
    dataset: Path, text_field: str = "text", index_path: Path | None = None
) -> dict[str, Any]:
    """Return exact file/text statistics and optionally write a text-only hash index."""
    file_hasher = hashlib.sha256()
    records = 0
    text_bytes = 0
    index_handle: TextIO | None = None
    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_handle = _open_index(index_path, "wt")
    try:
        for index, text in _iter_records(dataset, text_field, file_hasher=file_hasher):
            encoded = text.encode("utf-8")
            records += 1
            text_bytes += len(encoded)
            if index_handle is not None:
                row = {
                    "index": index,
                    "text_sha256": hashlib.sha256(encoded).hexdigest(),
                    "utf8_bytes": len(encoded),
                }
                index_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        if index_handle is not None:
            index_handle.close()

    return {
        "format": "jsonl",
        "text_field": text_field,
        "records": records,
        "size_bytes": dataset.stat().st_size,
        "text_utf8_bytes": text_bytes,
        "sha256": file_hasher.hexdigest(),
    }


def _distribution(manifest: dict[str, Any]) -> dict[str, Any]:
    distribution = manifest.get("distribution")
    if not isinstance(distribution, dict):
        raise DatasetManifestError("manifest.distribution must be an object")
    required = ("format", "text_field", "records", "size_bytes", "sha256")
    for key in required:
        if key not in distribution:
            raise DatasetManifestError(f"manifest.distribution.{key} is required")
    if distribution["format"] != "jsonl":
        raise DatasetManifestError("Only JSONL benchmark distributions are supported")
    if not isinstance(distribution["text_field"], str) or not distribution["text_field"]:
        raise DatasetManifestError("manifest.distribution.text_field must be non-empty")
    for key in ("records", "size_bytes"):
        if not isinstance(distribution[key], int) or distribution[key] <= 0:
            raise DatasetManifestError(f"manifest.distribution.{key} must be a positive integer")
    digest = distribution["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise DatasetManifestError("manifest.distribution.sha256 must be a SHA-256 digest")
    return distribution


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate the public contract and its optional record-index artifact."""
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise DatasetManifestError("manifest.schema_version must be 1")
    if not manifest.get("benchmark_id"):
        raise DatasetManifestError("manifest.benchmark_id is required")
    _distribution(manifest)

    record_index = manifest.get("record_index")
    if record_index is not None:
        if not isinstance(record_index, dict):
            raise DatasetManifestError("manifest.record_index must be an object")
        relative_path = record_index.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise DatasetManifestError("manifest.record_index.path is required")
        repository_root = manifest_path.parent.parent
        index_path = repository_root / relative_path
        if not index_path.is_file():
            raise DatasetManifestError(f"Record index not found: {index_path}")
        actual_digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
        if actual_digest != record_index.get("sha256"):
            raise DatasetManifestError(
                f"Record-index SHA-256 mismatch: {actual_digest} != {record_index.get('sha256')}"
            )
    return manifest


def verify_dataset(dataset: Path, manifest_path: Path) -> dict[str, Any]:
    """Fingerprint a local corpus and compare it with the published benchmark contract."""
    manifest = validate_manifest(manifest_path)
    expected = _distribution(manifest)
    actual = fingerprint_dataset(dataset, expected["text_field"])
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in ("format", "text_field", "records", "size_bytes", "sha256")
        if expected[key] != actual[key]
    }
    if mismatches:
        raise DatasetManifestError(
            "Dataset does not match the benchmark manifest:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entropy-dataset",
        description="Inspect or verify an Entropy Bench JSONL corpus",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="print a deterministic corpus fingerprint")
    describe.add_argument("dataset")
    describe.add_argument("--text-field", default="text")

    fingerprint = subparsers.add_parser(
        "fingerprint", help="fingerprint a corpus and optionally create a safe record-hash index"
    )
    fingerprint.add_argument("dataset")
    fingerprint.add_argument("--text-field", default="text")
    fingerprint.add_argument("--write-index")

    check = subparsers.add_parser("validate-manifest", help="validate a public benchmark manifest")
    check.add_argument("manifest", nargs="?", default="benchmarks/primary-news-2026-06-08.json")

    verify = subparsers.add_parser("verify", help="prove that a local corpus is the exact benchmark")
    verify.add_argument("dataset")
    verify.add_argument("--manifest", default="benchmarks/primary-news-2026-06-08.json")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command in {"describe", "fingerprint"}:
            index_path = Path(args.write_index) if args.command == "fingerprint" and args.write_index else None
            result = fingerprint_dataset(Path(args.dataset), args.text_field, index_path)
            if index_path is not None:
                result["record_index"] = str(index_path)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "validate-manifest":
            manifest = validate_manifest(Path(args.manifest))
            print(f"valid dataset manifest: {manifest['benchmark_id']}")
        elif args.command == "verify":
            result = verify_dataset(Path(args.dataset), Path(args.manifest))
            print("exact benchmark match")
            print(json.dumps(result, indent=2, sort_keys=True))
    except DatasetManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
