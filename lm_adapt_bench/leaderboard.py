"""Validate, render, and extend LM Adapt Bench leaderboards.

The tool intentionally uses only the Python standard library so leaderboard checks can run
without installing the GPU evaluation stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BOARD = Path("results/leaderboard.json")
DEFAULT_MARKDOWN = Path("LEADERBOARD.md")
TABLE_START = "<!-- leaderboard:start -->"
TABLE_END = "<!-- leaderboard:end -->"


class LeaderboardError(ValueError):
    """Raised when a board or submission violates the public schema."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LeaderboardError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LeaderboardError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LeaderboardError(f"Expected a JSON object in {path}")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeaderboardError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "a positive finite number" if positive else "finite"
        raise LeaderboardError(f"{label} must be {qualifier}")
    return number


def validate_board(board: dict[str, Any]) -> None:
    """Validate the canonical leaderboard and its rank invariants."""
    if board.get("schema_version") != 1:
        raise LeaderboardError("schema_version must be 1")
    if board.get("metric") != "adapted_bits_per_byte":
        raise LeaderboardError("metric must be adapted_bits_per_byte")
    if board.get("direction") != "lower_is_better":
        raise LeaderboardError("direction must be lower_is_better")

    dataset = board.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("benchmark_id"):
        raise LeaderboardError("dataset.benchmark_id is required")

    entries = board.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LeaderboardError("entries must be a non-empty list")

    seen_models: set[str] = set()
    previous_bpb = -math.inf
    for expected_rank, entry in enumerate(entries, start=1):
        prefix = f"entries[{expected_rank - 1}]"
        if not isinstance(entry, dict):
            raise LeaderboardError(f"{prefix} must be an object")
        if entry.get("rank") != expected_rank:
            raise LeaderboardError(f"{prefix}.rank must be {expected_rank}")
        model_id = entry.get("model_id")
        if not isinstance(model_id, str) or "/" not in model_id:
            raise LeaderboardError(f"{prefix}.model_id must be a repository-style model ID")
        if model_id in seen_models:
            raise LeaderboardError(f"duplicate model_id: {model_id}")
        seen_models.add(model_id)
        if not isinstance(entry.get("model_name"), str) or not entry["model_name"].strip():
            raise LeaderboardError(f"{prefix}.model_name is required")

        zero = _finite_number(entry.get("zero_shot_bpb"), f"{prefix}.zero_shot_bpb", positive=True)
        adapted = _finite_number(entry.get("adapted_bpb"), f"{prefix}.adapted_bpb", positive=True)
        reduction = _finite_number(entry.get("reduction_pct"), f"{prefix}.reduction_pct")
        expected_reduction = 100.0 * (zero - adapted) / zero
        if abs(reduction - expected_reduction) > 0.05:
            raise LeaderboardError(
                f"{prefix}.reduction_pct is inconsistent: {reduction} vs {expected_reduction:.6f}"
            )
        if adapted < previous_bpb:
            raise LeaderboardError("entries must be sorted by ascending adapted_bpb")
        previous_bpb = adapted


def validate_submission(submission: dict[str, Any]) -> None:
    """Validate a contribution generated from one benchmark result JSON."""
    if submission.get("schema_version") != 1:
        raise LeaderboardError("submission schema_version must be 1")
    required_strings = (
        "benchmark_id",
        "model_id",
        "model_name",
        "model_revision",
        "tokenizer_revision",
        "parameters",
        "source_result_sha256",
    )
    for key in required_strings:
        if not isinstance(submission.get(key), str) or not submission[key].strip():
            raise LeaderboardError(f"submission.{key} is required")
    digest = submission["source_result_sha256"]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
        raise LeaderboardError("submission.source_result_sha256 must be a SHA-256 hex digest")

    zero = _finite_number(submission.get("zero_shot_bpb"), "submission.zero_shot_bpb", positive=True)
    adapted = _finite_number(submission.get("adapted_bpb"), "submission.adapted_bpb", positive=True)
    reduction = _finite_number(submission.get("reduction_pct"), "submission.reduction_pct")
    expected_reduction = 100.0 * (zero - adapted) / zero
    if abs(reduction - expected_reduction) > 1e-6:
        raise LeaderboardError("submission.reduction_pct is inconsistent with its BPB values")


def render_table(board: dict[str, Any]) -> str:
    validate_board(board)
    rows = [
        "| Rank | Model | Parameters | Zero-shot BPB | Adapted BPB | Reduction |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for entry in board["entries"]:
        rows.append(
            "| {rank} | {name} | {parameters} | {zero:.3f} | **{adapted:.3f}** | {reduction:.1f}% |".format(
                rank=entry["rank"],
                name=entry["model_name"],
                parameters=entry["parameters"],
                zero=entry["zero_shot_bpb"],
                adapted=entry["adapted_bpb"],
                reduction=entry["reduction_pct"],
            )
        )
    return "\n".join(rows)


def sync_markdown(board: dict[str, Any], markdown_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    if TABLE_START not in text or TABLE_END not in text:
        raise LeaderboardError(f"{markdown_path} is missing leaderboard table markers")
    before, remainder = text.split(TABLE_START, 1)
    _, after = remainder.split(TABLE_END, 1)
    rendered = f"{before}{TABLE_START}\n\n{render_table(board)}\n\n{TABLE_END}{after}"
    markdown_path.write_text(rendered, encoding="utf-8")


def prepare_submission(args: argparse.Namespace) -> dict[str, Any]:
    result_path = Path(args.result)
    result = _read_json(result_path)
    model_id = result.get("model_id")
    zero = result.get("zero_shot_bpb")
    adapted = result.get("adapted_bpb", result.get("best_bpb"))
    if not isinstance(model_id, str) or not model_id:
        raise LeaderboardError("result.model_id is required")
    zero = _finite_number(zero, "result.zero_shot_bpb", positive=True)
    adapted = _finite_number(adapted, "result.best_bpb", positive=True)
    raw = result_path.read_bytes()
    submission = {
        "schema_version": 1,
        "benchmark_id": args.benchmark_id,
        "model_id": model_id,
        "model_name": args.model_name or model_id.rsplit("/", 1)[-1],
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "parameters": args.parameters,
        "zero_shot_bpb": zero,
        "adapted_bpb": adapted,
        "reduction_pct": 100.0 * (zero - adapted) / zero,
        "source_result_sha256": hashlib.sha256(raw).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_submission(submission)
    return submission


def add_submission(board: dict[str, Any], submission: dict[str, Any]) -> None:
    validate_board(board)
    validate_submission(submission)
    benchmark_id = board["dataset"]["benchmark_id"]
    if submission["benchmark_id"] != benchmark_id:
        raise LeaderboardError(
            f"submission targets {submission['benchmark_id']}, but board uses {benchmark_id}"
        )
    if any(entry["model_id"] == submission["model_id"] for entry in board["entries"]):
        raise LeaderboardError(f"model already exists: {submission['model_id']}")
    board["entries"].append(
        {
            "rank": 0,
            "model_id": submission["model_id"],
            "model_name": submission["model_name"],
            "model_revision": submission["model_revision"],
            "tokenizer_revision": submission["tokenizer_revision"],
            "parameters": submission["parameters"],
            "zero_shot_bpb": submission["zero_shot_bpb"],
            "adapted_bpb": submission["adapted_bpb"],
            "reduction_pct": submission["reduction_pct"],
            "source_result_sha256": submission["source_result_sha256"],
        }
    )
    board["entries"].sort(key=lambda entry: entry["adapted_bpb"])
    for rank, entry in enumerate(board["entries"], start=1):
        entry["rank"] = rank
    validate_board(board)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage an auditable LM Adapt Bench leaderboard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a canonical leaderboard")
    validate.add_argument("board", nargs="?", default=str(DEFAULT_BOARD))

    render = subparsers.add_parser("render", help="synchronize the Markdown leaderboard table")
    render.add_argument("board", nargs="?", default=str(DEFAULT_BOARD))
    render.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    render.add_argument("--check", action="store_true", help="fail if rendering would change Markdown")

    prepare = subparsers.add_parser("prepare-submission", help="convert a run result into a submission")
    prepare.add_argument("result")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--benchmark-id", default="primary-news-2026-06-08")
    prepare.add_argument("--model-name")
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--tokenizer-revision", required=True)
    prepare.add_argument("--parameters", required=True, help="display value such as 7B or 35B total")

    check_submission = subparsers.add_parser("validate-submission", help="validate a submission JSON")
    check_submission.add_argument("submission")

    add = subparsers.add_parser("add", help="add a validated submission and rerender the board")
    add.add_argument("submission")
    add.add_argument("--board", default=str(DEFAULT_BOARD))
    add.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "validate":
            validate_board(_read_json(Path(args.board)))
            print(f"valid leaderboard: {args.board}")
        elif args.command == "render":
            board = _read_json(Path(args.board))
            markdown = Path(args.markdown)
            original = markdown.read_text(encoding="utf-8")
            sync_markdown(board, markdown)
            changed = markdown.read_text(encoding="utf-8") != original
            if args.check and changed:
                markdown.write_text(original, encoding="utf-8")
                raise LeaderboardError(f"{markdown} is out of date; run entropy-leaderboard render")
            print(f"{'would update' if args.check and changed else 'rendered'} leaderboard: {markdown}")
        elif args.command == "prepare-submission":
            submission = prepare_submission(args)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
            print(f"wrote submission: {output}")
        elif args.command == "validate-submission":
            validate_submission(_read_json(Path(args.submission)))
            print(f"valid submission: {args.submission}")
        elif args.command == "add":
            board_path = Path(args.board)
            board = _read_json(board_path)
            add_submission(board, _read_json(Path(args.submission)))
            board_path.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
            sync_markdown(board, Path(args.markdown))
            print(f"added submission and updated {board_path} and {args.markdown}")
    except LeaderboardError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
