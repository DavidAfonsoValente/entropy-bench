"""Validate, render, and extend Entropy Bench leaderboards.

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
BENCHMARKS = Path("benchmarks")


def all_boards() -> list[Path]:
    """Every track's canonical board: results/leaderboard.json (news) plus results/leaderboard_*.json."""
    return sorted(Path("results").glob("leaderboard*.json"))


def _markers(board: dict[str, Any]) -> tuple[str, str]:
    """Each track owns its own table in LEADERBOARD.md, keyed by benchmark ID."""
    benchmark_id = board["dataset"]["benchmark_id"]
    return (f"<!-- leaderboard:start:{benchmark_id} -->", f"<!-- leaderboard:end:{benchmark_id} -->")


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
    start, end = _markers(board)
    text = markdown_path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise LeaderboardError(f"{markdown_path} is missing the table markers for {start[5:-4]}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    rendered = f"{before}{start}\n\n{render_table(board)}\n\n{end}{after}"
    markdown_path.write_text(rendered, encoding="utf-8")


def load_manifest(benchmark_id: str) -> dict[str, Any]:
    path = BENCHMARKS / f"{benchmark_id}.json"
    if not path.is_file():
        known = sorted(p.stem for p in BENCHMARKS.glob("*.json"))
        raise LeaderboardError(f"unknown benchmark_id {benchmark_id!r}; known: {known}")
    return _read_json(path)


def check_protocol(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Refuse a result that was not produced on this track's corpus with this track's recipe.

    A board compares like with like: a different corpus, context length or adaptation procedure
    (including a per-model hyperparameter search) is a different benchmark and needs its own board.
    """
    track = manifest["benchmark_id"]
    expected_sha = manifest["distribution"]["sha256"]
    if result.get("dataset_sha256") != expected_sha:
        raise LeaderboardError(
            f"result.dataset_sha256 does not match {track}'s corpus ({expected_sha[:12]}...); "
            "run on the exact benchmark file (tools/domain_transfer_eval.py records the hash)")
    evaluation = manifest.get("evaluation", {})
    if result.get("max_seq_len") != evaluation.get("max_sequence_length"):
        raise LeaderboardError(f"result.max_seq_len must be {evaluation.get('max_sequence_length')} for {track}")
    if result.get("mask_injected_special_tokens") is not evaluation.get("mask_injected_special_tokens"):
        raise LeaderboardError(f"result.mask_injected_special_tokens must match {track}")
    adaptation = manifest.get("adaptation", {})
    if adaptation.get("selection") == "fixed configuration shared by every model":
        want = {"r": adaptation["rank"], "alpha": adaptation["alpha"], "dropout": adaptation["dropout"],
                "lr": adaptation["learning_rate"], "effective_batch": adaptation["effective_batch_size"]}
        if result.get("lora") != want or result.get("train_steps") != adaptation["update_steps"]:
            raise LeaderboardError(
                f"{track} is a fixed-recipe track ({want}, {adaptation['update_steps']} steps); this result "
                "used a different adaptation and belongs on a separate board")


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
    check_protocol(result, load_manifest(args.benchmark_id))
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
    parser = argparse.ArgumentParser(description="Manage an auditable Entropy Bench leaderboard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate canonical leaderboards (default: every track)")
    validate.add_argument("boards", nargs="*")

    render = subparsers.add_parser("render", help="synchronize the Markdown leaderboard tables (default: every track)")
    render.add_argument("boards", nargs="*")
    render.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    render.add_argument("--check", action="store_true", help="fail if rendering would change Markdown")

    prepare = subparsers.add_parser("prepare-submission", help="convert a run result into a submission")
    prepare.add_argument("result")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--benchmark-id", required=True,
                         help="track to submit to, e.g. arxiv-math-2026-08-fixed-lora-v1 (public corpus) "
                              "or primary-news-2026-06-08-fixed-lora-v1 (maintainer-run)")
    prepare.add_argument("--model-name")
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--tokenizer-revision", required=True)
    prepare.add_argument("--parameters", required=True, help="display value such as 7B or 35B total")

    check_submission = subparsers.add_parser("validate-submission", help="validate a submission JSON")
    check_submission.add_argument("submission")

    add = subparsers.add_parser("add", help="add a validated submission and rerender the board")
    add.add_argument("submission")
    add.add_argument("--board", help="default: the board whose benchmark_id the submission names")
    add.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "validate":
            for path in [Path(b) for b in args.boards] or all_boards():
                validate_board(_read_json(path))
                print(f"valid leaderboard: {path}")
        elif args.command == "render":
            markdown = Path(args.markdown)
            original = markdown.read_text(encoding="utf-8")
            for path in [Path(b) for b in args.boards] or all_boards():
                sync_markdown(_read_json(path), markdown)
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
            submission = _read_json(Path(args.submission))
            if args.board:
                board_path = Path(args.board)
            else:
                matches = [p for p in all_boards()
                           if _read_json(p)["dataset"]["benchmark_id"] == submission.get("benchmark_id")]
                if len(matches) != 1:
                    raise LeaderboardError(f"no single board for benchmark_id {submission.get('benchmark_id')!r}")
                board_path = matches[0]
            board = _read_json(board_path)
            add_submission(board, submission)
            board_path.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
            sync_markdown(board, Path(args.markdown))
            print(f"added submission and updated {board_path} and {args.markdown}")
    except LeaderboardError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
