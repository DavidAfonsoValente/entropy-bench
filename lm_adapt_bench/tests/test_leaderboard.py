import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

from lm_adapt_bench.leaderboard import (
    LeaderboardError,
    add_submission,
    prepare_submission,
    render_table,
    validate_board,
    validate_submission,
)


ROOT = Path(__file__).resolve().parents[2]


def test_repository_board_is_valid():
    board = json.loads((ROOT / "results" / "leaderboard.json").read_text())
    validate_board(board)
    table = render_table(board)
    assert "Gemma-4-31B" in table
    assert table.count("\n|") == len(board["entries"]) + 1


def test_rejects_unsorted_board():
    board = json.loads((ROOT / "results" / "leaderboard.json").read_text())
    board["entries"][0], board["entries"][1] = board["entries"][1], board["entries"][0]
    with pytest.raises(LeaderboardError, match="rank must be"):
        validate_board(board)


def test_prepare_and_add_submission(tmp_path):
    result = {
        "model_id": "example/model-1b",
        "zero_shot_bpb": 1.0,
        "best_bpb": 0.7,
    }
    result_path = tmp_path / "result.json"
    raw = json.dumps(result).encode()
    result_path.write_bytes(raw)
    args = argparse.Namespace(
        result=str(result_path),
        benchmark_id="primary-news-2026-06-08",
        model_name="Example-1B",
        model_revision="abc123",
        tokenizer_revision="def456",
        parameters="1B",
    )
    submission = prepare_submission(args)
    validate_submission(submission)
    assert submission["source_result_sha256"] == hashlib.sha256(raw).hexdigest()

    board = json.loads((ROOT / "results" / "leaderboard.json").read_text())
    original = copy.deepcopy(board)
    add_submission(board, submission)
    assert len(board["entries"]) == len(original["entries"]) + 1
    assert [entry["rank"] for entry in board["entries"]] == list(range(1, len(board["entries"]) + 1))
    assert [entry["adapted_bpb"] for entry in board["entries"]] == sorted(
        entry["adapted_bpb"] for entry in board["entries"]
    )
