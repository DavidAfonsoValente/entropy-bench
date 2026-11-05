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


MATH = "arxiv-math-2026-08-fixed-lora-v1"


def _math_result(**overrides):
    manifest = json.loads((ROOT / "benchmarks" / f"{MATH}.json").read_text())
    result = {
        "model_id": "example/model-1b",
        "zero_shot_bpb": 1.0,
        "adapted_bpb": 0.7,
        "dataset_sha256": manifest["distribution"]["sha256"],
        "max_seq_len": 512,
        "mask_injected_special_tokens": True,
        "train_steps": 250,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "lr": 0.0001, "effective_batch": 32},
    }
    result.update(overrides)
    return result


def _prepare(tmp_path, result, monkeypatch):
    monkeypatch.chdir(ROOT)  # manifests are resolved relative to the repository root
    result_path = tmp_path / "result.json"
    raw = json.dumps(result).encode()
    result_path.write_bytes(raw)
    args = argparse.Namespace(result=str(result_path), benchmark_id=MATH, model_name="Example-1B",
                              model_revision="abc123", tokenizer_revision="def456", parameters="1B")
    return prepare_submission(args), raw


def test_prepare_and_add_submission(tmp_path, monkeypatch):
    submission, raw = _prepare(tmp_path, _math_result(), monkeypatch)
    validate_submission(submission)
    assert submission["source_result_sha256"] == hashlib.sha256(raw).hexdigest()

    board = json.loads((ROOT / "results" / "leaderboard_math.json").read_text())
    original = copy.deepcopy(board)
    add_submission(board, submission)
    assert len(board["entries"]) == len(original["entries"]) + 1
    assert [entry["rank"] for entry in board["entries"]] == list(range(1, len(board["entries"]) + 1))
    assert [entry["adapted_bpb"] for entry in board["entries"]] == sorted(
        entry["adapted_bpb"] for entry in board["entries"]
    )


@pytest.mark.parametrize("override,match", [
    ({"dataset_sha256": "0" * 64}, "dataset_sha256"),
    ({"train_steps": 1000}, "fixed-recipe track"),
    ({"lora": {"r": 64, "alpha": 128, "dropout": 0.0, "lr": 3e-4, "effective_batch": 16}}, "fixed-recipe track"),
    ({"max_seq_len": 1024}, "max_seq_len"),
])
def test_submission_must_match_the_track(tmp_path, monkeypatch, override, match):
    with pytest.raises(LeaderboardError, match=match):
        _prepare(tmp_path, _math_result(**override), monkeypatch)


def test_submission_rejected_by_the_other_track(tmp_path, monkeypatch):
    submission, _ = _prepare(tmp_path, _math_result(), monkeypatch)
    news = json.loads((ROOT / "results" / "leaderboard.json").read_text())
    with pytest.raises(LeaderboardError, match="but board uses"):
        add_submission(news, submission)


MATH_DOCS = "arxiv-math-2026-08-documents-v1"


def _doc_result(**doc_overrides):
    """A result as tools/domain_transfer_eval.py --zero-shot-only --doc-context 2048 writes it."""
    cell = json.loads((ROOT / "results" / "doc_protocol_cells" /
                       "math_doc__Qwen_Qwen2.5-0.5B.json").read_text())
    cell["model_id"] = "example/model-1b"
    cell["doc_protocol"].update(doc_overrides)
    return cell


def _prepare_doc(tmp_path, result, monkeypatch):
    monkeypatch.chdir(ROOT)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    args = argparse.Namespace(result=str(path), benchmark_id=MATH_DOCS, model_name="Example-1B",
                              model_revision="abc123", tokenizer_revision="def456", parameters="1B")
    return prepare_submission(args)


def test_document_boards_are_valid_and_complete():
    for name in ("leaderboard_news_documents.json", "leaderboard_math_documents.json"):
        board = json.loads((ROOT / "results" / name).read_text())
        validate_board(board)
        assert len(board["entries"]) == 17
        assert render_table(board).startswith("| Rank | Model | Parameters | Document BPB |")


def test_prepare_and_add_document_submission(tmp_path, monkeypatch):
    submission = _prepare_doc(tmp_path, _doc_result(), monkeypatch)
    assert submission["metric"] == "document_bits_per_byte" and "adapted_bpb" not in submission
    board = json.loads((ROOT / "results" / "leaderboard_math_documents.json").read_text())
    add_submission(board, submission)
    assert [e["doc_bpb"] for e in board["entries"]] == sorted(e["doc_bpb"] for e in board["entries"])


@pytest.mark.parametrize("override", [{"context": 512}, {"stride": 2048},
                                      {"zero_shot": {"bpb": 0.7, "bytes": 1000}}])
def test_document_submission_must_match_the_track(tmp_path, monkeypatch, override):
    with pytest.raises(LeaderboardError, match="scores whole documents"):
        _prepare_doc(tmp_path, _doc_result(**override), monkeypatch)


def test_document_submission_rejected_by_an_adapted_board(tmp_path, monkeypatch):
    submission = _prepare_doc(tmp_path, _doc_result(), monkeypatch)
    board = json.loads((ROOT / "results" / "leaderboard_math.json").read_text())
    board["dataset"]["benchmark_id"] = MATH_DOCS
    with pytest.raises(LeaderboardError, match="scored by"):
        add_submission(board, submission)
