"""The E9 collector must refuse a column that quietly violates the preregistration.

These test the ways a wrong column could pass, not the happy path. Each corresponds to a
hole found by exercising the collector against the real Probe B artifacts (job 56610639)
before the full 51-cell run was launched.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COLLECTOR = REPO / "tools" / "collect_e9.py"

JULY_GSM8K_HASH = "2330f4ebfcccaf66a892922df2819cdb1f118e448d076d3f42bdde4177678ac7"
ENV = ("PyTorch version: 2.4.1+cu118\n"
       "GPU models and configuration: GPU 0: NVIDIA A100-SXM-64GB\n"
       "Nvidia driver version: 535.274.02\n")

AUDIT = {
    "as_loaded": {"max_new_tokens": 2048}, "effective": {}, "stripped": {"max_new_tokens": 2048},
    "kept": {}, "versions": {"lm_eval": "0.4.12", "transformers": "5.11.0"},
    "model_class": "Qwen2ForCausalLM", "max_new_tokens_observed": 200, "generate_calls": 8,
}


def _cell(root, label, bench, *, limit=None, gpu="NVIDIA A100-SXM-64GB",
          batch="16", audit=True, score=0.34):
    d = root / label / bench / "run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_x.json").write_text(json.dumps({
        "results": {bench: {"exact_match,flexible-extract": score, "acc_norm,none": score,
                            "exact_match,custom-extract": score}},
        "versions": {bench: 3.0 if bench == "gsm8k" else 1.0},
        "task_hashes": {bench: JULY_GSM8K_HASH} if bench == "gsm8k" else {},
        "lm_eval_version": "0.4.12",
        "config": {"limit": limit, "batch_size": batch},
        "pretty_env_info": ENV.replace("NVIDIA A100-SXM-64GB", gpu),
    }))
    if audit:
        (root / label / bench / "gen_audit.json").write_text(json.dumps(AUDIT))


def _run(root):
    return subprocess.run([sys.executable, str(COLLECTOR), "--dry-run", "--cells", str(root)],
                          capture_output=True, text=True, cwd=REPO)


def test_a_limited_cell_is_never_collected_as_a_score(tmp_path):
    """Probe B's throughput cell has limit=5.0 and the same directory layout as a real one."""
    _cell(tmp_path, "Qwen2.5-0.5B", "gsm8k", limit=5.0)
    r = _run(tmp_path)
    assert r.returncode != 0
    assert "run with --limit and are not scores" in r.stdout


def test_a_cell_with_no_protocol_audit_is_rejected(tmp_path):
    _cell(tmp_path, "Qwen2.5-0.5B", "gsm8k", audit=False)
    r = _run(tmp_path)
    assert r.returncode != 0
    assert "no gen_audit.json" in r.stdout


def test_a_column_from_two_hosts_is_rejected(tmp_path):
    """The A100/H100 mixing this study measured, and once misreported."""
    _cell(tmp_path, "Qwen2.5-0.5B", "gsm8k")
    _cell(tmp_path, "Llama-3.2-1B", "gsm8k", gpu="NVIDIA H100 80GB HBM3")
    r = _run(tmp_path)
    assert r.returncode != 0
    assert "mixed GPUs" in r.stdout


def test_an_unparseable_gpu_string_is_a_problem_not_a_silent_pass(tmp_path):
    """A vacuous provenance check reads exactly like a passing one -- it must not."""
    _cell(tmp_path, "Qwen2.5-0.5B", "gsm8k", gpu="")
    r = _run(tmp_path)
    assert r.returncode != 0
    assert "vacuous" in r.stdout


def test_mixed_batch_sizes_are_rejected(tmp_path):
    """Declared deviation from July, but it must not vary WITHIN the column."""
    _cell(tmp_path, "Qwen2.5-0.5B", "gsm8k")
    _cell(tmp_path, "Llama-3.2-1B", "gsm8k", batch="1")
    r = _run(tmp_path)
    assert r.returncode != 0
    assert "mixed batch sizes" in r.stdout


def test_a_task_hash_mismatch_is_rejected(tmp_path):
    root = tmp_path
    _cell(root, "Qwen2.5-0.5B", "gsm8k")
    p = root / "Qwen2.5-0.5B" / "gsm8k" / "run" / "results_x.json"
    d = json.loads(p.read_text())
    d["task_hashes"]["gsm8k"] = "0" * 64
    p.write_text(json.dumps(d))
    r = _run(root)
    assert r.returncode != 0
    assert "task_hash mismatch" in r.stdout


def test_a_partial_column_is_never_staged(tmp_path):
    _cell(tmp_path, "Qwen2.5-0.5B", "gsm8k")
    r = _run(tmp_path)
    assert r.returncode != 0
    assert "not staging a partial column" in r.stdout


def test_a_renamed_label_cannot_silently_shrink_the_overlap_gate(tmp_path):
    """A full 51-cell column whose labels do not match July must not pass quietly.

    The agreement gate only constrains models it can pair with July. A typo or a rename
    would leave the pairing empty and the HALT rule inert, which reads exactly like a clean
    column -- the same failure mode as the vacuous GPU check.
    """
    for i in range(17):
        for b in ("gsm8k", "hellaswag", "mmlu_pro_1k"):
            _cell(tmp_path, f"NotAJulyLabel-{i}", b)
    r = _run(tmp_path)
    assert r.returncode != 0
    assert "would silently weaken the preregistered agreement gate" in r.stdout
