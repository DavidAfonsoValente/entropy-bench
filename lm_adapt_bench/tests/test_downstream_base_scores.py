"""The base-weights baseline must be recomputable in a release tree, where the raw fine-tune
generations (SCRATCH) are absent: base_task_scores() then reads the committed per-model scores."""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def test_base_task_scores_fall_back_to_committed(tmp_path, monkeypatch):
    monkeypatch.setenv("E11_DIR", str(tmp_path))  # an empty raw-cell directory
    import analyze_downstream
    ad = importlib.reload(analyze_downstream)
    committed = json.loads(ad.OUTPUT_JSON.read_text())["base_task_scores"]
    assert len(committed) == 15
    assert ad.base_task_scores() == committed
