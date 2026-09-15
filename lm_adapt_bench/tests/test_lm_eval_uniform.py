"""The E9 decode protocol is uniform across all seventeen models, or the column is void.

Driven by the generation_config.json actually shipped by the seventeen E9 checkpoints,
read off the staged HF cache on 2026-09-07 (docs/RUN_LEDGER.md, "E9 -- generation_config
survey"). The point of the test is the invariant, not the wrapper's internals: after
normalisation every model must decode under a byte-identical policy, differing only in
its own token identity.

Runs on CPU with no weights -- needs `transformers` only.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

GenerationConfig = pytest.importorskip("transformers").GenerationConfig
from lm_eval_uniform import KEEP, normalise_config  # noqa: E402

# Verbatim from the staged snapshots. The three Qwen3.5-Base checkpoints ship no
# generation_config.json at all, which HF represents as one built from the model config.
SHIPPED = {
    "Qwen/Qwen2.5-0.5B": {"bos_token_id": 151643, "do_sample": False, "eos_token_id": 151643, "max_new_tokens": 2048},
    "Qwen/Qwen2.5-1.5B": {"bos_token_id": 151643, "do_sample": False, "eos_token_id": 151643, "max_new_tokens": 2048},
    "Qwen/Qwen2.5-3B": {"bos_token_id": 151643, "do_sample": False, "eos_token_id": 151643, "max_new_tokens": 2048},
    "Qwen/Qwen2.5-7B": {"bos_token_id": 151643, "do_sample": False, "eos_token_id": 151643, "max_new_tokens": 2048},
    "Qwen/Qwen2.5-14B": {"bos_token_id": 151643, "do_sample": False, "eos_token_id": 151643, "max_new_tokens": 2048},
    "meta-llama/Llama-3.2-1B": {"_from_model_config": True, "bos_token_id": 128000, "do_sample": True, "eos_token_id": 128001, "temperature": 0.6, "top_p": 0.9},
    "LiquidAI/LFM2.5-1.2B-Base": {"_from_model_config": True, "bos_token_id": 1, "eos_token_id": 7, "pad_token_id": 0},
    "mistralai/Ministral-3-14B-Base-2512": {"bos_token_id": 1, "eos_token_id": 2, "max_length": 262144, "pad_token_id": 11},
    "google/gemma-4-12B": {"bos_token_id": 2, "do_sample": True, "eos_token_id": 1, "pad_token_id": 0, "suppress_tokens": [258883, 258882], "temperature": 1.0, "top_k": 64, "top_p": 0.95},
    "google/gemma-4-31B": {"bos_token_id": 2, "do_sample": True, "eos_token_id": 1, "pad_token_id": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95},
    "HuggingFaceTB/SmolLM2-1.7B": {"_from_model_config": True, "bos_token_id": 0, "eos_token_id": 0},
    "tiiuae/Falcon3-7B-Base": {"_from_model_config": True, "bos_token_id": 11, "eos_token_id": 11},
    "tiiuae/Falcon3-10B-Base": {"_from_model_config": True, "eos_token_id": 11},
    "mistralai/Mistral-Nemo-Base-2407": {"_from_model_config": True, "bos_token_id": 1, "eos_token_id": 2},
    "Qwen/Qwen3.5-4B-Base": {},
    "Qwen/Qwen3.5-9B-Base": {},
    "Qwen/Qwen3.5-35B-A3B-Base": {},
}

# The fields that change greedy output. do_sample/temperature/top_p/top_k are excluded on
# purpose: both generate_until tasks pass do_sample=false explicitly and lm_eval forwards
# it, so those were never live -- they are stripped for hygiene, not correctness.
# suppress_tokens is intentionally absent: it is KEPT, as part of the checkpoint's token
# identity. Stripping it let gemma-4-12B emit <image|>/<audio|> into gsm8k -- 22 empty
# completions and -9.9 points -- which is what put it here.
LIVE_UNDER_GREEDY = (
    "max_new_tokens", "min_new_tokens", "min_length", "num_beams",
    "begin_suppress_tokens", "bad_words_ids", "no_repeat_ngram_size",
    "repetition_penalty", "forced_eos_token_id", "length_penalty",
)


def _clean(spec):
    return normalise_config(GenerationConfig(**spec))[0]


@pytest.mark.parametrize("model_id", sorted(SHIPPED))
def test_no_field_that_alters_greedy_decoding_survives(model_id):
    clean = _clean(SHIPPED[model_id])
    ref = GenerationConfig()
    for field in LIVE_UNDER_GREEDY:
        assert getattr(clean, field) == getattr(ref, field), (
            f"{model_id} still carries {field}={getattr(clean, field)!r}"
        )


@pytest.mark.parametrize("model_id", sorted(SHIPPED))
def test_token_identity_is_preserved(model_id):
    spec = SHIPPED[model_id]
    clean = _clean(spec)
    for field in KEEP:
        assert getattr(clean, field) == spec.get(field), (
            f"{model_id}: {field} was not carried through"
        )


def test_every_model_decodes_under_the_same_policy():
    """The invariant the whole column rests on: identical except for token identity."""
    policies = {}
    for model_id, spec in SHIPPED.items():
        body = json.loads(_clean(spec).to_json_string(use_diff=True))
        for field in KEEP:
            body.pop(field, None)
        body.pop("transformers_version", None)
        policies[model_id] = json.dumps(body, sort_keys=True)
    distinct = set(policies.values())
    assert len(distinct) == 1, f"models decode under {len(distinct)} different policies: {policies}"


def test_bookkeeping_fields_are_not_reported_as_stripped():
    """_from_model_config records that transformers synthesised the config, not a decode
    policy. Six of the seventeen carry it; reporting it would be noise in every audit."""
    for model_id in ("meta-llama/Llama-3.2-1B", "HuggingFaceTB/SmolLM2-1.7B",
                     "tiiuae/Falcon3-10B-Base"):
        stripped = normalise_config(GenerationConfig(**SHIPPED[model_id]))[2]
        assert "_from_model_config" not in stripped, model_id
        assert "transformers_version" not in stripped, model_id


def test_the_known_confounds_are_the_ones_reported_as_stripped():
    """Regression guard on the survey itself, not just on the outcome."""
    assert normalise_config(GenerationConfig(**SHIPPED["Qwen/Qwen2.5-0.5B"]))[2] == {
        "do_sample": False, "max_new_tokens": 2048,
    }
    # gemma-4-12B's suppress_tokens must SURVIVE, not be stripped: it is what stops a unified
    # image+text model emitting modality tokens into a text benchmark.
    clean, _, stripped = normalise_config(GenerationConfig(**SHIPPED["google/gemma-4-12B"]))
    assert "suppress_tokens" not in stripped
    assert clean.suppress_tokens == [258883, 258882]
    # gemma-4-31B ships none, so it keeps none -- the asymmetry inside one family is real.
    assert normalise_config(GenerationConfig(**SHIPPED["google/gemma-4-31B"]))[0].suppress_tokens is None


def test_a_config_that_would_not_normalise_fails_loud():
    """If a future checkpoint ships something KEEP does not cover, the run must abort."""
    import lm_eval_uniform

    bad = GenerationConfig(repetition_penalty=1.2)
    saved = lm_eval_uniform.KEEP
    lm_eval_uniform.KEEP = saved + ("repetition_penalty",)  # simulate a careless widening
    try:
        with pytest.raises(RuntimeError, match="normalisation did not take"):
            normalise_config(bad)
    finally:
        lm_eval_uniform.KEEP = saved


# --------------------------------------------------------------------------------------
# tools/check_gen_audit.py -- the gate that decides whether a finished cell counts. It is
# what stands between a stale or non-uniform artifact and the published column, so it gets
# tested on the ways a cell could wrongly pass rather than only on the happy path.
# --------------------------------------------------------------------------------------

import subprocess  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "tools" / "check_gen_audit.py"

GOOD_AUDIT = {
    "as_loaded": {"bos_token_id": 151643, "do_sample": False, "eos_token_id": 151643,
                  "max_new_tokens": 2048},
    "effective": {"bos_token_id": 151643, "eos_token_id": 151643},
    "stripped": {"do_sample": False, "max_new_tokens": 2048},
    "kept": {"bos_token_id": 151643, "eos_token_id": 151643, "pad_token_id": None},
    "versions": {"lm_eval": "0.4.12", "transformers": "5.11.0", "torch": "2.4.1+cu118"},
    "model_class": "Qwen2ForCausalLM",
    "max_new_tokens_observed": 187,
    "generate_calls": 83,
}


def _check(tmp_path, audit, bench="gsm8k"):
    p = tmp_path / "gen_audit.json"
    p.write_text(json.dumps(audit))
    return subprocess.run([sys.executable, str(CHECKER), str(p), bench],
                          capture_output=True, text=True)


def test_a_clean_audit_passes(tmp_path):
    r = _check(tmp_path, GOOD_AUDIT)
    assert r.returncode == 0, r.stderr


def test_a_generation_over_julys_cap_is_rejected(tmp_path):
    """The failure this whole intervention exists to prevent."""
    bad = {**GOOD_AUDIT, "max_new_tokens_observed": 2048}
    r = _check(tmp_path, bad)
    assert r.returncode != 0
    assert "was NOT uniform" in r.stderr


def test_the_same_generation_length_is_fine_under_mmlus_larger_cap(tmp_path):
    """2048 is a violation for gsm8k and the intended protocol for mmlu_pro_1k."""
    assert _check(tmp_path, {**GOOD_AUDIT, "max_new_tokens_observed": 2048},
                  bench="mmlu_pro_1k").returncode == 0


def test_a_generating_task_that_recorded_no_generation_is_rejected(tmp_path):
    """Catches a wrapper whose _model_generate patch silently did not take."""
    bad = {k: v for k, v in GOOD_AUDIT.items() if k != "max_new_tokens_observed"}
    r = _check(tmp_path, bad)
    assert r.returncode != 0
    assert "recorded no generation" in r.stderr


def test_hellaswag_must_not_generate_at_all(tmp_path):
    """hellaswag is loglikelihood-only; a generate() call there means the wrong task ran."""
    r = _check(tmp_path, GOOD_AUDIT, bench="hellaswag")
    assert r.returncode != 0
    assert "loglikelihood-only" in r.stderr


def test_stack_drift_is_rejected(tmp_path):
    bad = {**GOOD_AUDIT, "versions": {**GOOD_AUDIT["versions"], "transformers": "5.14.1"}}
    r = _check(tmp_path, bad)
    assert r.returncode != 0
    assert "stack drift" in r.stderr


def test_a_decode_altering_field_surviving_into_effective_is_rejected(tmp_path):
    # repetition_penalty, not suppress_tokens: the latter is now deliberately preserved as part
    # of the checkpoint's token identity, so it is no longer an example of a forbidden field.
    bad = {**GOOD_AUDIT, "effective": {**GOOD_AUDIT["effective"], "repetition_penalty": 1.2}}
    r = _check(tmp_path, bad)
    assert r.returncode != 0
    assert "repetition_penalty" in r.stderr


def test_an_audit_from_an_older_wrapper_is_rejected(tmp_path):
    """A truncated or pre-instrumentation audit must not read as a pass."""
    bad = {k: v for k, v in GOOD_AUDIT.items() if k not in ("versions", "model_class")}
    r = _check(tmp_path, bad)
    assert r.returncode != 0
    assert "missing" in r.stderr


def test_a_missing_audit_is_rejected(tmp_path):
    r = subprocess.run([sys.executable, str(CHECKER), str(tmp_path / "absent.json"), "gsm8k"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "cannot read protocol audit" in r.stderr
