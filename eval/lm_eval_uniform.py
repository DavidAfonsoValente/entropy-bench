#!/usr/bin/env python
"""Run lm_eval's HF backend with a decode protocol that is identical across models.

Why this exists
---------------
E9 re-scores the whole public-benchmark column on Leonardo. No vLLM wheel compatible
with ``transformers>=5`` runs on this cluster's CUDA 12.2 driver, so the column moves to
lm_eval's ``hf`` backend -- and that backend is not protocol-uniform out of the box.

``HFLM._model_generate`` passes ``max_length`` to ``generate()`` but never
``max_new_tokens``. HF gives ``max_new_tokens`` priority over ``max_length``, so a
``max_new_tokens`` shipped in a model's ``generation_config.json`` silently outranks
lm_eval's length control. Measured across the seventeen E9 models (the table in
``docs/RUN_LEDGER.md``, "E9 -- the generation_config survey"):

  * the five Qwen2.5 models ship ``max_new_tokens: 2048`` -- observed live in probe
    56341304 as ``max_new_tokens (=2048)`` on a gsm8k cell whose July protocol was 256;
  * gemma-4-12B ships ``suppress_tokens: [258883, 258882]`` and gemma-4-31B does not,
    so two models in the same family decode over different output vocabularies;
  * Ministral-3 ships ``max_length: 262144`` (inert -- lm_eval passes max_length itself).

``do_sample``/``temperature``/``top_p``/``top_k`` (Llama-3.2, gemma-4) are NOT part of the
problem: both generate_until tasks set ``do_sample: false`` in their own config and
lm_eval forwards task ``generation_kwargs`` to ``generate()`` as explicit keyword
arguments, which outrank the loaded config (``normalize_gen_kwargs``,
``lm_eval/models/utils.py:607``: "All other kwargs passed through unchanged"). They are
stripped anyway so the effective config is identical across models.

What it does
------------
1. Wraps ``HFLM.__init__`` so that, once the model is built, the ``generation_config`` of
   **the object lm_eval will actually call ``generate()`` on** is replaced by a
   library-default ``GenerationConfig`` carrying only that checkpoint's token identity
   (``bos``/``eos``/``pad``). Note ``HFLM.model`` is a property that unwraps through
   Accelerate; mutating ``_model`` instead can set the attribute on a wrapper while
   generation reads the inner model's original config.
2. Wraps ``HFLM._model_generate`` to assert, on every call, that nothing has restored a
   ``max_new_tokens`` on that same object, and to record the **true** number of generated
   token ids (``out.shape[1] - context.shape[1]``). That is a direct measurement of the
   realised cap, not an inference from decoded text or from an absent warning.
3. Writes both to ``gen_audit.json`` per cell, so the protocol of every cell in the column
   is auditable after the fact.

Scope: ordinary decoder-only causal LMs, which is what all seventeen E9 models are. An
encoder-decoder, a custom ``GenerationConfig`` subclass, or a model that is not a
``GenerationMixin`` aborts rather than being silently normalised.

Usage
-----
    python eval/lm_eval_uniform.py --gen-audit PATH -- <normal lm_eval args>
"""

import json
import os
import sys

# Token identity is a property of the checkpoint and must survive; everything else in a
# loaded generation_config is a decode policy this column sets centrally.
#
# suppress_tokens is kept for the same reason as the token ids, and it was NOT obvious. A
# unified image+text checkpoint (gemma-4-12B here) suppresses its own modality tokens so that
# text-only decoding cannot emit them. Strip it and the model emits <image|>/<audio|> mid-answer:
# measured on gsm8k, 22 of 2638 completions came back empty and the score fell 9.9 points, while
# the same run with suppression active reproduces the reference column (0 empty, 0 markers).
# Suppressing tokens a text benchmark cannot consume is part of operating the checkpoint, not a
# decode policy this column gets to choose. The rule stays uniform across all seventeen models;
# only this one ships such a list.
KEEP = ("bos_token_id", "eos_token_id", "pad_token_id", "suppress_tokens")

# Fields that must sit at the library default once normalisation has run. Checked against a
# freshly constructed GenerationConfig rather than hardcoded literals, because the defaults
# are transformers-version dependent (min_length is None in 5.11, 0 in older releases) and a
# stale literal would fail a correct run. What this actually guards is KEEP: widen it by
# accident to something that alters decoding and the run aborts instead of scoring.
MUST_BE_DEFAULT = (
    "max_new_tokens", "min_new_tokens", "min_length", "max_length",
    "do_sample", "num_beams", "temperature", "top_p", "top_k",
    "repetition_penalty", "no_repeat_ngram_size",
    "begin_suppress_tokens", "bad_words_ids", "forced_eos_token_id", "length_penalty",
)


def _serialisable(cfg):
    """The non-default fields of a GenerationConfig, as plain JSON."""
    return json.loads(cfg.to_json_string(use_diff=True))


def _check_scope(model):
    """Abort on any model whose generation semantics this normalisation does not cover."""
    from transformers import GenerationConfig
    from transformers.generation import GenerationMixin

    if not isinstance(model, GenerationMixin):
        raise RuntimeError(f"{type(model).__name__} is not a GenerationMixin -- out of scope")
    # A vision-language model driven with text-only prompts is in scope: its decode path is
    # the same causal one, and nothing here feeds it images. Anything else is not.
    if getattr(model.config, "is_encoder_decoder", False):
        raise RuntimeError(
            f"{type(model).__name__} is an encoder-decoder; it may need decoder_start_token_id "
            "or forced_bos_token_id, which this normalisation would strip -- out of scope"
        )
    cfg = model.generation_config
    if type(cfg) is not GenerationConfig:
        raise RuntimeError(
            f"{type(model).__name__} uses a custom {type(cfg).__name__}; replacing it with the "
            "base class could drop fields its own generate() reads -- out of scope"
        )


def normalise_config(loaded):
    """Return (clean GenerationConfig, as-loaded-as-json, stripped-as-json).

    Pure surgery on a GenerationConfig, kept free of any model object so it can be tested
    against the seventeen real configs without loading weights. Raises if the result is not
    at library defaults on every decode-altering field.
    """
    from transformers import GenerationConfig

    if loaded is None:
        raise RuntimeError("model has no generation_config -- refusing to guess a protocol")

    before = _serialisable(loaded)
    clean = GenerationConfig()
    for field in KEEP:
        setattr(clean, field, getattr(loaded, field, None))

    reference = GenerationConfig()
    for field in MUST_BE_DEFAULT:
        default = getattr(reference, field)
        got = getattr(clean, field)
        if got != default:
            raise RuntimeError(
                f"generation_config normalisation did not take: {field}={got!r}, "
                f"expected {default!r}. The decode protocol is NOT uniform -- refusing to score."
            )

    # transformers_version and _from_model_config are bookkeeping, not decode policy: the
    # latter only records that transformers synthesised this config from the model config
    # rather than reading a generation_config.json. Reporting them as "stripped" would put
    # noise in the audit of every checkpoint that ships no generation_config.json.
    metadata = ("transformers_version", "_from_model_config")
    stripped = {k: v for k, v in before.items() if k not in KEEP and k not in metadata}
    return clean, before, stripped


def _normalise(lm, audit):
    # HFLM.model unwraps through Accelerate; this is the object _model_generate calls.
    model = lm.model
    _check_scope(model)
    clean, before, stripped = normalise_config(model.generation_config)
    model.generation_config = clean
    audit.update(
        # "as_loaded", not "shipped": transformers synthesises a GenerationConfig from the
        # model config when a checkpoint ships no generation_config.json (the three
        # Qwen3.5-Base models here), and this records what was loaded either way.
        as_loaded=before,
        effective=_serialisable(clean),
        stripped=stripped,
        kept={k: getattr(clean, k, None) for k in KEEP},
        model_class=type(model).__name__,
        auto_model_class=type(lm.AUTO_MODEL_CLASS).__name__
        if not hasattr(lm.AUTO_MODEL_CLASS, "__name__") else lm.AUTO_MODEL_CLASS.__name__,
        max_length_lm_eval_will_use=lm.max_length,
    )
    print("[uniform] model class :", type(model).__name__, flush=True)
    print("[uniform] as loaded   :", json.dumps(before, sort_keys=True), flush=True)
    print("[uniform] stripped    :", json.dumps(stripped, sort_keys=True), flush=True)
    print("[uniform] effective   :", json.dumps(audit["effective"], sort_keys=True), flush=True)
    print("[uniform] lm_eval max_length :", lm.max_length, flush=True)
    if not stripped:
        print("[uniform] nothing to strip -- this model already ran July's protocol", flush=True)


def _refuse_vlm_wrapper(pretrained, exc):
    """Abort with instructions instead of loading a checkpoint the wrong way.

    An earlier revision fell back to AutoModelForImageTextToText here. That LOADED
    Ministral-3-14B without any warning and scored 0.0000 on gsm8k and 0.2848 on hellaswag
    (chance is 0.25), because the checkpoint stores `language_model.model.*` and
    `language_model.lm_head.weight` while transformers 5.11's
    Mistral3ForConditionalGeneration expects `model.language_model.*` plus an lm_head it
    builds in __init__ -- disjoint key spaces, so the whole model was randomly initialised.

    A fallback that silently yields a random model is worse than no fallback: it passes
    exit-0, task_hash, the protocol audit and the cap check, and only a comparison against a
    known score catches it. So this refuses, loudly, and says what to do instead.
    """
    raise RuntimeError(
        f"{pretrained} is not loadable as a causal LM: {exc}\n"
        "Do NOT reach for AutoModelForImageTextToText -- for a checkpoint whose weights live "
        "under a `language_model.` prefix it loads a RANDOMLY INITIALISED model without "
        "warning and every gate in this pipeline still passes.\n"
        "If the checkpoint is a causal LM under a prefix, materialise a corrected copy with "
        "tools/materialise_prefixed_text_model.py (which verifies the key mapping exactly) "
        "and point the manifest at that path."
    )


def _patch(audit):
    from lm_eval.models.huggingface import HFLM

    original_init = HFLM.__init__
    original_generate = HFLM._model_generate
    original_create = HFLM._create_model

    def patched_create(self, pretrained, *args, **kwargs):
        try:
            return original_create(self, pretrained, *args, **kwargs)
        except ValueError as exc:
            if "Unrecognized configuration class" not in str(exc) or not isinstance(pretrained, str):
                raise
            # Ministral-3-14B ships a vision-language Mistral3Config that transformers 5.11
            # does not register for AutoModelForCausalLM, though its nested text_config
            # (Ministral3Config) is registered. July's vLLM loaded the same checkpoint and
            # drove its text tower without difficulty, so refusing it here would drop a model
            # from the preregistered overlap set because of a backend change forced on us.
            _refuse_vlm_wrapper(pretrained, str(exc).split("\n")[0])

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _normalise(self, audit)

    def patched_generate(self, *args, **kwargs):
        # Guard the object that is about to generate, not the one we mutated at init.
        live = self.model.generation_config
        if live.max_new_tokens is not None:
            raise RuntimeError(
                f"decode protocol drifted before generate(): max_new_tokens={live.max_new_tokens!r}"
            )
        out = original_generate(self, *args, **kwargs)
        context = kwargs.get("context", args[0] if args else None)
        if context is not None:
            produced = int(out.shape[1] - context.shape[1])
            audit["max_new_tokens_observed"] = max(
                audit.get("max_new_tokens_observed", 0), produced
            )
            audit["generate_calls"] = audit.get("generate_calls", 0) + 1
        return out

    HFLM.__init__ = patched_init
    HFLM._model_generate = patched_generate
    HFLM._create_model = patched_create


def _parse_args(argv):
    """Split off --gen-audit; everything else goes to lm_eval untouched."""
    audit_path = None
    rest = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            rest.extend(argv[i + 1:])
            break
        if arg == "--gen-audit" or arg.startswith("--gen-audit="):
            if audit_path is not None:
                sys.exit("FATAL: --gen-audit given more than once")
            if arg.startswith("--gen-audit="):
                audit_path = arg.split("=", 1)[1]
                i += 1
            else:
                if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                    sys.exit("FATAL: --gen-audit needs a path")
                audit_path = argv[i + 1]
                i += 2
            if not audit_path:
                sys.exit("FATAL: --gen-audit needs a non-empty path")
            continue
        rest.append(arg)
        i += 1
    return audit_path, rest


def main():
    audit_path, rest = _parse_args(sys.argv[1:])

    import lm_eval
    import torch
    import transformers

    audit = {
        "argv": rest,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "lm_eval": lm_eval.__version__,
        },
    }
    _patch(audit)

    from lm_eval.__main__ import cli_evaluate

    sys.argv = ["lm_eval"] + rest
    try:
        cli_evaluate()
    finally:
        # Written even on failure: a cell that died after loading still recorded which
        # protocol it was going to run under, which is what a post-mortem needs.
        if audit_path:
            os.makedirs(os.path.dirname(os.path.abspath(audit_path)), exist_ok=True)
            with open(audit_path, "w") as fh:
                json.dump(audit, fh, indent=2, sort_keys=True)
            obs = audit.get("max_new_tokens_observed")
            if obs is not None:
                print(f"[uniform] longest generation actually produced: {obs} tokens "
                      f"over {audit.get('generate_calls', 0)} batched calls", flush=True)
            print(f"[uniform] audit -> {audit_path}", flush=True)


if __name__ == "__main__":
    main()
