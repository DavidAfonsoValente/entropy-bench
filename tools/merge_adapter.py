#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and write a standalone checkpoint.

Needed because the benchmark harness (``eval/run_benchmark.py``) scores models through vLLM,
which wants a single set of weights. Adapting and then benchmarking is experiment E2 in
``docs/PLAN.md``: it converts the paper's mechanism claim -- that domain-matched adaptation trades
general calibration for task-relevant calibration -- from an inference about rank movement into a
direct measurement of accuracy before and after.

    python tools/merge_adapter.py --adapter out/news__Qwen_Qwen2_5-7B \\
        --out merged/news__Qwen_Qwen2_5-7B

The adapter directory must contain ``adapter_provenance.json`` written by
``tools/domain_transfer_eval.py --save-adapter``; the base model id is read from it so a merge can
never silently pair an adapter with the wrong base.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="directory written by --save-adapter")
    p.add_argument("--out", required=True, help="destination for the merged checkpoint")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def _load_base(base_id, dtype, device_map):
    """Return the object the adapter was trained against, exactly as the adaptation loaded it.

    Mirrors ``tools/domain_transfer_eval.py``'s loader, and the word "exactly" is the whole point.
    That script loads multimodal checkpoints -- Ministral-3-14B carries a Mistral3Config, Qwen3.5 a
    Qwen3_5Config, both refused by AutoModelForCausalLM -- through the image-text-to-text auto
    class and adapts the WRAPPER, not its language submodule. The saved adapters prove it: their
    target_modules include ``merging_layer``, ``linear_1`` and ``linear_2``, which exist only in the
    multimodal projector. So the merge must apply the adapter to the same wrapper and save that
    wrapper, which is also the checkpoint vLLM wants -- saving a bare text submodel writes a
    text-only config (Qwen3_5TextConfig) where vLLM requires the full one and engine init aborts.
    """
    import transformers
    from transformers import AutoModelForCausalLM
    common = dict(device_map=device_map, trust_remote_code=True)

    def _try(cls):
        for kw in ("dtype", "torch_dtype"):
            try:
                return cls.from_pretrained(base_id, **{kw: dtype}, **common)
            except TypeError:
                if kw == "torch_dtype":
                    raise
        return None

    try:
        return _try(AutoModelForCausalLM)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        print(f"[merge] {base_id} is not a plain causal LM; loading the multimodal wrapper",
              flush=True)

    cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if cls is None:
        raise SystemExit(f"cannot load {base_id}: no multimodal auto class available")
    return _try(cls)


SIDE_FILES = ("preprocessor_config.json", "processor_config.json", "chat_template.json",
              "params.json", "tekken.json")


def _resolve_side_files(base_id):
    """Locate the non-weight config files ``save_pretrained`` does not carry over.

    vLLM refuses a local multimodal checkpoint without ``preprocessor_config.json``, even though
    nothing in this benchmark ever passes it an image, and ``AutoProcessor`` cannot be used to
    regenerate it -- importing it raises ImportError in the runner image, which has no optional
    vision dependencies. So the files are copied verbatim from the base checkpoint. ``base_id`` may
    be a local directory (``hf_hub_download`` would treat that as a repo id and fail) or a cached
    Hub repo; both are handled, and a file that is simply absent upstream is reported as absent
    rather than invented.
    """
    found = {}
    local = Path(base_id)
    if local.is_dir():
        for name in SIDE_FILES:
            if (local / name).is_file():
                found[name] = local / name
        return found
    from huggingface_hub import hf_hub_download
    for name in SIDE_FILES:
        try:
            found[name] = Path(hf_hub_download(base_id, name, local_files_only=True))
        except Exception:
            continue
    return found


def main() -> None:
    a = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    adapter = Path(a.adapter)
    provenance_path = adapter / "adapter_provenance.json"
    if not provenance_path.is_file():
        raise SystemExit(
            f"{provenance_path} is missing. Only adapters written by "
            "tools/domain_transfer_eval.py --save-adapter can be merged, because the base model "
            "id has to come from the run that produced the adapter rather than from a flag."
        )
    provenance = json.loads(provenance_path.read_text())
    base_id = provenance["model_id"]
    print(f"[merge] base={base_id}  adapter={adapter}  corpus={provenance.get('corpus')}",
          flush=True)

    dtype = getattr(torch, a.dtype)
    base = _load_base(base_id, dtype, a.device_map)

    # Decide whether this checkpoint needs a preprocessor, and locate the files, BEFORE writing
    # anything. A wrapper that declares a vision or audio tower is unloadable by vLLM without its
    # preprocessor_config.json, and discovering that after save_pretrained has run would leave a
    # half-written checkpoint where a valid one used to be.
    config_keys = set(base.config.to_dict())
    needs_processor = bool(config_keys & {"vision_config", "audio_config"})
    extras = _resolve_side_files(base_id)
    if needs_processor and "preprocessor_config.json" not in extras:
        raise SystemExit(
            f"{base_id} declares a multimodal tower but no preprocessor_config.json is available "
            f"next to its weights, and vLLM refuses a local checkpoint without one. Synthesising "
            f"the file would mean inventing model configuration, so this merge is refused. "
            f"Nothing was written to {a.out}.")

    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()

    # Write into a clean directory. Re-merging over a previous run's output can leave that run's
    # tokenizer or processor files behind, producing a checkpoint that loads but preprocesses with
    # another model's configuration -- a wrong answer rather than a crash.
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    merged.save_pretrained(str(out), safe_serialization=True)

    # Prefer the tokenizer saved alongside the adapter: it is provably the one the BPB run used.
    try:
        AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True).save_pretrained(str(out))
    except Exception:
        AutoTokenizer.from_pretrained(base_id, trust_remote_code=True).save_pretrained(str(out))

    for name, src in extras.items():
        shutil.copy(src, out / name)
        print(f"[merge] copied {name} from the base checkpoint", flush=True)

    shutil.copy(provenance_path, out / "adapter_provenance.json")
    (out / "merge_provenance.json").write_text(json.dumps({
        "base_model_id": base_id,
        "adapter_dir": str(adapter),
        "dtype": a.dtype,
        "merged_from": provenance,
    }, indent=2) + "\n")

    n_files = len(list(out.glob("*.safetensors")))
    if n_files == 0:
        raise SystemExit(f"merge produced no safetensors shards in {out}")
    print(f"[merge] wrote {n_files} shard(s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
