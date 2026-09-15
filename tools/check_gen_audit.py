#!/usr/bin/env python
"""Validate an E9 cell's gen_audit.json: did this cell really run July's protocol?

`test -s` is not validation -- a truncated or stale audit passes it. This parses the
audit and asserts the things that would otherwise let a non-uniform cell into the column:
the pinned stack, that nothing decode-altering survived into the effective config, and --
the direct measurement -- that the longest sequence `generate()` actually produced fits
the benchmark's July cap.

    python tools/check_gen_audit.py <path/to/gen_audit.json> <benchmark>

Exits nonzero with a reason on any failure.
"""

import json
import sys

# July's effective max_gen_toks per benchmark. gsm8k sets none of its own, so lm_eval's
# default of 256 applied; mmlu_pro_1k's _default_template_yaml sets 2048. hellaswag is
# loglikelihood-only and never calls generate().
CAPS = {"gsm8k": 256, "mmlu_pro_1k": 2048, "hellaswag": None}
PINNED = {"lm_eval": "0.4.12", "transformers": "5.11.0"}


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    path, bench = argv[1], argv[2]
    if bench not in CAPS:
        sys.exit(f"FATAL: unknown benchmark {bench!r}")

    try:
        with open(path) as fh:
            audit = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"FATAL: cannot read protocol audit {path}: {exc}")

    problems = []

    for field in ("as_loaded", "effective", "stripped", "kept", "versions", "model_class"):
        if field not in audit:
            problems.append(f"audit is missing {field!r} -- truncated or from an older wrapper")
    if problems:
        sys.exit("FATAL: " + "; ".join(problems))

    got = {k: audit["versions"].get(k) for k in PINNED}
    if got != PINNED:
        problems.append(f"stack drift: expected {PINNED}, audit records {got}")

    # Nothing that alters greedy decoding may survive into the config generate() saw.
    # suppress_tokens is deliberately NOT here: it is preserved as part of the checkpoint's
    # token identity (see eval/lm_eval_uniform.py's KEEP), because stripping it lets a unified
    # multimodal model emit modality tokens into a text benchmark.
    for field in ("max_new_tokens", "do_sample", "num_beams",
                  "repetition_penalty", "no_repeat_ngram_size", "min_new_tokens",
                  "bad_words_ids", "forced_eos_token_id", "begin_suppress_tokens"):
        if field in audit["effective"]:
            problems.append(f"effective config still carries {field}={audit['effective'][field]!r}")

    cap = CAPS[bench]
    observed = audit.get("max_new_tokens_observed")
    calls = audit.get("generate_calls", 0)
    if cap is None:
        if calls:
            problems.append(f"{bench} is loglikelihood-only but generate() ran {calls} times")
    else:
        if observed is None:
            problems.append(f"{bench} generates, but the audit recorded no generation at all")
        elif observed > cap:
            problems.append(
                f"a generation produced {observed} tokens against July's cap of {cap} -- "
                "the protocol was NOT uniform"
            )

    if problems:
        for p in problems:
            print(f"FATAL: {p}", file=sys.stderr)
        return 1

    print(f"[audit] {path}: ok -- {audit['model_class']}, stripped {sorted(audit['stripped'])}, "
          f"longest generation {observed} tokens (July cap {cap}), {calls} generate calls")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
