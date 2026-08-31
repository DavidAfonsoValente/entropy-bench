#!/usr/bin/env python3
"""Sequential benchmark driver for the base-model comparison.

Runs an lm-evaluation-harness task across the paper's base models via the vLLM
backend, one model at a time, saving results incrementally (resumable) and
per-example samples (--log_samples). Produces a summary table at the end.

Benchmarks:
  mmlu_pro_1k : 5-shot CoT, generate_until, exact_match (custom-extract regex).
                Requires the custom task built by build_mmlu_pro_1k.py.
  hellaswag   : 10-shot, multiple_choice loglikelihood, acc_norm (full 10,042).

Examples:
  python run_benchmark.py --benchmark hellaswag --models all
  python run_benchmark.py --benchmark mmlu_pro_1k --models gemma-4-31B Qwen3.5-35B-MoE --tp 2

Notes:
  * BOS is left to each tokenizer's default (add_bos_token unset) -> Gemma/Llama/
    Mistral prepend BOS, Qwen does not. This is model-appropriate and identical
    across both benchmarks.
  * Models >~20B need tensor parallelism (--tp 2 on 2x A100-80GB).
  * Llama-3.2-1B is gated: point HF_TOKEN / ~/.cache/huggingface/token at an
    account with an approved Llama license.
"""
import argparse, os, subprocess, json, glob, time

# label -> (HF id, extra per-model vLLM args). All are BASE checkpoints.
MODELS = {
    "Qwen2.5-0.5B":    ("Qwen/Qwen2.5-0.5B",                  ""),
    "Qwen2.5-1.5B":    ("Qwen/Qwen2.5-1.5B",                  ""),
    "Qwen2.5-7B":      ("Qwen/Qwen2.5-7B",                    ""),
    "Qwen3.5-4B":      ("Qwen/Qwen3.5-4B-Base",               ""),
    "Qwen3.5-9B":      ("Qwen/Qwen3.5-9B-Base",               ""),
    "Qwen3.5-35B-MoE": ("Qwen/Qwen3.5-35B-A3B-Base",          "tensor_parallel_size=2"),
    "gemma-4-12B":     ("google/gemma-4-12B",                 ""),
    "gemma-4-31B":     ("google/gemma-4-31B",                 "tensor_parallel_size=2"),
    "Ministral-3-14B": ("mistralai/Ministral-3-14B-Base-2512", ""),
    "LFM2.5-1.2B":     ("LiquidAI/LFM2.5-1.2B-Base",          ""),
    "Llama-3.2-1B":    ("meta-llama/Llama-3.2-1B",            ""),  # gated
}

BENCH = {
    "mmlu_pro_1k": dict(task="mmlu_pro_1k", num_fewshot=5,  metric="exact_match,custom-extract"),
    "hellaswag":   dict(task="hellaswag",   num_fewshot=10, metric="acc_norm,none"),
    # GSM8K reports both strict-match and flexible-extract; flexible is the fair
    # headline for base models (they reason correctly but skip the '####' format).
    "gsm8k":       dict(task="gsm8k",       num_fewshot=5,  metric="exact_match,flexible-extract"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BENCH))
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--tp", type=int, default=None, help="override tensor_parallel_size for all models")
    ap.add_argument("--python", default=os.path.expanduser("~/vllm_env/bin/python"))
    ap.add_argument("--tasks-path", default=os.path.expanduser("~/tasks"),
                    help="--include_path for custom tasks (needed for mmlu_pro_1k)")
    ap.add_argument("--out", default=os.path.expanduser("~/results"))
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--model-paths", default=None, metavar="JSON",
                    help="JSON file mapping label -> local checkpoint directory. Scores those "
                         "checkpoints instead of the released base weights, under the identical "
                         "task, few-shot and extraction settings, so adapted and base numbers are "
                         "directly comparable. Used by experiment E2 (docs/PLAN.md) to measure "
                         "what adaptation does to task accuracy rather than inferring it from "
                         "rank movement.")
    ap.add_argument("--label-suffix", default="",
                    help="appended to output directory names, e.g. '__news-adapted', so adapted "
                         "runs never overwrite the published base-model results")
    args = ap.parse_args()

    overrides = {}
    if args.model_paths:
        overrides = json.load(open(args.model_paths))
        unknown = [k for k in overrides if k not in MODELS]
        if unknown:
            raise SystemExit("--model-paths has labels outside the cohort: %s" % sorted(unknown))
        for label, path in overrides.items():
            if not os.path.isdir(path):
                raise SystemExit("--model-paths: %s -> %s is not a directory" % (label, path))
        # Without a suffix the adapted run writes into the same output directory and result
        # table as the published base-model run, which either skips it as already-scored or
        # overwrites it. Either way the two become indistinguishable, so refuse.
        if not args.label_suffix:
            raise SystemExit(
                "--model-paths requires --label-suffix (e.g. --label-suffix __math-adapted). "
                "Without it the adapted scores share output paths and the results table with "
                "the published base-model run.")
    elif args.label_suffix:
        raise SystemExit("--label-suffix is only meaningful together with --model-paths")

    b = BENCH[args.benchmark]
    labels = list(MODELS) if args.models == ["all"] else args.models
    outroot = os.path.join(args.out, args.benchmark); os.makedirs(outroot, exist_ok=True)
    table_path = os.path.join(args.out, f"{args.benchmark}{args.label_suffix}_table.json")
    default_mml = 8192 if args.benchmark == "mmlu_pro_1k" else 4096
    mml = args.max_model_len or default_mml

    def load():
        try: return json.load(open(table_path))
        except Exception: return {}
    def save(t): json.dump(t, open(table_path, "w"), indent=2)
    def score(outdir):
        fs = sorted(glob.glob(f"{outdir}/*/results_*.json"))
        if not fs: return None, None
        r = json.load(open(fs[-1]))["results"].get(b["task"], {})
        m, s = b["metric"], b["metric"].replace(",", "_stderr,", 1)
        return r.get(m), r.get(s)

    tbl = load()
    if overrides:
        # Scoring released base weights inside a run labelled "adapted" would silently produce a
        # zero delta and look like a real measurement. Fail instead.
        unmapped = [x for x in labels if x not in overrides]
        if unmapped:
            raise SystemExit(
                "--model-paths given but no checkpoint for %s. Every requested model must have "
                "a local path, or its released base weights would be scored and reported under "
                "the '%s' label." % (sorted(unmapped), args.label_suffix))

    for label in labels:
        hf, extra = MODELS[label]
        # A local checkpoint replaces only the weights; task, few-shot count, extraction regex
        # and the tensor-parallel decision all stay exactly as the published base run used them.
        hf = overrides.get(label, hf)
        if tbl.get(label, {}).get("score") is not None:
            print(f"[skip] {label} = {tbl[label]['score']}", flush=True); continue
        tp = args.tp if args.tp is not None else \
            (2 if "tensor_parallel_size=2" in extra else 1)
        margs = f"pretrained={hf},dtype=bfloat16,trust_remote_code=True," \
                f"max_model_len={mml},gpu_memory_utilization={args.gpu_mem},tensor_parallel_size={tp}"
        tag = label + args.label_suffix
        outdir = os.path.join(outroot, tag); logf = os.path.join(args.out, f"log_{args.benchmark}_{tag}.log")
        cmd = [args.python, "-m", "lm_eval", "--model", "vllm", "--model_args", margs,
               "--tasks", b["task"], "--num_fewshot", str(b["num_fewshot"]),
               "--include_path", args.tasks_path, "--log_samples", "--output_path", outdir]
        env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
        print(f"\n===== {label} ({hf})  tp={tp} =====", flush=True); t0 = time.time()
        try:
            with open(logf, "w") as lf:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=10800, check=True)
            sc, se = score(outdir)
            tbl[label] = {"hf": hf, "score": sc, "stderr": se,
                          "metric": b["metric"], "minutes": round((time.time()-t0)/60, 1)}
            print(f"[done] {label}: {b['metric']}={sc}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {label} rc={e.returncode}; see {logf}", flush=True)
            tbl[label] = {"hf": hf, "score": None, "error": f"rc={e.returncode}"}
        except subprocess.TimeoutExpired:
            print(f"[timeout] {label}", flush=True); tbl[label] = {"hf": hf, "score": None, "error": "timeout"}
        save(tbl)

    print(f"\n===== {args.benchmark} summary ({b['metric']}) =====", flush=True)
    for label in labels:
        r = tbl.get(label, {})
        print(f"{label:18} {r.get('score')}  {r.get('error','') or ''}", flush=True)


if __name__ == "__main__":
    main()
