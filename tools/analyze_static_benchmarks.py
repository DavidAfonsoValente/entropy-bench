#!/usr/bin/env python3
"""Derive the paper's static-benchmark table and rank diagnostics.

The benchmark accuracies are committed in ``results/combined_bpb_vs_static.json``.
The BPB values used here come from the corrected, marker-masked, fixed-protocol
news rerun in ``results/domain_transfer``.  This keeps the headline comparison
on exactly the same protocol as the cross-domain experiment.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC_PATH = ROOT / "results" / "combined_bpb_vs_static.json"
DOMAIN_DIR = ROOT / "results" / "domain_transfer"
OUTPUT_JSON = ROOT / "results" / "static_benchmark_analysis.json"
OUTPUT_TEX = ROOT / "static_benchmark_table.tex"
OUTPUT_TSV = ROOT / "results" / "static_benchmark_scatter.tsv"
OUTPUT_FIGURE_TEX = ROOT / "figures" / "fig_benchmark_alignment.tex"

BENCHMARKS = {
    "mmlu_pro": {"label": "MMLU-Pro", "n": 1000, "raw_dir": "mmlu_pro_1k",
                 "raw_task": "mmlu_pro_1k", "raw_metric": "exact_match,custom-extract"},
    "hellaswag": {"label": "HellaSwag", "n": 10042, "raw_dir": "hellaswag",
                  "raw_task": "hellaswag", "raw_metric": "acc_norm,none"},
    "gsm8k": {"label": "GSM8K", "n": 1319, "raw_dir": "gsm8k",
              "raw_task": "gsm8k", "raw_metric": "exact_match,flexible-extract"},
}

MODEL_IDS = {
    "gemma-4-31B": "google/gemma-4-31B",
    "gemma-4-12B": "google/gemma-4-12B",
    "Ministral-3-14B": "mistralai/Ministral-3-14B-Base-2512",
    "Qwen3.5-35B-MoE": "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B-Base",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B-Base",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "LFM2.5-1.2B": "LiquidAI/LFM2.5-1.2B-Base",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
}

DISPLAY = {
    "gemma-4-31B": "Gemma-4-31B",
    "gemma-4-12B": "Gemma-4-12B",
    "Ministral-3-14B": "Ministral-3-14B",
    "Qwen3.5-35B-MoE": "Qwen-3.5-35B-MoE",
    "Qwen3.5-9B": "Qwen-3.5-9B",
    "Qwen2.5-7B": "Qwen-2.5-7B",
    "Qwen3.5-4B": "Qwen-3.5-4B",
    "Llama-3.2-1B": "Llama-3.2-1B",
    "Qwen2.5-1.5B": "Qwen-2.5-1.5B",
    "LFM2.5-1.2B": "LiquidAI-LFM2.5",
    "Qwen2.5-0.5B": "Qwen-2.5-0.5B",
}

PLOT_LABEL = {
    "gemma-4-31B": "G31",
    "gemma-4-12B": "G12",
    "Ministral-3-14B": "M14",
    "Qwen3.5-35B-MoE": "Q35",
    "Qwen3.5-9B": "Q9",
    "Qwen2.5-7B": "Q7",
    "Qwen3.5-4B": "Q4",
    "Llama-3.2-1B": "L1",
    "Qwen2.5-1.5B": "Q1.5",
    "LFM2.5-1.2B": "LFM",
    "Qwen2.5-0.5B": "Q0.5",
}


def load_raw_static(model: str, task: str) -> tuple[float, int]:
    """Exact (accuracy, n) for one model/task, read from the committed raw harness output
    under results/<raw_dir>/<model>/*/results_*.json (see eval/README.md)."""
    meta = BENCHMARKS[task]
    matches = sorted((ROOT / "results" / meta["raw_dir"] / model).glob("*/results_*.json"))
    if not matches:
        raise FileNotFoundError(f"no raw results_*.json for {model}/{task}")
    d = json.loads(matches[-1].read_text())
    row = d["results"][meta["raw_task"]]
    return row[meta["raw_metric"]], meta["n"]


def ranks(values: dict[str, float], *, lower_is_better: bool) -> dict[str, float]:
    """Return average ranks, preserving a sensible definition if ties appear."""
    ordered = sorted(values, key=values.get, reverse=not lower_is_better)
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for key in ordered[start:end]:
            result[key] = average_rank
        start = end
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator


def kendall_tau_a(xs: list[float], ys: list[float]) -> float:
    concordant = discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            product = (xs[i] - xs[j]) * (ys[i] - ys[j])
            concordant += product > 0
            discordant += product < 0
    return (concordant - discordant) / math.comb(len(xs), 2)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def two_proportion_z(p1: float, p2: float, n1: int, n2: int) -> float:
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se


def load_corrected_news_bpb() -> dict[str, float]:
    by_id = {}
    for path in sorted(DOMAIN_DIR.glob("news__*.json")):
        row = json.loads(path.read_text())
        if not row.get("mask_injected_special_tokens"):
            raise ValueError(f"{path} did not mask injected special-token targets")
        by_id[row["model_id"]] = row["adapted_bpb"]
    expected = set(MODEL_IDS.values())
    if set(by_id) != expected:
        raise ValueError(
            f"news BPB coverage mismatch: missing={sorted(expected - set(by_id))}, "
            f"extra={sorted(set(by_id) - expected)}"
        )
    return {label: by_id[model_id] for label, model_id in MODEL_IDS.items()}


def build_analysis() -> dict:
    static = json.loads(STATIC_PATH.read_text())
    if set(static) != set(MODEL_IDS):
        raise ValueError("static benchmark model set does not match the 11-model cohort")

    bpb = load_corrected_news_bpb()
    bpb_rank = ranks(bpb, lower_is_better=True)
    task_ranks = {
        task: ranks({m: static[m][task] for m in static}, lower_is_better=False)
        for task in BENCHMARKS
    }

    rows = []
    for model in sorted(static, key=bpb.get):
        scores = {}
        for task, metadata in BENCHMARKS.items():
            committed_accuracy = static[model][task]
            exact_accuracy, n = load_raw_static(model, task)
            # The committed table rounds to 3-4 decimals; the raw harness output must agree
            # with that rounding, or the committed number and the raw log have diverged.
            if abs(exact_accuracy - committed_accuracy) > 0.00051:
                raise ValueError(
                    f"{model}/{task}: committed accuracy {committed_accuracy} disagrees with "
                    f"raw log accuracy {exact_accuracy}"
                )
            successes = round(exact_accuracy * n)
            if abs(successes / n - exact_accuracy) > 1e-9:
                raise ValueError(f"{model}/{task}: raw accuracy is not an exact k/{n} fraction")
            low, high = wilson_interval(successes, n)
            scores[task] = {
                "accuracy": committed_accuracy,
                "correct_count": successes,
                "n": n,
                "wilson_95_low": low,
                "wilson_95_high": high,
                "rank": task_ranks[task][model],
                "rank_shift_vs_bpb": task_ranks[task][model] - bpb_rank[model],
            }
        rows.append(
            {
                "model": DISPLAY[model],
                "source_label": model,
                "model_id": MODEL_IDS[model],
                "adapted_bpb": bpb[model],
                "bpb_rank": bpb_rank[model],
                "benchmarks": scores,
            }
        )

    alignment = {}
    models = list(static)
    bpb_values = [bpb[m] for m in models]
    bpb_ranks = [bpb_rank[m] for m in models]
    for task, metadata in BENCHMARKS.items():
        accuracies = [static[m][task] for m in models]
        accuracy_ranks = [task_ranks[task][m] for m in models]
        tau = kendall_tau_a(bpb_values, [-x for x in accuracies])
        shifts = [abs(bpb_rank[m] - task_ranks[task][m]) for m in models]
        ordered = sorted(models, key=lambda m: static[m][task], reverse=True)
        leader, runner_up = ordered[:2]
        alignment[task] = {
            "label": metadata["label"],
            "pearson_bpb_vs_accuracy": pearson(bpb_values, accuracies),
            "spearman_rank_alignment": pearson(bpb_ranks, accuracy_ranks),
            "kendall_tau_bpb_vs_error": tau,
            "pairwise_order_agreement": (tau + 1) / 2,
            "identical_ranks": sum(shift == 0 for shift in shifts),
            "mean_absolute_rank_shift": sum(shifts) / len(shifts),
            "max_absolute_rank_shift": max(shifts),
            "observed_leader": DISPLAY[leader],
            "observed_runner_up": DISPLAY[runner_up],
            "leader_gap_percentage_points": 100 * (static[leader][task] - static[runner_up][task]),
            "leader_gap_independent_binomial_z": two_proportion_z(
                static[leader][task], static[runner_up][task], metadata["n"], metadata["n"]
            ),
        }

    return {
        "schema_version": 1,
        "bpb_source": "results/domain_transfer/news__*.json",
        "bpb_protocol": {
            "adaptation": "fixed LoRA configuration, 250 update steps",
            "injected_special_token_targets_masked": True,
            "direction": "lower_is_better",
        },
        "accuracy_source": str(STATIC_PATH.relative_to(ROOT)),
        "accuracy_note": (
            "Directly observed lm-evaluation-harness point estimates. Correct counts are exact, "
            "read from the committed raw per-model harness output under "
            "results/{gsm8k,hellaswag,mmlu_pro_1k}/, cross-checked against the rounded accuracy "
            "in results/combined_bpb_vs_static.json. Wilson intervals describe test-set sampling "
            "uncertainty, not run-to-run or prompt uncertainty."
        ),
        "benchmarks": BENCHMARKS,
        "rows": rows,
        "alignment": alignment,
    }


def format_rank(rank: float) -> str:
    return str(int(rank)) if rank.is_integer() else f"{rank:.1f}"


def render_tex(analysis: dict) -> str:
    leaders = {task: details["observed_leader"] for task, details in analysis["alignment"].items()}
    lines = ["% generated by tools/analyze_static_benchmarks.py"]
    for row in analysis["rows"]:
        bpb = f"{row['adapted_bpb']:.3f} ({format_rank(row['bpb_rank'])})"
        if row["bpb_rank"] == 1:
            bpb = "\\textbf{" + bpb + "}"
        cells = [row["model"], bpb]
        for task in BENCHMARKS:
            score = row["benchmarks"][task]
            value = f"{100 * score['accuracy']:.1f} ({format_rank(score['rank'])})"
            if row["model"] == leaders[task]:
                value = "\\textbf{" + value + "}"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def render_tsv(analysis: dict) -> str:
    columns = ["model", "adapted_bpb"]
    for task in BENCHMARKS:
        columns.extend((task, task + "_low", task + "_high"))
    columns.extend(("family", "plot_label"))
    lines = ["\t".join(columns)]
    for row in analysis["rows"]:
        source_label = row["source_label"]
        family = 1 if source_label.startswith("Qwen") else 2 if source_label.startswith("gemma") else 3
        task_values = []
        for task in BENCHMARKS:
            score = row["benchmarks"][task]
            task_values.extend(
                (
                    f"{score['accuracy']:.9f}",
                    f"{score['wilson_95_low']:.9f}",
                    f"{score['wilson_95_high']:.9f}",
                )
            )
        lines.append(
            "\t".join(
                [
                    row["model"],
                    f"{row['adapted_bpb']:.9f}",
                    *task_values,
                    str(family),
                    PLOT_LABEL[source_label],
                ]
            )
        )
    return "\n".join(lines) + "\n"


def render_figure_tex(analysis: dict) -> str:
    """Render a dependency-free three-panel TikZ plot with binomial intervals."""
    panels = (
        ("mmlu_pro", "MMLU-Pro", 0.05, 0.68, (0.2, 0.4, 0.6)),
        ("hellaswag", "HellaSwag", 0.48, 0.91, (0.5, 0.7, 0.9)),
        ("gsm8k", "GSM8K", 0.00, 0.93, (0.2, 0.5, 0.8)),
    )
    xmin, xmax, width, height, gap = 0.61, 0.92, 3.85, 2.90, 0.45
    x_ticks = (0.65, 0.75, 0.85)
    colors = {1: "benchqwen", 2: "benchgemma", 3: "benchother"}
    key_labels = {"G31", "G12", "Q35", "Q9"}
    label_shift = {
        ("mmlu_pro", "G31"): (0.08, -0.10),
        ("mmlu_pro", "G12"): (0.08, -0.10),
        ("mmlu_pro", "Q35"): (0.08, 0.09),
        ("mmlu_pro", "Q9"): (0.08, 0.09),
        ("hellaswag", "G31"): (0.08, 0.08),
        ("hellaswag", "G12"): (0.08, 0.08),
        ("hellaswag", "Q35"): (0.08, -0.11),
        ("hellaswag", "Q9"): (0.08, -0.11),
        ("gsm8k", "G31"): (0.08, -0.11),
        ("gsm8k", "G12"): (0.08, -0.11),
        ("gsm8k", "Q35"): (0.08, 0.09),
        ("gsm8k", "Q9"): (0.08, 0.09),
    }

    def tx(value: float) -> float:
        return width * (value - xmin) / (xmax - xmin)

    lines = [
        "% generated by tools/analyze_static_benchmarks.py",
        "\\definecolor{benchqwen}{HTML}{3B6FA0}",
        "\\definecolor{benchgemma}{HTML}{C07C33}",
        "\\definecolor{benchother}{HTML}{666666}",
        "\\begin{tikzpicture}[font=\\scriptsize]",
    ]
    for panel_index, (task, title, ymin, ymax, y_ticks) in enumerate(panels):
        xshift = panel_index * (width + gap)

        def ty(value: float) -> float:
            return height * (value - ymin) / (ymax - ymin)

        lines.extend(
            [
                f"\\begin{{scope}}[xshift={xshift:.3f}cm]",
                f"\\node[font=\\small\\bfseries] at ({width / 2:.3f},{height + 0.34:.3f}) {{{title}}};",
                f"\\draw[black!55] (0,0) -- ({width:.3f},0);",
                f"\\draw[black!55] (0,0) -- (0,{height:.3f});",
            ]
        )
        for tick in x_ticks:
            x = tx(tick)
            lines.append(f"\\draw[black!45] ({x:.3f},0) -- ({x:.3f},-0.055) node[below] {{{tick:.2f}}};")
        for tick in y_ticks:
            y = ty(tick)
            lines.append(
                f"\\draw[black!12] (0,{y:.3f}) -- ({width:.3f},{y:.3f});"
                f"\\draw[black!45] (0,{y:.3f}) -- (-0.055,{y:.3f}) node[left] {{{100*tick:.0f}}};"
            )
        if panel_index == 0:
            lines.append(f"\\node[rotate=90] at (-0.62,{height / 2:.3f}) {{accuracy (\\%) $\\uparrow$}};")

        for row in analysis["rows"]:
            source_label = row["source_label"]
            family = 1 if source_label.startswith("Qwen") else 2 if source_label.startswith("gemma") else 3
            point = row["benchmarks"][task]
            x = tx(row["adapted_bpb"])
            y, low, high = ty(point["accuracy"]), ty(point["wilson_95_low"]), ty(point["wilson_95_high"])
            color = colors[family]
            lines.append(f"\\draw[{color}!42,line width=0.55pt] ({x:.3f},{low:.3f}) -- ({x:.3f},{high:.3f});")
            lines.append(f"\\fill[{color}] ({x:.3f},{y:.3f}) circle (1.25pt);")
            label = PLOT_LABEL[source_label]
            if label in key_labels:
                dx, dy = label_shift[(task, label)]
                lines.append(
                    f"\\node[anchor=west,font=\\tiny,text={color}] at ({x + dx:.3f},{y + dy:.3f}) {{{label}}};"
                )
        lines.append("\\end{scope}")

    total_width = 3 * width + 2 * gap
    lines.extend(
        [
            f"\\node at ({total_width / 2:.3f},-0.47) {{adapted BPB $\\downarrow$}};",
            f"\\fill[benchqwen] ({total_width / 2 - 1.75:.3f},-0.88) circle (1.25pt);",
            f"\\node[anchor=west] at ({total_width / 2 - 1.63:.3f},-0.88) {{Qwen}};",
            f"\\fill[benchgemma] ({total_width / 2 - 0.45:.3f},-0.88) circle (1.25pt);",
            f"\\node[anchor=west] at ({total_width / 2 - 0.33:.3f},-0.88) {{Gemma}};",
            f"\\fill[benchother] ({total_width / 2 + 1.03:.3f},-0.88) circle (1.25pt);",
            f"\\node[anchor=west] at ({total_width / 2 + 1.15:.3f},-0.88) {{other}};",
            "\\end{tikzpicture}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()

    def canonicalize(value):
        """Remove platform-level libm drift from generated public artifacts."""
        if isinstance(value, float):
            return round(value, 12)
        if isinstance(value, list):
            return [canonicalize(item) for item in value]
        if isinstance(value, dict):
            return {key: canonicalize(item) for key, item in value.items()}
        return value

    analysis = canonicalize(build_analysis())
    rendered = {
        OUTPUT_JSON: json.dumps(analysis, indent=2) + "\n",
        OUTPUT_TEX: render_tex(analysis),
        OUTPUT_TSV: render_tsv(analysis),
        OUTPUT_FIGURE_TEX: render_figure_tex(analysis),
    }
    stale = [path for path, text in rendered.items() if not path.exists() or path.read_text() != text]
    if args.check:
        if stale:
            raise SystemExit("stale static-benchmark artifacts: " + ", ".join(map(str, stale)))
        print("static benchmark artifacts: current")
        return
    for path, text in rendered.items():
        path.write_text(text)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
