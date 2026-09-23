# Earlier wall-time-capped sweep (superseded)

These are the artifacts of the **first** news run, which selected each model's LoRA adapter with the
pipeline's multi-fidelity Optuna sweep under a wall-clock cap. **They do not produce any number in
the paper or on the leaderboard** (the reported results use the hand-picked configuration in
`lm_adapt_bench/configs/manual_hparams.yaml`, trained for a fixed 250 steps, because a full
per-model search of every model was beyond the compute budget), and they predate the
marker-masking correction, so their BPB values score the injected document-start token and are not
comparable with the published ones.

They are retained because `*/trials.csv` is the evidence for how the search must be budgeted.
Within a fixed wall-clock allocation the sweep completed

| Model | Completed trials |
|---|---:|
| Llama-3.2-1B | 18 |
| LiquidAI-LFM2.5-1.2B | 17 |
| Qwen-2.5-0.5B | 15 |
| Qwen-2.5-1.5B | 14 |
| Gemma-4-12B | 10 |
| Qwen-2.5-7B | 10 |
| Mistral Ministral-3-14B | 9 |
| Qwen-3.5-4B | 9 |
| Qwen-3.5-9B | 7 |
| Gemma-4-31B | **4** |
| Qwen-3.5-35B-MoE | **4** |

Search depth is inversely correlated with model size, so the sweep handed small models a
better-tuned adapter than large ones and biased adapted BPB by parameter count. That is why the
released pipeline (`--hparams sweep`, the default) budgets the search in trials, the same
`--n-trials` for every model, and applies no wall-time cap unless `--sweep-time-fraction` is set.

`final_checkpoint/` and `train_state/` directories are local only and are not committed
(see `.gitignore`); regenerate them with the pipeline if you need them.

Reproduce the table above with:

```bash
for f in results/legacy_sweep/*/trials.csv; do
  printf '%-38s %s\n' "$(basename "$(dirname "$f")")" "$(grep -c COMPLETE "$f")"
done
```
