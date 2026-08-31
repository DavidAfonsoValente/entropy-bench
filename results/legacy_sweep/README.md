# Legacy exploratory sweep (superseded)

These are the artifacts of the **first** news run, which selected each model's LoRA adapter with a
multi-fidelity Optuna sweep. **They do not produce any number in the paper or on the leaderboard**,
and they predate the marker-masking correction described in `paper_sota.tex` Section 3.3 — their
BPB values score the injected document-start token and are therefore not comparable with the
published ones.

They are retained for one reason: `*/trials.csv` is the evidence for the methodological decision
argued in Section 3.2. Within a fixed wall-clock allocation the sweep completed

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
headline protocol uses one fixed configuration for every cell instead.

`final_checkpoint/` and `train_state/` directories are local only and are not committed
(see `.gitignore`); regenerate them with the pipeline if you need them.

Reproduce the table above with:

```bash
for f in results/legacy_sweep/*/trials.csv; do
  printf '%-38s %s\n' "$(basename "$(dirname "$f")")" "$(grep -c COMPLETE "$f")"
done
```
