import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import optuna
from typing import List, Dict, Any, Optional

matplotlib.use("Agg")

def _get_short_id(model_id: str) -> str:
    return model_id.split("/")[-1]

def plot_ranking_bar(results: List[Dict[str, Any]], output_dir: str) -> str:
    # Filter out models with NaN best_bpb
    valid_results = [r for r in results if r.get("best_bpb") == r.get("best_bpb") and r.get("best_bpb") is not None]
    if not valid_results: return ""

    sorted_results = sorted(valid_results, key=lambda x: x.get("best_bpb", 0), reverse=True)
    model_ids = [_get_short_id(r["model_id"]) for r in sorted_results]
    zero_shot = [r.get("zero_shot_bpb", 0) for r in sorted_results]
    fine_tuned = [r.get("best_bpb", 0) for r in sorted_results]

    y = np.arange(len(model_ids))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(y + height/2, zero_shot, height, label='Zero-shot', color='skyblue')
    ax.barh(y - height/2, fine_tuned, height, label='Fine-tuned', color='steelblue')

    ax.set_xlabel('Bits Per Byte (lower is better)')
    ax.set_title('Model Comparison — BPB on Target Dataset')
    ax.set_yticks(y)
    ax.set_yticklabels(model_ids)
    ax.legend()

    for i, (zs, ft) in enumerate(zip(zero_shot, fine_tuned)):
        ax.text(zs, i + height/2, f' {zs:.2f}', va='center')
        ax.text(ft, i - height/2, f' {ft:.2f}', va='center')

    plt.tight_layout()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "ranking_bar.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_efficiency_scatter(results: List[Dict[str, Any]], output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    any_plotted = False
    for r in results:
        hours = r.get("training_time_seconds", 0) / 3600
        improvement = r.get("pct_improvement", 0)
        
        # Skip NaNs
        if hours != hours or improvement != improvement:
            continue
            
        label = _get_short_id(r["model_id"])
        ax.scatter(hours, improvement, label=label, s=100)
        ax.annotate(f" {label}\n ({r.get('bpb_per_gpu_hour', 0):.2f}/hr)", (hours, improvement))
        any_plotted = True

    if not any_plotted:
        plt.close()
        return ""

    ax.set_xlabel('GPU-hours')
    ax.set_ylabel('BPB Improvement (%)')
    ax.set_title('Adaptation Efficiency — BPB gain per GPU-hour')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.text(0.05, 0.95, "Top-left = most efficient", transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    path = os.path.join(output_dir, "figures", "efficiency_scatter.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_learning_curves(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Global comparison chart for all models."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    any_plotted = False
    for r in results:
        curve = r.get("bpb_curve", [])
        if not curve: continue
        
        # Filter NaNs from curve
        valid_curve = [(s, b) for s, b in curve if b == b and b is not None]
        if not valid_curve: continue
        
        steps, bpbs = zip(*valid_curve)
        ax.plot(steps, bpbs, marker='o', markersize=3, label=_get_short_id(r["model_id"]))
        any_plotted = True

    if not any_plotted:
        plt.close()
        return ""

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation BPB')
    ax.set_title('Global Adaptation Comparison — Learning Curves')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(output_dir, "figures", "learning_curves_all.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_individual_learning_curve(model_id: str, slug: str, curve: List[tuple], output_dir: str) -> str:
    """Specific chart for a single model's report section."""
    if not curve: return ""
    
    # Filter NaNs
    valid_curve = [(s, b) for s, b in curve if b == b and b is not None]
    if not valid_curve: return ""

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    steps, bpbs = zip(*valid_curve)
    ax.plot(steps, bpbs, color='darkblue', marker='o', markersize=4)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation BPB')
    ax.set_title(f'Learning Curve: {model_id}')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "figures", f"learning_curve_{slug}.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_sweep_scatter(model_id: str, slug: str, trials_df: pd.DataFrame, output_dir: str) -> str:
    cols = [c for c in trials_df.columns if c not in ["trial_number", "val_bpb", "state", "duration_seconds"]]
    n_params = len(cols)
    if n_params == 0: return ""
    
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=150)
    axes = axes.flatten()
    
    complete = trials_df[trials_df["state"] == "COMPLETE"]
    pruned = trials_df[trials_df["state"] == "PRUNED"]
    best_trial = complete.sort_values("val_bpb").iloc[0] if not complete.empty else None

    for i, col in enumerate(cols):
        ax = axes[i]
        
        # Check if the column is suitable for log scaling
        # (Must contain 'rate' or 'decay', be numeric, and have strictly positive values)
        use_log = False
        if "rate" in col.lower() or "decay" in col.lower():
            try:
                valid_vals = trials_df[col].dropna()
                if not valid_vals.empty and pd.api.types.is_numeric_dtype(valid_vals) and (valid_vals > 0).all():
                    use_log = True
            except Exception:
                pass
        
        # Add 'jitter' to categorical plots so vertical lines are readable
        x_complete = complete[col]
        x_pruned = pruned[col]
        
        # Only jitter if we are NOT using log scale
        if not use_log and (trials_df[col].dtype == 'object' or len(trials_df[col].unique()) < 10):
            # Add small random noise to X for visualization only
            spread = 0.05
            if trials_df[col].dtype != 'object':
                x_complete = x_complete + np.random.uniform(-spread, spread, size=len(x_complete))
                x_pruned = x_pruned + np.random.uniform(-spread, spread, size=len(x_pruned))
        
        ax.scatter(x_complete, complete["val_bpb"], c='#2b6cb0', alpha=0.6, s=20, label='Completed Trial')
        ax.scatter(x_pruned, [complete["val_bpb"].mean() if not complete.empty else 1.0] * len(pruned), 
                   c='#a0aec0', marker='x', alpha=0.4, s=20, label='Pruned Early')
        
        if best_trial is not None:
            ax.scatter(best_trial[col], best_trial["val_bpb"], c='#e53e3e', marker='*', s=150, edgecolors='black', label='Ideal Parameter', zorder=5)
            
        ax.set_title(f"Impact of {col}", fontsize=10, fontweight='bold')
        if use_log:
            ax.set_xscale('log')
            
    # Add a single unified legend for the whole grid
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
            
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"sweep_scatter_{slug}.png")

    try:
        plt.tight_layout()
    except Exception as e:
        if "log" in str(e).lower() or "positive" in str(e).lower():
            for ax in axes:
                try:
                    ax.set_xscale('linear')
                except Exception:
                    pass
            try:
                plt.tight_layout()
            except Exception:
                pass
        else:
            pass

    try:
        plt.savefig(path, bbox_inches='tight')
    except Exception as e:
        if "log" in str(e).lower() or "positive" in str(e).lower():
            for ax in axes:
                try:
                    ax.set_xscale('linear')
                except Exception:
                    pass
            try:
                plt.savefig(path, bbox_inches='tight')
            except Exception:
                pass
        else:
            pass

    plt.close()
    return path

def plot_parallel_coords(model_id: str, slug: str, trials_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "figures", f"parallel_coords_{slug}.png")
    complete = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if complete.empty: return ""
    
    cols = [c for c in trials_df.columns if c not in ["trial_number", "state", "duration_seconds"]]
    
    try:
        import plotly.graph_objects as go
        import kaleido
        
        dimensions = []
        for col in cols:
            dimensions.append(dict(label=col, values=complete[col]))
            
        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = complete['val_bpb'],
                           colorscale = 'Viridis_r',
                           showscale = True,
                           reversescale = False),
                dimensions = dimensions
            )
        )
        fig.update_layout(title=f"Parallel Coordinates — {model_id}")
        fig.write_image(path, engine="kaleido")
        return path
    except Exception as e:
        logging.getLogger("lm_adapt_bench").warning(f"Plotly/Kaleido failed, using fallback: {e}")
        # Fallback to a simple plot or skip
        return ""

def plot_hyperparameter_importance(model_id: str, slug: str, study: optuna.study.Study, output_dir: str) -> str:
    try:
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) < 3:
            return ""
            
        # Importance Plot
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        values = list(importances.values())
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        y = np.arange(len(params))
        ax.barh(y, values, color='teal')
        ax.set_yticks(y)
        ax.set_yticklabels(params)
        ax.set_xlabel('Importance')
        ax.set_title(f"Hyperparameter Importance — {model_id}")
        plt.tight_layout()
        path = os.path.join(output_dir, "figures", f"importance_{slug}.png")
        plt.savefig(path)
        plt.close()

        # CONTOUR PLOT (Landscape)
        # We try to use Optuna's built-in if plotly is available, else skip
        try:
            from optuna.visualization import plot_contour
            # Only plot the top 2 most important params for clarity
            top_params = sorted(importances, key=importances.get, reverse=True)[:2]
            fig_contour = plot_contour(study, params=top_params)
            contour_path = os.path.join(output_dir, "figures", f"contour_{slug}.png")
            fig_contour.write_image(contour_path, engine="kaleido")
        except:
            pass

        return path
    except Exception as e:
        logging.getLogger("lm_adapt_bench").warning(f"Importance plot failed: {e}")
        return ""
