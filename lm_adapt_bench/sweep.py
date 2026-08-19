import os
import json
import logging
import gc
import pandas as pd
import torch
import optuna
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config import TrainingConfig, SweepConfig, RunConfig
from .utils import get_model_and_tokeniser, slugify
from .trainer import Trainer

class SweepRunner:
    def __init__(self, model_id, datamodule, sweep_config: SweepConfig, training_config: TrainingConfig, 
                 device: torch.device, dtype: torch.dtype, avg_bytes_per_token: float, output_dir: str, run_config: RunConfig):
        self.model_id = model_id
        self.datamodule = datamodule
        self.sweep_config = sweep_config
        self.training_config = training_config
        self.device = device
        self.dtype = dtype
        self.avg_bytes_per_token = avg_bytes_per_token
        self.output_dir = output_dir
        self.run_config = run_config
        self.logger = logging.getLogger("lm_adapt_bench")
        self.slug = slugify(model_id)

    def _parse_search_space(self, trial) -> TrainingConfig:
        sampled = {}
        for name, spec in self.sweep_config.search_space.items():
            if spec.get("lora_only") and self.sweep_config.mode != "lora":
                continue
            
            p_type = spec.get("type")
            if p_type == "float":
                sampled[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif p_type == "int":
                sampled[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif p_type == "categorical":
                sampled[name] = trial.suggest_categorical(name, spec["choices"])
        
        # Special handling for lora_alpha_multiplier
        if "lora_alpha_multiplier" in sampled:
            sampled["lora_alpha"] = sampled["lora_r"] * sampled["lora_alpha_multiplier"]
            del sampled["lora_alpha_multiplier"]
            
        # Create TrainingConfig from sampled values, using defaults for missing ones
        config_dict = self.training_config.__dict__.copy()
        config_dict.update(sampled)
        return TrainingConfig(**config_dict)

    def _objective(self, trial) -> float:
        model, tokeniser = get_model_and_tokeniser(self.model_id, self.run_config, self.device, self.dtype)
        sampled_config = self._parse_search_space(trial)
        
        train_ds, val_ds, _ = self.datamodule.tokenize(self.model_id, tokeniser, self.datamodule.config.max_seq_len, self.output_dir)
        
        # Speed Fix: Cap validation set during sweep to 256 samples
        sweep_val_limit = 256
        if len(val_ds) > sweep_val_limit:
            import random
            indices = list(range(len(val_ds)))
            random.seed(42)
            subset_indices = random.sample(indices, sweep_val_limit)
            from torch.utils.data import Subset
            sweep_val_ds = Subset(val_ds, subset_indices)
        else:
            sweep_val_ds = val_ds

        trainer = Trainer(
            model, tokeniser, train_ds, sweep_val_ds, sampled_config, self.sweep_config, 
            self.device, self.avg_bytes_per_token, self.output_dir
        )
        
        if self.sweep_config.mode == "lora":
            from .utils import get_lora_target_modules
            targets = self.sweep_config.lora_targets or get_lora_target_modules(model)
            trainer._apply_lora(targets)
            
        try:
            result = trainer.train_sweep(n_steps=self.sweep_config.sweep_steps, trial=trial)
            val_bpb = result["val_bpb"]
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            val_bpb = float('inf')
        finally:
            # Aggressive cleanup
            del trainer
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
        return val_bpb

    def run(self) -> tuple[TrainingConfig, pd.DataFrame, optuna.Study, dict]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Successive Halving is the Industry Standard for 'Automated Budgets'
        # It starts many trials at low budget and promotes the best to high budget.
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=50,      # Start at 50 steps
            reduction_factor=3,   # Triple the budget for winners at each stage
            min_early_stopping_rate=0
        )

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            pruner=pruner,
            direction="minimize"
        )

        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"Sweep {self.model_id}", total=self.sweep_config.n_trials)

            def callback(study, trial):
                progress.update(task, advance=1)

            # Cap total sweep wall time so large models (>30 min/trial) cannot consume the
            # whole Slurm walltime here and starve the final adaptation. Optuna finishes the
            # in-flight trial then stops; n_trials remains the upper bound.
            sweep_timeout = getattr(self.sweep_config, "max_sweep_seconds", None)
            if sweep_timeout:
                logging.getLogger("lm_adapt_bench").info(
                    f"HP sweep for {self.model_id} capped at {sweep_timeout/3600:.2f}h "
                    f"(or {self.sweep_config.n_trials} trials, whichever first)."
                )
            study.optimize(self._objective, n_trials=self.sweep_config.n_trials,
                           timeout=sweep_timeout, callbacks=[callback])
            
        # Build trials_df
        trials = []
        for t in study.trials:
            d = {
                "trial_number": t.number,
                "val_bpb": t.value,
                "state": t.state.name,
                "duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete else None
            }
            d.update(t.params)
            trials.append(d)
        
        trials_df = pd.DataFrame(trials)
        
        # Build best TrainingConfig
        best_params = study.best_params.copy()
        if "lora_alpha_multiplier" in best_params:
            best_params["lora_alpha"] = best_params["lora_r"] * best_params["lora_alpha_multiplier"]
            del best_params["lora_alpha_multiplier"]
            
        config_dict = self.training_config.__dict__.copy()
        config_dict.update(best_params)
        best_config = TrainingConfig(**config_dict)
        
        model_output_dir = os.path.join(self.output_dir, self.slug)
        os.makedirs(model_output_dir, exist_ok=True)
        
        with open(os.path.join(model_output_dir, "best_config.json"), "w") as f:
            json.dump(best_config.__dict__, f, indent=2)
            
        trials_df.to_csv(os.path.join(model_output_dir, "trials.csv"), index=False)
        
        # Calculate Range Health
        range_health = {}
        for name in best_config.__dict__:
            spec = self.sweep_config.search_space.get(name)
            if spec and name in best_params:
                val = best_params[name]
                if spec["type"] == "float" or spec["type"] == "int":
                    low, high = spec["low"], spec["high"]
                    if high > low:
                        # If within 5% of boundaries, mark as 'constrained'
                        margin = (high - low) * 0.05
                        if abs(val - low) < margin or abs(val - high) < margin:
                            range_health[name] = "constrained"
                        else:
                            range_health[name] = "optimal"
                    else:
                        range_health[name] = "optimal"
                else:
                    range_health[name] = "optimal"
            else:
                range_health[name] = "fixed"
        
        return best_config, trials_df, study, range_health

    def check_stability(self, config_a: TrainingConfig, config_b: TrainingConfig) -> float:
        """Calculates a stability score (0-1). 1.0 = identical configs."""
        # Focus on Learning Rate (log space) and LoRA Rank
        import math
        lr_diff = abs(math.log10(config_a.learning_rate) - math.log10(config_b.learning_rate))
        r_diff = abs(config_a.lora_r - config_b.lora_r) / 32.0
        
        # Heuristic score: lower diff is better
        total_diff = (lr_diff / 2.0) + r_diff 
        return max(0.0, 1.0 - (total_diff / 2.0))
