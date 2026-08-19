import os
import time
import math
import logging
import shutil
from typing import Optional, List, Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import optuna

from .config import TrainingConfig, SweepConfig
from .evaluate import compute_bpb

class Trainer:
    def _is_large_multi_gpu(self) -> bool:
        is_multi_gpu = torch.cuda.device_count() > 1 if torch.cuda.is_available() else False
        is_large = getattr(self.model.config, "hidden_size", 0) > 4096 or "moe" in self.model.__class__.__name__.lower() or "35b" in getattr(self.model.config, "_name_or_path", "").lower()
        return is_large and is_multi_gpu

    def __init__(self, model, tokeniser, train_dataset, val_dataset, training_config: TrainingConfig, 
                 sweep_config: SweepConfig, device: torch.device, avg_bytes_per_token: float, output_dir: str):
        self.model = model
        self.tokeniser = tokeniser
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_config = training_config
        self.sweep_config = sweep_config
        self.device = device
        self.avg_bytes_per_token = avg_bytes_per_token
        self.output_dir = output_dir
        self.logger = logging.getLogger("lm_adapt_bench")
        
        # Detect dtype
        self.dtype = next(model.parameters()).dtype
        
        # Determine correct autocast dtype
        self.autocast_dtype = torch.float16
        if self.device.type == "cuda":
            if self.dtype == torch.bfloat16:
                self.autocast_dtype = torch.bfloat16
            elif torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
        
        # Optimization: Enable gradient checkpointing for VRAM efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if self._is_large_multi_gpu():
                self.logger.info("Skipping initial gradient checkpointing to prevent multi-GPU device-meta autograd bugs.")
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
            else:
                self.logger.info("Enabling gradient checkpointing for VRAM efficiency.")
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.eval_batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            num_workers=0
        )

    def _apply_lora(self, lora_targets: list[str]) -> None:
        self.logger.info(f"Applying LoRA with targets: {lora_targets}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=self.training_config.lora_dropout,
            target_modules=lora_targets,
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.autocast_dtype)
        
        # Ensure checkpointing stays active after wrapping, unless it is a large model on multi-GPU to avoid PyTorch device-meta autograd bugs
        if self._is_large_multi_gpu():
            self.logger.info("Disabling gradient checkpointing to prevent multi-GPU device-meta autograd bugs.")
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        elif hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def _build_optimiser_and_scheduler(self, n_total_steps: int):
        # Safety: Ensure at least 1 step to avoid division by zero in some schedulers
        n_total_steps = max(1, n_total_steps)
        self.optimiser = AdamW(
            self.model.parameters(), 
            lr=self.training_config.learning_rate, 
            weight_decay=self.training_config.weight_decay
        )
        num_warmup_steps = int(n_total_steps * self.training_config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimiser, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=n_total_steps
        )

    def train_sweep(self, n_steps: int, trial=None) -> dict:
        self.model.train()
        n_completed_steps = 0
        bpb_curve = []
        
        self._build_optimiser_and_scheduler(n_steps)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        iter_loader = iter(self.train_loader)
        
        for step in range(1, n_steps + 1):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(self.train_loader)
                batch = next(iter_loader)
                
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with torch.autocast(device_type=device_type, dtype=self.autocast_dtype):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / self.training_config.gradient_accumulation_steps
            
            loss.backward()
            
            if step % self.training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.gradient_clip)
                self.optimiser.step()
                self.scheduler.step()
                self.optimiser.zero_grad()
                n_completed_steps += 1
                
                # Speed Fix: Validate every 100 steps during sweep
                if n_completed_steps % 100 == 0 or n_completed_steps == n_steps:
                    val_bpb = compute_bpb(self.model, self.val_dataset, self.avg_bytes_per_token, self.device)
                    bpb_curve.append((n_completed_steps, val_bpb))
                    
                    # Divergence Guard for Sweep
                    if len(bpb_curve) >= 2 and val_bpb > bpb_curve[-2][1] * 1.5:
                        self.logger.warning(f"Sweep trial diverging (BPB {val_bpb:.4f}). Pruning.")
                        raise optuna.TrialPruned()

                    if trial is not None:
                        trial.report(val_bpb, step=n_completed_steps)
                        if trial.should_prune():
                            self.logger.info(f"Trial {trial.number} pruned at step {n_completed_steps}")
                            raise optuna.TrialPruned()
                            
        final_bpb = compute_bpb(self.model, self.val_dataset, self.avg_bytes_per_token, self.device)
        return {"val_bpb": final_bpb, "bpb_curve": bpb_curve, "steps": n_completed_steps}

    def train_full(self, lora_targets: Optional[list], mode: str) -> dict:
        if mode == "lora" and lora_targets:
            self._apply_lora(lora_targets)
            
        n_batches = len(self.train_loader)
        # Fix: Initialize scheduler for the MAX possible steps (including extensions)
        max_possible_steps = (n_batches * self.training_config.final_epochs * 2) // self.training_config.gradient_accumulation_steps
        
        self._build_optimiser_and_scheduler(max_possible_steps)
        
        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        self.model.train()
        bpb_curve = []
        global_step = 0
        grad_step = 0
        
        best_val_bpb = float('inf')
        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        patience = 10
        min_delta = 0.0001
        no_improvement_count = 0
        stopped_on_time = False

        # Per-model wall-time budget: stop gracefully (keeping the best checkpoint) before
        # the job's Slurm walltime is exhausted, so no model is ever killed mid-step.
        max_train_seconds = getattr(self.training_config, "max_train_seconds", None)

        for epoch in range(self.training_config.final_epochs * 2): # Allow double duration if improving
            for batch in self.train_loader:
                if max_train_seconds and (time.time() - start_time) > max_train_seconds:
                    self.logger.warning(
                        f"Training wall-time budget ({max_train_seconds/3600:.2f}h) reached at step {grad_step}. "
                        f"Stopping gracefully and keeping best checkpoint."
                    )
                    stopped_on_time = True
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.autocast(device_type=device_type, dtype=self.autocast_dtype):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / self.training_config.gradient_accumulation_steps
                
                if torch.isnan(loss):
                    self.logger.error("Loss exploded to NaN. Stopping.")
                    break

                loss.backward()
                global_step += 1
                
                if global_step % self.training_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.gradient_clip)
                    self.optimiser.step()
                    self.scheduler.step()
                    self.optimiser.zero_grad()
                    grad_step += 1
                    
                    # Validate every 500 steps to reduce Disk I/O
                    if grad_step % 500 == 0:
                        val_bpb = compute_bpb(self.model, self.val_dataset, self.avg_bytes_per_token, self.device)
                        bpb_curve.append((grad_step, val_bpb))
                        self.logger.info(f"Step {grad_step} - Val BPB: {val_bpb:.4f}")
                        
                        # Divergence Guard 1: BPB goes up significantly
                        if val_bpb > best_val_bpb * 1.1 and best_val_bpb != float('inf'):
                            self.logger.warning(f"Divergence detected (BPB {val_bpb:.4f} >> best {best_val_bpb:.4f}). Stopping.")
                            break
                        
                        # Divergence Guard 2: Trend Check (3 consecutive increases)
                        if len(bpb_curve) >= 3:
                            last_3 = [b for s, b in bpb_curve[-3:]]
                            if last_3[2] > last_3[1] > last_3[0]:
                                self.logger.warning("BPB curve trending upwards for 3 evals. Early exit to prevent divergence.")
                                break

                        if val_bpb < (best_val_bpb - min_delta):
                            best_val_bpb = val_bpb
                            no_improvement_count = 0
                            self.model.save_pretrained(os.path.join(self.output_dir, "best_checkpoint"))
                        else:
                            no_improvement_count += 1
                            
                        if no_improvement_count >= patience:
                            self.logger.info(f"Converged (no improvement > {min_delta} for {patience} evals). Stopping early.")
                            break
            else:
                if epoch >= self.training_config.final_epochs - 1 and no_improvement_count > 0:
                    self.logger.info("Target epochs reached and model has leveled off.")
                    break
                continue
            break

        training_time = time.time() - start_time
        peak_vram = 0.0
        if self.device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            
        best_ckpt_path = os.path.join(self.output_dir, "best_checkpoint")
        if os.path.exists(best_ckpt_path):
            self.logger.info("Reloading best checkpoint for final evaluation.")
            final_bpb = best_val_bpb
            shutil.copytree(best_ckpt_path, os.path.join(self.output_dir, "final_checkpoint"), dirs_exist_ok=True)
        else:
            final_bpb = compute_bpb(self.model, self.val_dataset, self.avg_bytes_per_token, self.device)
            self.model.save_pretrained(os.path.join(self.output_dir, "final_checkpoint"))
        
        # Recalculate stats AFTER LoRA/PEFT is applied
        from .utils import model_stats
        final_stats = model_stats(self.model)

        return {
            "val_bpb": final_bpb,
            "bpb_curve": bpb_curve,
            "training_time_seconds": training_time,
            "peak_gpu_memory_gb": peak_vram,
            "steps_trained": grad_step,
            "stopped_on_time": stopped_on_time,
            "model_stats": final_stats
        }

    def _restore_grad_checkpointing(self):
        """Re-apply the gradient-checkpointing policy after (re)wrapping with LoRA."""
        if self._is_large_multi_gpu():
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        elif hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def train_resumable(self, lora_targets: Optional[list], mode: str, state_dir: str, chain_index: int = 0, eval_every: int = 500, force_finalize: bool = False) -> dict:
        """Open-ended training that persists state and RESUMES across chained Slurm jobs until
        convergence (early-stop on val-BPB plateau). On a wall-time budget hit it saves state and
        returns converged=False so the launcher resubmits a follow-up job to continue.

        Resume carries adapter weights + step count + best-so-far + BPB curve. The optimiser is
        rebuilt fresh each link with a constant LR (warmup only on the first link), which is the
        correct schedule for training of unknown total length and sidesteps optimiser-state
        param-ordering fragility across reloads."""
        import json as _json
        os.makedirs(state_dir, exist_ok=True)
        meta_path = os.path.join(state_dir, "meta.json")
        adapter_path = os.path.join(state_dir, "adapter")
        best_adapter_path = os.path.join(state_dir, "best_adapter")

        resuming = os.path.exists(meta_path) and os.path.isdir(adapter_path)
        if resuming:
            with open(meta_path) as f:
                meta = _json.load(f)
            self.logger.info(f"Resuming training from {adapter_path} at step {meta['grad_step']} (chain link {chain_index}).")
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True)
            self.model.to(self.autocast_dtype)
            self._restore_grad_checkpointing()
            grad_step = meta["grad_step"]
            best_val_bpb = meta["best_val_bpb"] if meta.get("best_val_bpb") is not None else float('inf')
            no_improvement_count = meta.get("no_improvement_count", 0)
            bpb_curve = [tuple(x) for x in meta.get("bpb_curve", [])]
            prior_time = meta.get("total_train_time", 0.0)
        else:
            if mode == "lora" and lora_targets:
                self._apply_lora(lora_targets)
            grad_step = 0
            best_val_bpb = float('inf')
            no_improvement_count = 0
            bpb_curve = []
            prior_time = 0.0

        # Fresh optimiser each link. LR follows a cosine-to-floor schedule keyed on the
        # CUMULATIVE grad-step (grad_step already done + local step this link), so it anneals
        # smoothly within a link and keeps decaying across resumes -- no per-link re-warmup
        # spike. Warmup applies only at the very start (first link, cumulative step < warmup).
        self.optimiser = AdamW(self.model.parameters(),
                               lr=self.training_config.learning_rate,
                               weight_decay=self.training_config.weight_decay)
        target_steps = max(1, getattr(self.training_config, "target_train_steps", 8000))
        floor_ratio = float(getattr(self.training_config, "lr_floor_ratio", 0.1))
        n_batches = max(1, len(self.train_loader))
        approx_epoch_steps = max(1, (n_batches * self.training_config.final_epochs) // self.training_config.gradient_accumulation_steps)
        warmup = max(1, int(approx_epoch_steps * self.training_config.warmup_ratio))
        resume_offset = grad_step  # cumulative grad-steps already completed across prior links

        def _lr_lambda(local_step):
            s = resume_offset + local_step
            if s < warmup:
                return float(s) / float(warmup)
            progress = min(1.0, (s - warmup) / float(max(1, target_steps - warmup)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
            return floor_ratio + (1.0 - floor_ratio) * cosine     # peak -> floor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, _lr_lambda)
        self.logger.info(
            f"Cosine-to-floor LR: peak={self.training_config.learning_rate:.2e}, "
            f"floor={floor_ratio:.2f}x over {target_steps} cumulative steps "
            f"(warmup {warmup}, resume_offset {resume_offset})."
        )

        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.model.train()
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        patience = 10
        # Plateau threshold: an eval must beat the best by > min_delta to count as improvement.
        # 1e-3 (not 1e-4) so training declares convergence at a real plateau instead of grinding
        # extra chain-links/nodes for BPB noise at the LR floor (the 31B ran 4 links under 1e-4).
        min_delta = 0.001
        max_train_seconds = getattr(self.training_config, "max_train_seconds", None)
        global_step = 0
        converged = False
        stopped_on_time = False

        def save_state():
            self.model.save_pretrained(adapter_path)
            with open(meta_path, "w") as f:
                _json.dump({
                    "grad_step": grad_step,
                    "best_val_bpb": (None if best_val_bpb == float('inf') else best_val_bpb),
                    "no_improvement_count": no_improvement_count,
                    "bpb_curve": [list(x) for x in bpb_curve],
                    "total_train_time": prior_time + (time.time() - start_time),
                    "chain_index": chain_index,
                }, f)

        # Loop generously over epochs; real termination is early-stop or the wall-time budget.
        for epoch in range(self.training_config.final_epochs * 10):
            for batch in self.train_loader:
                if max_train_seconds and (time.time() - start_time) > max_train_seconds:
                    self.logger.warning(
                        f"Train-phase wall-time budget ({max_train_seconds/3600:.2f}h) reached at step {grad_step}. "
                        f"Saving state for resume in the next chained job."
                    )
                    stopped_on_time = True
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                with torch.autocast(device_type=device_type, dtype=self.autocast_dtype):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / self.training_config.gradient_accumulation_steps
                if torch.isnan(loss):
                    self.logger.error("Loss exploded to NaN. Terminating training.")
                    converged = True
                    break
                loss.backward()
                global_step += 1
                if global_step % self.training_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.gradient_clip)
                    self.optimiser.step()
                    self.scheduler.step()
                    self.optimiser.zero_grad()
                    grad_step += 1
                    if grad_step % eval_every == 0:
                        val_bpb = compute_bpb(self.model, self.val_dataset, self.avg_bytes_per_token, self.device)
                        bpb_curve.append((grad_step, val_bpb))
                        self.logger.info(f"Step {grad_step} - Val BPB: {val_bpb:.4f}")
                        if val_bpb < (best_val_bpb - min_delta):
                            best_val_bpb = val_bpb
                            no_improvement_count = 0
                            self.model.save_pretrained(best_adapter_path)
                        else:
                            no_improvement_count += 1
                        save_state()  # checkpoint each eval so a kill loses <=eval_every steps
                        if no_improvement_count >= patience:
                            self.logger.info(f"Converged (no improvement > {min_delta} for {patience} evals). Stopping.")
                            converged = True
                            break
            if converged or stopped_on_time:
                break

        save_state()
        training_time = prior_time + (time.time() - start_time)
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if self.device.type == "cuda" else 0.0

        if not converged and not stopped_on_time:
            # Ran out of epoch passes without plateau or timeout -> treat as done.
            converged = True

        # On the last allowed chain link, finalize from the best checkpoint even if the
        # plateau criterion wasn't reached, so the chain always yields a usable result.
        if force_finalize and not converged:
            self.logger.info("force_finalize: finalizing from best checkpoint despite no plateau (last chain link).")
            converged = True

        if converged:
            from .utils import model_stats
            if os.path.isdir(best_adapter_path):
                final_bpb = best_val_bpb
                shutil.copytree(best_adapter_path, os.path.join(self.output_dir, "final_checkpoint"), dirs_exist_ok=True)
            else:
                final_bpb = compute_bpb(self.model, self.val_dataset, self.avg_bytes_per_token, self.device)
                self.model.save_pretrained(os.path.join(self.output_dir, "final_checkpoint"))
            return {
                "converged": True,
                "val_bpb": final_bpb,
                "bpb_curve": bpb_curve,
                "training_time_seconds": training_time,
                "peak_gpu_memory_gb": peak_vram,
                "steps_trained": grad_step,
                "stopped_on_time": stopped_on_time,
                "model_stats": model_stats(self.model),
            }
        return {
            "converged": False,
            "bpb_curve": bpb_curve,
            "training_time_seconds": training_time,
            "peak_gpu_memory_gb": peak_vram,
            "steps_trained": grad_step,
            "stopped_on_time": True,
        }
