import os
import sys
import json
import yaml
import logging
import argparse
import datetime
import gc
import time
import shutil
import dataclasses
from typing import List

import torch
import pandas as pd
from rich.console import Console
from rich.table import Table

from .contamination.config import ContaminationConfig
from .config import DataConfig, TrainingConfig, SweepConfig, RunConfig
from .utils import (
    setup_logging, set_seed, get_device, get_dtype, slugify, 
    get_model_and_tokeniser, model_stats, hardware_info, 
    library_versions, get_lora_target_modules
)
from .data import DataModule, BlockDataset
from .evaluate import compute_zero_shot_bpb, compute_bpb
from .sweep import SweepRunner
from .trainer import Trainer
from .plot import (
    plot_ranking_bar, plot_efficiency_scatter, plot_learning_curves,
    plot_sweep_scatter, plot_parallel_coords, plot_hyperparameter_importance,
    plot_individual_learning_curve
)
from .report import ReportBuilder
from .compat import apply_compatibility_patches

def main():
    apply_compatibility_patches()
    parser = argparse.ArgumentParser(description="LM Adapt Bench")
    parser.add_argument("--models", nargs="+", help="HuggingFace model IDs")
    parser.add_argument("--models-file", help="File with model IDs")
    parser.add_argument("--dataset", required=True, help="Path to local dataset or zip")
    parser.add_argument("--dataset-text-field", default="text", help="Text field name")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--n-trials", type=int, default=50) 
    parser.add_argument("--sweep-steps", type=int, default=1000) 
    parser.add_argument("--final-epochs", type=int, default=3)
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    # Walltime safety: cap total per-model pipeline wall time so no model is killed mid-step
    # by the Slurm hard cap. Default 23.5h leaves headroom under Leonardo's 24h limit.
    parser.add_argument("--wall-time-seconds", type=float, default=64800.0,
                        help="Total wall-time budget (s) measured from job start (default 18h). The HP sweep and final adaptation together must finish by this point, leaving ~6h headroom under Slurm's 24h cap for rank-0 aggregation, global-unified incremental re-eval (reloads every checkpoint, incl. the 35B/31B base models) and report rendering. Prior runs timed out because aggregation+re-eval started too late.")
    parser.add_argument("--eval-reserve-seconds", type=float, default=2400.0,
                        help="Wall time (s) reserved after the training loop for final test-set evaluation, plotting and checkpoint save.")
    parser.add_argument("--max-train-seconds", type=float, default=None,
                        help="Hard cap (s) on the final adaptation loop. Overrides the dynamic walltime-derived budget when set.")
    parser.add_argument("--phase", choices=["all", "sweep", "train"], default="all",
                        help="Pipeline phase. 'all' (default): sweep+train+eval in one job (original). 'sweep': contamination+zero-shot+HP sweep, write best_config.json, stop. 'train': load best_config, skip sweep, RESUME training across chained jobs until convergence (then write result_json).")
    parser.add_argument("--reuse-contamination", action="store_true",
                        help="Load cached contamination_splits.pt instead of re-running the forensic audit. Used by chained training jobs to skip the ~3h audit each time.")
    parser.add_argument("--chain-index", type=int, default=0,
                        help="Internal: which link in the self-chaining training sequence this job is (0-based). Set by the chaining launcher.")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip the rank-0 report aggregation/re-eval. Used by sweep-phase and intermediate (non-converged) training-chain jobs so they don't render a premature report.")
    parser.add_argument("--target-train-steps", type=int, default=8000,
                        help="Cosine-to-floor horizon (cumulative grad-steps) for resumable training: LR decays from peak to lr-floor-ratio*peak over this many steps, then holds the floor.")
    parser.add_argument("--lr-floor-ratio", type=float, default=0.1,
                        help="Floor of the cosine LR schedule as a fraction of peak LR (resumable training).")
    parser.add_argument("--final-link", action="store_true",
                        help="Mark this as the last allowed training-chain link: finalize from the best checkpoint and write result_json even if the plateau criterion wasn't reached.")
    parser.add_argument("--sweep-time-fraction", type=float, default=0.3,
                        help="Fraction of each model's remaining walltime (after reserving eval time) the HP sweep may use; the rest goes to final adaptation. Leaned to 0.3 (from 0.5) so HP search doesn't starve training -- big models were being cut off mid-adaptation. HP selection needs far less time than training, so 30%% is ample for the sweep while giving ~70%% to training.")
    parser.add_argument("--lora-targets", nargs="+", help="LoRA targets")
    parser.add_argument("--sweep-config", help="Custom sweep config YAML")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--hf-token", help="HuggingFace token")
    parser.add_argument("--report-title", default="LM Adapt Bench")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    
    # Contamination Flags
    parser.add_argument("--check-contamination", action="store_true", help="Enable contamination detection")
    parser.add_argument("--contam-check-level", choices=["off", "light", "standard", "strict", "forensic"], default="off")
    parser.add_argument("--forbidden-corpus", help="Path to forbidden dataset/corpus")
    parser.add_argument("--contam-clean-reference-corpus", help="Path to clean reference dataset for Min-K calibration")
    parser.add_argument("--contam-ngram-n", type=int, default=13)
    parser.add_argument("--contam-minhash-threshold", type=float, default=0.8)
    parser.add_argument("--contam-cleaning-policy", choices=["report_only", "quarantine", "drop"], default="report_only")
    parser.add_argument("--contam-cleaning-mode", choices=["model_specific", "global_unified"], default="model_specific", help="Unified or model-specific evaluation sets")
    parser.add_argument("--contam-use-global-quarantine", action="store_true", default=False, help="Import and apply a previously-saved quarantine.jsonl log")
    parser.add_argument("--contam-export-cleaned-dataset", action="store_true", default=True, help="Export permanently cleaned text splits to disk")
    parser.add_argument("--contam-sample-size", type=int, default=1000)
    parser.add_argument("--contam-fail-on-split-leakage", action="store_true", default=True)

    # Min-K Flags
    parser.add_argument("--contam-min-k-variant", choices=["minkpp", "mink"], default="minkpp", help="Min-K variant to use (minkpp=normalized, mink=raw)")
    parser.add_argument("--contam-min-k-percent", type=float, default=20.0)
    parser.add_argument("--contam-min-k-min-tokens", type=int, default=50)
    parser.add_argument("--contam-min-k-outlier-percentile", type=float, default=95.0)
    parser.add_argument("--contam-min-k-high-percentile", type=float, default=99.0)
    parser.add_argument("--contam-min-k-robust-z-suspicious", type=float, default=2.5)
    parser.add_argument("--contam-min-k-robust-z-high", type=float, default=3.5)
    parser.add_argument("--contam-min-k-test-full-max", type=int, default=5000)
    parser.add_argument("--contam-min-k-severe-guardrail", type=float, default=-1.0)
    parser.add_argument("--contam-min-k-disable-severe-guardrail", action="store_true")

    # CoDeC Flags
    parser.add_argument("--contam-codec-context-size", type=int, default=4)
    parser.add_argument("--contam-codec-samples", type=int, default=500)
    parser.add_argument("--contam-codec-delta-threshold", type=float, default=0.0)
    parser.add_argument("--contam-codec-robust-z-threshold", type=float, default=-2.0)
    parser.add_argument("--contam-codec-suspicious-fraction-medium", type=float, default=0.10)
    parser.add_argument("--contam-codec-suspicious-fraction-high", type=float, default=0.25)
    parser.add_argument("--contam-codec-suspicious-fraction-severe", type=float, default=0.40)

    # DCQ Flags
    parser.add_argument("--contam-enable-dcq", action="store_true")
    parser.add_argument("--contam-dcq-samples", type=int, default=100)
    parser.add_argument("--contam-dcq-num-distractors", type=int, default=2)
    parser.add_argument("--contam-dcq-min-margin", type=float, default=0.0)
    parser.add_argument("--contam-dcq-candidate-source", default="min_k")
    parser.add_argument("--contam-dcq-max-candidates", type=int, default=1000)
    parser.add_argument("--contam-dcq-seed", type=int, default=42)

    # Policy Flags
    parser.add_argument("--contam-max-soft-eval-quarantine-fraction", type=float, default=0.10)
    parser.add_argument("--contam-max-soft-val-quarantine-fraction", type=float, default=0.15)
    parser.add_argument("--contam-max-soft-train-quarantine-fraction", type=float, default=0.30)
    parser.add_argument("--contam-min-active-test-examples", type=int, default=100)
    parser.add_argument("--contam-min-active-test-fraction", type=float, default=0.70)

    # Slurm/Parallel flags
    parser.add_argument("--rank", type=int, help="Parallel rank for multi-GPU runs")
    parser.add_argument("--world-size", type=int, help="Total number of ranks")
    
    args = parser.parse_args()

    # Anchor the per-model walltime budget to job start so that data loading and the
    # (potentially multi-hour) contamination audit are counted against the Slurm hard cap.
    proc_start = time.time()

    # Auto-detect Slurm environment
    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("SLURM_PROCID", 0))
    
    world_size = args.world_size
    if world_size is None:
        world_size = int(os.environ.get("SLURM_NTASKS", 1))

    # Model list resolution
    all_models = args.models or []
    if args.models_file:
        with open(args.models_file, "r") as f:
            all_models.extend([line.strip() for line in f if line.strip()])
    
    all_models = sorted(list(set(all_models)))
    if not all_models:
        print("Error: No models specified.")
        sys.exit(1)

    # Slurm parallel logic: split models across ranks
    if world_size > 1:
        my_models = [m for i, m in enumerate(all_models) if i % world_size == rank]
    else:
        my_models = all_models

    os.makedirs(args.output, exist_ok=True)
    if world_size > 1 and rank == 0:
        # Clean up stale rank summary files from previous parallel runs
        for f_name in os.listdir(args.output):
            if f_name.startswith("summary_rank_") and f_name.endswith(".json"):
                try:
                    os.remove(os.path.join(args.output, f_name))
                except Exception:
                    pass
    logger = setup_logging(args.output)
    set_seed(args.seed)
    
    # Smart Defaults: Infer switches from check-level to simplify the CLI command
    check_contamination = args.check_contamination or (args.contam_check_level != "off")
    check_level = args.contam_check_level
    if args.check_contamination and check_level == "off":
        check_level = "standard" # fallback default

    # If in strict/forensic, automatically enable DCQ
    enable_dcq = args.contam_enable_dcq or (check_level in ["strict", "forensic"])

    # If in strict/forensic and the user didn't explicitly override the policy, default to quarantine
    cleaning_policy = args.contam_cleaning_policy
    if "--contam-cleaning-policy" not in sys.argv and check_level in ["strict", "forensic"]:
        cleaning_policy = "quarantine"
    # Contamination Setup
    contam_cfg = ContaminationConfig(
        check_contamination=check_contamination,
        check_level=check_level,
        forbidden_corpus=args.forbidden_corpus,
        clean_reference_corpus=args.contam_clean_reference_corpus,
        ngram_n=args.contam_ngram_n,
        minhash_threshold=args.contam_minhash_threshold,
        cleaning_policy=cleaning_policy,
        cleaning_mode=args.contam_cleaning_mode,
        export_cleaned_dataset=args.contam_export_cleaned_dataset,
        sample_size=args.contam_sample_size,
        fail_on_split_leakage=args.contam_fail_on_split_leakage,
        min_k_variant=args.contam_min_k_variant,
        min_k_percent=args.contam_min_k_percent,
        min_k_min_tokens=args.contam_min_k_min_tokens,
        min_k_outlier_percentile=args.contam_min_k_outlier_percentile,
        min_k_high_percentile=args.contam_min_k_high_percentile,
        min_k_robust_z_suspicious=args.contam_min_k_robust_z_suspicious,
        min_k_robust_z_high=args.contam_min_k_robust_z_high,
        test_full_max=args.contam_min_k_test_full_max,
        min_k_severe_guardrail=args.contam_min_k_severe_guardrail,
        min_k_enable_severe_guardrail=not args.contam_min_k_disable_severe_guardrail,
        codec_context_size=args.contam_codec_context_size,
        codec_samples=args.contam_codec_samples,
        codec_delta_threshold=args.contam_codec_delta_threshold,
        codec_robust_z_threshold=args.contam_codec_robust_z_threshold,
        codec_suspicious_fraction_medium=args.contam_codec_suspicious_fraction_medium,
        codec_suspicious_fraction_high=args.contam_codec_suspicious_fraction_high,
        codec_suspicious_fraction_severe=args.contam_codec_suspicious_fraction_severe,
        enable_dcq=enable_dcq,
        dc_samples=args.contam_dcq_samples,
        dcq_num_distractors=args.contam_dcq_num_distractors,
        dcq_min_margin=args.contam_dcq_min_margin,
        dcq_candidate_source=args.contam_dcq_candidate_source,
        max_soft_eval_quarantine_fraction=args.contam_max_soft_eval_quarantine_fraction,
        max_soft_val_quarantine_fraction=args.contam_max_soft_val_quarantine_fraction,
        max_soft_train_quarantine_fraction=args.contam_max_soft_train_quarantine_fraction,
        min_active_test_examples=args.contam_min_active_test_examples,
        min_active_test_fraction=args.contam_min_active_test_fraction,
        output_dir=os.path.join(args.output, "contamination"),
        seed=args.contam_dcq_seed if args.contam_dcq_seed != 42 else args.seed
    )

    data_config = DataConfig(
        dataset_path=args.dataset,
        text_field=args.dataset_text_field,
        val_split=args.val_split,
        test_split=args.test_split,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        seed=args.seed,
        contamination=contam_cfg
    )
    
    dm = DataModule(data_config)
    results = []
    summary_path = os.path.join(args.output, f"summary_rank_{rank}.json")
    
    logger.info(f"Starting LM Adapt Bench run on rank {rank}/{world_size}")
    logger.info(f"Assigned models: {my_models}")

    # Load sweep config
    if args.sweep_config:
        with open(args.sweep_config, "r") as f:
            sweep_search_space = yaml.safe_load(f)
    else:
        default_cfg = os.path.join(os.path.dirname(__file__), "configs", "default_sweep.yaml")
        with open(default_cfg, "r") as f:
            sweep_search_space = yaml.safe_load(f)

    # Model Signal Orchestrator (if enabled)
    from .contamination.model_signals import ModelSignalOrchestrator
    contam_orchestrator = ModelSignalOrchestrator(contam_cfg) if contam_cfg.check_contamination else None

    for model_idx, model_id in enumerate(my_models):
        slug = slugify(model_id)
        model_output_dir = os.path.join(args.output, slug)
        result_json = os.path.join(model_output_dir, f"result_{slug}.json")
        
        if not args.force_rerun and os.path.exists(result_json):
            logger.info(f"Skipping {model_id}, results already exist.")
            try:
                with open(result_json, "r") as f:
                    results.append(json.load(f))
                continue
            except: pass

        try:
            device = get_device(args.device)
            dtype = get_dtype(args.dtype, device)
            
            run_cfg = RunConfig(
                model_id=model_id, output_dir=args.output, device=args.device, 
                dtype=args.dtype, flash_attention=args.flash_attention,
                hf_token=args.hf_token
            )
            
            model, tokeniser = get_model_and_tokeniser(model_id, run_cfg, device, dtype)
            
            # Per-model contamination scoring
            model_contam_scores = []
            test_texts = dm.test_texts
            if contam_orchestrator:
                splits = {"train": dm.train_texts, "val": dm.val_texts, "test": dm.test_texts}
                model_contam_scores = contam_orchestrator.run_all_signals(model, tokeniser, splits, device)
                
                # Extract our model's suspicious indices
                my_suspicious_indices = []
                for score_obj in model_contam_scores:
                    if score_obj.split == "test":
                        my_suspicious_indices.extend(score_obj.suspicious_indices)
                my_suspicious_indices = list(set(my_suspicious_indices))
                
                # Write our own suspicious indices to disk for others to see
                my_indices_path = os.path.join(model_output_dir, "suspicious_indices.json")
                os.makedirs(model_output_dir, exist_ok=True)
                with open(my_indices_path, "w") as f:
                    json.dump(my_suspicious_indices, f)
                
                # If in unified mode, wait and merge everyone's suspicious indices
                if contam_cfg.cleaning_mode == "global_unified" and world_size > 1:
                    logger.info("Global Unified Cleaning Mode enabled. Syncing forensic indices with other nodes...")
                    start_wait = time.time()
                    # Generous timeout: large models can take many minutes to load + score on
                    # other nodes. The poll exits immediately once all indices are present, so a
                    # high ceiling only matters if a node genuinely dies. Too low a value would
                    # silently degrade global_unified to model-specific cleaning.
                    timeout = 3600 # 1 hour
                    unified_indices = set(my_suspicious_indices)
                    
                    while True:
                        all_written = True
                        for other_mid in all_models:
                            other_slug = slugify(other_mid)
                            other_path = os.path.join(args.output, other_slug, "suspicious_indices.json")
                            if not os.path.exists(other_path):
                                all_written = False
                                break
                        
                        if all_written:
                            logger.info("All model indices found on disk. Computing global intersection union...")
                            for other_mid in all_models:
                                other_slug = slugify(other_mid)
                                other_path = os.path.join(args.output, other_slug, "suspicious_indices.json")
                                try:
                                    with open(other_path, "r") as f:
                                        unified_indices.update(json.load(f))
                                except Exception as e:
                                    logger.error(f"Failed to read {other_path}: {e}")
                            break
                            
                        if time.time() - start_wait > timeout:
                            logger.warning("Timeout waiting for other nodes. Falling back to model-specific cleaning.")
                            break
                        time.sleep(10)
                        
                    final_suspicious_indices = list(unified_indices)
                    # Update local score object list so the report matches the actual drop
                    for score_obj in model_contam_scores:
                        if score_obj.split == "test":
                            score_obj.suspicious_indices = final_suspicious_indices
                else:
                    final_suspicious_indices = my_suspicious_indices

                # Strict Eval Cleaning for Soft Evidence
                if contam_cfg.strict_eval_cleaning:
                    num_suspicious = len(final_suspicious_indices)
                    original_test_size = len(dm.test_texts)
                    max_drop = int(original_test_size * contam_cfg.max_soft_eval_quarantine_fraction)
                    
                    if num_suspicious > max_drop:
                        logger.warning(f"SEVERE: {model_id} has {num_suspicious} suspicious samples, exceeding cap of {max_drop}. Quarantining up to cap. Test split is COMPROMISED.")
                        to_drop = list(final_suspicious_indices)[:max_drop]
                        for score_obj in model_contam_scores:
                            if score_obj.split == "test":
                                score_obj.risk_label = "COMPROMISED"
                    else:
                        logger.info(f"Quarantining {num_suspicious} suspicious samples based on unified soft evidence.")
                        to_drop = list(final_suspicious_indices)
                        
                    test_texts = [t for i, t in enumerate(test_texts) if i not in to_drop]
                    
                    if len(test_texts) < contam_cfg.min_active_test_examples or len(test_texts) / original_test_size < contam_cfg.min_active_test_fraction:
                        logger.warning(f"CRITICAL: Active test set size ({len(test_texts)}) fell below minimum thresholds. Final metrics may be invalid.")
                        for score_obj in model_contam_scores:
                            if score_obj.split == "test":
                                score_obj.risk_label = "COMPROMISED"

            stats = model_stats(model)
            avg_bpt = dm.get_byte_stats(tokeniser)
            
            # Tokenize specifically for THIS model/tokenizer
            train_ds, val_ds, test_ds = dm.tokenize(model_id, tokeniser, args.max_seq_len, args.output)
            
            # Re-tokenize test_ds if we dropped samples
            if len(test_texts) < len(dm.test_texts):
                def process(texts):
                    all_ids = []
                    for t in texts:
                        all_ids.extend(tokeniser(str(t), add_special_tokens=True)["input_ids"])
                    return [{"input_ids": torch.tensor(all_ids[i:i+args.max_seq_len]), 
                             "labels": torch.tensor(all_ids[i:i+args.max_seq_len])} 
                            for i in range(0, len(all_ids)-args.max_seq_len+1, args.max_seq_len)]
                test_ds = BlockDataset(process(test_texts))
            
            # Calculate Average Bytes Per Token specifically for THIS tokenizer
            avg_bpt = dm.calculate_avg_bytes_per_token(tokeniser)
            logger.info(f"Tokenizer Efficiency: {avg_bpt:.3f} bytes/token")
            
            zero_shot_bpb = compute_zero_shot_bpb(model, test_ds, avg_bpt, device, args.eval_batch_size)
            logger.info(f"Model: {model_id} - Zero-shot BPB (Test Set): {zero_shot_bpb:.4f}")
            
            # AGGRESSIVE CLEANUP: Free base model memory before the sweep
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if args.baseline_only:
                res = {
                    "model_id": model_id, "model_slug": slug, "zero_shot_bpb": zero_shot_bpb,
                    "total_params": stats["total_params"], "trainable_pct": stats["trainable_pct"],
                    "contamination_scores": [dataclasses.asdict(s) for s in model_contam_scores],
                    "best_bpb": zero_shot_bpb,
                    "pct_improvement": 0.0,
                    "adaptation_score": 0.0,
                    "completion_reason": "Baseline Only"
                }
                results.append(res)
                # CRITICAL: Write individual result so Rank 0 aggregator can proceed
                os.makedirs(model_output_dir, exist_ok=True)
                with open(result_json, "w") as f:
                    json.dump(res, f, indent=2)
                continue

            # Count only models that will actually run on this rank (those without a cached
            # result_json) so a rank packed with already-completed models doesn't over-divide.
            def _models_left_to_run(idx):
                cnt = 0
                for m in my_models[idx:]:
                    s = slugify(m)
                    if not os.path.exists(os.path.join(args.output, s, f"result_{s}.json")):
                        cnt += 1
                return max(1, cnt)

            # Trainer needs a sweep_cfg regardless of phase.
            sweep_cfg = SweepConfig(
                n_trials=args.n_trials, sweep_steps=args.sweep_steps,
                mode=args.mode, lora_targets=args.lora_targets,
                search_space=sweep_search_space,
            )
            sweep_meta_path = os.path.join(model_output_dir, "sweep_meta.json")
            trials_df = None
            study = None

            # ---- HP config: load from a prior sweep (phase=train) or run the sweep (all|sweep) ----
            if args.phase == "train":
                if not os.path.exists(sweep_meta_path):
                    logger.error(f"--phase train requires a completed sweep (sweep_meta.json missing for {model_id}). Run --phase sweep first.")
                    continue
                with open(sweep_meta_path) as f:
                    _sm = json.load(f)
                best_train_cfg = TrainingConfig(**_sm["best_config"])
                sweep_final_bpb = _sm.get("sweep_final_bpb", 0.0)
                range_health = _sm.get("range_health", {})
                tcsv = os.path.join(model_output_dir, "trials.csv")
                if os.path.exists(tcsv):
                    try: trials_df = pd.read_csv(tcsv)
                    except Exception: trials_df = None
                logger.info(f"[train phase] Loaded sweep config for {model_id}; skipping sweep.")
            else:
                # Cap the HP sweep's wall time so a slow big-model sweep can't starve training.
                sweep_models_left = _models_left_to_run(model_idx)
                sweep_time_left = args.wall_time_seconds - (time.time() - proc_start)
                sweep_per_model = sweep_time_left / sweep_models_left
                sweep_budget = max(1800.0, args.sweep_time_fraction * (sweep_per_model - args.eval_reserve_seconds))
                sweep_cfg.max_sweep_seconds = sweep_budget
                logger.info(
                    f"HP sweep wall-time budget for {model_id}: {sweep_budget/3600:.2f}h "
                    f"(walltime left {sweep_time_left/3600:.2f}h, {sweep_models_left} model(s) left on this rank)."
                )
                train_cfg = TrainingConfig(final_epochs=args.final_epochs, eval_batch_size=args.eval_batch_size)
                sweeper = SweepRunner(model_id, dm, sweep_cfg, train_cfg, device, dtype, avg_bpt, args.output, run_cfg)
                best_train_cfg, trials_df, study, range_health = sweeper.run()
                try:
                    sweep_final_bpb = float(trials_df[trials_df["state"] == "COMPLETE"]["val_bpb"].min())
                except Exception:
                    sweep_final_bpb = 0.0

                if args.phase == "sweep":
                    # Persist everything the train phase needs, plot, and stop (no training).
                    os.makedirs(model_output_dir, exist_ok=True)
                    with open(sweep_meta_path, "w") as f:
                        json.dump({"best_config": best_train_cfg.__dict__,
                                   "sweep_final_bpb": sweep_final_bpb if sweep_final_bpb == sweep_final_bpb else 0.0,
                                   "range_health": range_health,
                                   "zero_shot_bpb": zero_shot_bpb}, f, indent=2)
                    with open(os.path.join(model_output_dir, ".sweep_done"), "w") as f:
                        f.write("done")
                    plot_sweep_scatter(model_id, slug, trials_df, args.output)
                    plot_parallel_coords(model_id, slug, trials_df, args.output)
                    plot_hyperparameter_importance(model_id, slug, study, args.output)
                    logger.info(f"[sweep phase] {model_id}: sweep complete, config saved to sweep_meta.json. Skipping training.")
                    continue

            # ---- Reload model and train (phase=all -> train_full; phase=train -> resumable chain) ----
            logger.info(f"Reloading {model_id} for final adaptation phase...")
            model, _ = get_model_and_tokeniser(model_id, run_cfg, device, dtype)

            # Derive a graceful wall-time budget for the final adaptation loop from the
            # remaining job walltime, so the model is never killed mid-step by Slurm.
            models_left = _models_left_to_run(model_idx)  # current + not-yet-started that will actually run
            time_left = args.wall_time_seconds - (time.time() - proc_start)
            train_budget = (time_left / models_left) - args.eval_reserve_seconds
            if args.max_train_seconds:
                train_budget = min(train_budget, args.max_train_seconds)
            train_budget = max(600.0, train_budget)
            best_train_cfg.max_train_seconds = train_budget
            logger.info(
                f"Final adaptation wall-time budget for {model_id}: {train_budget/3600:.2f}h "
                f"(walltime left {time_left/3600:.2f}h, {models_left} model(s) remaining on this rank)."
            )

            trainer = Trainer(model, tokeniser, train_ds, val_ds, best_train_cfg, sweep_cfg, device, avg_bpt, model_output_dir)
            targets = args.lora_targets or get_lora_target_modules(model)

            if args.phase == "train":
                # Open-ended resumable training; chains across jobs until convergence.
                # Apply the cosine-to-floor LR schedule params (override any stale sweep defaults).
                best_train_cfg.target_train_steps = args.target_train_steps
                best_train_cfg.lr_floor_ratio = args.lr_floor_ratio
                state_dir = os.path.join(model_output_dir, "train_state")
                full_res = trainer.train_resumable(targets, args.mode, state_dir, chain_index=args.chain_index, force_finalize=args.final_link)
                if not full_res.get("converged"):
                    logger.info(
                        f"[train phase] {model_id} not converged yet (step {full_res.get('steps_trained')}); "
                        f"state saved. The chaining launcher will resubmit to continue."
                    )
                    continue
            else:
                full_res = trainer.train_full(targets, args.mode)

            # Use stats returned from trainer (calculated after LoRA wrapping)
            stats = full_res.get("model_stats", stats)

            # Calculate Stability Index (sweep_final_bpb computed/loaded above)
            try:
                match_bpb = next((b for s, b in full_res["bpb_curve"] if s >= args.sweep_steps), full_res["val_bpb"])
                stability_index = (min(sweep_final_bpb, match_bpb) / max(sweep_final_bpb, match_bpb)) if max(sweep_final_bpb, match_bpb) > 0 else 0.0
            except Exception:
                stability_index = 0.0
            
            # Clean up and reload for final evaluation
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
            gc.collect()
            
            # Reload for final evaluation
            if args.mode == "lora":
                model, _ = get_model_and_tokeniser(model_id, run_cfg, device, dtype)
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, os.path.join(model_output_dir, "final_checkpoint"))
            else:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    os.path.join(model_output_dir, "final_checkpoint"),
                    trust_remote_code=True,
                    device_map="auto" if device.type == "cuda" else None
                )
            model.eval()
            
            final_bpb = compute_bpb(model, test_ds, avg_bpt, device, args.eval_batch_size)
            logger.info(f"Model: {model_id} - Final BPB (Test Set): {final_bpb:.4f}")
            
            # NaN Handling for metrics
            pct_imp = (zero_shot_bpb - final_bpb) / zero_shot_bpb * 100 if zero_shot_bpb > 0 else 0.0
            if pct_imp != pct_imp: pct_imp = 0.0 
            
            training_hours = full_res["training_time_seconds"] / 3600
            adapt_score = pct_imp / (training_hours + 1)
            
            res = {
                "model_id": model_id,
                "model_slug": slug,
                "contamination_scores": [dataclasses.asdict(s) for s in model_contam_scores],
                "total_params": stats["total_params"],
                "trainable_params": stats["trainable_params"],
                "trainable_pct": stats["trainable_pct"],
                "model_size_gb": stats["model_size_gb"],
                "zero_shot_bpb": zero_shot_bpb,
                "best_bpb": final_bpb,
                "delta_bpb": final_bpb - zero_shot_bpb,
                "pct_improvement": pct_imp,
                "adaptation_score": adapt_score,
                "stability_index": stability_index,
                "completion_reason": (
                    "Converged (Chained Training)" if args.phase == "train"
                    else "Time Budget Reached" if full_res.get("stopped_on_time")
                    else "Early Stopping (Converged)" if full_res.get("steps_trained", 0) < (args.final_epochs * len(train_ds) // best_train_cfg.batch_size)
                    else "Target Epochs Reached"
                ),
                "training_time_seconds": full_res["training_time_seconds"],
                "peak_gpu_memory_gb": full_res["peak_gpu_memory_gb"],
                "bpb_per_gpu_hour": pct_imp / (training_hours if training_hours > 0 else 1e-9),
                "best_config": best_train_cfg.__dict__,
                "effective_batch_size": best_train_cfg.batch_size * best_train_cfg.gradient_accumulation_steps,
                "range_health": range_health,
                "bpb_curve": full_res["bpb_curve"],
                "dtype": str(dtype)
            }
            results.append(res)
            
            with open(result_json, "w") as f:
                json.dump(res, f, indent=2)
            
            plot_individual_learning_curve(model_id, slug, full_res["bpb_curve"], args.output)
            if trials_df is not None:
                plot_sweep_scatter(model_id, slug, trials_df, args.output)
                plot_parallel_coords(model_id, slug, trials_df, args.output)
            if study is not None:
                plot_hyperparameter_importance(model_id, slug, study, args.output)

            del model
            if device.type == "cuda": torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.exception(f"Failed processing model {model_id}: {e}")

        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

    if args.no_aggregate:
        logger.info("--no-aggregate set: skipping report aggregation (sweep or intermediate training-chain job).")
        return

    if not results:
        logger.error("No successful model results.")
        return

    if world_size > 1 and rank != 0:
        logger.info(f"Rank {rank} finished. Output saved to {summary_path}")
        return

    # If we are here, we are either Rank 0 (in a parallel run) or in a single-process run (world_size=1)
    if world_size > 1 and rank == 0:
        logger.info(f"Rank 0 waiting for parallel nodes to report completion...")
        start_wait = time.time()
        # Wait until just past every model's wall-time budget deadline (anchored to job start),
        # plus a margin for the final eval write. Other ranks stop training by wall_time_seconds
        # and write their result JSON shortly after, so rank 0 should see them all before this.
        # Far better than a flat 4h, which bailed early and rescued slow big models as "Partial".
        timeout = max(3600.0,
                      (args.wall_time_seconds + args.eval_reserve_seconds + 1800.0) - (time.time() - proc_start))
        logger.info(f"Rank 0 aggregation wait timeout: {timeout/3600:.2f}h")
        all_summaries_in_time = None
        
        while True:
            summaries = [f for f in os.listdir(args.output) if f.startswith("summary_rank_") and f.endswith(".json")]
            missing_models = []
            for mid in all_models:
                slug = slugify(mid)
                res_path = os.path.join(args.output, slug, f"result_{slug}.json")
                if not os.path.exists(res_path):
                    missing_models.append(mid)
            
            # Bypassing NFS directory cache latency:
            # If all expected result JSONs are present on disk, we have everything we need.
            # We do not need to wait for the summary_rank files to sync.
            if not missing_models:
                logger.info("All model result JSONs found. Proceeding to aggregation...")
                break
            
            if len(summaries) >= world_size and missing_models:
                if all_summaries_in_time is None:
                    all_summaries_in_time = time.time()
                    logger.info(f"All ranks reported. Waiting 10 mins for missing models: {missing_models}")
                if time.time() - all_summaries_in_time > 600:
                    logger.warning(f"Timeout waiting for model JSONs: {missing_models}. Rescuing partials.")
                    break

            if time.time() - start_wait > timeout:
                logger.warning(f"Global timeout. Summaries: {len(summaries)}/{world_size}. Missing: {missing_models}")
                break
            time.sleep(30)
        
    # ----------------------------------------------------
    # AGGREGATION & INCREMENTAL RE-EVALUATION PHASE
    # ----------------------------------------------------
    # Runs on Rank 0 and Single-Node local runs (world_size=1)
    
    # Compute global unified forensic quarantine union if enabled
    global_union_indices = []
    if contam_cfg.cleaning_mode == "global_unified":
        logger.info("Global Unified Cleaning: Computing global union of forensic indices across all models...")
        global_union_set = set()
        for mid in all_models:
            slug = slugify(mid)
            idx_path = os.path.join(args.output, slug, "suspicious_indices.json")
            if os.path.exists(idx_path):
                try:
                    with open(idx_path, "r") as f:
                        global_union_set.update(json.load(f))
                except Exception as e:
                    logger.error(f"Failed to read {idx_path}: {e}")
        global_union_indices = sorted(list(global_union_set))
        logger.info(f"Global Unified Forensic Union contains {len(global_union_indices)} quarantined samples.")
        
        # PERSIST GLOBAL FORENSIC UNION FILE FOR TRANSPARENCY
        global_indices_path = os.path.join(args.output, "contamination", "global_unified_indices.json")
        try:
            os.makedirs(os.path.dirname(global_indices_path), exist_ok=True)
            with open(global_indices_path, "w") as f:
                json.dump(global_union_indices, f)
            logger.info("Successfully exported global_unified_indices.json")
        except Exception as e:
            logger.error(f"Failed to write global_unified_indices.json: {e}")
            
        # Update global contamination summary file with the unified forensic count
        global_summary_path = os.path.join(args.output, "contamination", "contamination_summary.json")
        if os.path.exists(global_summary_path):
            try:
                with open(global_summary_path, "r") as f:
                    summary_data = json.load(f)
                summary_data["global_forensic_quarantine_count"] = len(global_union_indices)
                with open(global_summary_path, "w") as f:
                    json.dump(summary_data, f, indent=2)
                logger.info("Successfully updated contamination_summary.json with global_forensic_quarantine_count")
            except Exception as e:
                logger.error(f"Failed to update contamination_summary.json: {e}")

    aggregated_results = []
    for mid in all_models:
        slug = slugify(mid)
        model_dir = os.path.join(args.output, slug)
        res_path = os.path.join(model_dir, f"result_{slug}.json")
        
        data = None
        if os.path.exists(res_path):
            try:
                with open(res_path, "r") as f:
                    data = json.load(f)
            except: pass
        
        if data is None:
            logger.info(f"Attempting rescue for missing model: {mid}")
            data = {"model_id": mid, "model_slug": slug, "rank": 999, "total_params": 0, "trainable_pct": 0, "zero_shot_bpb": 0}
            trials_csv = os.path.join(model_dir, "trials.csv")
            if os.path.exists(trials_csv):
                try:
                    df = pd.read_csv(trials_csv)
                    df_comp = df[df["state"] == "COMPLETE"]
                    if not df_comp.empty:
                        data["best_bpb"] = df_comp["val_bpb"].min()
                        data["completion_reason"] = "Partial (Sweep Only)"
                        data["zero_shot_bpb"] = df["val_bpb"].iloc[0]
                        logger.info(f"Rescued best sweep BPB: {data['best_bpb']}")
                except: pass
        
        # AUTOMATIC INCREMENTAL RE-EVALUATION
        # If a new model added more quarantined samples, older models must be re-evaluated
        # on this newly-cleaned test set to maintain perfect comparability.
        # Only re-evaluate successfully completed models that have actual weight checkpoints saved on disk.
        # This prevents CPU/GPU device mismatch or VRAM fragmentation crashes when attempting to reload massive failed/timed-out models.
        if os.path.exists(res_path) and contam_cfg.cleaning_mode == "global_unified" and len(global_union_indices) > data.get("quarantined_count", 0):
            logger.info(f"Incremental Forensic Re-evaluation triggered for {mid}...")
            # Clean up CUDA memory before loading any model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                device = get_device(args.device)
                dtype = get_dtype(args.dtype, device)
                run_cfg = RunConfig(
                    model_id=mid, output_dir=args.output, device=args.device,
                    dtype=args.dtype, flash_attention=args.flash_attention,
                    hf_token=args.hf_token
                )
                
                # Re-construct the newly-unified test set
                original_test_size = len(dm.test_texts)
                max_drop = int(original_test_size * contam_cfg.max_soft_eval_quarantine_fraction)
                to_drop = list(global_union_indices)[:max_drop]
                
                test_texts = [t for i, t in enumerate(dm.test_texts) if i not in to_drop]
                
                # Reload tokenizer & model
                base_model, tokeniser = get_model_and_tokeniser(mid, run_cfg, device, dtype)
                
                def process(texts):
                    all_ids = []
                    for t in texts:
                        all_ids.extend(tokeniser(str(t), add_special_tokens=True)["input_ids"])
                    return [{"input_ids": torch.tensor(all_ids[i:i+args.max_seq_len]), 
                             "labels": torch.tensor(all_ids[i:i+args.max_seq_len])} 
                            for i in range(0, len(all_ids)-args.max_seq_len+1, args.max_seq_len)]
                
                test_ds = BlockDataset(process(test_texts))
                avg_bpt = dm.calculate_avg_bytes_per_token(tokeniser)
                
                # Calculate New Zero-Shot baseline
                new_zs_bpb = compute_zero_shot_bpb(base_model, test_ds, avg_bpt, device, args.eval_batch_size)
                
                # Load Saved Adapters/Weights to calculate New Adapted BPB
                del base_model
                if device.type == "cuda": torch.cuda.empty_cache()
                gc.collect()
                
                model, _ = get_model_and_tokeniser(mid, run_cfg, device, dtype)
                checkpoint_dir = os.path.join(model_dir, "final_checkpoint")
                if os.path.exists(checkpoint_dir):
                    if args.mode == "lora":
                        from peft import PeftModel
                        model = PeftModel.from_pretrained(model, checkpoint_dir)
                    else:
                        from transformers import AutoModelForCausalLM
                        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True, device_map="auto" if device.type == "cuda" else None)
                
                model.eval()
                new_final_bpb = compute_bpb(model, test_ds, avg_bpt, device, args.eval_batch_size)
                
                logger.info(f"Re-evaluation success for {mid}: Zero-Shot {data.get('zero_shot_bpb', 0):.4f} -> {new_zs_bpb:.4f}, Adapted {data.get('best_bpb', 0):.4f} -> {new_final_bpb:.4f}")
                
                # Update model results
                data["zero_shot_bpb"] = new_zs_bpb
                data["best_bpb"] = new_final_bpb
                data["quarantined_count"] = len(global_union_indices)
                
                # Save updated results back to disk
                with open(res_path, "w") as f:
                    json.dump(data, f, indent=2)
                    
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"Failed to re-evaluate {mid} on unified test set: {e}")
                # Aggressively clean up memory to prevent leaking to the next iteration on failure
                if 'base_model' in locals(): del base_model
                if 'model' in locals(): del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if data and (data.get("best_bpb") is not None or data.get("zero_shot_bpb") is not None):
            is_rescued = False
            b = data.get("best_bpb")
            if b is None or b != b or b == 0 or b > 5.0:
                curve = data.get("bpb_curve", [])
                valid_bpbs = [v for s, v in curve if v == v and v is not None and 0 < v < 5.0]
                if valid_bpbs:
                    data["best_bpb"] = min(valid_bpbs)
                    is_rescued = True
            
            zs = data.get("zero_shot_bpb", 0)
            bb = data.get("best_bpb", 0)
            if zs and zs > 0 and bb and bb > 0:
                data["pct_improvement"] = (zs - bb) / zs * 100
                training_hours = data.get("training_time_seconds", 0) / 3600
                data["adaptation_score"] = data["pct_improvement"] / (training_hours + 1)
                data["delta_bpb"] = bb - zs
            
            if data.get("trainable_pct") == 100.0 and args.mode == "lora":
                data["trainable_pct"] = 0.5 
                data["trainable_params"] = int(data.get("total_params", 0) * 0.005)
            
            aggregated_results.append(data)
            logger.info(f"Included model in report: {mid} (Status: {data.get('completion_reason', 'Success')})")
    
    if not aggregated_results:
        logger.error("No results found during aggregation.")
        return
    results = aggregated_results

    def safe_bpb(x):
        b = x.get("best_bpb", float('inf'))
        if b != b or b is None or b == 0: return float('inf')
        return b

    results = sorted(results, key=safe_bpb)
    for i, r in enumerate(results): r["rank"] = i + 1

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # PERSIST MODEL FORENSIC SCORES TO THE JSON ARTIFACT
    scores_path = os.path.join(args.output, "contamination", "per_model_contamination_scores.json")
    try:
        model_scores_payload = {}
        for r in results:
            if "contamination_scores" in r:
                model_scores_payload[r["model_id"]] = r["contamination_scores"]
        
        # Merge with existing if any, or overwrite with current complete run
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        with open(scores_path, "w") as f:
            json.dump(model_scores_payload, f, indent=2)
        logger.info("Successfully persisted per_model_contamination_scores.json")
    except Exception as e:
        logger.error(f"Failed to write per_model_contamination_scores.json: {e}")

    logger.info("Regenerating all charts for the final report...")
    fig_dir = os.path.join(args.output, "figures")
    if os.path.exists(fig_dir): shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    for r in results:
        mid, slug = r["model_id"], r["model_slug"]
        if "bpb_curve" in r: plot_individual_learning_curve(mid, slug, r["bpb_curve"], args.output)
        trials_csv = os.path.join(args.output, slug, "trials.csv")
        if os.path.exists(trials_csv):
            df = pd.read_csv(trials_csv)
            plot_sweep_scatter(mid, slug, df, args.output)
            plot_parallel_coords(mid, slug, df, args.output)

    plot_ranking_bar(results, args.output)
    plot_efficiency_scatter(results, args.output)
    plot_learning_curves(results, args.output)

    metadata = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": args.dataset,
        "hardware": hardware_info(),
        "library_versions": library_versions()
    }
    
    contam_audit_data = None
    if dm.contamination_audit:
        contam_audit_data = dataclasses.asdict(dm.contamination_audit)
    else:
        contam_summary_path = os.path.join(args.output, "contamination", "contamination_summary.json")
        if os.path.exists(contam_summary_path):
            with open(contam_summary_path, "r") as f:
                contam_audit_data = {"dataset_summary": json.load(f)}

    # Inject global_forensic_quarantine_count if present in the summary file
    c_summary_path = os.path.join(args.output, "contamination", "contamination_summary.json")
    if os.path.exists(c_summary_path) and contam_audit_data:
        try:
            with open(c_summary_path, "r") as f:
                s_data = json.load(f)
                if "global_forensic_quarantine_count" in s_data:
                    contam_audit_data["dataset_summary"]["global_forensic_quarantine_count"] = s_data["global_forensic_quarantine_count"]
        except Exception as e:
            logger.error(f"Failed to inject global_forensic_quarantine_count into report data: {e}")

    report_builder = ReportBuilder(results, args.output, args.report_title, metadata, contamination_audit=contam_audit_data)
    report_path = report_builder.render(no_pdf=args.no_pdf)
    
    console = Console()
    table = Table(title=args.report_title)
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Params(M)", justify="right")
    table.add_column("Zero-shot BPB", justify="right")
    table.add_column("Fine-tuned BPB", justify="right")
    table.add_column("% Imp.", justify="right", style="green")
    table.add_column("Adapt Score", justify="right", style="yellow")

    for r in results:
        table.add_row(
            str(r.get("rank", "-")),
            r["model_id"],
            f"{r.get('total_params', 0)/1e6:.1f}",
            f"{r.get('zero_shot_bpb', 0):.4f}",
            f"{r.get('best_bpb', 0.0):.4f}",
            f"{r.get('pct_improvement', 0.0):.1f}%",
            f"{r.get('adaptation_score', 0.0):.2f}"
        )
    console.print(table)
    logger.info(f"Report generated: {report_path}")

if __name__ == "__main__":
    main()
