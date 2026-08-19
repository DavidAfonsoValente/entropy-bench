import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from .config import ContaminationConfig
from .schema import MinKSummary

class MinKDetector:
    def __init__(self, config: ContaminationConfig):
        self.config = config
        self.logger = logging.getLogger("lm_adapt_bench.contamination")

    def _compute_token_stats(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes raw logprobs and optionally normalized minkpp scores.
        logits: [batch, seq, vocab]
        labels: [batch, seq]
        mask: [batch, seq]
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # logp_true: [batch, seq]
        true_logprobs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        
        normalized_scores = None
        if self.config.min_k_variant == "minkpp":
            # Compute mean and std over the vocab distribution at each position
            # probs = exp(log_probs)
            probs = torch.exp(log_probs)
            
            # mean_logp = sum(probs * log_probs, dim=-1)
            mean_logp = torch.sum(probs * log_probs, dim=-1)
            
            # var_logp = sum(probs * (log_probs - mean_logp)^2, dim=-1)
            # Use sum(p * x^2) - (sum(p * x))^2 for efficiency
            mean_logp_sq = torch.sum(probs * (log_probs ** 2), dim=-1)
            var_logp = mean_logp_sq - (mean_logp ** 2)
            
            # Ensure no negative variance due to precision
            var_logp = torch.clamp(var_logp, min=0.0)
            std_logp = torch.sqrt(var_logp + self.config.min_k_std_epsilon)
            
            normalized_scores = (true_logprobs - mean_logp) / std_logp
            
        return true_logprobs, normalized_scores

    def score_samples(self, model, tokeniser, texts: List[str], device: torch.device) -> Tuple[List[Dict[str, Any]], MinKSummary]:
        """
        Computes Min-K% style or true Min-K++ scores with robust statistical thresholding.
        Returns a tuple of (sample_results, summary).
        """
        sample_results = []
        raw_scores = []
        model.eval()
        
        skipped_too_short = 0
        scoring_failed = 0
        
        is_minkpp = (self.config.min_k_variant == "minkpp")
        
        with torch.no_grad():
            total = len(texts)
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    self.logger.info(f"  [Min-K++] Scoring sample {i}/{total}...")
                
                # Cap at 2048 to prevent OOM
                max_len = min(2048, getattr(model.config, "max_position_embeddings", 2048))
                inputs = tokeniser(text, return_tensors="pt", truncation=True, max_length=max_len)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits
                
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:].contiguous()
                
                true_logp, norm_scores = self._compute_token_stats(shift_logits, shift_labels, shift_mask)
                
                # Get valid entries based on mask
                valid_true_logp = true_logp[shift_mask.bool()].to(torch.float32).cpu().numpy()
                valid_len = len(valid_true_logp)
                
                if valid_len < self.config.min_k_min_tokens:
                    skipped_too_short += 1
                    sample_results.append({
                        "valid": False,
                        "variant": self.config.min_k_variant,
                        "reason": "too_short_for_reliable_min_k",
                        "token_count_used": valid_len,
                        "min_k_score": 0.0
                    })
                    continue
                
                if valid_len == 0:
                    scoring_failed += 1
                    sample_results.append({
                        "valid": False, 
                        "variant": self.config.min_k_variant,
                        "reason": "empty_tokens", 
                        "token_count_used": 0, 
                        "min_k_score": 0.0
                    })
                    continue
                
                # Selection logic:
                # Based on Min-K intuition, we select the lowest K% of raw logprobs (hard tokens)
                # Then we aggregate either the raw logprobs (mink) or the normalized scores (minkpp)
                k = max(1, int(valid_len * (self.config.min_k_percent / 100.0)))
                
                # Find indices of lowest k raw logprobs
                hard_token_indices = np.argsort(valid_true_logp)[:k]
                
                if is_minkpp and norm_scores is not None:
                    valid_norm_scores = norm_scores[shift_mask.bool()].to(torch.float32).cpu().numpy()
                    # Aggregate normalized scores over hard token positions
                    score = float(np.mean(valid_norm_scores[hard_token_indices]))
                else:
                    # Raw Min-K: aggregate raw logprobs over hard token positions
                    score = float(np.mean(valid_true_logp[hard_token_indices]))
                
                # Higher score is more suspicious for both variants
                # (less negative raw logp OR higher relative likelihood)
                sample_results.append({
                    "valid": True,
                    "variant": self.config.min_k_variant,
                    "min_k_score": score,
                    "token_count_used": valid_len,
                    "reason": "scored"
                })
                raw_scores.append(score)
                
                if i % 50 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()

        # Compute statistics and summary
        valid_n = len(raw_scores)
        summary = MinKSummary(
            variant=self.config.min_k_variant,
            normalized_scoring=is_minkpp,
            valid_n=valid_n,
            skipped_too_short=skipped_too_short,
            scoring_failed=scoring_failed,
            score_direction=self.config.min_k_score_direction
        )
        
        # ... (rest of thresholding logic from previous implementaton) ...
        # I'll keep the existing MAD/robust-z calibration
        if valid_n < 30:
            summary.calibration_confidence = "INSUFFICIENT"
        elif valid_n < 100:
            summary.calibration_confidence = "LOW"
        else:
            summary.calibration_confidence = "HIGH"

        if valid_n > 0:
            raw_scores_np = np.array(raw_scores)
            summary.p95 = float(np.percentile(raw_scores_np, self.config.min_k_outlier_percentile))
            summary.p99 = float(np.percentile(raw_scores_np, self.config.min_k_high_percentile))
            summary.median = float(np.median(raw_scores_np))
            summary.mad = float(np.median(np.abs(raw_scores_np - summary.median)))
            
            mad_multiplier = 1.4826
            
            outlier_count = 0
            suspicious_count = 0
            high_suspicious_count = 0
            severe_guardrail_count = 0
            
            for res in sample_results:
                if not res["valid"]: continue
                
                score = res["min_k_score"]
                pct_rank = float((np.sum(raw_scores_np <= score) / valid_n) * 100.0)
                res["percentile_rank"] = pct_rank
                
                robust_z = 0.0
                if summary.mad > 0:
                    robust_z = (score - summary.median) / (summary.mad * mad_multiplier)
                res["robust_z"] = robust_z
                
                classification = "none"
                # Only apply raw guardrail to raw variant
                is_raw_variant = (not is_minkpp)
                if is_raw_variant and self.config.min_k_enable_severe_guardrail and score > self.config.min_k_severe_guardrail:
                    classification = "severe_guardrail"
                    severe_guardrail_count += 1
                elif pct_rank >= self.config.min_k_high_percentile and robust_z >= self.config.min_k_robust_z_high:
                    classification = "high_suspicious"
                    high_suspicious_count += 1
                elif pct_rank >= self.config.min_k_high_percentile or (pct_rank >= self.config.min_k_outlier_percentile and robust_z >= self.config.min_k_robust_z_suspicious):
                    classification = "suspicious"
                    suspicious_count += 1
                elif pct_rank >= self.config.min_k_outlier_percentile:
                    classification = "outlier"
                    outlier_count += 1
                    
                res["classification"] = classification
            
            summary.outlier_fraction = outlier_count / valid_n
            summary.suspicious_fraction = suspicious_count / valid_n
            summary.high_suspicious_fraction = high_suspicious_count / valid_n
            summary.severe_guardrail_fraction = severe_guardrail_count / valid_n

        return sample_results, summary
