import torch
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
import logging
from .config import ContaminationConfig
from .schema import CodecSummary

class CoDeCDetector:
    def __init__(self, config: ContaminationConfig):
        self.config = config
        self.logger = logging.getLogger("lm_adapt_bench.contamination")

    def _get_nll(self, model, tokeniser, text: str, prefix: str, device: torch.device) -> float:
        full_text = prefix + text if prefix else text
        # Cap at 2048 to prevent memory blowouts with few-shot context
        max_len = min(2048, getattr(model.config, "max_position_embeddings", 2048))
        inputs = tokeniser(full_text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # We only want the NLL of the 'text' part
        if prefix:
            # Re-tokenize prefix with same max_len to get accurate slice point
            prefix_ids = tokeniser(prefix, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]
            prefix_len = prefix_ids.shape[1]
        else:
            prefix_len = 0
            
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits # [batch, seq, vocab]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Loss per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Mask out padding
            loss = loss * shift_mask.view(-1)
            
            # Only average loss for the tokens belonging to 'text'
            # Note: prefix_len - 1 because shift_labels starts from index 1 (predicting t0)
            start_idx = max(0, prefix_len - 1)
            target_loss = loss[start_idx:]
            target_mask = shift_mask.view(-1)[start_idx:]
            
            valid_loss = target_loss[target_mask.bool()]
            
            if len(valid_loss) == 0:
                return 0.0
            return float(torch.mean(valid_loss).cpu().item())

    def score_dataset(self, model, tokeniser, texts: List[str], device: torch.device) -> Tuple[List[Dict[str, Any]], CodecSummary]:
        """
        CoDeC: Measure NLL delta when context is added.
        Calibrated version using robust statistics (Median/MAD).
        """
        if len(texts) < self.config.codec_context_size + 1:
            self.logger.warning(f"Not enough texts ({len(texts)}) for CoDeC context size {self.config.codec_context_size}")
            return [], CodecSummary(num_samples=0, context_size=self.config.codec_context_size, calibration_confidence="INSUFFICIENT")

        local_rng = random.Random(self.config.seed)
        all_indices = list(range(len(texts)))
        sample_indices = local_rng.sample(all_indices, min(len(texts), self.config.codec_samples))
        
        sample_results = []
        raw_deltas = []
        
        model.eval()
        total = len(sample_indices)
        for i, idx in enumerate(sample_indices):
            if i % 50 == 0:
                self.logger.info(f"  [CoDeC] Scoring sample {i}/{total}...")
            
            target_text = texts[idx]
            
            # Select K context samples (not the target itself)
            other_indices = [j for j in all_indices if j != idx]
            context_indices = local_rng.sample(other_indices, self.config.codec_context_size)
            context_prefix = "\n\n".join([texts[j] for j in context_indices]) + "\n\n"
            
            nll_base = self._get_nll(model, tokeniser, target_text, "", device)
            nll_context = self._get_nll(model, tokeniser, target_text, context_prefix, device)
            
            # CoDeC logic: delta = nll_context - nll_base
            # suspicious if delta < 0 (confidence increases with context)
            delta = nll_context - nll_base
            raw_deltas.append(delta)
            
            sample_results.append({
                "original_index": idx,
                "base_score": nll_base,
                "contextual_score": nll_context,
                "delta": delta,
                "context_indices": context_indices
            })
            
            if i % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()
            
        deltas = np.array(raw_deltas, dtype=np.float32)
        valid_n = len(deltas)
        
        summary = CodecSummary(
            num_samples=valid_n,
            context_size=self.config.codec_context_size,
            threshold_used=self.config.codec_delta_threshold
        )
        
        if valid_n < 30:
            summary.calibration_confidence = "INSUFFICIENT"
        elif valid_n < 100:
            summary.calibration_confidence = "LOW"
        else:
            summary.calibration_confidence = "HIGH"

        if valid_n > 0:
            summary.codec_delta_mean = float(np.mean(deltas))
            summary.codec_delta_median = float(np.median(deltas))
            summary.codec_delta_std = float(np.std(deltas))
            
            # Scientific Calibration: Instead of 0.0, we use a robust outlier threshold
            # MAD (Median Absolute Deviation)
            mad = float(np.median(np.abs(deltas - summary.codec_delta_median)))
            mad_multiplier = 1.4826
            
            suspicious_count = 0
            for res in sample_results:
                delta = res["delta"]
                
                # Robust Z-score: how many MADs is this delta from the median?
                # We only care about NEGATIVE outliers (unusually large confidence gain)
                robust_z = 0.0
                if mad > 0:
                    robust_z = (delta - summary.codec_delta_median) / (mad * mad_multiplier)
                
                res["robust_z"] = robust_z
                
                # A sample is suspicious if:
                # 1. Delta is absolute negative (required by paper)
                # 2. Delta is a statistical outlier compared to the rest of the split
                is_suspicious = (delta < self.config.codec_delta_threshold) and (robust_z < self.config.codec_robust_z_threshold)
                
                res["suspicious"] = is_suspicious
                if is_suspicious:
                    suspicious_count += 1
                    res["reason"] = f"CoDeC robust Z-score ({robust_z:.2f}) indicates anomalous confidence gain."
                else:
                    res["reason"] = ""

            summary.suspicious_fraction = suspicious_count / valid_n
            
            # Set split-level risk label
            if summary.suspicious_fraction >= self.config.codec_suspicious_fraction_severe:
                summary.risk_label = "SEVERE"
            elif summary.suspicious_fraction >= self.config.codec_suspicious_fraction_high:
                summary.risk_label = "HIGH"
            elif summary.suspicious_fraction >= self.config.codec_suspicious_fraction_medium:
                summary.risk_label = "MEDIUM"
            else:
                summary.risk_label = "LOW"
                
        return sample_results, summary
