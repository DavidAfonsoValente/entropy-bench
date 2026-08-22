import logging
import torch
from typing import List, Dict, Any, Optional, Tuple
from .min_k import MinKDetector
from .codec import CoDeCDetector
from .dcq import DCQDetector
from .config import ContaminationConfig
from .schema import ModelContaminationScore

class ModelSignalOrchestrator:
    def __init__(self, config: ContaminationConfig):
        self.config = config
        self.logger = logging.getLogger("lm_adapt_bench.contamination")
        self.min_k = MinKDetector(config)
        self.codec = CoDeCDetector(config)
        self.dcq = DCQDetector(config)

    def run_all_signals(self, model, tokeniser, splits: Dict[str, List[str]], device: torch.device) -> List[ModelContaminationScore]:
        model_id = model.config._name_or_path
        scores = []
        
        for split_name, texts in splits.items():
            if not texts: continue
            
            # GATE: Model-specific signals only run in standard mode or higher
            if self.config.check_level not in ["standard", "strict", "forensic"]:
                continue

            self.logger.info(f"Running model-specific signals for {model_id} on {split_name} split")
            
            # 1. Min-K Scoring
            sample_size = self.config.sample_size
            if split_name == "test":
                # For dry-runs/standard benchmarks, 512 samples is enough for a robust signal.
                # Only use 5000 in full forensic mode.
                default_test_max = 512
                if self.config.check_level == "forensic":
                    default_test_max = self.config.test_full_max
                sample_size = max(sample_size, default_test_max)
            
            sampled_texts = texts[:sample_size]
            mk_results, mk_summary = self.min_k.score_samples(model, tokeniser, sampled_texts, device)
            
            # 2. CoDeC Scoring
            codec_results, codec_summary = self.codec.score_dataset(model, tokeniser, texts, device)
            
            # 3. DCQ Confirmation
            dcq_summary = None
            suspicious_indices = set()
            
            # Determine candidates for DCQ if strict/forensic
            dcq_candidates = []
            dcq_source = "none"
            
            if self.config.check_level in ["strict", "forensic"] or self.config.enable_dcq:
                # Get Min-K highly suspicious
                for idx, res in enumerate(mk_results):
                    if res.get("classification") in ["suspicious", "high_suspicious", "severe_guardrail"]:
                        dcq_candidates.append((idx, sampled_texts[idx]))
                
                # If CoDeC is high, maybe add CoDeC suspicious too
                if codec_summary.risk_label in ["HIGH", "SEVERE"]:
                    for res in codec_results:
                        if res.get("suspicious"):
                            idx = res["original_index"]
                            if idx not in [c[0] for c in dcq_candidates]:
                                dcq_candidates.append((idx, texts[idx]))
                
                if dcq_candidates:
                    dcq_source = "min_k_and_codec_suspicious"
                    candidate_texts = [c[1] for c in dcq_candidates]
                    dcq_results, dcq_summary = self.dcq.run_quiz(model, tokeniser, candidate_texts, device, candidate_source=dcq_source)
                    
                    # If DCQ is positive, we flag the index for quarantine
                    for i, res in enumerate(dcq_results):
                        if res.get("is_positive"):
                            orig_idx = dcq_candidates[res["original_index"]][0]
                            suspicious_indices.add(orig_idx)

            # 4. Fallback: if not strict mode, but we want to log indices for severe cases
            if self.config.check_level not in ["strict", "forensic"] and not self.config.enable_dcq:
                # Standard mode: Min-K severe guardrail might still trigger a drop depending on config,
                # but usually soft signals alone don't drop. We'll pass the severe ones up just in case.
                for idx, res in enumerate(mk_results):
                    if res.get("classification") == "severe_guardrail":
                        suspicious_indices.add(idx)

            risk_score, risk_label, notes = self._aggregate_model_risk(mk_summary, codec_summary, dcq_summary)
            
            score_obj = ModelContaminationScore(
                model_id=model_id,
                split=split_name,
                min_k_summary=mk_summary,
                codec_summary=codec_summary,
                dcq_summary=dcq_summary,
                suspicious_fraction=codec_summary.suspicious_fraction if codec_summary else 0.0,
                risk_score=risk_score,
                risk_label=risk_label,
                sample_size=len(sampled_texts),
                suspicious_indices=list(suspicious_indices),
                notes=notes
            )
            scores.append(score_obj)
            
        return scores

    def _aggregate_model_risk(self, min_k, codec, dcq) -> Tuple[float, str, List[str]]:
        risk_score = 0.0
        notes = []
        
        # CoDeC signal
        if codec:
            sf = codec.suspicious_fraction
            risk_score += sf * 0.5
            if codec.risk_label in ["HIGH", "SEVERE"]:
                notes.append(f"CoDeC risk is {codec.risk_label} (suspicious fraction {sf:.2f}).")
                
        # Min-K signal
        if min_k:
            if min_k.high_suspicious_fraction > 0.05:
                risk_score += 0.3
                notes.append(f"Min-K has elevated high-suspicious fraction ({min_k.high_suspicious_fraction:.2f}).")
            if min_k.severe_guardrail_fraction > 0.01:
                risk_score += 0.2
                notes.append(f"Min-K triggered severe guardrail for {min_k.severe_guardrail_fraction:.2%} of samples.")

        # DCQ signal
        if dcq and dcq.num_scored > 0:
            if dcq.risk_label == "HIGH":
                risk_score += 0.4
                notes.append(
                    f"DCQ preference rate ({dcq.dcq_accuracy:.2f}) flagged perturbation-sensitive "
                    "samples; this is a conservative risk signal, not proof of memorization."
                )

        risk_label = "LOW"
        if risk_score > self.config.risk_high_threshold: risk_label = "HIGH"
        elif risk_score > self.config.risk_medium_threshold: risk_label = "MEDIUM"
        
        return min(1.0, risk_score), risk_label, notes
