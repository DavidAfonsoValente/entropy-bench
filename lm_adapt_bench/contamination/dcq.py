import torch
import numpy as np
import random
import re
from typing import List, Dict, Any, Tuple, Optional
import logging
from .config import ContaminationConfig
from .schema import DCQSummary

class DCQDetector:
    def __init__(self, config: ContaminationConfig):
        self.config = config
        self.logger = logging.getLogger("lm_adapt_bench.contamination")

    def _perturb_text(self, text: str, rng: random.Random) -> str:
        """Simple word-level perturbations for the 'quiz' distractors."""
        words = text.split()
        if len(words) < 5:
            return text + " [noise]" # fallback
            
        # Strategy: Randomly swap two adjacent words
        perturbed = list(words)
        idx = rng.randint(0, len(words) - 2)
        perturbed[idx], perturbed[idx+1] = perturbed[idx+1], perturbed[idx]
        
        # Strategy: Replace a random word with a placeholder
        idx2 = rng.randint(0, len(words) - 1)
        perturbed[idx2] = "[PAD]"
        
        return " ".join(perturbed)

    def _get_logprob(self, model, tokeniser, text: str, device: torch.device) -> float:
        # Cap at 2048 for safety
        max_len = min(2048, getattr(model.config, "max_position_embeddings", 2048))
        inputs = tokeniser(text, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            return -outputs.loss.item() # Return log-likelihood (neg loss)

    def run_quiz(self, model, tokeniser, texts: List[str], device: torch.device, candidate_source: str = "random") -> Tuple[List[Dict[str, Any]], DCQSummary]:
        """
        DCQ: If model 'chooses' original over perturbed versions, it might be memorized.
        Deterministic using local RNG.
        """
        if not texts:
            return [], DCQSummary()

        local_rng = random.Random(self.config.seed)
        
        # In actual candidate selection, the caller should pass the pre-filtered texts.
        # Here we just sample if we exceed num_samples.
        num_to_score = min(len(texts), self.config.dc_samples)
        sample_indices = local_rng.sample(range(len(texts)), num_to_score)
        
        sample_results = []
        margins = []
        original_selected_count = 0
        
        model.eval()
        total = len(sample_indices)
        for i, idx in enumerate(sample_indices):
            if i % 25 == 0:
                self.logger.info(f"  [DCQ Quiz] Running quiz {i}/{total}...")
            
            original = texts[idx]
            
            options = [original]
            for _ in range(self.config.dcq_num_distractors):
                options.append(self._perturb_text(original, local_rng))
                
            # Randomize order
            option_indices = list(range(len(options)))
            local_rng.shuffle(option_indices)
            
            shuffled_options = [options[j] for j in option_indices]
            original_option_index = option_indices.index(0)
            
            option_scores = []
            for opt in shuffled_options:
                option_scores.append(self._get_logprob(model, tokeniser, opt, device))
                
            selected_option_index = int(np.argmax(option_scores))
            
            # Margin is diff between highest and second highest
            sorted_scores = sorted(option_scores, reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            
            is_original_selected = (selected_option_index == original_option_index)
            is_positive = is_original_selected and margin >= self.config.dcq_min_margin
            
            if is_positive:
                original_selected_count += 1
                
            sample_results.append({
                "original_index": idx,
                "original_option_index": original_option_index,
                "selected_option_index": selected_option_index,
                "original_was_selected": is_original_selected,
                "option_scores": option_scores,
                "score_margin": margin,
                "num_distractors": self.config.dcq_num_distractors,
                "is_positive": is_positive,
                "reason": "Original selected" if is_positive else "Distractor selected or margin too low"
            })
            margins.append(margin)
            
            if i % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

        accuracy = original_selected_count / num_to_score if num_to_score > 0 else 0.0
        expected_chance = 1.0 / (1.0 + self.config.dcq_num_distractors)
        
        risk_label = "LOW"
        if accuracy > self.config.dcq_accuracy_threshold:
            risk_label = "HIGH"

        summary = DCQSummary(
            num_candidates=len(texts),
            num_scored=num_to_score,
            original_selected_count=original_selected_count,
            dcq_accuracy=accuracy,
            mean_margin=float(np.mean(margins)) if margins else 0.0,
            positive_fraction=accuracy,
            candidate_source=candidate_source,
            risk_label=risk_label
        )
        
        return sample_results, summary
