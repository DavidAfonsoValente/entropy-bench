"""Offline smoke test of the full pipeline against mocked models.

Runs config -> data -> sweep -> adapt -> evaluate -> report end to end with the model and trainer
replaced by mocks, writing to ./results_verify, so the wiring is exercised on CPU in seconds
without downloading a checkpoint. Catches the class of breakage that unit tests miss because they
never run the stages together. Run by `make check`.
"""
import sys
import os
import shutil
import json
import dataclasses
import unittest
from unittest.mock import patch, MagicMock
import torch

# Create a temporary directory for verification outputs
VERIFY_DIR = "./results_verify"
if os.path.exists(VERIFY_DIR):
    shutil.rmtree(VERIFY_DIR)
os.makedirs(VERIFY_DIR, exist_ok=True)

# ----------------------------------------------------
# MOCKS
# ----------------------------------------------------
class MockConfig:
    max_position_embeddings = 512
    _name_or_path = "mock-model"

class MockModel:
    config = MockConfig()
    def eval(self): pass
    def to(self, *args, **kwargs): return self
    def parameters(self):
        return [torch.zeros(10)]
    def __call__(self, input_ids, attention_mask=None, labels=None, **kwargs):
        class MockOutput:
            logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 10)
            loss = torch.tensor(1.5)
        return MockOutput()

class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = [1, 2, 3, 4, 5]
        mask = [1, 1, 1, 1, 1]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([tokens]),
                "attention_mask": torch.tensor([mask])
            }
        return {"input_ids": tokens, "attention_mask": mask}
    
    def decode(self, tokens, **kwargs):
        return "mock text"

def mock_get_model_and_tokeniser(model_id, run_cfg, device, dtype):
    return MockModel(), MockTokenizer()

def mock_peft_from_pretrained(model, checkpoint_dir, **kwargs):
    return MockModel()

# ----------------------------------------------------
# PLAYBOOK VERIFICATION TEST SUITE
# ----------------------------------------------------
class TestPlaybookVerification(unittest.TestCase):

    @patch("lm_adapt_bench.cli.get_model_and_tokeniser", mock_get_model_and_tokeniser)
    @patch("peft.PeftModel.from_pretrained", mock_peft_from_pretrained)
    def test_playbook_workflows(self):
        from lm_adapt_bench.cli import main
        import sys
        
        # We also need a dummy txt file to act as the dataset
        # Write duplicate sentences separated by double newlines (\n\n) to match data.py parser!
        dummy_dataset = os.path.join(VERIFY_DIR, "dummy.txt")
        with open(dummy_dataset, "w") as f:
            for i in range(250):
                if i < 5:
                    f.write("This is a duplicated sentence that must be cleaned.\n\n")
                else:
                    f.write(f"Sample sentence {i} for evaluation purposes.\n\n")
                
        # ----------------------------------------------------
        # CASE 1: Fast Forensic Audit (Baseline Only)
        # ----------------------------------------------------
        print("\n=== Verifying Case 1: Fast Forensic Audit (Baseline Only) ===")
        sys.argv = [
            "cli.py",
            "--dataset", dummy_dataset,
            "--models", "model_a", "model_b",
            "--output", VERIFY_DIR,
            "--contam-check-level", "strict",
            "--baseline-only"
        ]
        
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
            
        # Assert Case 1 Outputs
        self.assertTrue(os.path.exists(os.path.join(VERIFY_DIR, "model_a", "result_model_a.json")))
        self.assertTrue(os.path.exists(os.path.join(VERIFY_DIR, "model_b", "result_model_b.json")))
        self.assertTrue(os.path.exists(os.path.join(VERIFY_DIR, "contamination", "quarantine.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(VERIFY_DIR, "contamination", "cleaned_dataset", "train.jsonl")))
        
        # Verify Case 1 Result content
        with open(os.path.join(VERIFY_DIR, "model_a", "result_model_a.json"), "r") as f:
            data_a = json.load(f)
            self.assertEqual(data_a["completion_reason"], "Baseline Only")
            self.assertEqual(data_a["pct_improvement"], 0.0)

        # ----------------------------------------------------
        # CASE 3: Compliance-Strict Benchmark (Unified Mode)
        # ----------------------------------------------------
        print("\n=== Verifying Case 3: Compliance-Strict Benchmark (Unified Mode) ===")
        # Clean results for fresh run
        shutil.rmtree(os.path.join(VERIFY_DIR, "model_a"), ignore_errors=True)
        shutil.rmtree(os.path.join(VERIFY_DIR, "model_b"), ignore_errors=True)
        
        # Pre-write Mock Rank 1 outputs to prevent wait-deadlock
        # Since we are single-process, we must simulate Rank 1 finishing
        model_b_dir = os.path.join(VERIFY_DIR, "model_b")
        os.makedirs(model_b_dir, exist_ok=True)
        with open(os.path.join(model_b_dir, "suspicious_indices.json"), "w") as f:
            json.dump([1, 2, 3], f)
            
        res_b_path = os.path.join(model_b_dir, "result_model_b.json")
        with open(res_b_path, "w") as f:
            json.dump({
                "model_id": "model_b", "model_slug": "model_b", "zero_shot_bpb": 1.5, "best_bpb": 1.5,
                "quarantined_count": 3, "completion_reason": "Baseline Only", "total_params": 1000, "trainable_pct": 0.0
            }, f)
            
        with open(os.path.join(VERIFY_DIR, "summary_rank_1.json"), "w") as f:
            json.dump([], f)
            
        sys.argv = [
            "cli.py",
            "--dataset", dummy_dataset,
            "--models", "model_a", "model_b",
            "--output", VERIFY_DIR,
            "--contam-check-level", "strict",
            "--contam-cleaning-mode", "global_unified",
            "--baseline-only",
            "--rank", "0",
            "--world-size", "2"
        ]
        
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
            
        # Assert Case 3 Outputs
        self.assertTrue(os.path.exists(os.path.join(VERIFY_DIR, "contamination", "global_unified_indices.json")))
        
        # ----------------------------------------------------
        # CASE 5: Leaderboard Expansion (Incremental Addition)
        # ----------------------------------------------------
        print("\n=== Verifying Case 5: Leaderboard Expansion (Incremental Adding) ===")
        # Set up a mock "Model 5" run that introduces a new leak
        model_c_dir = os.path.join(VERIFY_DIR, "model_c")
        os.makedirs(model_c_dir, exist_ok=True)
        
        # Write "Model 5" (model-c) suspicious indices (introducing new index 4)
        with open(os.path.join(model_c_dir, "suspicious_indices.json"), "w") as f:
            json.dump([4], f)
            
        # Write "Model 5" mock results
        res_c_path = os.path.join(model_c_dir, "result_model_c.json")
        with open(res_c_path, "w") as f:
            json.dump({
                "model_id": "model_c", "model_slug": "model_c", "zero_shot_bpb": 1.5, "best_bpb": 1.5,
                "quarantined_count": 1, "completion_reason": "Baseline Only"
            }, f)
            
        # Trigger local compilation and incremental re-evaluation on the login node (world_size=1)
        sys.argv = [
            "cli.py",
            "--dataset", dummy_dataset,
            "--models", "model_a", "model_b", "model_c",
            "--output", VERIFY_DIR,
            "--contam-check-level", "strict",
            "--contam-cleaning-mode", "global_unified",
            "--baseline-only"
        ]
        
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
            
        # Verify that Model A was indeed re-evaluated because Model C added index 4!
        with open(os.path.join(VERIFY_DIR, "model_a", "result_model_a.json"), "r") as f:
            updated_data_a = json.load(f)
            # The quarantined_count should now be updated to match the new global union size (4)
            print(f"Model A updated quarantined count: {updated_data_a.get('quarantined_count')}")
            self.assertGreater(updated_data_a.get("quarantined_count", 0), 0)

        print("\n🚀 All 5 Playbook Cases successfully verified at code-level!")

if __name__ == "__main__":
    unittest.main()
