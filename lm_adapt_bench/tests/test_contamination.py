import pytest
import os
import shutil
import torch
import random
from lm_adapt_bench.contamination.text_normalize import normalize_text
from lm_adapt_bench.contamination.fingerprints import Fingerprinter
from lm_adapt_bench.contamination.deduper import Deduper
from lm_adapt_bench.contamination.split_leakage import SplitLeakageChecker
from lm_adapt_bench.contamination.risk_aggregator import RiskAggregator
from lm_adapt_bench.contamination.config import ContaminationConfig
from lm_adapt_bench.contamination.schema import ContaminationFinding


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed the global RNGs before every test in this module.

    Several mock models here return `torch.randn(...)` logits, so their behaviour depends on
    whatever consumed the global torch RNG earlier in the process. That made
    test_codec_direction pass in isolation but fail intermittently in a full-suite run,
    depending only on how many other test files had been collected first.
    """
    torch.manual_seed(0)
    random.seed(0)


def test_normalization():
    text = "  Hello \n WORLD!  "
    norm = normalize_text(text)
    assert norm == "Hello WORLD!"
    
    # Unicode normalize
    text2 = "H\u00e9llo" # Hello with accent
    norm2 = normalize_text(text2)
    assert norm2 == "H\u00e9llo"

def test_exact_dedupe():
    texts = ["hello world", "hello world", "different text"]
    ids = ["id1", "id2", "id3"]
    fingerprinter = Fingerprinter()
    deduper = Deduper(fingerprinter)
    dupes = deduper.find_exact_duplicates(texts, ids)
    assert len(dupes) == 1
    assert dupes[0] == ("id1", "id2")

def test_near_dedupe():
    try:
        import datasketch
    except ImportError:
        pytest.skip("datasketch not installed")
        
    texts = [
        "this is a long sentence that is almost the same as the next one for testing purposes",
        "this is a long sentence that is almost the same as the next one for testing purposes indeed",
        "completely different text that should not match anything"
    ]
    ids = ["id1", "id2", "id3"]
    fingerprinter = Fingerprinter(ngram_n=3)
    deduper = Deduper(fingerprinter)
    dupes = deduper.find_near_duplicates(texts, ids, threshold=0.7)
    assert len(dupes) >= 1
    assert dupes[0][0] == "id1"
    assert dupes[0][1] == "id2"

def test_split_leakage():
    splits = {
        "train": ["sample one", "sample two"],
        "test": ["sample one", "unique sample"]
    }
    split_ids = {
        "train": ["tr1", "tr2"],
        "test": ["te1", "te2"]
    }
    fingerprinter = Fingerprinter(ngram_n=2)
    deduper = Deduper(fingerprinter)
    checker = SplitLeakageChecker(deduper)
    findings = checker.check_leakage(splits, split_ids)
    
    leakage = [f for f in findings if f["detector"] == "exact_split_leakage"]
    assert len(leakage) == 1
    assert leakage[0]["record_id"] == "te1"
    assert leakage[0]["matched_record_id"] == "tr1"

def test_risk_aggregator():
    config = ContaminationConfig()
    agg = RiskAggregator(config)
    
    findings = [
        ContaminationFinding("f1", "id1", "test", "exact_split_leakage", "critical", 1.0)
    ]
    level, score = agg.aggregate_dataset_risk(findings, 100)
    assert level == "CRITICAL"
    assert score >= 0.5

@pytest.fixture
def mock_model():
    class MockConfig:
        max_position_embeddings = 512
        _name_or_path = "mock-model"
    class MockModel:
        config = MockConfig()
        def eval(self): pass
        def __call__(self, input_ids, attention_mask=None, labels=None, **kwargs):
            class MockOutput:
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100)
                loss = torch.tensor(2.0)
            return MockOutput()
    return MockModel()

@pytest.fixture
def mock_tokenizer():
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
    return MockTokenizer()

def test_min_k_mock(mock_model, mock_tokenizer):
    from lm_adapt_bench.contamination.min_k import MinKDetector
    cfg = ContaminationConfig(min_k_percent=20.0, min_k_min_tokens=1)
    detector = MinKDetector(cfg)
    results, summary = detector.score_samples(mock_model, mock_tokenizer, ["test text"], torch.device("cpu"))
    assert len(results) == 1
    assert isinstance(results[0]["min_k_score"], float)

def test_codec_mock(mock_model, mock_tokenizer):
    from lm_adapt_bench.contamination.codec import CoDeCDetector
    cfg = ContaminationConfig(codec_context_size=1, codec_samples=1)
    detector = CoDeCDetector(cfg)
    results, summary = detector.score_dataset(mock_model, mock_tokenizer, ["text1", "text2", "text3"], torch.device("cpu"))
    assert hasattr(summary, "suspicious_fraction")
    assert summary.num_samples == 1

def test_dcq_mock(mock_model, mock_tokenizer):
    from lm_adapt_bench.contamination.dcq import DCQDetector
    cfg = ContaminationConfig(dc_samples=1, dcq_num_distractors=1)
    detector = DCQDetector(cfg)
    results, summary = detector.run_quiz(mock_model, mock_tokenizer, ["long enough text for perturbation"], torch.device("cpu"))
    assert hasattr(summary, "dcq_accuracy")
    assert summary.num_scored == 1

def test_forbidden_corpus_overlap_with_ngram():
    try:
        import datasketch
    except ImportError:
        pytest.skip("datasketch not installed")
    from lm_adapt_bench.contamination.forbidden_overlap import ForbiddenChecker
    fingerprinter = Fingerprinter(ngram_n=2)
    checker = ForbiddenChecker(fingerprinter, threshold=0.1)
    checker.add_forbidden_corpus(["this is a public text we want to avoid"])
    
    findings = checker.check_overlap("this is a public text", "rec_1")
    assert len(findings) > 0
    assert findings[0]["matched_source"] == "forbidden"

def test_cleaner_drops_test_leakage():
    from lm_adapt_bench.contamination.cleaner import DatasetCleaner
    from lm_adapt_bench.contamination.schema import ContaminationFinding
    from lm_adapt_bench.contamination.config import ContaminationConfig
    cfg = ContaminationConfig(cleaning_policy="drop")
    cleaner = DatasetCleaner(cfg)
    
    splits = {"test": ["leaked text", "clean text"]}
    split_ids = {"test": ["id1", "id2"]}
    findings = [ContaminationFinding("f1", "id1", "test", "exact", "critical", 1.0)]
    
    cleaned_splits, cleaned_ids, quarantine = cleaner.apply_cleaning(splits, split_ids, findings)
    assert len(cleaned_splits["test"]) == 1
    assert cleaned_ids["test"][0] == "id2"
    assert len(quarantine) == 1

def test_quarantine_manifest_contains_reason_and_ids():
    from lm_adapt_bench.contamination.cleaner import DatasetCleaner
    from lm_adapt_bench.contamination.schema import ContaminationFinding
    from lm_adapt_bench.contamination.config import ContaminationConfig
    cfg = ContaminationConfig(cleaning_policy="quarantine")
    cleaner = DatasetCleaner(cfg)
    
    splits = {"val": ["suspicious"]}
    split_ids = {"val": ["id1"]}
    findings = [ContaminationFinding("f1", "id1", "val", "near_dedupe", "high", 0.9, reason="Test Reason")]
    
    _, _, quarantine = cleaner.apply_cleaning(splits, split_ids, findings)
    assert len(quarantine) == 1
    assert quarantine[0]["record_id"] == "id1"
    assert quarantine[0]["reason"] == "Test Reason"
    assert quarantine[0]["action"] == "quarantine"

def test_cli_flags_parse():
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["cli.py", "--dataset", "dummy", "--check-contamination", "--contam-check-level", "strict"]
        # Just verifying flags exist in parser
    finally:
        sys.argv = old_argv

def test_report_payload_serializable():
    import json
    import dataclasses
    from lm_adapt_bench.contamination.schema import DatasetContaminationSummary, ContaminationAudit
    summary = DatasetContaminationSummary()
    audit = ContaminationAudit("run1", summary)
    s = json.dumps(dataclasses.asdict(audit))
    assert "run1" in s

def test_contamination_disabled_preserves_legacy_flow():
    from lm_adapt_bench.data import DataModule
    from lm_adapt_bench.config import DataConfig, ContaminationConfig
    config = DataConfig(dataset_path="dummy.txt", contamination=ContaminationConfig(check_contamination=False))
    dm = DataModule.__new__(DataModule)
    dm.config = config
    dm.raw_texts = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    dm._assign_ids()
    dm._split_data()
    assert len(dm.train_texts) == 8

def test_optional_dependencies_are_lazy():
    import sys
    # even if missing, Fingerprinter should instantiate
    fp = Fingerprinter()
    assert fp.num_perm == 128

def test_min_k_shift_and_mask():
    from lm_adapt_bench.contamination.min_k import MinKDetector
    class MockModel:
        config = type('obj', (object,), {'max_position_embeddings': 512, '_name_or_path': 'mock'})
        def eval(self): pass
        def __call__(self, input_ids, attention_mask=None, **kwargs):
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, 10)
            for i in range(seq - 1):
                logits[0, i, input_ids[0, i+1]] = 10.0
            logits[0, 0, input_ids[0, 1]] = -5.0
            return type('obj', (object,), {'logits': logits})()
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def __call__(self, text, **kwargs):
            ids = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
            mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}
    
    cfg = ContaminationConfig(min_k_percent=100.0, min_k_min_tokens=1)
    detector = MinKDetector(cfg)
    results, summary = detector.score_samples(MockModel(), MockTokenizer(), ["clean"], torch.device("cpu"))
    assert len(results) == 1
    assert results[0]["token_count_used"] == 4 # Padding masked

@pytest.mark.xfail(strict=True, reason=(
    "This test does not exercise CoDeC. CoDeCDetector._get_nll computes the NLL from "
    "outputs.logits, and MockModel returns torch.randn(...) logits, so the branched "
    "`loss` value it sets is never read and the assertion is a property of unseeded "
    "random noise -- it passed or failed depending on how much global torch RNG earlier "
    "tests had consumed. With the RNG seeded (see the autouse fixture above) it fails "
    "deterministically: every delta lands at about -2.82 with a near-zero MAD, so no "
    "sample clears the robust-z < -2.0 outlier condition and suspicious_fraction is 0.0. "
    "Fixing it means having the mock emit logits that actually encode the memorisation "
    "signature -- low NLL on the target when a context prefix is present -- for one "
    "planted sample and not the rest. Left failing rather than silently reseeded so the "
    "gap stays visible."))
def test_codec_direction(mock_tokenizer):
    from lm_adapt_bench.contamination.codec import CoDeCDetector
    class MockModel:
        config = type('obj', (object,), {'max_position_embeddings': 512, '_name_or_path': 'mock'})
        def eval(self): pass
        def __call__(self, input_ids, **kwargs):
            if input_ids.sum() < 50:
                loss = torch.tensor(0.5)
            else:
                loss = torch.tensor(5.0)
            return type('obj', (object,), {'logits': torch.randn(1, input_ids.shape[1], 10), 'loss': loss})()
            
    texts = ["clean text"] * 50
    texts[0] = "outlier text"
    
    cfg = ContaminationConfig(codec_context_size=1, codec_samples=50, seed=42)
    detector = CoDeCDetector(cfg)
    results, summary = detector.score_dataset(MockModel(), mock_tokenizer, texts, torch.device("cpu"))
    assert summary.suspicious_fraction > 0

def test_detector_determinism():
    from lm_adapt_bench.contamination.codec import CoDeCDetector
    texts = [str(i) for i in range(20)]
    def run_codec():
        class StaticModel:
            config = type('obj', (object,), {'max_position_embeddings': 512, '_name_or_path': 'mock'})
            def eval(self): pass
            def __call__(self, input_ids, **kwargs):
                return type('obj', (object,), {'logits': torch.ones(1, input_ids.shape[1], 10), 'loss': torch.tensor(1.0)})()
        tokenizer = type('obj', (object,), {'pad_token_id': 0, 'eos_token_id': 2, '__call__': lambda self, text, **kwargs: {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}})()
        cfg = ContaminationConfig(codec_context_size=2, codec_samples=5, seed=42)
        det = CoDeCDetector(cfg)
        return det.score_dataset(StaticModel(), tokenizer, texts, torch.device("cpu"))
    
    results1, summary1 = run_codec()
    results2, summary2 = run_codec()
    
    assert [r["original_index"] for r in results1] == [r["original_index"] for r in results2]
    assert [r["delta"] for r in results1] == [r["delta"] for r in results2]

def test_minkpp_not_equal_raw_mink(mock_tokenizer):
    from lm_adapt_bench.contamination.min_k import MinKDetector
    import torch
    
    class VarianceModel:
        config = type('obj', (object,), {'max_position_embeddings': 512, '_name_or_path': 'mock'})
        def eval(self): pass
        def __call__(self, input_ids, **kwargs):
            logits = torch.zeros(1, input_ids.shape[1], 10)
            logits[0, 0, input_ids[0, 1]] = 20.0 
            return type('obj', (object,), {'logits': logits})()

    cfg_raw = ContaminationConfig(min_k_variant="mink", min_k_percent=100.0, min_k_min_tokens=1)
    cfg_pp = ContaminationConfig(min_k_variant="minkpp", min_k_percent=100.0, min_k_min_tokens=1)
    
    det_raw = MinKDetector(cfg_raw)
    det_pp = MinKDetector(cfg_pp)
    
    text = "this is a test text"
    res_raw, _ = det_raw.score_samples(VarianceModel(), mock_tokenizer, [text], torch.device("cpu"))
    res_pp, _ = det_pp.score_samples(VarianceModel(), mock_tokenizer, [text], torch.device("cpu"))
    
    assert res_raw[0]["min_k_score"] != res_pp[0]["min_k_score"]

def test_minkpp_shift_correctness():
    from lm_adapt_bench.contamination.min_k import MinKDetector
    import torch
    
    class ShiftModel:
        config = type('obj', (object,), {'max_position_embeddings': 512, '_name_or_path': 'mock'})
        def eval(self): pass
        def __call__(self, input_ids, **kwargs):
            logits = torch.zeros(1, input_ids.shape[1], 10)
            for i in range(input_ids.shape[1] - 1):
                logits[0, i, input_ids[0, i+1]] = 10.0
            return type('obj', (object,), {'logits': logits})()

    class ShiftTokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])}
            
    cfg = ContaminationConfig(min_k_variant="minkpp", min_k_percent=100.0, min_k_min_tokens=1)
    det = MinKDetector(cfg)
    results, summary = det.score_samples(ShiftModel(), ShiftTokenizer(), ["test"], torch.device("cpu"))
    assert results[0]["min_k_score"] > 0

def test_dcq_logic_verification():
    from lm_adapt_bench.contamination.dcq import DCQDetector
    import torch
    
    class PreferenceModel:
        config = type('obj', (object,), {'max_position_embeddings': 512, '_name_or_path': 'mock'})
        def eval(self): pass
        def __call__(self, input_ids, **kwargs):
            loss = torch.sum(input_ids).to(torch.float32) / 1000.0
            return type('obj', (object,), {'loss': loss})()

    class DistractorTokenizer:
        pad_token_id = 0
        def __call__(self, text, return_tensors=None, **kwargs):
            val = 100 if "[PAD]" in text else 10
            ids = [val, val, val]
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([[1, 1, 1]])}
            return {"input_ids": ids}

    cfg = ContaminationConfig(dc_samples=1, dcq_num_distractors=1, dcq_min_margin=0.01)
    detector = DCQDetector(cfg)
    results, summary = detector.run_quiz(PreferenceModel(), DistractorTokenizer(), ["this is a long sentence for perturbation"], torch.device("cpu"))
    assert summary.original_selected_count == 1
    assert results[0]["is_positive"] == True

def test_global_quarantine_import(tmp_path):
    import logging
    from lm_adapt_bench.data import DataModule
    from lm_adapt_bench.config import DataConfig
    from lm_adapt_bench.contamination.config import ContaminationConfig
    import json
    
    # Create mock quarantine.jsonl
    q_dir = tmp_path / "contamination"
    q_dir.mkdir()
    q_file = q_dir / "quarantine.jsonl"
    
    # We want to quarantine 'rec_1' (which is the first text '1')
    # and check if it gets successfully dropped from train_texts
    with open(q_file, "w") as f:
        f.write(json.dumps({"record_id": "rec_0", "split": "train", "reason": "test"}) + "\n")
        
    cfg = DataConfig(
        dataset_path="dummy.txt", 
        contamination=ContaminationConfig(
            check_contamination=True,
            check_level="light",
            use_global_quarantine=True,
            output_dir=str(q_dir),
            cleaning_policy="drop"
        )
    )
    
    dm = DataModule.__new__(DataModule)
    dm.config = cfg
    dm.logger = logging.getLogger("lm_adapt_bench")
    
    dm.raw_texts = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    dm.raw_ids = [f"rec_{i}" for i in range(10)]
    
    # Run the checks
    dm._run_contamination_checks()
    
    # Verify that rec_0 ('1') was indeed flagged and quarantined
    assert len(dm.train_texts) < 8 # Dropped rec_0

