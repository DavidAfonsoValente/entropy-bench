import os
import json
import csv
import random
import hashlib
import logging
import tempfile
import shutil
import zipfile
from datetime import datetime
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, Dataset as HFDataset
from .config import DataConfig
from .contamination.text_normalize import normalize_text
from .contamination.fingerprints import Fingerprinter
from .contamination.deduper import Deduper
from .contamination.forbidden_overlap import ForbiddenChecker
from .contamination.split_leakage import SplitLeakageChecker
from .contamination.cleaner import DatasetCleaner
from .contamination.risk_aggregator import RiskAggregator
from .contamination.audit import AuditWriter
from .contamination.schema import ContaminationFinding, DatasetContaminationSummary, ContaminationAudit

class BlockDataset(Dataset):
    def __init__(self, blocks: List[Dict[str, torch.Tensor]]):
        self.blocks = blocks
    def __len__(self): return len(self.blocks)
    def __getitem__(self, i): return self.blocks[i]

class DataModule:
    def __init__(self, data_config: DataConfig):
        self.config = data_config
        self.logger = logging.getLogger("lm_adapt_bench")
        self.raw_texts: List[str] = []
        self.raw_ids: List[str] = []
        self.train_texts: List[str] = []
        self.train_ids: List[str] = []
        self.val_texts: List[str] = []
        self.val_ids: List[str] = []
        self.test_texts: List[str] = []
        self.test_ids: List[str] = []
        self.contamination_audit = None
        
        self._load_raw_texts()
        self._assign_ids()
        
        if self.config.contamination.check_contamination:
            self._run_contamination_checks()
        else:
            self._split_data()

    def _load_raw_texts(self) -> None:
        if self.config.hf_dataset_name:
            self.logger.info(f"Loading HF dataset: {self.config.hf_dataset_name}")
            ds = load_dataset(self.config.hf_dataset_name, self.config.hf_dataset_config, split=self.config.hf_dataset_split)
            self.raw_texts = [x[self.config.text_field] for x in ds if self.config.text_field in x]
        else:
            path = self.config.dataset_path
            if not os.path.exists(path): raise FileNotFoundError(f"Dataset path not found: {path}")

            if path.endswith(".zip"):
                self.logger.info(f"Extracting zip dataset: {path}")
                extract_dir = tempfile.mkdtemp(prefix="lm_bench_")
                try:
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    self.logger.info("--- Zip Contents Discovery ---")
                    found_any_file = False
                    
                    # 1. Check if it's a HuggingFace 'save_to_disk' directory
                    for root, dirs, files in os.walk(extract_dir):
                        if "dataset_info.json" in files and any(f.endswith(".arrow") for f in files):
                            self.logger.info(f"Detected HuggingFace Arrow dataset in: {root}")
                            try:
                                hf_ds = load_from_disk(root)
                                if self.config.text_field in hf_ds.column_names:
                                    self.raw_texts.extend(hf_ds[self.config.text_field])
                                    found_any_file = True
                                    continue # Move to next directory if any
                            except Exception as e:
                                self.logger.warning(f"Failed to load Arrow dataset via load_from_disk: {e}")

                    # 2. If not already loaded, search for individual files
                    if not found_any_file:
                        exclude_names = {"dataset_info.json", "metadata.json", "state.json", "license", "readme"}
                        for root, dirs, files in os.walk(extract_dir):
                            for file in files:
                                file_l = file.lower()
                                if file_l.endswith((".txt", ".jsonl", ".csv", ".json", ".parquet", ".arrow")):
                                    if file_l in exclude_names: continue
                                    real_path = os.path.join(root, file)
                                    self.logger.info(f"Reading file: {file}")
                                    try:
                                        self._read_file(real_path)
                                        found_any_file = True
                                    except Exception as e:
                                        self.logger.warning(f"Could not read {file}: {e}")
                    
                    if not found_any_file:
                        raise ValueError(f"No valid data files found in zip.")
                finally:
                    shutil.rmtree(extract_dir)
            else:
                self._read_file(path)

        if not self.raw_texts:
            raise ValueError(f"No text data found. Field: '{self.config.text_field}'")

        if self.config.max_samples:
            random.seed(self.config.seed)
            random.shuffle(self.raw_texts)
            self.raw_texts = self.raw_texts[:self.config.max_samples]

    def _assign_ids(self) -> None:
        """Assigns stable IDs to raw texts based on hash of normalized text."""
        self.raw_ids = []
        for i, text in enumerate(self.raw_texts):
            norm = normalize_text(text)
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
            self.raw_ids.append(f"rec_{i}_{h}")

    def _split_data(self) -> None:
        n_total = len(self.raw_texts)
        indices = list(range(n_total))
        random.seed(self.config.seed)
        random.shuffle(indices)
        
        n_test = int(n_total * self.config.test_split)
        n_val = int(n_total * self.config.val_split)
        
        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val:]
        
        self.test_texts = [self.raw_texts[i] for i in test_idx]
        self.test_ids = [self.raw_ids[i] for i in test_idx]
        
        self.val_texts = [self.raw_texts[i] for i in val_idx]
        self.val_ids = [self.raw_ids[i] for i in val_idx]
        
        self.train_texts = [self.raw_texts[i] for i in train_idx]
        self.train_ids = [self.raw_ids[i] for i in train_idx]

    def _run_contamination_checks(self) -> None:
        c_cfg = self.config.contamination
        
        # Determine parallel rank and world size to avoid race conditions
        import sys
        import time
        rank = 0
        world_size = 1
        if "--rank" in sys.argv:
            try:
                rank = int(sys.argv[sys.argv.index("--rank") + 1])
            except Exception:
                pass
        else:
            rank = int(os.environ.get("SLURM_PROCID", 0))

        if "--world-size" in sys.argv:
            try:
                world_size = int(sys.argv[sys.argv.index("--world-size") + 1])
            except Exception:
                pass
        else:
            world_size = int(os.environ.get("SLURM_NTASKS", 1))

        # Reuse-contamination fast path: if a previous run already produced the deterministic
        # splits, every rank (incl. rank 0) loads them and skips the multi-hour forensic audit.
        # Used by chained training jobs so each job doesn't pay the ~3h audit again.
        reuse = ("--reuse-contamination" in sys.argv) or (os.environ.get("LM_REUSE_CONTAMINATION") == "1")
        if reuse and c_cfg.output_dir:
            flag_path = os.path.join(c_cfg.output_dir, ".contamination_complete")
            splits_path = os.path.join(c_cfg.output_dir, "contamination_splits.pt")
            if os.path.exists(flag_path) and os.path.exists(splits_path):
                self.logger.info("Reuse-contamination enabled: loading cached splits, skipping forensic audit.")
                try:
                    splits_data = torch.load(splits_path, weights_only=False)
                    self.train_texts = splits_data["train_texts"]; self.train_ids = splits_data["train_ids"]
                    self.val_texts = splits_data["val_texts"]; self.val_ids = splits_data["val_ids"]
                    self.test_texts = splits_data["test_texts"]; self.test_ids = splits_data["test_ids"]
                    self.contamination_audit = splits_data.get("contamination_audit")
                    return
                except Exception as e:
                    self.logger.error(f"Reuse-contamination load failed ({e}); running full audit.")
            else:
                self.logger.warning("Reuse-contamination requested but no cached splits found; running full audit.")

        if world_size > 1 and rank != 0:
            if c_cfg.output_dir:
                flag_path = os.path.join(c_cfg.output_dir, ".contamination_complete")
                splits_path = os.path.join(c_cfg.output_dir, "contamination_splits.pt")
                self.logger.info(f"Rank {rank} waiting for Rank 0 to complete contamination checks...")
                while not os.path.exists(flag_path) or not os.path.exists(splits_path):
                    time.sleep(2)
                
                self.logger.info(f"Rank {rank} loading contamination splits from {splits_path}")
                try:
                    splits_data = torch.load(splits_path, weights_only=False)
                    self.train_texts = splits_data["train_texts"]
                    self.train_ids = splits_data["train_ids"]
                    self.val_texts = splits_data["val_texts"]
                    self.val_ids = splits_data["val_ids"]
                    self.test_texts = splits_data["test_texts"]
                    self.test_ids = splits_data["test_ids"]
                    self.contamination_audit = splits_data.get("contamination_audit")
                    self.logger.info(f"Rank {rank} successfully loaded splits from Rank 0.")
                    return
                except Exception as e:
                    self.logger.error(f"Rank {rank} failed to load contamination splits: {e}. Falling back to default split...")
            
            # Fallback if no output_dir or load failed
            self._split_data()
            return

        # Rest of the method is for Rank 0 or single-node run
        self.logger.info("Starting contamination detection subsystem...")
        fingerprinter = Fingerprinter(num_perm=c_cfg.num_perm, ngram_n=c_cfg.ngram_n)
        deduper = Deduper(fingerprinter)
        findings = []
        
        # 0. Import previously-saved quarantine log if enabled
        if c_cfg.use_global_quarantine and c_cfg.output_dir:
            q_file = os.path.join(c_cfg.output_dir, "quarantine.jsonl")
            if os.path.exists(q_file):
                self.logger.info(f"Importing previous quarantine log from {q_file}...")
                try:
                    with open(q_file, "r") as f:
                        for line in f:
                            if line.strip():
                                q_data = json.loads(line)
                                # Map the split from quarantine log (or use 'raw' as default)
                                split_map = q_data.get("split", "raw")
                                findings.append(ContaminationFinding(
                                    finding_id=f"imported_{q_data['record_id']}",
                                    record_id=q_data["record_id"],
                                    split=split_map,
                                    detector="imported_quarantine",
                                    severity="high",
                                    score=1.0,
                                    reason=f"Imported from previous quarantine log: {q_data.get('reason', 'N/A')}"
                                ))
                except Exception as e:
                    self.logger.error(f"Failed to import previous quarantine log: {e}")

        # 1. Exact deduplication on raw data
        exact_dupes = deduper.find_exact_duplicates(self.raw_texts, self.raw_ids)
        for canonical, duplicate in exact_dupes:
            findings.append(ContaminationFinding(
                finding_id=f"exact_{duplicate}", record_id=duplicate, split="raw",
                detector="exact_dedupe", severity="low", score=1.0,
                matched_record_id=canonical, reason="Exact duplicate in raw dataset"
            ))

        # 2. Near-duplicate detection (standard or higher)
        if c_cfg.check_level in ["standard", "strict", "forensic"]:
            near_dupes = deduper.find_near_duplicates(self.raw_texts, self.raw_ids, threshold=c_cfg.minhash_threshold)
            for id1, id2, score in near_dupes:
                findings.append(ContaminationFinding(
                    finding_id=f"near_{id2}", record_id=id2, split="raw",
                    detector="near_dedupe", severity="low", score=score,
                    matched_record_id=id1, reason=f"Near-duplicate (score={score:.2f}) in raw dataset"
                ))

        # 3. Forbidden corpus check
        if c_cfg.forbidden_corpus:
            checker = ForbiddenChecker(fingerprinter, threshold=c_cfg.minhash_threshold)
            # We need to load forbidden corpus. For simplicity, reusing a small loader logic here or 
            # assuming it's a list of texts if we had a helper.
            # For this implementation, let's assume forbidden_corpus is a path.
            forbidden_texts = self._load_external_texts(c_cfg.forbidden_corpus)
            checker.add_forbidden_corpus(forbidden_texts, source_name=os.path.basename(c_cfg.forbidden_corpus))
            
            for text, rid in zip(self.raw_texts, self.raw_ids):
                f_overlaps = checker.check_overlap(text, rid)
                for o in f_overlaps:
                    findings.append(ContaminationFinding(
                        finding_id=f"forbidden_{rid}_{o['matched_source']}",
                        record_id=rid, split="raw", detector=o['detector'],
                        severity=o['severity'], score=o['score'],
                        matched_source=o['matched_source'], reason=f"Overlap with forbidden corpus: {o['matched_source']}"
                    ))

        # Split data temporarily to check leakage
        self._split_data()
        
        # Update findings with split info if split was 'raw'
        for f in findings:
            if f.split == "raw":
                if f.record_id in self.train_ids: f.split = "train"
                elif f.record_id in self.val_ids: f.split = "val"
                elif f.record_id in self.test_ids: f.split = "test"

        # 4. Split Leakage Check
        leakage_checker = SplitLeakageChecker(deduper)
        splits = {"train": self.train_texts, "val": self.val_texts, "test": self.test_texts}
        split_ids = {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids}
        leak_findings = leakage_checker.check_leakage(splits, split_ids, threshold=c_cfg.minhash_threshold)
        for lf in leak_findings:
            findings.append(ContaminationFinding(
                finding_id=f"leak_{lf['record_id']}", **lf
            ))

        # 5. Risk Aggregation & Cleaning
        aggregator = RiskAggregator(c_cfg)
        risk_level, risk_score = aggregator.aggregate_dataset_risk(findings, len(self.raw_texts))
        
        cleaner = DatasetCleaner(c_cfg)
        cleaned_splits, cleaned_ids, quarantine = cleaner.apply_cleaning(splits, split_ids, findings)
        
        self.train_texts, self.train_ids = cleaned_splits["train"], cleaned_ids["train"]
        self.val_texts, self.val_ids = cleaned_splits["val"], cleaned_ids["val"]
        self.test_texts, self.test_ids = cleaned_splits["test"], cleaned_ids["test"]
        
        # Summary & Audit
        summary = DatasetContaminationSummary(
            original_rows=len(self.raw_texts),
            cleaned_rows=len(self.train_texts) + len(self.val_texts) + len(self.test_texts),
            duplicates_removed=len([f for f in findings if "dedupe" in f.detector]),
            quarantined_rows=len(quarantine),
            test_rows_dropped=len([q for q in quarantine if q["split"] == "test"]),
            split_leakage_count=len([f for f in findings if "leakage" in f.detector]),
            forbidden_overlap_count=len([f for f in findings if "forbidden" in f.detector]),
            risk_level=risk_level,
            risk_score=risk_score,
            findings_count=len(findings)
        )
        
        self.contamination_audit = ContaminationAudit(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            dataset_summary=summary,
            findings=findings
        )
        
        if c_cfg.output_dir:
            writer = AuditWriter(c_cfg.output_dir)
            writer.write_audit(self.contamination_audit)
            writer.write_quarantine(quarantine)
            
            # Export permanently cleaned text splits
            if c_cfg.export_cleaned_dataset:
                export_dir = os.path.join(c_cfg.output_dir, "cleaned_dataset")
                os.makedirs(export_dir, exist_ok=True)
                for split_name, texts in [("train", self.train_texts), ("val", self.val_texts), ("test", self.test_texts)]:
                    path = os.path.join(export_dir, f"{split_name}.jsonl")
                    with open(path, "w", encoding="utf-8") as f:
                        for text in texts:
                            f.write(json.dumps({self.config.text_field: text}) + "\n")
                self.logger.info(f"Permanently cleaned text splits exported to {export_dir}")

            # Save contamination splits file for other parallel ranks to load
            if world_size > 1:
                splits_path = os.path.join(c_cfg.output_dir, "contamination_splits.pt")
                splits_data = {
                    "train_texts": self.train_texts,
                    "train_ids": self.train_ids,
                    "val_texts": self.val_texts,
                    "val_ids": self.val_ids,
                    "test_texts": self.test_texts,
                    "test_ids": self.test_ids,
                    "contamination_audit": self.contamination_audit
                }
                # Remove completion flag first if exists to prevent early wake up
                flag_path = os.path.join(c_cfg.output_dir, ".contamination_complete")
                if os.path.exists(flag_path):
                    try:
                        os.remove(flag_path)
                    except Exception:
                        pass
                
                self.logger.info(f"Saving contamination splits to {splits_path}")
                torch.save(splits_data, splits_path)
                
                # Write completion flag
                with open(flag_path, "w") as f:
                    f.write("done\n")
                self.logger.info(f"Wrote completion flag to {flag_path}")
            
        self.logger.info(f"Contamination check complete. Risk Level: {risk_level}. Cleaned dataset size: {summary.cleaned_rows}")

    def _load_external_texts(self, path: str) -> List[str]:
        """Helper to load texts from an external path (forbidden corpus)."""
        # Create a temporary DataConfig and DataModule to reuse loading logic
        tmp_cfg = DataConfig(dataset_path=path, text_field=self.config.text_field)
        tmp_dm = DataModule(tmp_cfg)
        return tmp_dm.raw_texts

    def _read_file(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                self.raw_texts.extend([p.strip() for p in f.read().split("\n\n") if p.strip()])
        elif ext in [".jsonl", ".json"]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                try:
                    data = json.loads(content)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if isinstance(item, dict) and self.config.text_field in item:
                            self.raw_texts.append(str(item[self.config.text_field]))
                except json.JSONDecodeError:
                    f.seek(0)
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict) and self.config.text_field in data:
                                self.raw_texts.append(str(data[self.config.text_field]))
                        except: pass
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.config.text_field in row:
                        self.raw_texts.append(str(row[self.config.text_field]))
        elif ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            if self.config.text_field in df.columns:
                self.raw_texts.extend(df[self.config.text_field].astype(str).tolist())
        elif ext == ".arrow":
            import datasets
            ds = datasets.Dataset.from_file(path)
            if self.config.text_field in ds.column_names:
                self.raw_texts.extend(ds[self.config.text_field])

    def calculate_avg_bytes_per_token(self, tokeniser) -> float:
        """Calculates model-specific byte efficiency for BPB normalization."""
        texts = self.test_texts or self.val_texts or self.train_texts
        if not texts: return 4.0 # Fallback
        
        # Sample up to 500 texts for speed
        sample_texts = texts[:500]
        total_bytes = sum(len(str(t).encode("utf-8")) for t in sample_texts)
        
        # Count tokens accurately (ignoring padding/special tokens if needed, 
        # but here we want the raw model ratio)
        total_tokens = 0
        for t in sample_texts:
            tokens = tokeniser(str(t), add_special_tokens=False)["input_ids"]
            total_tokens += len(tokens)
            
        return total_bytes / max(total_tokens, 1)

    def get_byte_stats(self, tokeniser) -> float:
        # We prefer using the test set for calculating the average bytes per token
        # as it is the final evaluation set. If empty, fall back to val, then train.
        texts = self.test_texts or self.val_texts or self.train_texts
        if not texts: return 1.0
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        total_tokens = sum(len(tokeniser(t, add_special_tokens=True)["input_ids"]) for t in texts)
        return total_bytes / max(total_tokens, 1)

    def _cache_path(self, model_id: str, max_seq_len: int, output_dir: str) -> Path:
        content_hash = hashlib.sha256(self.config.dataset_path.encode()).hexdigest()[:16]
        # v3: label masking of injected special tokens changes the cached labels, so it has
        # to be part of the key -- otherwise a v2 cache silently reintroduces the old
        # behaviour. See docs/TOKEN_GAIN_FINDINGS.md.
        mask = getattr(self.config, "mask_injected_special_tokens", True)
        key = (f"{model_id}|{content_hash}|{self.config.text_field}|{max_seq_len}"
               f"|mask={int(bool(mask))}|v3")
        cache_dir = Path(output_dir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".pt")

    def tokenize(self, model_id: str, tokeniser, max_seq_len: int, output_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
        cache_path = self._cache_path(model_id, max_seq_len, output_dir)
        if cache_path.exists():
            data = torch.load(cache_path, weights_only=False)
            return BlockDataset(data["train"]), BlockDataset(data["val"]), BlockDataset(data["test"])
        # How many special tokens does this tokenizer prepend per document? Gemma injects
        # <bos>, Mistral <s>, Llama-3 <|begin_of_text|>, LFM2.5 <|startoftext|>; the Qwen
        # tokenizers inject nothing.
        n_injected = 0
        if getattr(self.config, "mask_injected_special_tokens", True):
            probe = "hello world"
            n_injected = max(0, len(tokeniser(probe, add_special_tokens=True)["input_ids"])
                             - len(tokeniser(probe, add_special_tokens=False)["input_ids"]))
            if n_injected:
                self.logger.info(
                    "Masking %d injected special token(s) per document out of the loss "
                    "(kept as context)", n_injected)

        def process(texts):
            all_ids, all_labels = [], []
            for t in texts:
                ids = tokeniser(str(t), add_special_tokens=True)["input_ids"]
                labels = list(ids)
                # Keep the marker as context but do not score the model on predicting it.
                for k in range(min(n_injected, len(labels))):
                    labels[k] = -100
                all_ids.extend(ids)
                all_labels.extend(labels)
            return [{"input_ids": torch.tensor(all_ids[i:i+max_seq_len]),
                     "labels": torch.tensor(all_labels[i:i+max_seq_len])}
                    for i in range(0, len(all_ids)-max_seq_len+1, max_seq_len)]
        train, val, test = process(self.train_texts), process(self.val_texts), process(self.test_texts)
        torch.save({"train": train, "val": val, "test": test}, cache_path)
        return BlockDataset(train), BlockDataset(val), BlockDataset(test)
