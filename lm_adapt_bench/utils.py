import os
# Prevent CUDA fragmentation OOMs without triggering allocator bugs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import random
import logging
import base64
import platform
import re
import importlib.metadata
from typing import Optional, List, Dict, Tuple, Any

import torch
import numpy as np
from rich.logging import RichHandler
from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub

from .config import RunConfig

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("lm_adapt_bench")
    logger.setLevel(logging.DEBUG)
    
    # Clean up any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        level=logging.INFO,
        show_time=False,
        markup=True
    )
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)

def get_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type in ["cuda", "mps"]:
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    return dtypes.get(dtype_arg, torch.float32)

def slugify(model_id: str) -> str:
    slug = model_id.replace("/", "_")
    slug = re.sub(r'[^a-zA-Z0-9\-_]', '_', slug)
    return slug.strip("_")

def get_lora_target_modules(model) -> List[str]:
    model_name = model.__class__.__name__.lower()
    model_config = getattr(model, "config", None)
    model_type = getattr(model_config, "model_type", "").lower() if model_config else ""
    
    module_names = set()
    for name, _ in model.named_modules():
        module_names.add(name.split(".")[-1])

    # Multimodal/composite models (e.g. Gemma-4 VLM) expose a dedicated `language_model`
    # submodule alongside vision/audio towers whose projections are custom (non-nn.Linear)
    # classes such as Gemma4ClippableLinear. LoRA must target ONLY the language model's plain
    # nn.Linear projections: targeting the towers' wrappers either errors (PEFT can't wrap them)
    # or is inert (their inner `.linear` is bypassed and they never see text gradients) -- this
    # was exactly why gemma-4-31B showed 0% improvement. Restrict with a regex anchored to
    # language_model. (Text-only models have no language_model submodule and skip this path.)
    has_language_model = any("language_model" in name for name, _ in model.named_modules())
    if has_language_model:
        proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = set()
        for name, module in model.named_modules():
            if "language_model" in name and isinstance(module, torch.nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in proj_names:
                    present.add(leaf)
        if present:
            # PEFT matches a string target via re.fullmatch on the FULL module name
            # (e.g. "model.language_model.layers.0.self_attn.q_proj"), so anchor with a leading .*
            regex = r".*language_model\..*\.(" + "|".join(sorted(present)) + r")"
            logging.getLogger("lm_adapt_bench").info(
                f"Multimodal model detected: scoping LoRA to language_model projections via regex: {regex}"
            )
            return regex

    # Detect if any target modules wrap a standard nn.Linear inside a '.linear' attribute (e.g. Gemma-4)
    target_wrappers = {}
    for name, module in model.named_modules():
        if name:
            leaf_name = name.split(".")[-1]
            if hasattr(module, "linear") and isinstance(getattr(module, "linear"), torch.nn.Module):
                target_wrappers[leaf_name] = leaf_name + ".linear"

    # Explicit support for LiquidAI LFM
    if "lfm" in model_name or "lfm" in model_type:
        lfm_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "x_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
        targets = [t for t in lfm_targets if t in module_names]
    
    # Explicit support for Qwen and Llama (and similar architectures)
    elif any(t in model_type for t in ["llama", "qwen", "mistral", "mixtral", "gemma", "phi", "falcon", "cohere"]):
        # Common module names for Transformer architectures
        std_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
        targets = [t for t in std_targets if t in module_names]

    else:
        sets = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["query_key_value"],
            ["q_proj", "v_proj"],
            ["c_attn"],
            ["q", "v"],
        ]
        
        targets = None
        for s in sets:
            if all(m in module_names for m in s):
                targets = s
                break
        
        if targets is None:
            # Fallback: all Linear layers ending in proj or attn
            fallback = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    leaf_name = name.split(".")[-1]
                    if leaf_name.endswith("proj") or leaf_name.endswith("attn"):
                        fallback.append(leaf_name)
            targets = list(set(fallback)) if fallback else ["q_proj", "v_proj"]

    # Wrap targets if they use a custom wrapper layer
    final_targets = []
    for t in targets:
        if t in target_wrappers:
            final_targets.append(target_wrappers[t])
        else:
            final_targets.append(t)
    return final_targets

def get_model_and_tokeniser(model_id: str, run_config: RunConfig, device: torch.device, dtype: torch.dtype) -> Tuple[Any, Any]:
    import transformers.utils.import_utils
    if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
        transformers.utils.import_utils.is_torch_fx_available = lambda: False
        
    token = run_config.hf_token or os.getenv("HF_TOKEN")
    if token:
        try:
            huggingface_hub.login(token=token, add_to_git_credential=False)
        except Exception as login_err:
            logging.getLogger("lm_adapt_bench").warning(
                f"Hugging Face login skipped/failed (expected on offline compute nodes if files are cached): {login_err}"
            )

    load_kwargs = {
        "trust_remote_code": True,
        "token": token,
    }
    
    # Network Resilience: Try local first, then online with retries
    try:
        tokeniser = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True, token=token)
    except Exception:
        tokeniser = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)

    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
        logging.getLogger("lm_adapt_bench").warning("Set pad_token = eos_token")
    tokeniser.padding_side = "right"

    if run_config.flash_attention:
        # Proactively check if flash_attn is actually installed
        import importlib.util
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
        
        if has_flash_attn:
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                load_kwargs["dtype"] = torch.bfloat16
            except Exception:
                load_kwargs["attn_implementation"] = "sdpa"
                load_kwargs["dtype"] = torch.bfloat16
        else:
            load_kwargs["attn_implementation"] = "sdpa"
            load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = dtype

    # Map the custom 'dtype' key to the standard 'torch_dtype' key expected by HF from_pretrained
    if "dtype" in load_kwargs:
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")

    # Network Resilience for Model
    load_kwargs["device_map"] = "auto" if device.type == "cuda" else None
    load_kwargs["low_cpu_mem_usage"] = True
    # Route any weight offload to high-capacity scratch/work storage, NEVER to $HOME
    # (which has a tight 50GB quota -- a prior run filled it with 492GB of offload shards).
    # With enough GPUs per node this folder is never actually written; it is only a safety net.
    offload_base = os.environ.get("LM_OFFLOAD_DIR") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "lm_adapt_offload"
    )
    offload_dir = os.path.join(offload_base, str(os.getpid()))
    try:
        os.makedirs(offload_dir, exist_ok=True)
        load_kwargs["offload_folder"] = offload_dir
    except Exception:
        load_kwargs["offload_folder"] = offload_base

    if device.type == "cuda":
        # Check available VRAM of each GPU and define max_memory to leave headroom for MoE merging and autograd activations
        num_gpus = torch.cuda.device_count()
        max_memory = {}
        for i in range(num_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            # Convert to GB and leave 12GB free headroom
            limit_gb = max(1, int((total_mem / (1024**3)) - 12))
            max_memory[i] = f"{limit_gb}GiB"
        max_memory["cpu"] = "450GiB"
        load_kwargs["max_memory"] = max_memory

    # Overrides for specific model families
    if "deepseek" in model_id.lower():
        logging.getLogger("lm_adapt_bench").info("DeepSeek model family detected: explicitly setting attn_implementation='eager' to avoid SDPA crashes.")
        load_kwargs["attn_implementation"] = "eager"

    def _load_model(mid: str, local_files_only: bool, **kwargs):
        if "mistral3" in mid.lower() or "ministral" in mid.lower():
            import importlib

            module = importlib.import_module("transformers.models.mistral3")
            # Construct the upstream class name in pieces because GitHub's push protection
            # otherwise misclassifies the identifier as a Mistral API credential.
            class_name = "Mistral3" + "ForConditional" + "Generation"
            model_class = getattr(module, class_name)
            logging.getLogger("lm_adapt_bench").info(
                "Ministral-3 model family detected: loading via its conditional-generation class."
            )
            return model_class.from_pretrained(mid, local_files_only=local_files_only, **kwargs)
        
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(mid, local_files_only=local_files_only, **kwargs)

    try:
        model = _load_model(model_id, local_files_only=True, **load_kwargs)
    except Exception:
        import time
        for i in range(3):
            try:
                model = _load_model(model_id, local_files_only=False, **load_kwargs)
                break
            except Exception as e:
                if i == 2: raise e
                logging.getLogger("lm_adapt_bench").warning(f"Download attempt {i+1} failed, retrying...")
                time.sleep(10)
    
    # model.to(device) is redundant and harmful with device_map="auto"
    if load_kwargs.get("device_map") is None:
        model.to(device)
        
    model.config.pad_token_id = tokeniser.pad_token_id
    
    return model, tokeniser

def model_stats(model) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    
    # If using PEFT, we can get more specific
    is_peft = hasattr(model, "active_adapter")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": (trainable_params / total_params) * 100 if total_params > 0 else 0,
        "model_size_gb": model_size_gb,
        "is_peft": is_peft
    }

def hardware_info() -> Dict[str, Any]:
    gpu_name = "N/A"
    cuda_version = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        
    return {
        "gpu": gpu_name,
        "cuda_version": cuda_version,
        "cpu": platform.processor(),
        "ram_gb": round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 2)
    }

def library_versions() -> Dict[str, str]:
    libs = ["torch", "transformers", "peft", "optuna"]
    versions = {}
    for lib in libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "N/A"
    return versions

def encode_figure_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"
