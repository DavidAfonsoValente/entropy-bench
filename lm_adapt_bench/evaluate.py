import math
import logging
import torch
from torch.utils.data import DataLoader

def compute_bpb(model, val_dataset, avg_bytes_per_token: float, device: torch.device, batch_size: int = 4) -> float:
    model.eval()

    # Adaptive batching: if we OOM, we try again with batch_size 1
    try:
        return _run_evaluation(model, val_dataset, avg_bytes_per_token, device, batch_size)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logging.getLogger("lm_adapt_bench").warning("OOM during eval. Retrying with batch_size=1")
        return _run_evaluation(model, val_dataset, avg_bytes_per_token, device, 1)

def _run_evaluation(model, val_dataset, avg_bytes_per_token, device, batch_size):
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    pad_token_id = getattr(model.config, "pad_token_id", -100)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # Determine correct autocast dtype to prevent float16 overflows
    autocast_dtype = torch.float16
    if device_type == "cuda":
        model_dtype = getattr(model, "dtype", None)
        if model_dtype == torch.bfloat16:
            autocast_dtype = torch.bfloat16
        elif torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            # Standard Causal LM evaluation: shift labels by 1
            labels = batch["labels"].to(device)

            if device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            # Mask out padding from the count
            shift_labels = labels[..., 1:].contiguous()
            n_predicted = (shift_labels != -100).sum().item()

            total_loss += loss.item() * n_predicted
            total_tokens += n_predicted

    if total_tokens == 0: return 0.0
    return float((total_loss / total_tokens / math.log(2)) / avg_bytes_per_token)

def compute_zero_shot_bpb(model, val_dataset, avg_bytes_per_token: float, device: torch.device, batch_size: int = 16) -> float:
    return compute_bpb(model, val_dataset, avg_bytes_per_token, device, batch_size)
