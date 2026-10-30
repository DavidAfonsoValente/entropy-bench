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


def doc_windows(token_docs, context: int, stride: int):
    """Split tokenised documents into scoring windows that never cross a document boundary.

    Each document is scored with its own start-of-document context and nothing before it. A
    document longer than ``context`` is covered by a sliding window: every window after the first
    starts ``stride`` tokens later, and scores only the tokens no earlier window scored, so each
    token is predicted once, from at least ``context - stride`` tokens of its own document.

    ``token_docs`` is a list of ``(ids, n_prefix)`` pairs; the first ``n_prefix`` ids (an injected
    BOS) are context only. Returns ``(ids, score_from)`` pairs: positions ``>= score_from`` are
    scored as next-token targets.
    """
    assert 0 < stride <= context, (stride, context)
    out = []
    for ids, n_prefix in token_docs:
        first = max(n_prefix, 1)  # position 0 has nothing to predict it
        if len(ids) <= first:
            continue
        start, scored_to = 0, first
        while True:
            end = min(start + context, len(ids))
            out.append((ids[start:end], scored_to - start))
            if end == len(ids):
                break
            scored_to = end
            start = end - (context - stride)
    return out


def compute_doc_bpb(model, tokeniser, texts, device, context: int = 2048, stride: int = 1024,
                    token_budget: int = 16384) -> dict:
    """BPB over whole documents, each in its own windows, with the tokenizer's own document start.

    Unlike ``compute_bpb`` (fixed blocks cut from a packed stream, most starting mid-document),
    every document here begins with the BOS its tokenizer injects (none for Qwen) and is scored in
    full, so the denominator is the UTF-8 byte count of ``texts`` and is identical for every model.
    Returns ``{"bpb", "nats", "bytes", "tokens", "windows"}``.
    """
    model.eval()
    probe = "hello world"
    n_prefix = max(0, len(tokeniser(probe, add_special_tokens=True)["input_ids"])
                   - len(tokeniser(probe, add_special_tokens=False)["input_ids"]))
    docs = [(tokeniser(t, add_special_tokens=True)["input_ids"], n_prefix) for t in texts]
    windows = sorted(doc_windows(docs, context, stride), key=lambda w: len(w[0]))
    pad = tokeniser.pad_token_id if tokeniser.pad_token_id is not None else 0
    nats, tokens, i = 0.0, 0, 0
    with torch.no_grad():
        while i < len(windows):
            width = len(windows[i][0])
            j = i
            while j < len(windows) and (j - i + 1) * len(windows[j][0]) <= max(token_budget, width):
                j += 1
            batch = windows[i:j]
            width = len(batch[-1][0])
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for r, (w, score_from) in enumerate(batch):
                ids[r, :len(w)] = torch.tensor(w)
                mask[r, :len(w)] = 1
                labels[r, score_from:len(w)] = ids[r, score_from:len(w)]
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            target = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=-100, reduction="sum")
            nats += loss.item()
            tokens += int((target != -100).sum().item())
            i = j
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return {"bpb": nats / math.log(2) / n_bytes, "nats": nats, "bytes": n_bytes,
            "tokens": tokens, "windows": len(windows)}
