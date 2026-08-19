"""Narrow compatibility patches required by specific model-library combinations."""

from __future__ import annotations

import logging


logger = logging.getLogger("lm_adapt_bench")
_PATCHED = False


def apply_compatibility_patches() -> None:
    """Apply idempotent Transformers/PEFT compatibility patches before model loading."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import transformers.integrations.moe

        module = transformers.integrations.moe
        original = module._grouped_mm_fallback
        if not getattr(original, "_lm_adapt_bench_patched", False):

            def patched_grouped_mm_fallback(input_tensor, weight, *args, **kwargs):
                if input_tensor.dtype != weight.dtype:
                    input_tensor = input_tensor.to(weight.dtype)
                return original(input_tensor, weight, *args, **kwargs)

            patched_grouped_mm_fallback._lm_adapt_bench_patched = True
            module._grouped_mm_fallback = patched_grouped_mm_fallback
            logger.info("Applied Transformers MoE dtype compatibility patch.")
    except Exception as exc:
        logger.debug("Transformers MoE compatibility patch not needed: %s", exc)

    try:
        import transformers.core_model_loading as core_loading

        converter = core_loading.WeightConverter
        if not hasattr(converter, "_lm_adapt_bench_patched"):
            original_init = converter.__init__

            def patched_init(self, *args, **kwargs):
                kwargs.pop("distributed_operation", None)
                kwargs.pop("quantization_operation", None)
                return original_init(self, *args, **kwargs)

            converter.__init__ = patched_init
            if not hasattr(converter, "distributed_operation"):
                converter.distributed_operation = None
            if not hasattr(converter, "quantization_operation"):
                converter.quantization_operation = None
            converter._lm_adapt_bench_patched = True
            logger.info("Applied PEFT/Transformers WeightConverter compatibility patch.")
    except Exception as exc:
        logger.debug("PEFT/Transformers compatibility patch not needed: %s", exc)

    _PATCHED = True
