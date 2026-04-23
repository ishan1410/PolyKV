"""
Architecture-agnostic utilities for layer device detection and layer counting.
Supports: Llama, Mistral, Qwen2, Gemma, Phi-3, Falcon, GPT-2, GPT-Neo,
          GPT-J, GPT-NeoX, OPT, BLOOM, Phi-2, and generic fallback.
"""
import torch


def get_layer_device(model, layer_idx: int) -> torch.device:
    """
    Return the device that transformer layer `layer_idx` lives on.
    Tries known architecture patterns in order, falls back to first parameter.
    """
    patterns = [
        # Llama, Mistral, Qwen2, Gemma, Phi-3
        lambda m, i: m.model.layers[i].self_attn.q_proj.weight.device,
        # Mistral variant
        lambda m, i: m.model.layers[i].attention.q_proj.weight.device,
        # OPT
        lambda m, i: m.model.decoder.layers[i].self_attn.q_proj.weight.device,
        # GPT-2, GPT-Neo
        lambda m, i: m.transformer.h[i].attn.c_attn.weight.device,
        # GPT-J
        lambda m, i: m.transformer.h[i].attn.q_proj.weight.device,
        # GPT-NeoX
        lambda m, i: m.gpt_neox.layers[i].attention.query_key_value.weight.device,
        # BLOOM, Falcon
        lambda m, i: m.transformer.h[i].self_attention.query_key_value.weight.device,
        # Phi-2
        lambda m, i: m.model.layers[i].mixer.Wqkv.weight.device,
    ]
    for pattern in patterns:
        try:
            return pattern(model, layer_idx)
        except (AttributeError, IndexError, KeyError):
            continue
    # Universal fallback
    return next(model.parameters()).device


def get_num_layers(model) -> int:
    """
    Return the number of transformer layers in the model.
    """
    # config-based (most reliable)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)

    # structural fallback
    patterns = [
        lambda m: len(m.model.layers),           # Llama family
        lambda m: len(m.model.decoder.layers),   # OPT
        lambda m: len(m.transformer.h),          # GPT-2 / Falcon / BLOOM
        lambda m: len(m.gpt_neox.layers),        # GPT-NeoX
    ]
    for pattern in patterns:
        try:
            return pattern(model)
        except (AttributeError, TypeError):
            continue

    raise ValueError(
        "Could not determine number of layers. "
        "Use a supported architecture or pass num_layers explicitly."
    )


def get_first_device(model) -> torch.device:
    """Device of the first transformer layer (where input embeddings live)."""
    try:
        return get_layer_device(model, 0)
    except Exception:
        return next(model.parameters()).device
