"""Shared SFT training helpers: freeze policy and 8-bit optimizer."""

from __future__ import annotations

import torch
import torch.optim as optim


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def freeze_for_sft(
    model,
    *,
    freeze_vision: bool = True,
    unfreeze_lm_blocks: int = 0,
) -> tuple[int, int]:
    """
    Configure trainable parameters.

    - MP: always trainable
    - ViT: frozen when ``freeze_vision`` (default)
    - LM: all blocks trainable if ``unfreeze_lm_blocks <= 0``,
      otherwise only the last N transformer blocks (+ norm + head)
    """
    set_requires_grad(model, False)
    set_requires_grad(model.MP, True)

    if not freeze_vision:
        set_requires_grad(model.vision_encoder, True)

    dec = model.decoder
    if unfreeze_lm_blocks <= 0:
        set_requires_grad(dec, True)
    else:
        if hasattr(dec, "blocks"):
            n = len(dec.blocks)
            for i in range(max(0, n - unfreeze_lm_blocks), n):
                set_requires_grad(dec.blocks[i], True)
        if hasattr(dec, "norm"):
            set_requires_grad(dec.norm, True)
        if hasattr(dec, "head"):
            set_requires_grad(dec.head, True)
        if hasattr(dec, "token_embedding"):
            set_requires_grad(dec.token_embedding, True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def sft_param_groups(model, lr_mp: float, lr_backbones: float) -> list[dict]:
    mp_params = [p for p in model.MP.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    groups: list[dict] = []
    if mp_params:
        groups.append({"params": mp_params, "lr": lr_mp, "name": "mp"})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": lr_backbones, "name": "decoder"})
    return groups


def build_sft_optimizer(
    param_groups: list[dict],
    *,
    weight_decay: float,
    use_8bit: bool,
) -> optim.Optimizer:
    if not param_groups:
        raise ValueError("No trainable parameters — check freeze_for_sft settings")
    if use_8bit:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(param_groups, weight_decay=weight_decay)
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
