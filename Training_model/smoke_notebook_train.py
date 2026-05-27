"""Notebook-style training: frozen backbone + log-loss (same as sft_pipeline section 5-6)."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "nanoVLM"), str(ROOT / "Datasets"), str(ROOT / "Training_model")):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_logloss import action_embedding_matrix, optimal_set_log_loss, positive_action_mask
from data.processors import get_image_processor, get_tokenizer
from minigrid_collator import MiniGridSFTCollator, batch_prompt_tensors, last_hidden_after_prompt
from minigrid_sft_dataset import MiniGridSFTDataset, load_objects, precompute_trajectories, split_objects
from models.vision_language_model import VisionLanguageModel


def set_requires_grad(module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def freeze_for_sft(model, *, vit_last_n: int, lm_last_n: int) -> None:
    set_requires_grad(model, False)
    set_requires_grad(model.MP, True)
    vit = model.vision_encoder
    if hasattr(vit, "blocks"):
        n = len(vit.blocks)
        for i in range(max(0, n - vit_last_n), n):
            set_requires_grad(vit.blocks[i], True)
    if hasattr(vit, "layer_norm"):
        set_requires_grad(vit.layer_norm, True)
    dec = model.decoder
    if hasattr(dec, "blocks"):
        n = len(dec.blocks)
        for i in range(max(0, n - lm_last_n), n):
            set_requires_grad(dec.blocks[i], True)
    if hasattr(dec, "norm"):
        set_requires_grad(dec.norm, True)
    if hasattr(dec, "head"):
        set_requires_grad(dec.head, True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    objects = load_objects(ROOT / "Datasets" / "dataset.json")[:6]
    train_objs, val_objs = split_objects(objects, 0.2, 0)
    train_cache = precompute_trajectories(train_objs)
    val_cache = precompute_trajectories(val_objs)

    model = VisionLanguageModel.from_pretrained(str(ROOT / "checkpoints" / "nanoVLM-222M"))
    freeze_for_sft(model, vit_last_n=2, lm_last_n=2)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train:,}")

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    train_ds = MiniGridSFTDataset(train_objs, tokenizer, image_processor, train_cache)
    collator = MiniGridSFTCollator(tokenizer, model.cfg.lm_max_length)
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collator, drop_last=True)
    model.to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )
    model.train()
    losses = []
    for step, batch in enumerate(loader):
        if step >= 15:
            break
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        prompt_ids, prompt_mask = batch_prompt_tensors(
            batch, tokenizer=tokenizer, max_length=model.cfg.lm_max_length, device=device
        )
        action_emb = action_embedding_matrix(model, tokenizer, device)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = last_hidden_after_prompt(model, prompt_ids, images, attention_mask=prompt_mask)
                pos = positive_action_mask(batch.get("allowed_actions"), h.size(0), device, dtype=h.dtype)
                emb_loss = optimal_set_log_loss(h, action_emb, pos)
                _, ce_loss = model(input_ids, images, attention_mask, targets=labels)
                loss = emb_loss + ce_loss
        else:
            h = last_hidden_after_prompt(model, prompt_ids, images, attention_mask=prompt_mask)
            pos = positive_action_mask(batch.get("allowed_actions"), h.size(0), device, dtype=h.dtype)
            emb_loss = optimal_set_log_loss(h, action_emb, pos)
            _, ce_loss = model(input_ids, images, attention_mask, targets=labels)
            loss = emb_loss + ce_loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"step {step+1} loss={loss.item():.4f}")

    print(f"mean loss first5={sum(losses[:5])/min(5,len(losses)):.4f} last={losses[-1]:.4f}")
    print("NOTEBOOK-STYLE TRAIN OK")


if __name__ == "__main__":
    main()
