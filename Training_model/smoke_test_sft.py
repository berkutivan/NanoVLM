"""
Smoke test for SFT pipeline (notebook-equivalent logic, minimal run).
Run from repo root: python Training_model/smoke_test_sft.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "Training_model"
DATASETS_DIR = ROOT / "Datasets"
NANOVLM_DIR = ROOT / "nanoVLM"

for p in (str(NANOVLM_DIR), str(DATASETS_DIR), str(TRAINING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.utils.data import DataLoader

from action_logloss import (
    action_embedding_matrix,
    optimal_set_log_loss,
    positive_action_mask,
)
from data.processors import get_image_processor, get_tokenizer
from minigrid_collator import (
    MiniGridSFTCollator,
    batch_prompt_tensors,
    last_hidden_after_prompt,
)
from minigrid_sft_dataset import (
    MiniGridSFTDataset,
    load_objects,
    precompute_trajectories,
    split_objects,
)
from models.vision_language_model import VisionLanguageModel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset_path = ROOT / "Datasets" / "dataset.json"
    if not dataset_path.exists():
        raise SystemExit(f"Missing {dataset_path}")

    objects = load_objects(dataset_path)[:4]
    train_objs, val_objs = split_objects(objects, val_ratio=0.25, seed=0)
    print(f"mazes train={len(train_objs)} val={len(val_objs)}")

    train_cache = precompute_trajectories(train_objs)
    val_cache = precompute_trajectories(val_objs)
    print(f"steps train={sum(len(v) for v in train_cache.values())}")

    ckpt = ROOT / "checkpoints" / "nanoVLM-222M"
    if not (ckpt / "model.safetensors").exists():
        print(f"Local weights missing at {ckpt}, trying HF hub lusxvr/nanoVLM-222M ...")
        ckpt_path = "lusxvr/nanoVLM-222M"
    else:
        ckpt_path = str(ckpt)

    model = VisionLanguageModel.from_pretrained(ckpt_path)
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    train_ds = MiniGridSFTDataset(train_objs, tokenizer, image_processor, train_cache)
    collator = MiniGridSFTCollator(tokenizer, model.cfg.lm_max_length)
    probe = collator([train_ds[0]])
    required = {"prompt_input_ids", "prompt_attention_mask", "prompt_texts", "allowed_actions"}
    missing = required - set(probe.keys())
    if missing:
        raise SystemExit(f"Collator missing keys: {missing}")
    print("collator keys OK:", sorted(probe.keys()))

    loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collator)
    batch = next(iter(loader))
    model.to(device)
    model.train()

    images = batch["image"].to(device)
    prompt_ids, prompt_mask = batch_prompt_tensors(
        batch, tokenizer=tokenizer, max_length=model.cfg.lm_max_length, device=device
    )
    allowed = batch.get("allowed_actions")

    action_emb = action_embedding_matrix(model, tokenizer, device)
    h = last_hidden_after_prompt(model, prompt_ids, images, attention_mask=prompt_mask)
    pos = positive_action_mask(allowed, h.size(0), device, dtype=h.dtype)
    loss = optimal_set_log_loss(h, action_emb, pos)
    loss.backward()
    print(f"loss={loss.item():.4f} | h.shape={tuple(h.shape)} | grad_ok=True")

    # One optimizer step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    opt.zero_grad()
    h = last_hidden_after_prompt(model, prompt_ids, images, attention_mask=prompt_mask)
    loss = optimal_set_log_loss(h, action_emb.detach(), pos)
    loss.backward()
    opt.step()
    print("optimizer step OK")

    # Greedy decode sanity
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            [train_ds[0]["text_data"]],
            padding=True,
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=model.cfg.lm_max_length,
        )
        gen = model.generate(
            enc["input_ids"].to(device),
            images[:1],
            enc["attention_mask"].to(device),
            max_new_tokens=8,
        )
        pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    print("generate sample pred:", repr(pred[:80]))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
