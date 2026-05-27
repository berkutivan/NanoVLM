"""
SFT fine-tuning of nanoVLM-222M on MiniGrid expert trajectories.

Fine-tuning strategy
--------------------
1. Load the *full* pretrained VLM from checkpoints/nanoVLM-222M (vision + LM + MP).
2. Do NOT re-initialize from backbone weights only — that would be continued pre-training.
3. Optimize all parameters with two learning rates (nanoVLM convention):
   - modality projector (MP): higher LR — adapts vision tokens to the new task quickly;
   - vision encoder + language decoder: lower LR — preserves general VLM features while adapting.
4. Loss: optimal-set embedding log-loss + CE on answer tokens (trains ``lm_head``).
   Validation reports embedding accuracy (aligned with log-loss) and constrained greedy gen accuracy.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
NANOVLM_DIR = ROOT / "nanoVLM"
DATASETS_DIR = ROOT / "Datasets"

for p in (str(NANOVLM_DIR), str(DATASETS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_logloss import (  # noqa: E402
    ACTION_ORDER,
    action_embedding_matrix,
    action_first_token_ids,
    count_allowed_hits,
    optimal_set_log_loss,
    positive_action_mask,
    predict_actions_from_hidden,
    actions_from_first_token_ids,
)
from data.processors import get_image_processor, get_tokenizer  # noqa: E402
from minigrid_collator import (  # noqa: E402
    MiniGridSFTCollator,
    batch_prompt_tensors,
    last_hidden_after_prompt,
)
from models.vision_language_model import VisionLanguageModel  # noqa: E402

from minigrid_sft_dataset import (  # noqa: E402
    MiniGridSFTDataset,
    load_objects,
    precompute_trajectories,
    split_objects,
)
from sft_config import SFTConfig  # noqa: E402

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(msg: str) -> None:
    print(msg, flush=True)


def get_lr(step: int, max_lr: float, max_steps: int) -> float:
    min_lr = max_lr * 0.1
    warmup = max(1, int(max_steps * 0.03))
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    decay = (step - warmup) / max(max_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (max_lr - min_lr)


def first_grid_action(text: str) -> str:
    """First whitespace-delimited token if it is a valid MiniGrid action (no fuzzy substring match)."""
    t = (text or "").strip().lower().split()
    if not t:
        return ""
    w = t[0].rstrip(".,;:!?")
    return w if w in ACTION_ORDER else ""


@torch.no_grad()
def eval_metrics(
    model: VisionLanguageModel,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 8,
) -> tuple[float, float, float]:
    """
    Returns (val_logloss, val_emb_acc, val_gen_acc).

    val_emb_acc: argmax over action embeddings (aligned with training loss).
    val_gen_acc: greedy decode, first token restricted to action prefixes.
    """
    model.eval()
    total_loss = 0.0
    emb_hits = 0
    gen_hits = 0
    n_samples = 0
    n_batches = 0

    action_emb = action_embedding_matrix(model, tokenizer, device)
    first_action_toks = action_first_token_ids(tokenizer, device)

    for batch in loader:
        images = batch["image"].to(device)
        prompt_ids, prompt_mask = batch_prompt_tensors(
            batch,
            tokenizer=tokenizer,
            max_length=model.cfg.lm_max_length,
            device=device,
        )
        allowed = batch.get("allowed_actions")

        h = last_hidden_after_prompt(model, prompt_ids, images, attention_mask=prompt_mask)
        pos = positive_action_mask(allowed, h.size(0), device, dtype=h.dtype)
        total_loss += optimal_set_log_loss(h, action_emb, pos).item()
        n_batches += 1

        emb_preds = predict_actions_from_hidden(h, action_emb)
        emb_hits += count_allowed_hits(emb_preds, allowed)

        gen = model.generate(
            prompt_ids,
            images,
            prompt_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            restrict_first_token_to=first_action_toks,
        )
        gen_preds = actions_from_first_token_ids(gen[:, 0], first_action_toks)
        gen_hits += count_allowed_hits(gen_preds, allowed)
        n_samples += h.size(0)

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    emb_acc = emb_hits / n_samples if n_samples else 0.0
    gen_acc = gen_hits / n_samples if n_samples else 0.0
    return avg_loss, emb_acc, gen_acc


def train_sft(cfg: SFTConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    log("=" * 60)
    log("nanoVLM SFT — MiniGrid expert fine-tuning")
    log("=" * 60)
    log(f"Device: {device} | AMP: {use_amp}")

    objects = load_objects(Path(cfg.dataset_path))
    if cfg.max_objects is not None:
        objects = objects[: cfg.max_objects]
    log(f"Mazes in dataset: {len(objects)}")

    train_objs, val_objs = split_objects(objects, cfg.val_ratio, cfg.seed)
    log(f"Train mazes: {len(train_objs)} | Val mazes: {len(val_objs)}")

    log("Precomputing expert trajectories (BFS on x,y,dir)...")
    t0 = time.time()
    train_cache = precompute_trajectories(train_objs)
    val_cache = precompute_trajectories(val_objs)
    train_steps = sum(len(v) for v in train_cache.values())
    val_steps = sum(len(v) for v in val_cache.values())
    log(
        f"Steps: train={train_steps}, val={val_steps} "
        f"(built in {time.time() - t0:.1f}s)"
    )

    log(f"Loading pretrained VLM from: {cfg.pretrained_path}")
    model = VisionLanguageModel.from_pretrained(cfg.pretrained_path)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Parameters: {n_params:,} total, {trainable:,} trainable (fine-tune all)")

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    train_ds = MiniGridSFTDataset(train_objs, tokenizer, image_processor, train_cache)
    val_ds = MiniGridSFTDataset(val_objs, tokenizer, image_processor, val_cache)

    collator = MiniGridSFTCollator(tokenizer, model.cfg.lm_max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= cfg.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    param_groups = [
        {"params": model.MP.parameters(), "lr": cfg.lr_mp, "name": "mp"},
        {
            "params": list(model.decoder.parameters()) + list(model.vision_encoder.parameters()),
            "lr": cfg.lr_backbones,
            "name": "backbones",
        },
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    model.to(device)
    if cfg.compile_model and device.type == "cuda":
        model = torch.compile(model)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = max(
        1,
        (len(train_loader) // cfg.grad_accum_steps) * cfg.epochs,
    )
    global_step = 0
    best_val_acc = -1.0

    log(
        f"Training: {cfg.epochs} epochs, batch_size={cfg.batch_size}, "
        f"lr_mp={cfg.lr_mp}, lr_backbones={cfg.lr_backbones}"
    )
    log(
        "Loss: optimal-set log-loss + CE on answer tokens "
        f"(ce_weight={cfg.ce_loss_weight})"
    )
    log("-" * 60)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_ids, prompt_mask = batch_prompt_tensors(
                batch,
                tokenizer=tokenizer,
                max_length=model.cfg.lm_max_length,
                device=device,
            )
            allowed = batch.get("allowed_actions")

            action_emb = action_embedding_matrix(model, tokenizer, device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    h = last_hidden_after_prompt(
                        model, prompt_ids, images, attention_mask=prompt_mask
                    )
                    pos = positive_action_mask(allowed, h.size(0), device, dtype=h.dtype)
                    emb_loss = optimal_set_log_loss(h, action_emb, pos)
                    _, ce_loss = model(input_ids, images, attention_mask, targets=labels)
                    loss = emb_loss + cfg.ce_loss_weight * ce_loss
                loss = loss / cfg.grad_accum_steps
                loss.backward()
            else:
                h = last_hidden_after_prompt(
                    model, prompt_ids, images, attention_mask=prompt_mask
                )
                pos = positive_action_mask(allowed, h.size(0), device, dtype=h.dtype)
                emb_loss = optimal_set_log_loss(h, action_emb, pos)
                _, ce_loss = model(input_ids, images, attention_mask, targets=labels)
                loss = emb_loss + cfg.ce_loss_weight * ce_loss
                loss = loss / cfg.grad_accum_steps
                loss.backward()

            do_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.param_groups[0]["lr"] = get_lr(global_step, cfg.lr_mp, max_steps)
                optimizer.param_groups[1]["lr"] = get_lr(
                    global_step, cfg.lr_backbones, max_steps
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            loss_val = loss.item() * cfg.grad_accum_steps
            epoch_loss += loss_val
            epoch_batches += 1

            if global_step > 0 and global_step % cfg.log_every == 0:
                lr_mp = optimizer.param_groups[0]["lr"]
                lr_bb = optimizer.param_groups[1]["lr"]
                log(
                    f"[epoch {epoch + 1}/{cfg.epochs}] step {global_step} | "
                    f"train_logloss={loss_val:.4f} | lr_mp={lr_mp:.2e} lr_bb={lr_bb:.2e}"
                )

            if global_step > 0 and global_step % cfg.eval_every == 0:
                val_loss, val_emb_acc, val_gen_acc = eval_metrics(
                    model, tokenizer, val_loader, device
                )
                log(
                    f"  >> val_logloss={val_loss:.4f} | val_emb_acc={val_emb_acc * 100:.2f}% "
                    f"| val_gen_acc={val_gen_acc * 100:.2f}%"
                )
                if val_gen_acc > best_val_acc:
                    best_val_acc = val_gen_acc
                    save_path = out_dir / "best"
                    model.save_pretrained(str(save_path))
                    log(f"  >> saved best checkpoint to {save_path}")

        avg_train = epoch_loss / max(epoch_batches, 1)
        val_loss, val_emb_acc, val_gen_acc = eval_metrics(
            model, tokenizer, val_loader, device
        )
        elapsed = time.time() - t_epoch
        log(
            f"Epoch {epoch + 1} done in {elapsed:.1f}s | "
            f"avg_train_logloss={avg_train:.4f} | val_logloss={val_loss:.4f} | "
            f"val_emb_acc={val_emb_acc * 100:.2f}% | val_gen_acc={val_gen_acc * 100:.2f}%"
        )

        if cfg.save_every_epoch:
            ep_path = out_dir / f"epoch_{epoch + 1}"
            model.save_pretrained(str(ep_path))
            log(f"Checkpoint: {ep_path}")

    final_path = out_dir / "last"
    model.save_pretrained(str(final_path))
    log("=" * 60)
    if best_val_acc >= 0:
        log(f"Done. Best val gen acc: {best_val_acc * 100:.2f}%")
    else:
        log("Done. No checkpoint saved by val_gen_acc (eval_every may be larger than steps).")
    log(f"Last weights: {final_path}")
    log("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT fine-tune nanoVLM on MiniGrid")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lr-mp", type=float, default=None)
    parser.add_argument("--lr-backbones", type=float, default=None)
    args = parser.parse_args()

    cfg = SFTConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_objects is not None:
        cfg.max_objects = args.max_objects
    if args.pretrained is not None:
        cfg.pretrained_path = args.pretrained
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.lr_mp is not None:
        cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        cfg.lr_backbones = args.lr_backbones

    train_sft(cfg)


if __name__ == "__main__":
    main()
