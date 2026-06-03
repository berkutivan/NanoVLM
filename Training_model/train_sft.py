"""
SFT fine-tuning of nanoVLM-222M on MiniGrid expert trajectories.

Fine-tuning strategy
--------------------
1. Load the *full* pretrained VLM from checkpoints/nanoVLM-222M (vision + LM + MP).
2. Do NOT re-initialize from backbone weights only — that would be continued pre-training.
3. Optimize MP + language decoder (ViT frozen by default) with two learning rates:
   - modality projector (MP): higher LR — adapts vision tokens to the new task quickly;
   - language decoder: lower LR — adapts LM while ViT features stay fixed.
4. Loss: optimal-set embedding log-loss + CE on answer tokens (trains ``lm_head``).
   BF16 AMP + optional 8-bit AdamW for faster training / lower VRAM.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
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
    filter_curated_cache,
    filter_minari_cache,
    load_curated_trajectories,
    load_objects,
    precompute_minari_for_keys,
    precompute_trajectories,
    split_curated_episodes,
    split_minari_episodes,
    subsample_list,
    subsample_dataset,
    split_objects,
)
from sft_config import SFTConfig  # noqa: E402
from sft_train_utils import (  # noqa: E402
    build_sft_optimizer,
    freeze_for_sft,
    sft_param_groups,
    trainable_parameters,
)

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


def _curated_manifest_path(cfg: SFTConfig) -> Path:
    return Path(cfg.curated_dataset_path) / "manifest.json"


def _build_curated_datasets(cfg: SFTConfig, tokenizer, image_processor):
    curated_dir = Path(cfg.curated_dataset_path)
    manifest, full_cache = load_curated_trajectories(curated_dir)
    all_ids = sorted(full_cache.keys())
    train_ids, val_ids = split_curated_episodes(all_ids, cfg.val_ratio, cfg.seed)
    val_ids = subsample_list(val_ids, cfg.val_subsample, cfg.seed + 1)
    log(
        f"Curated episodes: total={manifest.get('n_episodes', len(all_ids))} "
        f"train={len(train_ids)} val={len(val_ids)}"
    )
    log(
        f"  FOV-filtered steps: {manifest.get('n_steps', '?')} "
        f"(direct={manifest.get('n_direct_steps', '?')}, "
        f"explore={manifest.get('n_explore_steps', '?')})"
    )

    train_cache = filter_curated_cache(full_cache, train_ids)
    val_cache = filter_curated_cache(full_cache, val_ids)
    train_steps = sum(len(v) for v in train_cache.values())
    val_steps = sum(len(v) for v in val_cache.values())
    log(f"  split steps: train={train_steps} val={val_steps}")

    train_ds = MiniGridSFTDataset.from_curated(
        tokenizer,
        image_processor,
        trajectory_cache=train_cache,
    )
    if cfg.train_subsample < 1.0:
        n_before = len(train_ds)
        train_ds = subsample_dataset(train_ds, cfg.train_subsample, cfg.seed + 2)
        log(
            f"train_ds: {n_before} steps → {len(train_ds)} "
            f"(train_subsample={cfg.train_subsample})"
        )
    val_ds = MiniGridSFTDataset.from_curated(
        tokenizer,
        image_processor,
        trajectory_cache=val_cache,
    )
    return train_ds, val_ds


def _build_minari_datasets(cfg: SFTConfig, tokenizer, image_processor):
    from minari_adapter import iterate_episode_ids, load_minari_dataset  # noqa: WPS433

    all_keys: list[tuple[str, int]] = []
    for dataset_id in cfg.minari_datasets:
        ds = load_minari_dataset(dataset_id, download=cfg.minari_download)
        for ep_id in iterate_episode_ids(ds, cfg.max_episodes_per_dataset):
            all_keys.append((dataset_id, ep_id))

    train_keys, val_keys = split_minari_episodes(all_keys, cfg.val_ratio, cfg.seed)
    val_keys = subsample_list(val_keys, cfg.val_subsample, cfg.seed + 1)
    log(
        f"Minari episodes: total={len(all_keys)} train={len(train_keys)} "
        f"val={len(val_keys)} (val_subsample={cfg.val_subsample})"
    )

    log("Replaying Minari episodes with RGB partial observations...")
    t0 = time.time()
    replay_keys = train_keys + val_keys
    full_cache = precompute_minari_for_keys(
        replay_keys,
        download=False,
        tile_size=cfg.minari_tile_size,
        log_every=cfg.replay_log_every,
    )
    train_cache = filter_minari_cache(full_cache, train_keys)
    val_cache = filter_minari_cache(full_cache, val_keys)
    train_steps = sum(len(v) for v in train_cache.values())
    val_steps = sum(len(v) for v in val_cache.values())
    log(
        f"Replayed steps: train={train_steps} ({len(train_keys)} episodes), "
        f"val={val_steps} ({len(val_keys)} episodes) "
        f"in {time.time() - t0:.1f}s"
    )

    train_ds = MiniGridSFTDataset.from_minari(
        cfg.minari_datasets,
        tokenizer,
        image_processor,
        trajectory_cache=train_cache,
        download=False,
        tile_size=cfg.minari_tile_size,
    )
    if cfg.train_subsample < 1.0:
        n_before = len(train_ds)
        train_ds = subsample_dataset(train_ds, cfg.train_subsample, cfg.seed + 2)
        log(
            f"train_ds: {n_before} steps → {len(train_ds)} "
            f"(train_subsample={cfg.train_subsample}, episodes={len(train_keys)})"
        )
    val_ds = MiniGridSFTDataset.from_minari(
        cfg.minari_datasets,
        tokenizer,
        image_processor,
        trajectory_cache=val_cache,
        download=False,
        tile_size=cfg.minari_tile_size,
    )
    return train_ds, val_ds


def train_sft(cfg: SFTConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    log("=" * 60)
    log("nanoVLM SFT — MiniGrid / BabyAI expert fine-tuning")
    log("=" * 60)
    log(f"Device: {device} | AMP: {use_amp}")

    log(f"Loading pretrained VLM from: {cfg.pretrained_path}")
    model = VisionLanguageModel.from_pretrained(cfg.pretrained_path)
    trainable, n_params = freeze_for_sft(
        model,
        freeze_vision=cfg.freeze_vision,
        unfreeze_lm_blocks=cfg.unfreeze_lm_blocks,
    )
    log(
        f"Parameters: {n_params:,} total, {trainable:,} trainable "
        f"(freeze_vision={cfg.freeze_vision}, unfreeze_lm_blocks={cfg.unfreeze_lm_blocks})"
    )

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    if _curated_manifest_path(cfg).is_file():
        log(f"Data source: curated BabyAI ({cfg.curated_dataset_path})")
        train_ds, val_ds = _build_curated_datasets(cfg, tokenizer, image_processor)
    elif cfg.minari_datasets:
        log(f"Data source: Minari ({len(cfg.minari_datasets)} datasets)")
        train_ds, val_ds = _build_minari_datasets(cfg, tokenizer, image_processor)
    else:
        objects = load_objects(Path(cfg.dataset_path))
        if cfg.max_objects is not None:
            objects = objects[: cfg.max_objects]
        log(f"Data source: JSON mazes ({len(objects)} objects)")

        train_objs, val_objs = split_objects(objects, cfg.val_ratio, cfg.seed)
        val_objs = subsample_list(val_objs, cfg.val_subsample, cfg.seed + 1)
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

    param_groups = sft_param_groups(model, cfg.lr_mp, cfg.lr_backbones)
    optimizer = build_sft_optimizer(
        param_groups,
        weight_decay=cfg.weight_decay,
        use_8bit=cfg.use_8bit_optimizer and device.type == "cuda",
    )
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

    opt_label = "AdamW8bit" if cfg.use_8bit_optimizer and device.type == "cuda" else "AdamW"
    log(
        f"Training: {cfg.epochs} epochs, batch_size={cfg.batch_size}, "
        f"lr_mp={cfg.lr_mp}, lr_decoder={cfg.lr_backbones}, optimizer={opt_label}"
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
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), cfg.max_grad_norm)
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
    parser.add_argument(
        "--no-freeze-vision",
        action="store_true",
        help="Train ViT weights (default: ViT frozen)",
    )
    parser.add_argument(
        "--no-8bit-optimizer",
        action="store_true",
        help="Use standard AdamW instead of bitsandbytes 8-bit",
    )
    parser.add_argument(
        "--unfreeze-lm-blocks",
        type=int,
        default=None,
        help="Train only last N LM blocks (0 = full decoder, default from config)",
    )
    parser.add_argument(
        "--no-minari",
        action="store_true",
        help="Use legacy Datasets/dataset.json instead of Minari BabyAI data",
    )
    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="Cap episodes replayed per Minari dataset (debug/smoke)",
    )
    args = parser.parse_args()

    cfg = SFTConfig()
    if args.no_minari:
        cfg.minari_datasets = []
    if args.max_episodes_per_dataset is not None:
        cfg.max_episodes_per_dataset = args.max_episodes_per_dataset
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
    if args.no_freeze_vision:
        cfg.freeze_vision = False
    if args.no_8bit_optimizer:
        cfg.use_8bit_optimizer = False
    if args.unfreeze_lm_blocks is not None:
        cfg.unfreeze_lm_blocks = args.unfreeze_lm_blocks

    train_sft(cfg)


if __name__ == "__main__":
    main()
