"""
GRPO (Group Relative Policy Optimization) for MiniGrid VLM policies.

Paper: https://arxiv.org/abs/2402.03300
- Sample G rollouts per environment seed (group).
- Advantage_i = (R_i - mean(R)) / (std(R) + eps)
- Loss = -E[advantage * log pi(a|s)] + beta * KL(pi || pi_ref)
  where pi_ref is the frozen SFT checkpoint.
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
NANOVLM_DIR = ROOT / "nanoVLM"
for p in (str(NANOVLM_DIR), str(TRAINING_DIR), str(ROOT / "Datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.processors import get_image_processor, get_tokenizer  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402
from grpo_policy import GrpoVlmPolicy  # noqa: E402
from grpo_rollout import (  # noqa: E402
    collect_grpo_group,
    empty_env_id,
    group_advantages,
    iter_training_seeds,
    make_empty_rgb_env,
    make_manifest_rgb_env,
    rollout_episode,
)
from minigrid_live_eval import (  # noqa: E402
    LiveEvalConfig,
    MiniGridVlmPolicy,
    evaluate_manifest_episodes,
    print_live_eval_summary,
)
from minigrid_sft_dataset import load_curated_trajectories, split_curated_episodes  # noqa: E402
from models.vision_language_model import VisionLanguageModel  # noqa: E402
from sft_train_utils import (  # noqa: E402
    build_sft_optimizer,
    freeze_for_sft,
    sft_param_groups,
    trainable_parameters,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_sft_ckpt(cfg: GRPOConfig) -> Path:
    ckpt = Path(cfg.sft_checkpoint)
    if ckpt.is_dir() and (ckpt / "model.safetensors").is_file():
        return ckpt
    for name in ("best", "last", "epoch_3", "epoch_2", "epoch_1"):
        alt = Path(cfg.sft_checkpoint).parent / name
        if (alt / "model.safetensors").is_file():
            return alt
    raise FileNotFoundError(f"SFT checkpoint not found: {cfg.sft_checkpoint}")


def _load_models(cfg: GRPOConfig, device: torch.device):
    ckpt = _resolve_sft_ckpt(cfg)
    log(f"Loading policy from SFT: {ckpt}")
    model = VisionLanguageModel.from_pretrained(str(ckpt)).to(device)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    return model, ref_model, tokenizer, image_processor


def evaluate_empty_env(
    policy: GrpoVlmPolicy,
    cfg: GRPOConfig,
    *,
    n_episodes: int,
) -> dict[str, float]:
    successes: list[bool] = []
    returns: list[float] = []
    policy.eval()
    for i in range(n_episodes):
        env = make_empty_rgb_env(size=cfg.empty_size, tile_size=cfg.tile_size)
        try:
            traj = rollout_episode(
                env,
                policy,
                seed=cfg.seed + 1000 + i,
                max_steps=cfg.max_agent_steps,
                env_id=empty_env_id(cfg.empty_size),
            )
            successes.append(traj.success)
            returns.append(traj.return_)
        finally:
            env.close()
    policy.train()
    n = max(len(successes), 1)
    return {
        "success_rate": sum(successes) / n,
        "mean_return": sum(returns) / n,
        "mean_length": 0.0,
    }


def grpo_loss_for_group(
    group: list[EpisodeTrajectory],
    advantages: list[float],
    policy: GrpoVlmPolicy,
    kl_coef: float,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=policy.device)
    n_terms = 0
    for traj, adv in zip(group, advantages):
        if traj.length == 0:
            continue
        scale = float(adv) / traj.length
        for step in traj.steps:
            lp, kl = policy.step_log_prob_train(step)
            loss = loss - scale * lp + kl_coef * kl
            n_terms += 1
    if n_terms == 0:
        return loss
    return loss / n_terms


def evaluate_sft_style(
    model,
    tokenizer,
    image_processor,
    device: torch.device,
    manifest: dict,
    val_ids: list[int],
    *,
    n_episodes: int,
    tile_size: int,
    output_mode: str = "action",
    ref_model=None,
) -> dict[str, float]:
    if output_mode == "action":
        policy = MiniGridVlmPolicy(model, tokenizer, image_processor, device)
    else:
        policy = GrpoVlmPolicy(
            model,
            ref_model or model,
            tokenizer,
            image_processor,
            device,
            output_mode=output_mode,
        )
    cfg = LiveEvalConfig(
        tile_size=tile_size,
        n_eval_episodes=n_episodes,
        verbose_first=False,
        visualize_first=False,
    )
    summary = evaluate_manifest_episodes(manifest, val_ids, policy, config=cfg)
    return {
        "success_rate": summary.success_rate,
        "mean_return": summary.mean_return,
        "mean_length": summary.mean_length,
    }


def train_grpo(cfg: GRPOConfig) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    out_dir = Path(cfg.output_dir) / cfg.output_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ref_model, tokenizer, image_processor = _load_models(cfg, device)
    freeze_for_sft(
        model,
        freeze_vision=cfg.freeze_vision,
        unfreeze_lm_blocks=cfg.unfreeze_lm_blocks,
    )
    optimizer = build_sft_optimizer(
        sft_param_groups(model, cfg.lr_mp, cfg.lr_backbones),
        weight_decay=cfg.weight_decay,
        use_8bit=cfg.use_8bit_optimizer and device.type == "cuda",
    )

    policy = GrpoVlmPolicy(
        model,
        ref_model,
        tokenizer,
        image_processor,
        device,
        output_mode=cfg.output_mode,
        temperature=cfg.temperature,
        max_new_tokens_text=cfg.max_new_tokens_text,
    )

    manifest: dict | None = None
    val_ids: list[int] = []
    train_meta: list[dict] = []
    dataset_cache: dict[str, Any] = {}

    if cfg.env_mode == "manifest":
        manifest_path = Path(cfg.curated_manifest_path)
        manifest, _ = load_curated_trajectories(manifest_path.parent)
        all_ids = sorted({int(e["id"]) for e in manifest["episodes"]})
        train_ids, val_ids = split_curated_episodes(all_ids, val_ratio=0.1, seed=cfg.seed)
        if cfg.train_episode_ids is not None:
            train_ids = [i for i in cfg.train_episode_ids if i in train_ids]
        id_to_meta = {int(e["id"]): e for e in manifest["episodes"]}
        train_meta = [id_to_meta[i] for i in train_ids if i in id_to_meta]

    seeds = iter_training_seeds(cfg.seed, cfg.num_updates * cfg.episodes_per_update)
    history: dict[str, list] = {
        "update": [],
        "loss": [],
        "mean_return": [],
        "success_rate": [],
        "eval_success_rate": [],
        "eval_mean_return": [],
    }

    log("=" * 60)
    log(f"GRPO training | mode={cfg.output_mode} | env={cfg.env_mode}")
    log(f"  group_size={cfg.group_size} | updates={cfg.num_updates} | kl_coef={cfg.kl_coef}")
    log("=" * 60)

    seed_idx = 0
    for update in range(cfg.num_updates):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        update_returns: list[float] = []
        update_successes: list[bool] = []
        update_loss_val = 0.0
        groups_done = 0

        for _ in range(cfg.episodes_per_update):
            reset_options = None

            if cfg.env_mode == "empty":
                seed = seeds[seed_idx]
                seed_idx += 1
                env_id = f"MiniGrid-Empty-{cfg.empty_size}x{cfg.empty_size}-v0"

                def env_factory():
                    return make_empty_rgb_env(size=cfg.empty_size, tile_size=cfg.tile_size)

            else:
                ep_meta = train_meta[update % len(train_meta)]
                env_id = f"curated-{ep_meta['id']}"
                _env, _meta, _ = make_manifest_rgb_env(
                    ep_meta, tile_size=cfg.tile_size, dataset_cache=dataset_cache
                )
                _env.close()
                seed = _meta.get("seed", seeds[seed_idx % len(seeds)])
                seed_idx += 1
                reset_options = _meta.get("options")

                def env_factory(epm=ep_meta):
                    env, _m, _k = make_manifest_rgb_env(
                        epm, tile_size=cfg.tile_size, dataset_cache=dataset_cache
                    )
                    return env

            group = collect_grpo_group(
                env_factory,
                policy,
                seed=seed,
                group_size=cfg.group_size,
                max_steps=cfg.max_agent_steps,
                env_id=env_id,
                reset_options=reset_options,
            )

            returns = [g.return_ for g in group]
            update_returns.extend(returns)
            update_successes.extend(g.success for g in group)
            advantages = group_advantages(returns, eps=cfg.adv_eps, clip=cfg.clip_adv)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = grpo_loss_for_group(group, advantages, policy, cfg.kl_coef)
            else:
                loss = grpo_loss_for_group(group, advantages, policy, cfg.kl_coef)

            if loss.requires_grad:
                (loss / cfg.episodes_per_update).backward()
            update_loss_val += float(loss.detach().item())
            groups_done += 1

        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), cfg.max_grad_norm)
        optimizer.step()

        mean_ret = sum(update_returns) / max(len(update_returns), 1)
        succ = sum(update_successes) / max(len(update_successes), 1)
        history["update"].append(update)
        history["loss"].append(update_loss_val / max(groups_done, 1))
        history["mean_return"].append(mean_ret)
        history["success_rate"].append(succ)

        if update % cfg.log_every == 0:
            log(
                f"[{update}/{cfg.num_updates}] loss={history['loss'][-1]:.4f} | "
                f"rollout_return={mean_ret:.3f} | rollout_success={succ * 100:.1f}%"
            )

        if update > 0 and update % cfg.eval_every == 0:
            model.eval()
            if manifest is not None:
                metrics = evaluate_sft_style(
                    model,
                    tokenizer,
                    image_processor,
                    device,
                    manifest,
                    val_ids,
                    n_episodes=cfg.eval_episodes,
                    tile_size=cfg.tile_size,
                    output_mode=cfg.output_mode,
                    ref_model=ref_model,
                )
            else:
                metrics = evaluate_empty_env(
                    policy, cfg, n_episodes=cfg.eval_episodes
                )
            history["eval_success_rate"].append(metrics["success_rate"])
            history["eval_mean_return"].append(metrics["mean_return"])
            log(
                f"  >> eval success={metrics['success_rate'] * 100:.1f}% | "
                f"return={metrics['mean_return']:.4f}"
            )
            policy.train()

        if update > 0 and update % cfg.save_every == 0:
            model.save_pretrained(str(out_dir / f"step_{update}"))

    model.save_pretrained(str(out_dir / "last"))
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    log("=" * 60)
    log(f"GRPO done -> {out_dir}")
    log("=" * 60)
    return history
