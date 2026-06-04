"""
Live MiniGrid evaluation: env -> VLM agent -> step -> env -> ...

Metrics follow official MiniGrid docs:
  - success reward: 1 - 0.9 * (step_count / max_steps)
  - failure reward: 0
  - terminated / truncated (Gymnasium step API)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 — register envs
import numpy as np
import torch
from PIL import Image

TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
for p in (str(TRAINING_DIR), str(ROOT / "Datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_logloss import (  # noqa: E402
    ACTION_ORDER,
    action_first_token_ids,
    actions_from_first_token_ids,
)
from minari_adapter import ACTION_NAME_TO_ID, load_minari_dataset, make_rgb_env  # noqa: E402
from minigrid_sft_dataset import PROMPT_TEMPLATE  # noqa: E402


@dataclass
class LiveEvalConfig:
    tile_size: int = 8
    max_agent_steps: int | None = None
    n_eval_episodes: int = 10
    verbose_first: bool = True
    visualize_first: bool = True
    max_plot_frames: int = 8


@dataclass
class EpisodeRollout:
    curated_id: int
    dataset_id: str
    theme: str
    base_mission: str
    steps: int = 0
    return_: float = 0.0
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    failed_zero_reward: bool = False
    step_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LiveEvalSummary:
    results: list[EpisodeRollout]
    success_rate: float
    mean_return: float
    mean_length: float
    fail_zero: int


class MiniGridVlmPolicy:
    """VLM policy: (RGB, mission) -> BabyAI action name."""

    def __init__(self, model, tokenizer, image_processor, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.model.eval()
        self._first_action_toks = action_first_token_ids(tokenizer, device)

    @staticmethod
    def _parse_action(raw: str, gen_first_tok: int, first_action_toks: torch.Tensor, device) -> str:
        mapped = actions_from_first_token_ids(
            torch.tensor([gen_first_tok], device=device), first_action_toks
        )[0]
        if mapped:
            return mapped
        t = (raw or "").strip().lower().split()
        if not t:
            return "forward"
        w = t[0].rstrip(".,;:!?")
        return w if w in ACTION_ORDER else "forward"

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, mission: str) -> tuple[str, str]:
        """Return (action_name, raw_decode)."""
        prompt = PROMPT_TEMPLATE.format(mission=mission)
        enc = self.tokenizer(
            [prompt],
            padding=True,
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.model.cfg.lm_max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        img_t = self.image_processor(Image.fromarray(rgb).convert("RGB")).unsqueeze(0).to(self.device)
        gen = self.model.generate(
            input_ids,
            img_t,
            attention_mask,
            max_new_tokens=8,
            do_sample=False,
            restrict_first_token_to=self._first_action_toks,
        )
        raw = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        action = self._parse_action(raw, int(gen[0, 0].item()), self._first_action_toks, self.device)
        return action, raw


def _make_env_for_episode(
    ep_meta: dict[str, Any],
    *,
    tile_size: int,
    dataset_cache: dict[str, Any],
) -> tuple[gym.Env, dict[str, Any]]:
    dataset_id = ep_meta["dataset_id"]
    episode_id = int(ep_meta["episode_id"])
    if dataset_id not in dataset_cache:
        dataset_cache[dataset_id] = load_minari_dataset(dataset_id, download=True)
    dataset = dataset_cache[dataset_id]
    meta = list(dataset.storage.get_episode_metadata([episode_id]))[0]
    env = make_rgb_env(dataset, tile_size=tile_size)
    return env, meta


def run_agent_episode(
    ep_meta: dict[str, Any],
    policy: MiniGridVlmPolicy,
    *,
    tile_size: int = 8,
    max_agent_steps: int | None = None,
    dataset_cache: dict[str, Any] | None = None,
    verbose: bool = False,
    collect_frames: bool = False,
) -> EpisodeRollout:
    """
    Full loop: reset -> (obs -> agent -> env.step) -> ...

    Stops on terminated/truncated, zero final reward (failure), or step limit.
    """
    cache = dataset_cache if dataset_cache is not None else {}
    env, meta = _make_env_for_episode(ep_meta, tile_size=tile_size, dataset_cache=cache)
    seed = meta.get("seed")
    options = meta.get("options")
    obs, _ = env.reset(seed=seed, options=options)
    mission = str(obs["mission"])

    env_max = int(getattr(env.unwrapped, "max_steps", 256))
    step_limit = env_max if max_agent_steps is None else min(int(max_agent_steps), env_max)

    out = EpisodeRollout(
        curated_id=int(ep_meta["id"]),
        dataset_id=ep_meta["dataset_id"],
        theme=ep_meta.get("theme", "?"),
        base_mission=ep_meta.get("base_mission", mission),
    )

    total_reward = 0.0
    terminated = truncated = False

    try:
        for t in range(step_limit):
            rgb = np.asarray(obs["image"], dtype=np.uint8)
            if hasattr(policy, "act"):
                action_name, _lp, _kl, extra = policy.act(rgb, mission, sample=False)
                raw = extra.get("raw", action_name)
            else:
                action_name, raw = policy.predict(rgb, mission)
            action_id = ACTION_NAME_TO_ID.get(action_name, ACTION_NAME_TO_ID["forward"])

            if verbose:
                print(
                    f"  step {t:3d} | action={action_name:8s} | "
                    f"mission={mission[:70]!r}"
                )

            frame: dict[str, Any] = {"step": t, "action": action_name, "raw": raw, "mission": mission}
            if collect_frames:
                frame["image"] = rgb
            out.step_log.append(frame)

            obs, reward, terminated, truncated, _info = env.step(action_id)
            total_reward += float(reward)
            mission = str(obs["mission"])
            out.steps = t + 1

            if terminated or truncated:
                if float(reward) <= 0.0:
                    out.failed_zero_reward = True
                break

        out.return_ = total_reward
        out.terminated = terminated
        out.truncated = truncated
        out.success = terminated and not truncated and total_reward > 0.0
    finally:
        env.close()

    return out


def evaluate_manifest_episodes(
    manifest: dict[str, Any],
    episode_ids: list[int],
    policy: MiniGridVlmPolicy,
    *,
    config: LiveEvalConfig | None = None,
) -> LiveEvalSummary:
    """Run live rollouts on curated manifest episodes selected by curated id."""
    cfg = config or LiveEvalConfig()
    id_set = set(int(i) for i in episode_ids)
    eval_meta = [e for e in manifest["episodes"] if int(e["id"]) in id_set][: cfg.n_eval_episodes]

    print(f"Evaluating {len(eval_meta)} episodes in live MiniGrid (tile_size={cfg.tile_size})")

    dataset_cache: dict[str, Any] = {}
    results: list[EpisodeRollout] = []

    for i, ep_meta in enumerate(eval_meta):
        verbose = cfg.verbose_first and i == 0
        collect = cfg.visualize_first and i == 0
        if verbose:
            print("=" * 60)
            print(
                f"Episode curated_id={ep_meta['id']} | {ep_meta['theme']} | "
                f"{ep_meta.get('base_mission', '')[:80]}"
            )
        rollout = run_agent_episode(
            ep_meta,
            policy,
            tile_size=cfg.tile_size,
            max_agent_steps=cfg.max_agent_steps,
            dataset_cache=dataset_cache,
            verbose=verbose,
            collect_frames=collect,
        )
        results.append(rollout)
        status = "SUCCESS" if rollout.success else ("TIMEOUT" if rollout.truncated else "FAIL")
        print(
            f"[{i + 1}/{len(eval_meta)}] id={rollout.curated_id} | {status} | "
            f"steps={rollout.steps} | return={rollout.return_:.3f} | "
            f"terminated={rollout.terminated} truncated={rollout.truncated}"
        )

    n = len(results)
    return LiveEvalSummary(
        results=results,
        success_rate=sum(r.success for r in results) / max(n, 1),
        mean_return=sum(r.return_ for r in results) / max(n, 1),
        mean_length=sum(r.steps for r in results) / max(n, 1),
        fail_zero=sum(r.failed_zero_reward for r in results),
    )


def print_live_eval_summary(summary: LiveEvalSummary) -> None:
    n = len(summary.results)
    print("\n" + "=" * 60)
    print("MiniGrid live eval (official metrics)")
    print(f"  episodes       : {n}")
    print(f"  success rate   : {summary.success_rate * 100:.1f}%")
    print(f"  mean return    : {summary.mean_return:.4f}  (success: 1 - 0.9*steps/max_steps)")
    print(f"  mean length    : {summary.mean_length:.1f} steps")
    print(f"  zero-reward end: {summary.fail_zero}/{n}  (agent did not complete task)")
    print("=" * 60)
    for row in summary.results:
        print(
            f"  id={row.curated_id:3d} theme={row.theme:12s} "
            f"steps={row.steps:3d} return={row.return_:.3f} "
            f"success={row.success} term={row.terminated} trunc={row.truncated}"
        )


def plot_rollout_frames(rollout: EpisodeRollout, *, max_frames: int = 8) -> None:
    import matplotlib.pyplot as plt

    frames = [s for s in rollout.step_log if "image" in s]
    n_show = min(max_frames, len(frames))
    if not n_show:
        return

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    for ax, step in zip(axes, frames[:n_show]):
        ax.imshow(step["image"])
        ax.set_title(f"{step['step']}: {step['action']}", fontsize=9)
        ax.axis("off")
    plt.suptitle(
        f"Live rollout id={rollout.curated_id} | return={rollout.return_:.3f} | success={rollout.success}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()
