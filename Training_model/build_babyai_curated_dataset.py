"""
Build a curated BabyAI SFT dataset: 100 episodes, filtered steps.

Keeps only steps where key or finish (Goal / goto Ball) is in the agent FOV,
or where one explore move (left/right/forward) would reveal such a target
(in that case the mission gets ``You should explore space``).

Output: ``Datasets/babyai_curated/manifest.json`` + ``episodes/*.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "Datasets"
TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATASETS_DIR / "babyai_curated"

for p in (str(DATASETS_DIR), str(TRAINING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from babyai_fov import classify_step_mission  # noqa: E402
from maze_expert import ExpertStep  # noqa: E402
from minari_adapter import (  # noqa: E402
    MINARI_MAZE_DATASETS,
    action_name,
    ensure_h5py,
    iterate_episode_ids,
    load_minari_dataset,
    make_rgb_env,
)

# Diverse BabyAI themes / difficulty (Minari optimal-v0 on HuggingFace).
CURATED_SOURCE_DATASETS: tuple[str, ...] = MINARI_MAZE_DATASETS + (
    "minigrid/BabyAI-GoToRedBallGrey/optimal-v0",
    "minigrid/BabyAI-GoToLocal/optimal-v0",
    "minigrid/BabyAI-GoTo/optimal-v0",
    "minigrid/BabyAI-Unlock/optimal-v0",
    "minigrid/BabyAI-UnlockPickup/optimal-v0",
)


@dataclass
class EpisodeStats:
    dataset_id: str
    episode_id: int
    n_total_steps: int
    n_kept_steps: int
    n_direct: int
    n_explore: int


def _theme_label(dataset_id: str) -> str:
    name = dataset_id.split("/")[-1].replace("/optimal-v0", "")
    if "KeyCorridor" in name:
        return "key_corridor"
    if "GoToObjMaze" in name:
        return "goto_maze"
    if "GoTo" in name:
        return "goto"
    if "Unlock" in name:
        return "unlock"
    if "Open" in name:
        return "open"
    return "other"


def replay_episode_filtered(
    dataset: Any,
    episode: Any,
    episode_metadata: dict[str, Any],
    *,
    tile_size: int,
) -> tuple[list[ExpertStep], EpisodeStats]:
    seed = episode_metadata.get("seed")
    options = episode_metadata.get("options")
    prefix_ids: list[int] = []

    def make_env_fn():
        return make_rgb_env(dataset, tile_size=tile_size)

    env = make_env_fn()
    try:
        obs, _ = env.reset(seed=seed, options=options)
        base_mission = str(obs["mission"])
        kept: list[ExpertStep] = []
        n_direct = 0
        n_explore = 0

        for step_index, action in enumerate(episode.actions):
            aid = int(action)
            mission, tag = classify_step_mission(
                base_mission,
                env,
                make_env_fn=make_env_fn,
                seed=seed,
                options=options,
                prefix_action_ids=prefix_ids,
            )
            if mission is not None:
                kept.append(
                    ExpertStep(
                        image=np.asarray(obs["image"], dtype=np.uint8),
                        mission=mission,
                        action=action_name(aid),
                        action_id=aid,
                        allowed_actions=(action_name(aid),),
                        step_index=len(kept),
                    )
                )
                if tag == "direct":
                    n_direct += 1
                else:
                    n_explore += 1

            obs, _, terminated, truncated, _ = env.step(aid)
            prefix_ids.append(aid)
            if terminated or truncated:
                break
    finally:
        env.close()

    stats = EpisodeStats(
        dataset_id="",
        episode_id=int(episode.id),
        n_total_steps=len(episode.actions),
        n_kept_steps=len(kept),
        n_direct=n_direct,
        n_explore=n_explore,
    )
    return kept, stats


def scan_episode(
    dataset_id: str,
    episode_id: int,
    *,
    download: bool,
    tile_size: int,
) -> EpisodeStats | None:
    dataset = load_minari_dataset(dataset_id, download=download)
    metas = list(dataset.storage.get_episode_metadata([episode_id]))
    if not metas:
        return None
    episodes = list(dataset.iterate_episodes(episode_indices=[episode_id]))
    if not episodes:
        return None
    _, stats = replay_episode_filtered(
        dataset, episodes[0], metas[0], tile_size=tile_size
    )
    stats = EpisodeStats(
        dataset_id=dataset_id,
        episode_id=episode_id,
        n_total_steps=stats.n_total_steps,
        n_kept_steps=stats.n_kept_steps,
        n_direct=stats.n_direct,
        n_explore=stats.n_explore,
    )
    return stats if stats.n_kept_steps > 0 else None


def select_episodes_stratified(
    dataset_ids: tuple[str, ...],
    n_episodes: int,
    seed: int,
    *,
    download: bool,
    tile_size: int,
    max_scan_per_dataset: int = 80,
    min_pool_per_dataset: int = 8,
    log_every: int = 20,
) -> list[tuple[str, int]]:
    """Round-robin pick episodes with ≥1 kept step across themes."""
    rng = np.random.default_rng(seed)
    pools: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for dataset_id in dataset_ids:
        try:
            ds = load_minari_dataset(dataset_id, download=download)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {dataset_id}: {exc}", flush=True)
            continue
        ep_ids = iterate_episode_ids(ds)
        rng.shuffle(ep_ids)
        scan_ids = ep_ids[:max_scan_per_dataset] if max_scan_per_dataset else ep_ids
        print(
            f"  scanning {dataset_id} ({len(scan_ids)}/{len(ep_ids)} episodes)...",
            flush=True,
        )
        for i, ep_id in enumerate(scan_ids):
            stats = scan_episode(
                dataset_id, ep_id, download=False, tile_size=tile_size
            )
            if stats is not None:
                pools[dataset_id].append((dataset_id, ep_id))
            if len(pools[dataset_id]) >= min_pool_per_dataset:
                print(
                    f"    [{dataset_id}] pool={len(pools[dataset_id])} "
                    f"(stop early after {i + 1} scans)",
                    flush=True,
                )
                break
            if log_every and (i + 1) % log_every == 0:
                print(
                    f"    [{dataset_id}] scanned {i + 1}/{len(scan_ids)}, "
                    f"pool={len(pools[dataset_id])}",
                    flush=True,
                )

    available = [d for d in dataset_ids if pools.get(d)]
    if not available:
        raise RuntimeError("No episodes passed FOV filter — check Minari install/data")

    selected: list[tuple[str, int]] = []
    idx = {d: 0 for d in available}
    while len(selected) < n_episodes:
        progressed = False
        for dataset_id in available:
            pool = pools[dataset_id]
            if idx[dataset_id] >= len(pool):
                continue
            selected.append(pool[idx[dataset_id]])
            idx[dataset_id] += 1
            progressed = True
            if len(selected) >= n_episodes:
                break
        if not progressed:
            break

    if len(selected) < n_episodes:
        print(
            f"Warning: only {len(selected)} episodes available (requested {n_episodes})",
            flush=True,
        )
    return selected


def save_episode_npz(path: Path, steps: list[ExpertStep], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.stack([s.image for s in steps], axis=0)
    actions = np.array([s.action for s in steps], dtype=object)
    action_ids = np.array([s.action_id for s in steps], dtype=np.int32)
    missions = np.array([s.mission for s in steps], dtype=object)
    np.savez_compressed(
        path,
        images=images,
        actions=actions,
        action_ids=action_ids,
        missions=missions,
        meta_json=json.dumps(meta),
    )


def build_curated_dataset(
    *,
    output_dir: Path,
    n_episodes: int = 100,
    seed: int = 0,
    tile_size: int = 8,
    download: bool = True,
    dataset_ids: tuple[str, ...] | None = None,
    max_scan_per_dataset: int = 80,
) -> Path:
    ensure_h5py()
    output_dir = Path(output_dir)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    sources = dataset_ids or CURATED_SOURCE_DATASETS
    print(f"Selecting {n_episodes} episodes from {len(sources)} BabyAI datasets...", flush=True)
    t0 = time.time()
    selected = select_episodes_stratified(
        sources,
        n_episodes,
        seed,
        download=download,
        tile_size=tile_size,
        max_scan_per_dataset=max_scan_per_dataset,
    )
    print(f"Selected {len(selected)} episodes in {time.time() - t0:.1f}s", flush=True)

    manifest_episodes: list[dict[str, Any]] = []
    total_steps = 0
    total_direct = 0
    total_explore = 0

    for cur_id, (dataset_id, episode_id) in enumerate(selected):
        dataset = load_minari_dataset(dataset_id, download=False)
        metas = list(dataset.storage.get_episode_metadata([episode_id]))
        episodes = list(dataset.iterate_episodes(episode_indices=[episode_id]))
        steps, raw_stats = replay_episode_filtered(
            dataset, episodes[0], metas[0], tile_size=tile_size
        )
        stats = EpisodeStats(
            dataset_id=dataset_id,
            episode_id=episode_id,
            n_total_steps=raw_stats.n_total_steps,
            n_kept_steps=raw_stats.n_kept_steps,
            n_direct=raw_stats.n_direct,
            n_explore=raw_stats.n_explore,
        )

        npz_name = f"{cur_id:04d}.npz"
        ep_meta = {
            "dataset_id": dataset_id,
            "episode_id": episode_id,
            "theme": _theme_label(dataset_id),
            "base_mission": str(steps[0].mission).split(" You should explore space")[0],
            "n_kept_steps": len(steps),
            "n_direct": stats.n_direct,
            "n_explore": stats.n_explore,
        }
        save_episode_npz(episodes_dir / npz_name, steps, ep_meta)

        manifest_episodes.append(
            {
                "id": cur_id,
                "file": f"episodes/{npz_name}",
                **ep_meta,
            }
        )
        total_steps += len(steps)
        total_direct += stats.n_direct
        total_explore += stats.n_explore
        if (cur_id + 1) % 10 == 0:
            print(f"  saved {cur_id + 1}/{len(selected)} episodes", flush=True)

    manifest = {
        "version": 1,
        "format": "babyai_curated",
        "tile_size": tile_size,
        "n_episodes": len(manifest_episodes),
        "n_steps": total_steps,
        "n_direct_steps": total_direct,
        "n_explore_steps": total_explore,
        "seed": seed,
        "source_datasets": list(sources),
        "episodes": manifest_episodes,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(
        f"Done: {manifest_path}\n"
        f"  episodes={len(manifest_episodes)} steps={total_steps} "
        f"(direct={total_direct}, explore={total_explore})",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated BabyAI FOV-filtered dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Minari dataset ids (default: CURATED_SOURCE_DATASETS)",
    )
    parser.add_argument(
        "--max-scan-per-dataset",
        type=int,
        default=80,
        help="Cap episode scans per Minari dataset during selection (speed)",
    )
    args = parser.parse_args()

    ids = tuple(args.datasets) if args.datasets else None
    build_curated_dataset(
        output_dir=Path(args.output_dir),
        n_episodes=args.n_episodes,
        seed=args.seed,
        tile_size=args.tile_size,
        download=not args.no_download,
        dataset_ids=ids,
        max_scan_per_dataset=args.max_scan_per_dataset,
    )


if __name__ == "__main__":
    main()
