"""
Show pre-sample #N: reset env from layout, display RGB obs via matplotlib.

https://minigrid.farama.org/ — RGBImgPartialObsWrapper + plt.imshow
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from maze_presample import MazePreSampleBuilder, MazePreSampleConfig, load_dataset


def replay(index: int, render_mode: str | None = "human") -> None:
    data = load_dataset()
    objects = data.get("objects", [])
    if not objects:
        print("dataset.json empty — run create_sample.py", file=sys.stderr)
        sys.exit(1)
    if index < 0 or index >= len(objects):
        print(f"index 0..{len(objects) - 1}", file=sys.stderr)
        sys.exit(1)

    obj = objects[index]
    layout = obj["layout"]
    cfg = MazePreSampleConfig(
        size=layout["size"],
        n_walls=len(layout["walls"]),
        seed=obj["seed"],
        agent_start_pos=tuple(layout["agent_start"]),
        agent_start_dir=layout["agent_start_dir"],
        render_mode=render_mode,
        tile_size=obj["predicate_space"]["observation"]["tile_size"],
        fixed_walls=[tuple(w) for w in layout["walls"]],
        fixed_goal=tuple(layout["goal_pos"]),
    )
    env = MazePreSampleBuilder(cfg).make_env()
    obs, _ = env.reset(seed=cfg.seed)

    print(f"id={obj['id']} mission={obj['mission']} goal={layout['goal_pos']}")

    if render_mode == "human":
        env.unwrapped.render()

    plt.imshow(obs["image"])
    plt.title(obj["mission"])
    plt.axis("off")
    plt.show()
    env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("index", type=int)
    p.add_argument("--no-pygame", action="store_true", help="Only matplotlib, no env.render()")
    args = p.parse_args()
    replay(args.index, render_mode=None if args.no_pygame else "human")


if __name__ == "__main__":
    main()
