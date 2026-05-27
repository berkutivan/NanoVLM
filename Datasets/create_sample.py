"""Append one pre-sample to Datasets/dataset.json."""

from __future__ import annotations

import argparse

from maze_presample import append_presample


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--n-walls", type=int, default=None)
    args = p.parse_args()

    sample = append_presample(seed=args.seed, size=args.size, n_walls=args.n_walls)
    layout = sample["layout"]
    print(
        f"Added id={sample['id']} size={layout['size']} "
        f"walls={len(layout['walls'])} goal={layout['goal_pos']}"
    )


if __name__ == "__main__":
    main()
