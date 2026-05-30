"""Download Minari BabyAI maze datasets listed in sft_config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training_model"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from minari_adapter import MINARI_MAZE_DATASETS, ensure_h5py, load_minari_dataset  # noqa: E402
from sft_config import DEFAULT_MINARI_DATASETS  # noqa: E402


def main() -> None:
    ensure_h5py()
    parser = argparse.ArgumentParser(description="Download Minari BabyAI datasets")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_MINARI_DATASETS),
        help="Minari dataset ids (default: sft_config.DEFAULT_MINARI_DATASETS)",
    )
    parser.add_argument(
        "--all-maze",
        action="store_true",
        help="Download full MINARI_MAZE_DATASETS list",
    )
    args = parser.parse_args()
    ids = list(MINARI_MAZE_DATASETS) if args.all_maze else args.datasets

    for dataset_id in ids:
        print(f"Downloading {dataset_id} ...", flush=True)
        ds = load_minari_dataset(dataset_id, download=True)
        print(
            f"  OK: episodes={ds.total_episodes} steps={ds.total_steps}",
            flush=True,
        )


if __name__ == "__main__":
    main()
