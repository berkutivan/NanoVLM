"""Execute sft_pipeline.ipynb cells (skip Colab-only), with smoke training settings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = Path(__file__).resolve().parent / "sft_pipeline.ipynb"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"

SKIP_MARKERS = (
    "!git clone",
    "%cd /content",
    "!unzip /content",
    "!pip install",
)


def should_skip(source: str) -> bool:
    return any(m in source for m in SKIP_MARKERS)


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    os.chdir(ROOT)
    sys.path[:0] = [
        str(ROOT / "nanoVLM"),
        str(ROOT / "Datasets"),
        str(ROOT / "Training_model"),
    ]

    # Inject smoke overrides before training cell
    overrides = """
MAX_OBJECTS = 6
EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 1
LOG_EVERY = 5
EVAL_EVERY = 20
N_MAZES = 0
"""

    g: dict = {"__name__": "__main__"}
    failed = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if should_skip(src):
            print(f"[skip cell {i}] Colab setup")
            continue
        if "## 6." in "".join(nb["cells"][i - 1].get("source", [])) if i > 0 else False:
            exec(overrides, g)
        if "for epoch in range(EPOCHS):" in src and "MAX_OBJECTS" not in g:
            exec(overrides, g)
        print(f"[run cell {i}] {src[:60].strip().replace(chr(10), ' ')}...")
        try:
            exec(compile(src, f"<cell {i}>", "exec"), g)
        except Exception as e:
            failed = (i, e)
            print(f"[FAIL cell {i}] {type(e).__name__}: {e}")
            break

    if failed:
        raise SystemExit(f"Notebook execution failed at cell {failed[0]}: {failed[1]}")
    print("NOTEBOOK SMOKE RUN PASSED")


if __name__ == "__main__":
    main()
