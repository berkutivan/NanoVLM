"""Download or refresh local nanoVLM-222M weights (optional helper)."""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nanoVLM"))

from models.vision_language_model import VisionLanguageModel  # noqa: E402

OUT = ROOT / "checkpoints" / "nanoVLM-222M"
HUB = "lusxvr/nanoVLM-222M"


def main() -> None:
    source = OUT if OUT.exists() else HUB
    print(f"Loading from {source} ...")
    model = VisionLanguageModel.from_pretrained(str(source))
    model.save_pretrained(str(OUT))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
