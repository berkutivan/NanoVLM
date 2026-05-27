# NanoVLM MiniGrid SFT (Colab-ready)

This repository is packaged to work after a plain `git clone` in Google Colab (no submodules / nested git deps).

## Colab quickstart

```bash
git clone https://github.com/berkutivan/NanoVLM.git
cd NanoVLM
pip install -r requirements.txt
```

### Run the SFT notebook

Open `Training_model/sft_pipeline.ipynb` and run from top to bottom.

### Run scripts

From the repo root:

```bash
python nanoVLM/generate.py --image nanoVLM/assets/image.png --prompt "What is this?"
python Training_model/train_sft.py
```

## Notes

- `checkpoints/`, `.venv/` and `*.zip` are intentionally ignored by git.
- Pretrained weights are pulled from Hugging Face Hub by default (see `Training_model/sft_pipeline.ipynb`).
