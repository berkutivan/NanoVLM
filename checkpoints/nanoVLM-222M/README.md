---
license: apache-2.0
library_name: nanovlm
tags:
- vision-language
- multimodal
- pytorch
- small-model
- efficient
- research
- VLM
model_name: nanoVLM
datasets:
- HuggingFaceM4/the_cauldron
metrics:
- accuracy
pipeline_tag: image-text-to-text
---
  **nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. 
  Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. 
  It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), 
  resulting in a compact 222M parameter model. The model achieves 35.3% accuracy on MMStar after training for ~6 hours on a single H100 GPU 
  using 1.7M samples from [the cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) dataset, 
  making it a strong baseline for low-resource VLM research.

  The model is ideal for researchers and developers interested in exploring VLM training with minimal computational overhead, 
  and serves as a perfect starting point for tinkering with multimodal architectures.

**Model Architecture:**
  - Vision Transformer (SigLIP-B/16)
  - Causal Language Model (SmolLM2)
  - Modality Projection Layer

**Training:**
  - Trained on ~1.7M samples from the `the_cauldron` dataset
  - 6 hours on a single NVIDIA H100 GPU
  - Resulting model size: 222M parameters

**Evaluation:**
  - MMStar Accuracy: 35.3%

**Usage:**  
Usable through the nanoVLM repository: https://github.com/huggingface/nanoVLM  
For more details, see: https://github.com/huggingface/nanoVLM?tab=readme-ov-file#hub-integration
  ```python
  from models.vision_language_model import VisionLanguageModel
  model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")
  ```