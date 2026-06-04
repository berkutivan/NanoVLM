from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")
model.save_pretrained("checkpoints/nanoVLM-222M")  # локальная копия