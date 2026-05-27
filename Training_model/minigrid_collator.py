"""Collator compatible with transformers>=5 (no batch_encode_plus)."""

from __future__ import annotations

import torch


def batch_prompt_tensors(
    batch: dict,
    *,
    tokenizer,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prompt token tensors for log-loss (hidden state at end of prompt only).

    Prefer ``prompt_input_ids`` / ``prompt_attention_mask`` from the collator.
    Fall back to on-the-fly tokenization of ``prompt_texts`` (e.g. stale DataLoader).
    """
    if "prompt_input_ids" in batch and "prompt_attention_mask" in batch:
        return (
            batch["prompt_input_ids"].to(device),
            batch["prompt_attention_mask"].to(device),
        )

    texts = batch.get("prompt_texts")
    if not texts:
        keys = sorted(batch.keys())
        raise KeyError(
            "Batch has no prompt_input_ids / prompt_texts. "
            f"Got keys: {keys}. Re-run the dataset/collator cell after updating minigrid_collator.py."
        )

    enc = tokenizer(
        texts,
        padding="max_length",
        padding_side="left",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def last_hidden_after_prompt(
    model,
    input_ids: torch.Tensor,
    image: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Hidden state at the last real text token of the prompt (before any answer).

    Works with stock ``VisionLanguageModel`` (no patch to nanoVLM required).
    """
    if hasattr(model, "last_hidden_after_prompt"):
        return model.last_hidden_after_prompt(input_ids, image, attention_mask=attention_mask)

    image_embd = model.vision_encoder(image)
    image_embd = model.MP(image_embd)
    token_embd = model.decoder.token_embedding(input_ids)
    combined_embd = torch.cat((image_embd, token_embd), dim=1)

    if attention_mask is not None:
        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)
        image_attention_mask = torch.ones(
            (batch_size, img_seq_len),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask_full = torch.cat((image_attention_mask, attention_mask), dim=1)
    else:
        attention_mask_full = None

    hidden = model.decoder(combined_embd, attention_mask_full)
    hidden_text = hidden[:, image_embd.size(1) :, :]

    if attention_mask is None:
        return hidden_text[:, -1, :]

    b = hidden_text.size(0)
    last_idx = attention_mask.sum(dim=1).long().clamp(min=1) - 1
    row = torch.arange(b, device=hidden_text.device)
    return hidden_text[row, last_idx, :]


class MiniGridSFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        images = torch.stack([item["image"] for item in batch])
        texts = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]
        allowed_actions = [item.get("allowed_actions") for item in batch]

        input_sequences = [f"{texts[i]}{answers[i]}" for i in range(len(batch))]

        encoded_full_sequences = self.tokenizer(
            input_sequences,
            padding="max_length",
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        prompt_only = self.tokenizer(
            texts,
            padding="max_length",
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded_full_sequences["input_ids"]
        attention_mask = encoded_full_sequences["attention_mask"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone()
        labels[:, -1] = -100

        original_lengths = [
            len(self.tokenizer.encode(seq, add_special_tokens=False)) for seq in input_sequences
        ]

        for i in range(len(batch)):
            question_length = len(
                self.tokenizer.encode(texts[i], add_special_tokens=False)
            )

            if original_lengths[i] > self.max_length:
                labels[i, :] = -100
                continue

            first_token_pos = attention_mask[i].nonzero(as_tuple=True)[0][0].item()
            question_end = first_token_pos + question_length - 1
            labels[i, :question_end] = -100

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "allowed_actions": allowed_actions,
            "prompt_texts": texts,
            "prompt_input_ids": prompt_only["input_ids"],
            "prompt_attention_mask": prompt_only["attention_mask"],
        }
