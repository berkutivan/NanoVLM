"""Collator compatible with transformers>=5 (no batch_encode_plus)."""

import torch


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
        }
