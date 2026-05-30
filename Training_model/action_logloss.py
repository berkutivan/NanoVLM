"""
Binary log-loss on "optimal action set" using prompt hidden state vs action embeddings.

Optimal actions come from maze_expert (shortest path on (x, y, direction) graph).
Positive class = any optimal action; probability = max cosine similarity between
the model state vector and embeddings of positive actions (mapped to (0, 1) for log-loss).

Reference action embeddings are detached so gradients flow through the prompt representation only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ACTION_ORDER = (
    "left",
    "right",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
)

try:
    from minigrid_collator import last_hidden_after_prompt  # noqa: E402
except ImportError:
    def last_hidden_after_prompt(  # type: ignore[misc]
        model,
        input_ids: torch.Tensor,
        image: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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


@torch.no_grad()
def action_embedding_matrix(model, tokenizer, device: torch.device) -> torch.Tensor:
    """
    One vector per action, same formatting as the dataset answer prefix: ``" {action}"``.
    Rows follow ACTION_ORDER.
    """
    weight = model.decoder.token_embedding.weight
    rows: list[torch.Tensor] = []
    for a in ACTION_ORDER:
        ids = tokenizer.encode(f" {a}", add_special_tokens=False)
        if not ids:
            raise ValueError(f"tokenizer produced no ids for action {a!r}")
        idx = torch.tensor(ids, device=device, dtype=torch.long)
        vecs = weight[idx]
        rows.append(vecs.mean(dim=0))
    return torch.stack(rows, dim=0)


def positive_action_mask(
    allowed_per_sample: list[list[str]] | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Shape [B, len(ACTION_ORDER)], 1.0 where the action is in the optimal set for that sample."""
    m = torch.zeros(batch_size, len(ACTION_ORDER), device=device, dtype=dtype)
    if allowed_per_sample is None:
        return m.fill_(1.0)
    for i, acts in enumerate(allowed_per_sample):
        if i >= batch_size:
            break
        s = set(acts or ())
        for j, name in enumerate(ACTION_ORDER):
            if name in s:
                m[i, j] = 1.0
    return m


def optimal_set_log_loss(
    h: torch.Tensor,
    action_emb: torch.Tensor,
    positive_mask: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    h: [B, H] last prompt hidden (gradients OK).
    action_emb: [len(ACTION_ORDER), H] reference vectors (expected detached).
    positive_mask: [B, len(ACTION_ORDER)] 1 for optimal actions at this state.

    For each sample, sims = cosine_sim(h, e_k) for k in ACTION_ORDER.
    p = max_{k: mask_k=1} sims_k  (best alignment to any correct action).
    Map cosine similarity from [-1, 1] to (0, 1) via (sim + 1) / 2, then -log(p) with clamping.

    Note: "cosine distance" is often 1 - cosine similarity; taking the maximum similarity to
    any positive action matches minimizing the minimum distance to the positive set.
    """
    ref = action_emb.detach()
    # Cosine formula (no separate F.normalize on h beyond what cosine_similarity does internally).
    sims = F.cosine_similarity(h.unsqueeze(1), ref.unsqueeze(0), dim=-1)
    masked = sims.masked_fill(positive_mask < 0.5, float("-inf"))
    max_sim, _ = masked.max(dim=1)
    bad = ~torch.isfinite(max_sim)
    if bad.any():
        max_sim = torch.where(bad, sims.max(dim=1).values, max_sim)
    p = ((max_sim + 1.0) * 0.5).clamp(eps, 1.0 - eps)
    return (-torch.log(p)).mean()


def action_cosine_similarities(h: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
    """Cosine similarity [B, len(ACTION_ORDER)] between prompt hidden states and action embeddings."""
    ref = action_emb.detach()
    return F.cosine_similarity(h.unsqueeze(1), ref.unsqueeze(0), dim=-1)


def predict_actions_from_hidden(h: torch.Tensor, action_emb: torch.Tensor) -> list[str]:
    """Argmax over BabyAI actions by embedding cosine similarity."""
    sims = action_cosine_similarities(h, action_emb)
    indices = sims.argmax(dim=1).tolist()
    return [ACTION_ORDER[i] for i in indices]


@torch.no_grad()
def action_first_token_ids(tokenizer, device: torch.device) -> torch.Tensor:
    """First subword id for each answer prefix ``" {action}"`` (for constrained decode)."""
    ids: list[int] = []
    for a in ACTION_ORDER:
        encoded = tokenizer.encode(f" {a}", add_special_tokens=False)
        if not encoded:
            raise ValueError(f"tokenizer produced no ids for action {a!r}")
        ids.append(encoded[0])
    return torch.tensor(ids, device=device, dtype=torch.long)


def count_allowed_hits(
    predicted: list[str],
    allowed_per_sample: list[list[str] | None],
) -> int:
    hits = 0
    for pred, allowed in zip(predicted, allowed_per_sample):
        opts = allowed if allowed is not None else list(ACTION_ORDER)
        if pred in set(opts):
            hits += 1
    return hits


def actions_from_first_token_ids(
    token_ids: torch.Tensor,
    first_token_ids: torch.Tensor,
) -> list[str]:
    """Map generated first token ids to action names (restricted decode)."""
    mapping = {
        int(tid): ACTION_ORDER[i] for i, tid in enumerate(first_token_ids.tolist())
    }
    return [mapping.get(int(t), "") for t in token_ids.tolist()]
