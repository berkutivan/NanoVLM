"""VLM policy with sampling and log-probs for GRPO."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from action_logloss import ACTION_ORDER, action_first_token_ids, last_hidden_after_prompt
from grpo_prompts import build_prompt, parse_action_from_text


class GrpoVlmPolicy:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        image_processor,
        device: torch.device,
        *,
        output_mode: str = "action",
        temperature: float = 1.0,
        max_new_tokens_text: int = 64,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.output_mode = output_mode
        self.temperature = temperature
        self.max_new_tokens_text = max_new_tokens_text
        self._first_action_toks = action_first_token_ids(tokenizer, device)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def _encode_prompt(self, mission: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        text = build_prompt(mission, self.output_mode)
        enc = self.tokenizer(
            [text],
            padding=True,
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.model.cfg.lm_max_length,
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), text

    def _image_tensor(self, rgb) -> torch.Tensor:
        return self.image_processor(Image.fromarray(rgb).convert("RGB")).unsqueeze(0).to(self.device)

    def _first_action_log_probs(self, model, h: torch.Tensor) -> torch.Tensor:
        """Log-probs over 7 actions via LM head on last prompt hidden state."""
        if not model.decoder.lm_use_tokens:
            logits = model.decoder.head(h)
        else:
            logits = h
        sub = logits[:, self._first_action_toks]
        return F.log_softmax(sub / self.temperature, dim=-1)

    @torch.no_grad()
    def _act_action_mode(self, rgb, mission: str, *, sample: bool) -> tuple[str, float, float, dict]:
        prompt_ids, prompt_mask, prompt_text = self._encode_prompt(mission)
        img_t = self._image_tensor(rgb)

        h = last_hidden_after_prompt(self.model, prompt_ids, img_t, attention_mask=prompt_mask)
        log_probs = self._first_action_log_probs(self.model, h).squeeze(0)
        probs = log_probs.exp()

        if sample:
            idx = torch.multinomial(probs, num_samples=1).item()
        else:
            idx = int(probs.argmax().item())
        action = ACTION_ORDER[idx]

        with torch.no_grad():
            h_ref = last_hidden_after_prompt(self.ref_model, prompt_ids, img_t, attention_mask=prompt_mask)
            log_probs_ref = self._first_action_log_probs(self.ref_model, h_ref).squeeze(0)
        kl = float(((log_probs - log_probs_ref).exp() * (log_probs - log_probs_ref)).sum().item())

        return action, float(log_probs[idx].item()), kl, {"prompt_text": prompt_text}

    @torch.no_grad()
    def _act_text_mode(self, rgb, mission: str, *, sample: bool) -> tuple[str, float, float, dict]:
        prompt_ids, prompt_mask, prompt_text = self._encode_prompt(mission)
        img_t = self._image_tensor(rgb)

        gen = self.model.generate(
            prompt_ids,
            img_t,
            prompt_mask,
            max_new_tokens=self.max_new_tokens_text,
            do_sample=sample,
        )
        raw = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        action = parse_action_from_text(raw)

        lp, kl, completion_ids = self._sequence_log_prob_and_kl(
            self.model, self.ref_model, prompt_ids, prompt_mask, img_t, gen[0]
        )
        return action, lp, kl, {
            "prompt_text": prompt_text,
            "raw": raw,
            "completion_ids": completion_ids,
        }

    def _sequence_log_prob_and_kl(
        self,
        model,
        ref_model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        image: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> tuple[float, float, list[int]]:
        """Sum log-prob of completion tokens (teacher forcing on generated ids)."""
        comp = completion_ids.unsqueeze(0)
        input_ids = torch.cat([prompt_ids, comp], dim=1)
        attn = torch.cat(
            [prompt_mask, torch.ones_like(comp, dtype=prompt_mask.dtype)],
            dim=1,
        )
        img_seq = model.vision_encoder(image)
        img_seq = model.MP(img_seq)
        tok_emb = model.decoder.token_embedding(input_ids)
        combined = torch.cat([img_seq, tok_emb], dim=1)
        img_len = img_seq.size(1)
        img_mask = torch.ones((1, img_len), device=attn.device, dtype=attn.dtype)
        full_mask = torch.cat([img_mask, attn], dim=1)

        hidden = model.decoder(combined, full_mask)
        if not model.decoder.lm_use_tokens:
            logits = model.decoder.head(hidden)
        else:
            logits = hidden
        logits = logits[:, img_len:-1, :]
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # only completion segment
        prompt_len = prompt_ids.size(1)
        comp_lp = token_lp[:, prompt_len - 1 :]
        total_lp = float(comp_lp.sum().item())
        comp_list = comp.squeeze(0).tolist()

        with torch.no_grad():
            hidden_r = ref_model.decoder(combined, full_mask)
            if not ref_model.decoder.lm_use_tokens:
                logits_r = ref_model.decoder.head(hidden_r)
            else:
                logits_r = hidden_r
            logits_r = logits_r[:, img_len:-1, :]
            log_probs_r = F.log_softmax(logits_r, dim=-1)
            token_lp_r = log_probs_r.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            comp_lp_r = token_lp_r[:, prompt_len - 1 :]
            kl_tokens = (comp_lp.exp() * (comp_lp - comp_lp_r)).sum()
            kl_val = float(kl_tokens.item())

        return total_lp, kl_val, comp_list

    @torch.no_grad()
    def act(self, rgb, mission: str, *, sample: bool = True) -> tuple[str, float, float, dict[str, Any]]:
        if self.output_mode == "text_action":
            return self._act_text_mode(rgb, mission, sample=sample)
        return self._act_action_mode(rgb, mission, sample=sample)

    @torch.no_grad()
    def predict(self, rgb, mission: str) -> tuple[str, str]:
        """Greedy action for live eval (compatible with MiniGridVlmPolicy)."""
        action, _lp, _kl, extra = self.act(rgb, mission, sample=False)
        return action, extra.get("raw", action)

    def step_log_prob_train(self, step) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute log_prob (+ KL) with gradients for one rollout step."""
        prompt_ids, prompt_mask, _ = self._encode_prompt(step.mission)
        img_t = self._image_tensor(step.rgb)

        if self.output_mode == "action":
            h = last_hidden_after_prompt(self.model, prompt_ids, img_t, attention_mask=prompt_mask)
            log_probs = self._first_action_log_probs(self.model, h).squeeze(0)
            idx = ACTION_ORDER.index(step.action)
            lp = log_probs[idx]

            with torch.no_grad():
                h_ref = last_hidden_after_prompt(
                    self.ref_model, prompt_ids, img_t, attention_mask=prompt_mask
                )
                log_probs_ref = self._first_action_log_probs(self.ref_model, h_ref).squeeze(0)
            kl = (log_probs.exp() * (log_probs - log_probs_ref)).sum()
            return lp, kl

        if not step.completion_ids:
            raise ValueError("text_action rollout step missing completion_ids")
        comp = torch.tensor(step.completion_ids, device=self.device, dtype=prompt_ids.dtype)
        lp_sum, kl_sum, _ = self._sequence_log_prob_and_kl_train(
            prompt_ids, prompt_mask, img_t, comp
        )
        return lp_sum, kl_sum

    def _sequence_log_prob_and_kl_train(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        image: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        comp = completion_ids.unsqueeze(0)
        input_ids = torch.cat([prompt_ids, comp], dim=1)
        attn = torch.cat(
            [prompt_mask, torch.ones_like(comp, dtype=prompt_mask.dtype)],
            dim=1,
        )
        img_seq = self.model.vision_encoder(image)
        img_seq = self.model.MP(img_seq)
        tok_emb = self.model.decoder.token_embedding(input_ids)
        combined = torch.cat([img_seq, tok_emb], dim=1)
        img_len = img_seq.size(1)
        img_mask = torch.ones((1, img_len), device=attn.device, dtype=attn.dtype)
        full_mask = torch.cat([img_mask, attn], dim=1)

        hidden = self.model.decoder(combined, full_mask)
        if not self.model.decoder.lm_use_tokens:
            logits = self.model.decoder.head(hidden)
        else:
            logits = hidden
        logits = logits[:, img_len:-1, :]
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.size(1)
        comp_lp = token_lp[:, prompt_len - 1 :]
        total_lp = comp_lp.sum()

        with torch.no_grad():
            hidden_r = self.ref_model.decoder(combined, full_mask)
            if not self.ref_model.decoder.lm_use_tokens:
                logits_r = self.ref_model.decoder.head(hidden_r)
            else:
                logits_r = hidden_r
            logits_r = logits_r[:, img_len:-1, :]
            log_probs_r = F.log_softmax(logits_r, dim=-1)
            token_lp_r = log_probs_r.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            comp_lp_r = token_lp_r[:, prompt_len - 1 :]
        kl = (comp_lp.exp() * (comp_lp - comp_lp_r)).sum()
        return total_lp, kl, completion_ids.tolist()
