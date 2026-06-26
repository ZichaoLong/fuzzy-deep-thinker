from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TinyDecoderConfig:
    vocab_size: int
    max_seq_len: int = 768
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


class TinyDecoder(nn.Module):
    def __init__(self, config: TinyDecoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TinyDecoderBlock(config.d_model, config.n_heads, config.dropout) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.latent_norm = nn.LayerNorm(config.d_model)
        self.latent_proj = nn.Linear(config.d_model, config.d_model)

    def forward_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_embeds(self.token_embedding(input_ids))

    def forward_embeds(self, input_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input_embeds.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_embeds.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = input_embeds + self.position_embedding(positions)
        mask = _causal_mask(seq_len, input_embeds.device)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.final_norm(hidden)

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def text_loss(self, input_ids: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_ids(input_ids)
        logits = self.logits_from_hidden(hidden)
        token_losses = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
        token_losses = token_losses.view_as(labels)
        denom = label_mask.float().sum().clamp_min(1.0)
        return (token_losses * label_mask.float()).sum() / denom

    def continuous_loss(
        self,
        prefix_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        num_steps: int,
        mode: str,
        soft_temperature: float = 1.0,
    ) -> torch.Tensor:
        if answer_ids.numel() == 0:
            raise ValueError("answer_ids must not be empty")

        seq_embeds = self.token_embedding(prefix_ids.unsqueeze(0))
        for _ in range(num_steps):
            hidden = self.forward_embeds(seq_embeds)
            last_hidden = hidden[:, -1, :]
            if mode == "latent":
                next_embed = self.latent_proj(self.latent_norm(last_hidden))
            elif mode == "soft":
                logits = self.logits_from_hidden(last_hidden)
                probs = torch.softmax(logits / soft_temperature, dim=-1)
                next_embed = probs @ self.token_embedding.weight
            else:
                raise ValueError(f"Unknown continuous mode: {mode}")
            seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)

        answer_input_ids = answer_ids[:-1]
        if answer_input_ids.numel() > 0:
            answer_embeds = self.token_embedding(answer_input_ids.unsqueeze(0))
            seq_embeds = torch.cat([seq_embeds, answer_embeds], dim=1)

        hidden = self.forward_embeds(seq_embeds)
        logits = self.logits_from_hidden(hidden)
        start = prefix_ids.numel() + num_steps - 1
        end = start + answer_ids.numel()
        supervised_logits = logits[:, start:end, :].squeeze(0)
        return F.cross_entropy(supervised_logits, answer_ids)

    def candidate_nll(self, prefix_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        if candidate_ids.numel() == 0:
            raise ValueError("candidate_ids must not be empty")
        input_ids = torch.cat([prefix_ids, candidate_ids[:-1]], dim=0).unsqueeze(0)
        hidden = self.forward_ids(input_ids)
        logits = self.logits_from_hidden(hidden).squeeze(0)
        start = prefix_ids.numel() - 1
        end = start + candidate_ids.numel()
        candidate_logits = logits[start:end, :]
        token_losses = F.cross_entropy(candidate_logits, candidate_ids, reduction="none")
        return token_losses.mean()

    def continuous_candidate_nll(
        self,
        prefix_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        num_steps: int,
        mode: str,
        soft_temperature: float = 1.0,
    ) -> torch.Tensor:
        if candidate_ids.numel() == 0:
            raise ValueError("candidate_ids must not be empty")

        seq_embeds = self.token_embedding(prefix_ids.unsqueeze(0))
        for _ in range(num_steps):
            hidden = self.forward_embeds(seq_embeds)
            last_hidden = hidden[:, -1, :]
            if mode == "latent":
                next_embed = self.latent_proj(self.latent_norm(last_hidden))
            elif mode == "soft":
                logits = self.logits_from_hidden(last_hidden)
                probs = torch.softmax(logits / soft_temperature, dim=-1)
                next_embed = probs @ self.token_embedding.weight
            else:
                raise ValueError(f"Unknown continuous mode: {mode}")
            seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)

        candidate_input_ids = candidate_ids[:-1]
        if candidate_input_ids.numel() > 0:
            candidate_embeds = self.token_embedding(candidate_input_ids.unsqueeze(0))
            seq_embeds = torch.cat([seq_embeds, candidate_embeds], dim=1)

        hidden = self.forward_embeds(seq_embeds)
        logits = self.logits_from_hidden(hidden).squeeze(0)
        start = prefix_ids.numel() + num_steps - 1
        end = start + candidate_ids.numel()
        token_losses = F.cross_entropy(logits[start:end, :], candidate_ids, reduction="none")
        return token_losses.mean()

    @torch.no_grad()
    def generate_text(self, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            hidden = self.forward_ids(ids.unsqueeze(0))
            logits = self.logits_from_hidden(hidden[:, -1, :])
            next_id = int(torch.argmax(logits, dim=-1).item())
            ids = torch.cat([ids, torch.tensor([next_id], device=ids.device, dtype=ids.dtype)])
            if next_id == 0:
                break
        return ids[input_ids.numel() :].tolist()

    @torch.no_grad()
    def generate_continuous(
        self,
        prefix_ids: torch.Tensor,
        num_steps: int,
        mode: str,
        max_new_tokens: int,
        soft_temperature: float = 1.0,
    ) -> list[int]:
        seq_embeds = self.token_embedding(prefix_ids.unsqueeze(0))
        for _ in range(num_steps):
            hidden = self.forward_embeds(seq_embeds)
            last_hidden = hidden[:, -1, :]
            if mode == "latent":
                next_embed = self.latent_proj(self.latent_norm(last_hidden))
            elif mode == "soft":
                logits = self.logits_from_hidden(last_hidden)
                probs = torch.softmax(logits / soft_temperature, dim=-1)
                next_embed = probs @ self.token_embedding.weight
            else:
                raise ValueError(f"Unknown continuous mode: {mode}")
            seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)

        generated = []
        for _ in range(max_new_tokens):
            hidden = self.forward_embeds(seq_embeds)
            logits = self.logits_from_hidden(hidden[:, -1, :])
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_id)
            next_embed = self.token_embedding(torch.tensor([[next_id]], device=seq_embeds.device))
            seq_embeds = torch.cat([seq_embeds, next_embed], dim=1)
            if next_id == 0:
                break
        return generated


class TinyDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.attn_norm(hidden)
        batch_size, seq_len, d_model = hidden.shape
        qkv = self.qkv(hidden)
        q, k, v = qkv.chunk(3, dim=-1)
        q = _split_heads(q, self.n_heads)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        hidden = residual + self.dropout(self.out_proj(attn))

        residual = hidden
        hidden = self.mlp_norm(hidden)
        hidden = residual + self.dropout(self.mlp(hidden))
        return hidden


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    return x.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
