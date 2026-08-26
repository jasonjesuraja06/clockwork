"""Qwen2 causal LM reimplemented in plain PyTorch, weight-compatible with Hugging Face."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from clockwork.engine.attn_metadata import AttentionMetadata
from clockwork.kernels.attention import (
    naive_attention,
    paged_attention_decode,
    paged_attention_prefill,
)


def _head_dim(config) -> int:
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


def _rope_theta(config) -> float:
    # transformers 5.x stores rope_theta inside config.rope_parameters.
    params = getattr(config, "rope_parameters", None)
    if params:
        rope_type = params.get("rope_type", "default")
        if rope_type != "default":
            raise NotImplementedError(f"rope_type {rope_type!r} is not supported")
        return float(params["rope_theta"])
    return float(config.rope_theta)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise NotImplementedError(f"hidden_act {hidden_act!r} is not supported")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = _head_dim(config)
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        if kv_cache is None:
            out = naive_attention(q, k, v, self.scaling, causal=True)
        else:
            kv_cache.write(self.layer_idx, attn_metadata.slot_mapping, k, v)
            k_cache = kv_cache.k_cache[self.layer_idx]
            v_cache = kv_cache.v_cache[self.layer_idx]
            if attn_metadata.is_prefill:
                chunks = []
                start = 0
                for i, query_len in enumerate(attn_metadata.query_lens):
                    chunks.append(
                        paged_attention_prefill(
                            q[start : start + query_len],
                            k_cache,
                            v_cache,
                            attn_metadata.seq_block_tables[i],
                            int(attn_metadata.ctx_lens[i]),
                            self.scaling,
                        )
                    )
                    start += query_len
                out = torch.cat(chunks, dim=0)
            else:
                out = paged_attention_decode(
                    q,
                    k_cache,
                    v_cache,
                    attn_metadata.block_tables,
                    attn_metadata.ctx_lens,
                    self.scaling,
                )
        return self.o_proj(out.reshape(num_tokens, -1))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._rope_theta = _rope_theta(config)
        self._rope_head_dim = _head_dim(config)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        head_dim = self._rope_head_dim
        return 1.0 / (
            self._rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

    def _apply(self, fn, recurse=True):
        # Keep inv_freq float32 under model-wide dtype casts (HF does the same);
        # a half-precision inv_freq visibly degrades cos/sin at long positions,
        # so recompute rather than round-trip through the lower precision.
        module = super()._apply(fn, recurse)
        if module.inv_freq.dtype != torch.float32:
            module.inv_freq = module._compute_inv_freq().to(module.inv_freq.device)
        return module

    def rope_cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(positions, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, kv_cache, attn_metadata)
        return self.norm(hidden_states)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, hf_config, attention_backend: str = "torch"):
        """Build the model from a Hugging Face Qwen2 config; accepts the HF state dict unchanged."""
        super().__init__()
        if getattr(hf_config, "use_sliding_window", False):
            raise NotImplementedError("sliding-window attention is not supported")
        self.config = hf_config
        self.attention_backend = attention_backend
        self.model = Qwen2Model(hf_config)
        self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)
        if getattr(hf_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        """Run the flat token batch and return last-token logits, [num_seqs, vocab_size]."""
        hidden_states = self.model(input_ids, positions, kv_cache, attn_metadata)
        if attn_metadata is not None:
            indices = attn_metadata.logits_indices
        else:
            indices = torch.tensor([hidden_states.shape[0] - 1], device=hidden_states.device)
        return self.lm_head(hidden_states[indices])
