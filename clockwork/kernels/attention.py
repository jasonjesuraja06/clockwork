"""Attention entry points: dense reference plus paged prefill and decode in torch."""

import torch

from clockwork.kernels.triton_paged_attn import HAS_TRITON, triton_paged_attention_decode

__all__ = [
    "HAS_TRITON",
    "naive_attention",
    "paged_attention_decode",
    "paged_attention_decode_torch",
    "paged_attention_prefill",
    "resolve_backend",
]


def resolve_backend(name: str = "auto") -> str:
    """Pick the attention backend: torch unless Triton and CUDA are both usable."""
    triton_usable = HAS_TRITON and torch.cuda.is_available()
    if name == "torch":
        return "torch"
    if name == "triton":
        if not triton_usable:
            raise RuntimeError(
                "attention backend 'triton' requested but it needs both CUDA and an "
                f"installed Triton (HAS_TRITON={HAS_TRITON}, "
                f"cuda={torch.cuda.is_available()})"
            )
        return "triton"
    if name == "auto":
        return "triton" if triton_usable else "torch"
    raise ValueError(f"unknown attention backend {name!r}; expected auto, torch, or triton")


def _repeat_kv(k: torch.Tensor, v: torch.Tensor, num_heads: int, head_axis: int):
    num_kv_heads = k.shape[head_axis]
    if num_kv_heads == num_heads:
        return k, v
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads {num_heads} not a multiple of num_kv_heads {num_kv_heads}")
    rep = num_heads // num_kv_heads
    return k.repeat_interleave(rep, dim=head_axis), v.repeat_interleave(rep, dim=head_axis)


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Dense reference attention; q [q_len, heads, dim], k and v [kv_len, kv_heads, dim]."""
    q_len, num_heads, _ = q.shape
    kv_len = k.shape[0]
    k, v = _repeat_kv(k, v, num_heads, head_axis=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    if causal:
        # Query row i sits at absolute position (kv_len - q_len + i), so it may
        # attend to key positions 0 .. kv_len - q_len + i inclusive.
        offset = kv_len - q_len
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos + offset, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v)


def _gather_context(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table,
    ctx_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = k_cache.shape[1]
    num_kv_heads, head_dim = k_cache.shape[2], k_cache.shape[3]
    blocks = torch.as_tensor(block_table, dtype=torch.long, device=k_cache.device)
    num_needed = -(-ctx_len // block_size)
    blocks = blocks[:num_needed]
    k = k_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:ctx_len]
    v = v_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:ctx_len]
    return k, v


def paged_attention_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table,
    ctx_len: int,
    scale: float,
) -> torch.Tensor:
    """Prefill attention over a paged cache with the query aligned to the context end."""
    if q.shape[0] > ctx_len:
        raise ValueError(f"query_len {q.shape[0]} exceeds ctx_len {ctx_len}")
    k, v = _gather_context(k_cache, v_cache, block_table, ctx_len)
    return naive_attention(q, k, v, scale, causal=True)


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    ctx_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Single-token decode attention over a paged cache for a batch of sequences."""
    if q.is_cuda and resolve_backend("auto") == "triton":
        return triton_paged_attention_decode(q, k_cache, v_cache, block_tables, ctx_lens, scale)
    return paged_attention_decode_torch(q, k_cache, v_cache, block_tables, ctx_lens, scale)


def paged_attention_decode_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    ctx_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Torch decode path; also the correctness reference for the Triton kernel."""
    batch, num_heads, head_dim = q.shape
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    tables = block_tables.to(torch.long)
    max_ctx = tables.shape[1] * block_size
    k = k_cache[tables].reshape(batch, max_ctx, num_kv_heads, head_dim)
    v = v_cache[tables].reshape(batch, max_ctx, num_kv_heads, head_dim)
    k, v = _repeat_kv(k, v, num_heads, head_axis=2)
    scores = torch.einsum("bhd,bkhd->bhk", q, k) * scale
    k_pos = torch.arange(max_ctx, device=q.device).view(1, 1, -1)
    invalid = k_pos >= ctx_lens.to(torch.long).view(batch, 1, 1)
    scores = scores.masked_fill(invalid, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhk,bkhd->bhd", probs, v)
