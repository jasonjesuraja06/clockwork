"""Triton paged-attention decode kernel; imports cleanly without Triton."""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        block_tables_ptr,
        ctx_lens_ptr,
        scale,
        num_q_per_kv,
        block_size,
        stride_qb,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_ob,
        stride_oh,
        stride_bt,
        HEAD_DIM: tl.constexpr,
        PADDED_BLOCK_SIZE: tl.constexpr,
        PADDED_HEAD_DIM: tl.constexpr,
    ):
        # One program per (sequence, query head). The cache holds K/V as
        # [num_blocks, block_size, num_kv_heads, head_dim]; the block table maps
        # logical block i of this sequence to a physical block id, so tiles are
        # gathered straight from the paged cache, one block per iteration,
        # without materializing the context.
        seq = tl.program_id(0)
        head = tl.program_id(1)
        kv_head = head // num_q_per_kv

        ctx_len = tl.load(ctx_lens_ptr + seq)
        offs = tl.arange(0, PADDED_BLOCK_SIZE)
        d = tl.arange(0, PADDED_HEAD_DIM)
        d_mask = d < HEAD_DIM

        q = tl.load(q_ptr + seq * stride_qb + head * stride_qh + d, mask=d_mask, other=0.0)
        q = q.to(tl.float32)

        # Online-softmax invariant: after each tile, running_max is the max
        # score seen so far, exp_sum is sum(exp(score - running_max)), and acc
        # is the V accumulation on the same scale, so acc / exp_sum equals the
        # full softmax attention once the last tile is folded in.
        running_max = float("-inf")
        exp_sum = 0.0
        acc = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)

        for logical_block in range(0, tl.cdiv(ctx_len, block_size)):
            block_id = tl.load(block_tables_ptr + seq * stride_bt + logical_block).to(tl.int64)
            token_mask = (offs < block_size) & (logical_block * block_size + offs < ctx_len)
            tile_mask = token_mask[:, None] & d_mask[None, :]

            k_tile = tl.load(
                k_ptr
                + block_id * stride_kb
                + offs[:, None] * stride_ks
                + kv_head * stride_kh
                + d[None, :],
                mask=tile_mask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k_tile * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, float("-inf"))

            new_max = tl.maximum(running_max, tl.max(scores, axis=0))
            correction = tl.exp(running_max - new_max)
            probs = tl.exp(scores - new_max)
            exp_sum = exp_sum * correction + tl.sum(probs, axis=0)

            v_tile = tl.load(
                v_ptr
                + block_id * stride_vb
                + offs[:, None] * stride_vs
                + kv_head * stride_vh
                + d[None, :],
                mask=tile_mask,
                other=0.0,
            ).to(tl.float32)
            acc = acc * correction + tl.sum(probs[:, None] * v_tile, axis=0)
            running_max = new_max

        out = acc / exp_sum
        tl.store(
            out_ptr + seq * stride_ob + head * stride_oh + d,
            out.to(out_ptr.dtype.element_ty),
            mask=d_mask,
        )


def triton_paged_attention_decode(q, k_cache, v_cache, block_tables, ctx_lens, scale):
    """Decode attention on the Triton backend; requires CUDA and an installed Triton."""
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed; use resolve_backend('torch') and paged_attention_decode"
        )
    if not q.is_cuda:
        raise RuntimeError("the Triton decode kernel requires CUDA tensors")
    batch, num_heads, head_dim = q.shape
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads {num_heads} not a multiple of num_kv_heads {num_kv_heads}")
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("k_cache and v_cache must be contiguous over head_dim")

    q = q.contiguous()
    block_tables = block_tables.to(device=q.device, dtype=torch.int32).contiguous()
    ctx_lens = ctx_lens.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.empty_like(q)

    grid = (batch, num_heads)
    _paged_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        block_tables,
        ctx_lens,
        scale,
        num_heads // num_kv_heads,
        block_size,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        block_tables.stride(0),
        HEAD_DIM=head_dim,
        PADDED_BLOCK_SIZE=triton.next_power_of_2(block_size),
        PADDED_HEAD_DIM=triton.next_power_of_2(head_dim),
    )
    return out
