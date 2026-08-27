import pytest
import torch

from clockwork.kernels import attention, triton_paged_attn
from clockwork.kernels.attention import (
    HAS_TRITON,
    paged_attention_decode,
    paged_attention_decode_torch,
    resolve_backend,
)

TRITON_USABLE = HAS_TRITON and torch.cuda.is_available()

CASES = [
    # (num_heads, num_kv_heads, head_dim, block_size, ctx_lens)
    (4, 4, 16, 4, [5, 12, 9]),
    (8, 2, 32, 4, [1, 7, 16, 3]),
    (6, 3, 64, 8, [23, 8, 31]),
    (12, 2, 128, 16, [17, 64, 49, 2]),
]


def build_case(num_heads, num_kv_heads, head_dim, block_size, ctx_lens, device="cpu"):
    # Sequences 0 and 1 share their first physical block out of order, block
    # tables are non-contiguous, and free blocks hold huge values so any
    # masking or gather bug lands far outside tolerance.
    torch.manual_seed(sum(ctx_lens) + num_heads * head_dim)
    batch = len(ctx_lens)
    blocks_per_seq = [-(-ctx // block_size) for ctx in ctx_lens]
    num_blocks = sum(blocks_per_seq) + 4

    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    k_cache = torch.full(shape, 100.0, device=device)
    v_cache = torch.full(shape, 100.0, device=device)

    free = list(torch.randperm(num_blocks).tolist())
    shared_first = free.pop()
    tables = []
    for i, ctx_len in enumerate(ctx_lens):
        table = [free.pop() for _ in range(blocks_per_seq[i])]
        if i < 2:
            table[0] = shared_first
        tables.append(table)
        k = torch.randn(ctx_len, num_kv_heads, head_dim, device=device)
        v = torch.randn(ctx_len, num_kv_heads, head_dim, device=device)
        for pos in range(ctx_len):
            blk, off = table[pos // block_size], pos % block_size
            k_cache[blk, off] = k[pos]
            v_cache[blk, off] = v[pos]

    q = torch.randn(batch, num_heads, head_dim, device=device)
    max_blocks = max(blocks_per_seq)
    block_tables = torch.zeros(batch, max_blocks, dtype=torch.int32, device=device)
    for i, table in enumerate(tables):
        block_tables[i, : len(table)] = torch.tensor(table, dtype=torch.int32)
    return q, k_cache, v_cache, block_tables, torch.tensor(ctx_lens, dtype=torch.int32)


def tiled_decode_transliteration(q, k_cache, v_cache, block_tables, ctx_lens, scale):
    # Line-for-line torch transliteration of _paged_decode_kernel: one
    # (sequence, head) program, one block-sized tile per iteration, and the
    # same online-softmax running max, exp sum, and accumulator updates. It
    # validates the kernel's loop structure on CPU where Triton cannot run.
    batch, num_heads, head_dim = q.shape
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    num_q_per_kv = num_heads // num_kv_heads
    out = torch.empty_like(q)
    for seq in range(batch):
        ctx_len = int(ctx_lens[seq])
        for head in range(num_heads):
            kv_head = head // num_q_per_kv
            q_vec = q[seq, head].to(torch.float32)
            running_max = torch.tensor(float("-inf"))
            exp_sum = torch.tensor(0.0)
            acc = torch.zeros(head_dim)
            offs = torch.arange(block_size)
            for logical_block in range(-(-ctx_len // block_size)):
                block_id = int(block_tables[seq, logical_block])
                token_mask = logical_block * block_size + offs < ctx_len
                k_tile = k_cache[block_id, :, kv_head].to(torch.float32)
                scores = (k_tile * q_vec).sum(-1) * scale
                scores = torch.where(token_mask, scores, torch.tensor(float("-inf")))
                new_max = torch.maximum(running_max, scores.max())
                correction = torch.exp(running_max - new_max)
                probs = torch.exp(scores - new_max)
                exp_sum = exp_sum * correction + probs.sum()
                v_tile = v_cache[block_id, :, kv_head].to(torch.float32)
                acc = acc * correction + (probs[:, None] * v_tile).sum(0)
                running_max = new_max
            out[seq, head] = (acc / exp_sum).to(q.dtype)
    return out


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,block_size,ctx_lens", CASES)
def test_transliteration_matches_torch_decode(
    num_heads, num_kv_heads, head_dim, block_size, ctx_lens
):
    q, k_cache, v_cache, block_tables, ctx = build_case(
        num_heads, num_kv_heads, head_dim, block_size, ctx_lens
    )
    scale = head_dim**-0.5
    out = tiled_decode_transliteration(q, k_cache, v_cache, block_tables, ctx, scale)
    ref = paged_attention_decode_torch(q, k_cache, v_cache, block_tables, ctx, scale)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_transliteration_single_token_context():
    # ctx_len 1 exercises the first-tile case where the running max starts at
    # -inf and the correction factor must collapse to zero, not NaN.
    q, k_cache, v_cache, block_tables, ctx = build_case(4, 2, 16, 4, [1, 1])
    scale = 16**-0.5
    out = tiled_decode_transliteration(q, k_cache, v_cache, block_tables, ctx, scale)
    ref = paged_attention_decode_torch(q, k_cache, v_cache, block_tables, ctx, scale)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_decode_on_cpu_never_routes_to_triton(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("triton kernel called for cpu tensors")

    monkeypatch.setattr(attention, "triton_paged_attention_decode", _boom)
    q, k_cache, v_cache, block_tables, ctx = build_case(4, 2, 16, 4, [5, 3])
    scale = 16**-0.5
    out = paged_attention_decode(q, k_cache, v_cache, block_tables, ctx, scale)
    ref = paged_attention_decode_torch(q, k_cache, v_cache, block_tables, ctx, scale)
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(TRITON_USABLE, reason="Triton and CUDA are available")
def test_triton_unavailable_raises_clearly():
    with pytest.raises(RuntimeError, match="triton"):
        resolve_backend("triton")
    with pytest.raises(RuntimeError, match="Triton is not installed"):
        triton_paged_attn.triton_paged_attention_decode(None, None, None, None, None, 1.0)


@pytest.mark.gpu
@pytest.mark.skipif(not TRITON_USABLE, reason="requires CUDA and Triton")
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,block_size,ctx_lens", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_triton_kernel_matches_torch(
    num_heads, num_kv_heads, head_dim, block_size, ctx_lens, dtype
):
    q, k_cache, v_cache, block_tables, ctx = build_case(
        num_heads, num_kv_heads, head_dim, block_size, ctx_lens, device="cuda"
    )
    q, k_cache, v_cache = q.to(dtype), k_cache.to(dtype), v_cache.to(dtype)
    scale = head_dim**-0.5
    out = triton_paged_attn.triton_paged_attention_decode(
        q, k_cache, v_cache, block_tables, ctx.to("cuda"), scale
    )
    ref = paged_attention_decode_torch(q, k_cache, v_cache, block_tables, ctx.to("cuda"), scale)
    assert out.shape == (len(ctx_lens), num_heads, head_dim)
    assert out.dtype == dtype
    tol = 1e-5 if dtype is torch.float32 else 2e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.gpu
@pytest.mark.skipif(not TRITON_USABLE, reason="requires CUDA and Triton")
def test_decode_routes_to_triton_on_cuda(monkeypatch):
    calls = []
    real = triton_paged_attn.triton_paged_attention_decode

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(attention, "triton_paged_attention_decode", _spy)
    q, k_cache, v_cache, block_tables, ctx = build_case(8, 2, 64, 16, [9, 33], device="cuda")
    scale = 64**-0.5
    out = paged_attention_decode(q, k_cache, v_cache, block_tables, ctx.to("cuda"), scale)
    assert calls, "paged_attention_decode did not route through the triton kernel"
    ref = paged_attention_decode_torch(q, k_cache, v_cache, block_tables, ctx.to("cuda"), scale)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
