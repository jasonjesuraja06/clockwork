import pytest
import torch

from clockwork.kernels import triton_paged_attn
from clockwork.kernels.attention import (
    HAS_TRITON,
    naive_attention,
    paged_attention_decode,
    paged_attention_prefill,
    resolve_backend,
)

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
BLOCK_SIZE = 4
NUM_BLOCKS = 16
SCALE = HEAD_DIM**-0.5


def write_to_cache(k_cache, v_cache, block_table, k, v, start=0):
    for i in range(k.shape[0]):
        pos = start + i
        blk = block_table[pos // BLOCK_SIZE]
        off = pos % BLOCK_SIZE
        k_cache[blk, off] = k[i]
        v_cache[blk, off] = v[i]


def empty_cache():
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return torch.zeros(shape), torch.zeros(shape)


def test_prefill_full_sequence_matches_naive(seed_all):
    ctx_len = 13
    q = torch.randn(ctx_len, NUM_HEADS, HEAD_DIM)
    k = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
    block_table = [7, 2, 11, 5]
    k_cache, v_cache = empty_cache()
    write_to_cache(k_cache, v_cache, block_table, k, v)

    out = paged_attention_prefill(q, k_cache, v_cache, block_table, ctx_len, SCALE)
    ref = naive_attention(q, k, v, SCALE, causal=True)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_prefill_with_context_offset_matches_dense_slice(seed_all):
    ctx_len = 21
    query_len = 6
    q_full = torch.randn(ctx_len, NUM_HEADS, HEAD_DIM)
    k = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
    block_table = [9, 0, 14, 3, 6, 1]
    k_cache, v_cache = empty_cache()
    write_to_cache(k_cache, v_cache, block_table, k, v)

    q_suffix = q_full[ctx_len - query_len :]
    out = paged_attention_prefill(q_suffix, k_cache, v_cache, block_table, ctx_len, SCALE)
    ref = naive_attention(q_full, k, v, SCALE, causal=True)[ctx_len - query_len :]
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_prefill_rejects_query_longer_than_context(seed_all):
    q = torch.randn(5, NUM_HEADS, HEAD_DIM)
    k_cache, v_cache = empty_cache()
    with pytest.raises(ValueError):
        paged_attention_prefill(q, k_cache, v_cache, [0], 3, SCALE)


def test_decode_varied_ctx_lens_and_block_layouts(seed_all):
    # Sequences 0 and 2 share physical block 6 as their first block, the block
    # tables are out of order and non-contiguous, and two ctx_lens are not
    # multiples of BLOCK_SIZE.
    ctx_lens = [5, 12, 9]
    tables = [[6, 3], [2, 9, 1], [6, 4, 8]]
    shared = (
        torch.randn(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        torch.randn(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
    )

    k_cache, v_cache = empty_cache()
    dense_kv = []
    for ctx_len, table in zip(ctx_lens, tables, strict=True):
        k = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
        v = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
        if table[0] == 6:
            k[:BLOCK_SIZE] = shared[0]
            v[:BLOCK_SIZE] = shared[1]
        write_to_cache(k_cache, v_cache, table, k, v)
        dense_kv.append((k, v))

    batch = len(ctx_lens)
    q = torch.randn(batch, NUM_HEADS, HEAD_DIM)
    max_blocks = max(len(t) for t in tables)
    block_tables = torch.zeros(batch, max_blocks, dtype=torch.int32)
    for i, table in enumerate(tables):
        block_tables[i, : len(table)] = torch.tensor(table, dtype=torch.int32)

    out = paged_attention_decode(
        q, k_cache, v_cache, block_tables, torch.tensor(ctx_lens, dtype=torch.int32), SCALE
    )

    for i, (k, v) in enumerate(dense_kv):
        ref = naive_attention(q[i : i + 1], k, v, SCALE, causal=True)
        torch.testing.assert_close(out[i : i + 1], ref, atol=1e-6, rtol=1e-6)


def test_decode_padding_blocks_do_not_leak(seed_all):
    # Point every padding slot at a block full of huge values; masking must
    # keep them out of the softmax.
    ctx_len = 3
    table = [5, 7]
    k_cache, v_cache = empty_cache()
    k = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM)
    write_to_cache(k_cache, v_cache, table, k, v)
    k_cache[7] = 100.0
    v_cache[7] = 100.0
    k_cache[5, ctx_len:] = 100.0
    v_cache[5, ctx_len:] = 100.0

    q = torch.randn(1, NUM_HEADS, HEAD_DIM)
    out = paged_attention_decode(
        q,
        k_cache,
        v_cache,
        torch.tensor([table], dtype=torch.int32),
        torch.tensor([ctx_len], dtype=torch.int32),
        SCALE,
    )
    ref = naive_attention(q, k, v, SCALE, causal=True)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_naive_attention_mha_matches_sdpa(seed_all):
    q = torch.randn(10, NUM_HEADS, HEAD_DIM)
    k = torch.randn(10, NUM_HEADS, HEAD_DIM)
    v = torch.randn(10, NUM_HEADS, HEAD_DIM)
    out = naive_attention(q, k, v, SCALE, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2), is_causal=True, scale=SCALE
    ).permute(1, 0, 2)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_resolve_backend():
    assert resolve_backend("torch") == "torch"
    with pytest.raises(ValueError):
        resolve_backend("mps")
    if not (HAS_TRITON and torch.cuda.is_available()):
        assert resolve_backend("auto") == "torch"
        with pytest.raises(RuntimeError):
            resolve_backend("triton")
    else:
        assert resolve_backend("auto") == "triton"
        assert resolve_backend("triton") == "triton"


@pytest.mark.skipif(triton_paged_attn.HAS_TRITON, reason="Triton is installed")
def test_triton_stub_raises_without_triton():
    with pytest.raises(RuntimeError):
        triton_paged_attn.triton_paged_attention_decode(None, None, None, None, None, 1.0)
