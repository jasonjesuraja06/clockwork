import pytest
import torch

from clockwork.kvcache import PagedKVCache

NUM_LAYERS = 3
NUM_BLOCKS = 8
BLOCK_SIZE = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8


@pytest.fixture
def cache() -> PagedKVCache:
    return PagedKVCache(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
    )


def slot_mapping_for(block_table: list[int], start: int, end: int) -> torch.Tensor:
    slots = [
        block_table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE for pos in range(start, end)
    ]
    return torch.tensor(slots, dtype=torch.int64)


def rand_kv(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (num_tokens, NUM_KV_HEADS, HEAD_DIM)
    return torch.randn(shape), torch.randn(shape)


def test_cache_shapes(cache):
    assert len(cache.k_cache) == NUM_LAYERS
    assert len(cache.v_cache) == NUM_LAYERS
    expected = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    for layer in range(NUM_LAYERS):
        assert cache.k_cache[layer].shape == expected
        assert cache.v_cache[layer].shape == expected


def test_write_then_gather_round_trip_out_of_order_table(seed_all, cache):
    # Out-of-order physical blocks with a partial last block (10 tokens over 3 blocks).
    block_table = [5, 1, 6]
    ctx_len = 10
    k, v = rand_kv(ctx_len)
    slots = slot_mapping_for(block_table, 0, ctx_len)
    for layer in range(NUM_LAYERS):
        cache.write(layer, slots, k + layer, v - layer)
    for layer in range(NUM_LAYERS):
        got_k, got_v = cache.gather(layer, block_table, ctx_len)
        assert got_k.shape == (ctx_len, NUM_KV_HEADS, HEAD_DIM)
        assert got_v.shape == (ctx_len, NUM_KV_HEADS, HEAD_DIM)
        torch.testing.assert_close(got_k, k + layer)
        torch.testing.assert_close(got_v, v - layer)


def test_incremental_writes_round_trip(seed_all, cache):
    # Prefill 6 tokens, then decode 5 more one at a time across a block boundary.
    block_table = [3, 0, 7]
    k, v = rand_kv(11)
    cache.write(0, slot_mapping_for(block_table, 0, 6), k[:6], v[:6])
    for pos in range(6, 11):
        slots = slot_mapping_for(block_table, pos, pos + 1)
        cache.write(0, slots, k[pos : pos + 1], v[pos : pos + 1])
    got_k, got_v = cache.gather(0, block_table, 11)
    torch.testing.assert_close(got_k, k)
    torch.testing.assert_close(got_v, v)


def test_gather_full_blocks(seed_all, cache):
    block_table = [2, 4]
    ctx_len = 2 * BLOCK_SIZE
    k, v = rand_kv(ctx_len)
    cache.write(1, slot_mapping_for(block_table, 0, ctx_len), k, v)
    got_k, got_v = cache.gather(1, block_table, ctx_len)
    torch.testing.assert_close(got_k, k)
    torch.testing.assert_close(got_v, v)


def test_gather_ignores_trailing_table_entries(seed_all, cache):
    # A block table longer than ctx_len needs must not affect the gather.
    block_table = [6, 2, 5, 7]
    ctx_len = 5
    k, v = rand_kv(ctx_len)
    cache.write(0, slot_mapping_for(block_table, 0, ctx_len), k, v)
    got_k, got_v = cache.gather(0, block_table, ctx_len)
    torch.testing.assert_close(got_k, k)
    torch.testing.assert_close(got_v, v)


def test_copy_block_copies_all_layers(seed_all, cache):
    src, dst = 1, 6
    k, v = rand_kv(BLOCK_SIZE)
    for layer in range(NUM_LAYERS):
        cache.write(layer, slot_mapping_for([src], 0, BLOCK_SIZE), k * (layer + 1), v * (layer + 1))
    cache.copy_block(src, dst)
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(cache.k_cache[layer][dst], cache.k_cache[layer][src])
        torch.testing.assert_close(cache.v_cache[layer][dst], cache.v_cache[layer][src])


def test_copy_block_isolation(seed_all, cache):
    src, dst = 0, 3
    k, v = rand_kv(BLOCK_SIZE)
    for layer in range(NUM_LAYERS):
        cache.write(layer, slot_mapping_for([src], 0, BLOCK_SIZE), k, v)
    cache.copy_block(src, dst)
    src_k = [cache.k_cache[layer][src].clone() for layer in range(NUM_LAYERS)]
    src_v = [cache.v_cache[layer][src].clone() for layer in range(NUM_LAYERS)]
    new_k, new_v = rand_kv(BLOCK_SIZE)
    for layer in range(NUM_LAYERS):
        cache.write(layer, slot_mapping_for([dst], 0, BLOCK_SIZE), new_k, new_v)
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(cache.k_cache[layer][src], src_k[layer])
        torch.testing.assert_close(cache.v_cache[layer][src], src_v[layer])
        torch.testing.assert_close(cache.k_cache[layer][dst], new_k.to(torch.float32))
        torch.testing.assert_close(cache.v_cache[layer][dst], new_v.to(torch.float32))


def test_layers_are_independent(seed_all, cache):
    block_table = [4]
    k0, v0 = rand_kv(BLOCK_SIZE)
    cache.write(0, slot_mapping_for(block_table, 0, BLOCK_SIZE), k0, v0)
    for layer in range(1, NUM_LAYERS):
        assert torch.all(cache.k_cache[layer] == 0)
        assert torch.all(cache.v_cache[layer] == 0)
