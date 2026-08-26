import pytest
import torch

from clockwork.engine.sequence import Sequence
from clockwork.kvcache import AllocatorOutOfMemory, BlockAllocator, PagedKVCache
from clockwork.kvcache.block_manager import BlockManager

NUM_BLOCKS = 16
BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 1
HEAD_DIM = 2
# watermark 0.25 * 16 blocks = 4 headroom blocks, chosen so thresholds are exact.
WATERMARK = 0.25
WATERMARK_BLOCKS = 4


@pytest.fixture
def alloc() -> BlockAllocator:
    return BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)


@pytest.fixture
def kv() -> PagedKVCache:
    return PagedKVCache(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
    )


@pytest.fixture
def mgr(alloc, kv) -> BlockManager:
    return BlockManager(alloc, kv_cache=kv, watermark=WATERMARK)


def make_seq(seq_id: str, num_tokens: int) -> Sequence:
    return Sequence(seq_id, list(range(num_tokens)))


def prefill(mgr: BlockManager, seq: Sequence) -> None:
    mgr.allocate(seq)
    seq.num_computed_tokens = len(seq)


def decode_step(mgr: BlockManager, seq: Sequence, token: int = 0) -> list[tuple[int, int]]:
    # Engine order: sample and append the token, materialize its slot, compute.
    seq.append_token(token)
    pairs = mgr.append_slots(seq)
    seq.num_computed_tokens += 1
    return pairs


def fork_child(mgr: BlockManager, parent: Sequence, child_id: str) -> Sequence:
    child = Sequence(child_id, parent.token_ids())
    mgr.fork(parent, child)
    child.num_computed_tokens = parent.num_computed_tokens
    return child


def fill_block(kv: PagedKVCache, block_id: int) -> None:
    for layer in range(kv.num_layers):
        kv.k_cache[layer][block_id].fill_(100.0 * layer + block_id)
        kv.v_cache[layer][block_id].fill_(-(100.0 * layer + block_id) - 1.0)


def test_allocate_covers_all_prompt_tokens(mgr, alloc):
    seq = make_seq("a", 9)
    prefill(mgr, seq)
    assert len(seq.block_table) == 3
    assert len(set(seq.block_table)) == 3
    assert alloc.num_free_blocks == NUM_BLOCKS - 3
    for block_id in seq.block_table:
        assert alloc.refcount(block_id) == 1


def test_allocate_skips_radix_prepopulated_blocks(mgr, alloc):
    seq = make_seq("a", 6)
    cached = alloc.allocate()
    seq.block_table = [cached]
    seq.num_computed_tokens = 4
    seq.num_cached_tokens = 4
    mgr.allocate(seq)
    assert len(seq.block_table) == 2
    assert seq.block_table[0] == cached
    assert alloc.num_free_blocks == NUM_BLOCKS - 2


def test_can_allocate_exactly_at_watermark(mgr):
    # 12 blocks needed, 16 free: 16 - 12 == WATERMARK_BLOCKS exactly.
    seq = make_seq("a", 12 * BLOCK_SIZE)
    assert mgr.can_allocate(seq)
    mgr.allocate(seq)
    assert mgr.num_free_blocks() == WATERMARK_BLOCKS


def test_can_allocate_one_below_watermark(mgr):
    # 13 blocks needed, 16 free: 16 - 13 == WATERMARK_BLOCKS - 1.
    seq = make_seq("a", 12 * BLOCK_SIZE + 1)
    assert not mgr.can_allocate(seq)


def test_can_allocate_counts_only_new_blocks(mgr, alloc):
    # 12 free blocks. A fresh 9-block request breaches the watermark, but the
    # same request with 4 blocks already covered by a radix hit needs only 5.
    cached = alloc.allocate_many(4)
    fresh = make_seq("fresh", 9 * BLOCK_SIZE)
    assert not mgr.can_allocate(fresh)
    hit = make_seq("hit", 9 * BLOCK_SIZE)
    hit.block_table = list(cached)
    hit.num_computed_tokens = 4 * BLOCK_SIZE
    hit.num_cached_tokens = 4 * BLOCK_SIZE
    assert mgr.can_allocate(hit)


def test_append_into_partial_block_allocates_nothing(mgr, alloc):
    seq = make_seq("a", 5)
    prefill(mgr, seq)
    free_before = alloc.num_free_blocks
    for position in (5, 6, 7):
        pairs = decode_step(mgr, seq)
        assert pairs == []
        assert seq.num_computed_tokens == position + 1
    assert len(seq.block_table) == 2
    assert alloc.num_free_blocks == free_before


def test_append_at_block_boundary_allocates_one_block(mgr, alloc):
    seq = make_seq("a", 8)
    prefill(mgr, seq)
    assert len(seq.block_table) == 2
    free_before = alloc.num_free_blocks
    pairs = decode_step(mgr, seq)
    assert pairs == []
    assert len(seq.block_table) == 3
    assert alloc.num_free_blocks == free_before - 1
    assert mgr.slot_mapping(seq, 8, 9) == [seq.block_table[2] * BLOCK_SIZE]


def test_cow_on_shared_last_block_copies_data(mgr, alloc, kv):
    a = make_seq("a", 6)
    prefill(mgr, a)
    for block_id in a.block_table:
        fill_block(kv, block_id)
    b = fork_child(mgr, a, "b")
    src_expected = a.block_table[1]
    src_snapshot = [kv.k_cache[layer][src_expected].clone() for layer in range(NUM_LAYERS)]

    pairs = decode_step(mgr, a)
    assert len(pairs) == 1
    src, dst = pairs[0]
    assert src == src_expected
    assert dst != src
    assert a.block_table[1] == dst
    assert b.block_table[1] == src
    assert alloc.refcount(src) == 1
    assert alloc.refcount(dst) == 1
    for layer in range(NUM_LAYERS):
        assert torch.equal(kv.k_cache[layer][dst], kv.k_cache[layer][src])
        assert torch.equal(kv.v_cache[layer][dst], kv.v_cache[layer][src])

    # Writing a's new token into the private copy must not touch the sharer.
    slots = torch.tensor(mgr.slot_mapping(a, 6, 7), dtype=torch.int64)
    payload = torch.full((1, NUM_KV_HEADS, HEAD_DIM), 777.0)
    kv.write(0, slots, payload, payload)
    assert torch.equal(kv.k_cache[0][dst][2], payload[0])
    for layer in range(NUM_LAYERS):
        assert torch.equal(kv.k_cache[layer][src], src_snapshot[layer])


def test_fork_then_divergent_decode_isolates_sequences(mgr, alloc):
    a = make_seq("a", 6)
    prefill(mgr, a)
    b = fork_child(mgr, a, "b")
    assert b.block_table == a.block_table
    for block_id in a.block_table:
        assert alloc.refcount(block_id) == 2

    pairs_a = decode_step(mgr, a, token=10)
    assert len(pairs_a) == 1
    # b's last block became private when a copied out, so b appends in place.
    pairs_b = decode_step(mgr, b, token=20)
    assert pairs_b == []
    assert a.block_table[0] == b.block_table[0]
    assert a.block_table[1] != b.block_table[1]
    assert mgr.slot_mapping(a, 6, 7) != mgr.slot_mapping(b, 6, 7)
    assert alloc.refcount(a.block_table[0]) == 2
    assert alloc.refcount(a.block_table[1]) == 1
    assert alloc.refcount(b.block_table[1]) == 1


def test_free_returns_every_block(mgr, alloc):
    a = make_seq("a", 6)
    prefill(mgr, a)
    b = fork_child(mgr, a, "b")
    decode_step(mgr, a)
    decode_step(mgr, b)
    mgr.free(a)
    assert a.block_table == []
    assert alloc.num_free_blocks < NUM_BLOCKS
    mgr.free(b)
    assert b.block_table == []
    assert alloc.num_free_blocks == NUM_BLOCKS
    for block_id in range(NUM_BLOCKS):
        assert alloc.refcount(block_id) == 0


def test_free_twice_raises(mgr):
    seq = make_seq("a", 5)
    prefill(mgr, seq)
    mgr.free(seq)
    with pytest.raises(ValueError):
        mgr.free(seq)


def test_free_then_reallocate_after_preemption(mgr, alloc):
    seq = make_seq("a", 5)
    prefill(mgr, seq)
    decode_step(mgr, seq)
    mgr.free(seq)
    seq.reset_for_recompute()
    mgr.allocate(seq)
    assert len(seq.block_table) == 2
    mgr.free(seq)
    assert alloc.num_free_blocks == NUM_BLOCKS


def test_append_oom_on_new_block_raises_cleanly(mgr, alloc):
    seq = make_seq("a", NUM_BLOCKS * BLOCK_SIZE)
    prefill(mgr, seq)
    assert alloc.num_free_blocks == 0
    assert not mgr.can_append(seq)
    seq.append_token(0)
    with pytest.raises(AllocatorOutOfMemory):
        mgr.append_slots(seq)
    assert len(seq.block_table) == NUM_BLOCKS
    assert alloc.num_free_blocks == 0


def test_append_oom_on_cow_raises_cleanly(mgr, alloc):
    a = make_seq("a", 6)
    prefill(mgr, a)
    fork_child(mgr, a, "b")
    filler = make_seq("filler", (NUM_BLOCKS - 2) * BLOCK_SIZE)
    prefill(mgr, filler)
    assert alloc.num_free_blocks == 0
    assert not mgr.can_append(a)
    table_before = list(a.block_table)
    a.append_token(0)
    with pytest.raises(AllocatorOutOfMemory):
        mgr.append_slots(a)
    assert a.block_table == table_before
    assert alloc.refcount(a.block_table[1]) == 2


def test_can_append_partial_private_block_needs_no_free_blocks(mgr, alloc):
    seq = make_seq("a", 5)
    prefill(mgr, seq)
    filler = make_seq("filler", (NUM_BLOCKS - 2) * BLOCK_SIZE)
    prefill(mgr, filler)
    assert alloc.num_free_blocks == 0
    assert mgr.can_append(seq)
    for _ in range(3):
        decode_step(mgr, seq)
    assert not mgr.can_append(seq)


def test_slot_mapping_layout(mgr):
    seq = make_seq("a", 6)
    prefill(mgr, seq)
    b0, b1 = seq.block_table
    expected = [b0 * BLOCK_SIZE + i for i in range(4)] + [b1 * BLOCK_SIZE, b1 * BLOCK_SIZE + 1]
    assert mgr.slot_mapping(seq, 0, 6) == expected
    assert mgr.slot_mapping(seq, 4, 6) == expected[4:]
    assert mgr.slot_mapping(seq, 3, 3) == []
    with pytest.raises(ValueError):
        mgr.slot_mapping(seq, 0, 2 * BLOCK_SIZE + 1)
    with pytest.raises(ValueError):
        mgr.slot_mapping(seq, -1, 2)
