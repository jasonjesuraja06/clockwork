import random
from collections import Counter

import pytest

from clockwork.kvcache import BlockAllocator
from clockwork.radix import PrefixMatch, RadixPrefixCache, RadixTree

BLOCK_SIZE = 4
NUM_BLOCKS = 64


@pytest.fixture
def alloc() -> BlockAllocator:
    return BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)


@pytest.fixture
def cache(alloc) -> RadixPrefixCache:
    return RadixPrefixCache(alloc, BLOCK_SIZE)


def tree_prefix_map(tree: RadixTree) -> dict[tuple[int, ...], int]:
    out: dict[tuple[int, ...], int] = {}
    stack = [(tree.root, ())]
    while stack:
        node, prefix = stack.pop()
        for i, block_id in enumerate(node.block_ids):
            out[prefix + tuple(node.token_ids[: (i + 1) * tree.block_size])] = block_id
        full = prefix + tuple(node.token_ids)
        for child in node.children.values():
            stack.append((child, full))
    return out


def test_match_on_empty_tree(cache):
    result = cache.match(list(range(10)))
    assert result == PrefixMatch(0, [])
    assert cache.stats.queries == 1
    assert cache.stats.prompt_tokens == 10
    assert cache.stats.hit_tokens == 0


def test_insert_trims_partial_block():
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    cache = RadixPrefixCache(alloc, 16)
    tokens = list(range(35))
    blocks = alloc.allocate_many(3)
    cache.insert(tokens, blocks)
    assert cache.tree.num_blocks() == 2
    assert alloc.refcount(blocks[0]) == 2
    assert alloc.refcount(blocks[1]) == 2
    assert alloc.refcount(blocks[2]) == 1
    result = cache.match(tokens)
    assert result.num_tokens == 32
    assert result.block_ids == blocks[:2]


def test_never_matches_full_sequence(cache, alloc):
    tokens = list(range(8))
    blocks = alloc.allocate_many(2)
    cache.insert(tokens, blocks)
    result = cache.match(tokens)
    assert result.num_tokens == BLOCK_SIZE
    assert result.block_ids == [blocks[0]]
    result = cache.match(tokens + [99])
    assert result.num_tokens == 8
    assert result.block_ids == blocks
    assert cache.match(tokens[:BLOCK_SIZE]).num_tokens == 0
    assert cache.match([]).num_tokens == 0


def test_match_increfs_before_return(cache, alloc):
    tokens = list(range(8))
    blocks = alloc.allocate_many(2)
    cache.insert(tokens, blocks)
    result = cache.match(tokens + [99])
    for block_id in result.block_ids:
        assert alloc.refcount(block_id) == 3
    cache.release(result.block_ids)
    for block_id in result.block_ids:
        assert alloc.refcount(block_id) == 2


def test_mid_edge_split(cache, alloc):
    seq_a = [0] * 4 + [1] * 4 + [2] * 4
    seq_b = [0] * 4 + [1] * 4 + [3] * 4
    blocks_a = alloc.allocate_many(3)
    blocks_b = alloc.allocate_many(3)
    cache.insert(seq_a, blocks_a)
    cache.insert(seq_b, blocks_b)
    assert cache.tree.num_blocks() == 4
    for block_id in blocks_a:
        assert alloc.refcount(block_id) == 2
    assert alloc.refcount(blocks_b[0]) == 1
    assert alloc.refcount(blocks_b[1]) == 1
    assert alloc.refcount(blocks_b[2]) == 2
    upper = cache.tree.root.children[(0, 0, 0, 0)]
    assert upper.num_blocks() == 2
    assert upper.block_ids == blocks_a[:2]
    assert len(upper.children) == 2
    result = cache.match(seq_a + [9])
    assert (result.num_tokens, result.block_ids) == (12, blocks_a)
    cache.release(result.block_ids)
    result = cache.match(seq_b + [9])
    assert (result.num_tokens, result.block_ids) == (12, [blocks_a[0], blocks_a[1], blocks_b[2]])
    cache.release(result.block_ids)


def test_divergence_within_first_block(cache, alloc):
    seq_a = [1, 0, 0, 0, 7, 7, 7, 7]
    seq_b = [1, 9, 9, 9, 8, 8, 8, 8]
    blocks_a = alloc.allocate_many(2)
    blocks_b = alloc.allocate_many(2)
    cache.insert(seq_a, blocks_a)
    cache.insert(seq_b, blocks_b)
    assert len(cache.tree.root.children) == 2
    result = cache.match(seq_a + [5])
    assert (result.num_tokens, result.block_ids) == (8, blocks_a)
    cache.release(result.block_ids)
    result = cache.match(seq_b + [5])
    assert (result.num_tokens, result.block_ids) == (8, blocks_b)
    cache.release(result.block_ids)


def test_duplicate_insert_keeps_tree_blocks(cache, alloc):
    tokens = list(range(8))
    blocks_a = alloc.allocate_many(2)
    blocks_b = alloc.allocate_many(2)
    cache.insert(tokens, blocks_a)
    cache.insert(tokens, blocks_b)
    for block_id in blocks_b:
        assert alloc.refcount(block_id) == 1
    # The caller keeps sole ownership of its duplicates and frees them itself.
    alloc.free_many(blocks_b)
    result = cache.match(tokens + [99])
    assert result.block_ids == blocks_a
    cache.release(result.block_ids)


def test_lru_eviction_order(cache, alloc):
    seq1, seq2, seq3 = [[t] * 8 for t in (1, 2, 3)]
    blocks1 = alloc.allocate_many(2)
    blocks2 = alloc.allocate_many(2)
    blocks3 = alloc.allocate_many(2)
    cache.insert(seq1, blocks1)
    cache.insert(seq2, blocks2)
    cache.insert(seq3, blocks3)
    result = cache.match(seq1 + [9])
    cache.release(result.block_ids)
    # seq2 is now least recently used; eviction is whole-leaf so both blocks go.
    assert cache.evict(1) == 2
    for block_id in blocks2:
        assert alloc.refcount(block_id) == 1
    for block_id in blocks1 + blocks3:
        assert alloc.refcount(block_id) == 2
    assert cache.evict(1) == 2
    for block_id in blocks3:
        assert alloc.refcount(block_id) == 1
    assert cache.evict(10) == 2
    assert cache.tree.num_blocks() == 0
    assert cache.evict(1) == 0


def test_locked_nodes_survive_eviction(cache, alloc):
    seq1, seq2 = [1] * 8, [2] * 8
    blocks1 = alloc.allocate_many(2)
    blocks2 = alloc.allocate_many(2)
    cache.insert(seq1, blocks1)
    cache.insert(seq2, blocks2)
    hold = cache.match(seq1 + [9])
    assert hold.block_ids == blocks1
    assert cache.tree.evictable_blocks() == 2
    assert cache.evict(100) == 2
    for block_id in blocks2:
        assert alloc.refcount(block_id) == 1
    result = cache.match(seq1 + [9])
    assert result.block_ids == blocks1
    cache.release(result.block_ids)
    cache.release(hold.block_ids)
    assert cache.tree.evictable_blocks() == 2
    assert cache.evict(100) == 2
    assert cache.tree.num_blocks() == 0


def test_lock_follows_blocks_through_split(cache, alloc):
    seq_a = list(range(12))
    blocks_a = alloc.allocate_many(3)
    cache.insert(seq_a, blocks_a)
    hold = cache.match(seq_a)
    assert hold.block_ids == blocks_a[:2]
    seq_b = seq_a[:8] + [50, 51, 52, 53]
    blocks_b = alloc.allocate_many(3)
    cache.insert(seq_b, blocks_b)
    # The insert split the locked node at block 2: the held span stays locked while
    # the two divergent tails remain evictable.
    assert cache.tree.evictable_blocks() == 2
    assert cache.evict(100) == 2
    result = cache.match(seq_a[:8] + [9])
    assert result.block_ids == blocks_a[:2]
    cache.release(result.block_ids)
    cache.release(hold.block_ids)
    assert cache.evict(100) == 2
    alloc.free_many(blocks_a)
    alloc.free_many(blocks_b)
    assert alloc.num_free_blocks == NUM_BLOCKS


def test_hit_rate_math(cache, alloc):
    assert cache.stats.hit_rate == 0.0
    cache.match(list(range(9)))
    assert cache.stats.hit_rate == 0.0
    blocks = alloc.allocate_many(2)
    cache.insert(list(range(8)), blocks)
    cache.match(list(range(9)))
    assert cache.stats.queries == 2
    assert cache.stats.prompt_tokens == 18
    assert cache.stats.hit_tokens == 8
    assert cache.stats.hit_rate == pytest.approx(8 / 18)


def test_disabled_cache(alloc):
    cache = RadixPrefixCache(alloc, BLOCK_SIZE, enabled=False)
    blocks = alloc.allocate_many(2)
    cache.insert(list(range(8)), blocks)
    assert cache.tree.num_blocks() == 0
    for block_id in blocks:
        assert alloc.refcount(block_id) == 1
    result = cache.match(list(range(9)))
    assert result == PrefixMatch(0, [])
    assert cache.stats.queries == 1
    assert cache.stats.prompt_tokens == 9
    assert cache.stats.hit_tokens == 0


def test_refcount_accounting_full_cycle(cache, alloc):
    seq_a = list(range(12))
    blocks_a = alloc.allocate_many(3)
    cache.insert(seq_a, blocks_a)
    hold = cache.match(seq_a)
    assert (hold.num_tokens, hold.block_ids) == (8, blocks_a[:2])
    seq_b = seq_a[:8] + [90, 91, 92, 93]
    blocks_b = alloc.allocate_many(3)
    cache.insert(seq_b, blocks_b)
    alloc.free_many(blocks_a)
    alloc.free_many(blocks_b)
    cache.release(hold.block_ids)
    assert cache.evict(100) == 4
    assert cache.tree.num_blocks() == 0
    assert alloc.num_free_blocks == NUM_BLOCKS


def test_reset_drops_tree_references(cache, alloc):
    blocks = alloc.allocate_many(2)
    cache.insert(list(range(8)), blocks)
    hold = cache.match(list(range(9)))
    cache.release(hold.block_ids)
    cache.reset()
    for block_id in blocks:
        assert alloc.refcount(block_id) == 1
    assert cache.tree.num_blocks() == 0
    assert cache.stats.queries == 0
    assert cache.match(list(range(9))).num_tokens == 0
    alloc.free_many(blocks)


def test_fuzz_against_reference():
    rng = random.Random(20260826)
    block_size = 4
    num_blocks = 160
    alloc = BlockAllocator(num_blocks, block_size)
    cache = RadixPrefixCache(alloc, block_size)
    ref: dict[tuple[int, ...], int] = {}
    seqs: list[tuple[list[int], list[int]]] = []
    holds: list[list[int]] = []
    expected = {"queries": 0, "prompt": 0, "hit": 0}

    def check_refcounts() -> None:
        tree_blocks = set(cache.tree.blocks())
        held = Counter()
        for hold in holds:
            held.update(hold)
        owned = Counter()
        for _, blks in seqs:
            owned.update(blks)
        for block_id in range(num_blocks):
            want = int(block_id in tree_blocks) + held[block_id] + owned[block_id]
            assert alloc.refcount(block_id) == want

    def expected_match(tokens: list[int]) -> tuple[int, list[int]]:
        if not tokens:
            return 0, []
        cap = (len(tokens) - 1) // block_size * block_size
        blocks = []
        i = 1
        while i * block_size <= cap and tuple(tokens[: i * block_size]) in ref:
            blocks.append(ref[tuple(tokens[: i * block_size])])
            i += 1
        return len(blocks) * block_size, blocks

    def random_tokens() -> list[int]:
        if seqs and rng.random() < 0.7:
            base, _ = rng.choice(seqs)
            cut = rng.randrange(len(base) + 1)
            tail = [rng.randrange(4) for _ in range(rng.randrange(13))]
            return base[:cut] + tail
        return [rng.randrange(4) for _ in range(rng.randrange(1, 41))]

    for _ in range(600):
        op = rng.random()
        if op < 0.35:
            tokens = random_tokens() or [0]
            n_full = len(tokens) // block_size
            if (alloc.num_free_blocks < n_full or len(seqs) >= 10) and seqs:
                _, blks = seqs.pop(0)
                alloc.free_many(blks)
            if alloc.num_free_blocks >= n_full:
                blks = alloc.allocate_many(n_full)
                cache.insert(tokens, blks)
                for i in range(n_full):
                    ref.setdefault(tuple(tokens[: (i + 1) * block_size]), blks[i])
                seqs.append((tokens, blks))
        elif op < 0.6:
            tokens = random_tokens() or [0]
            want_tokens, want_blocks = expected_match(tokens)
            result = cache.match(tokens)
            expected["queries"] += 1
            expected["prompt"] += len(tokens)
            expected["hit"] += want_tokens
            assert result.num_tokens == want_tokens
            assert result.block_ids == want_blocks
            if result.block_ids:
                holds.append(list(result.block_ids))
        elif op < 0.75:
            if holds:
                cache.release(holds.pop(rng.randrange(len(holds))))
        elif op < 0.9:
            if seqs:
                _, blks = seqs.pop(rng.randrange(len(seqs)))
                alloc.free_many(blks)
        else:
            before = tree_prefix_map(cache.tree)
            removed = cache.evict(rng.randrange(1, 9))
            after = tree_prefix_map(cache.tree)
            assert set(after.items()) <= set(before.items())
            assert len(before) - len(after) == removed
            surviving = set(after.values())
            for hold in holds:
                assert set(hold) <= surviving
            ref = dict(after)
        check_refcounts()

    assert cache.stats.queries == expected["queries"]
    assert cache.stats.prompt_tokens == expected["prompt"]
    assert cache.stats.hit_tokens == expected["hit"]
    for hold in holds:
        cache.release(hold)
    for _, blks in seqs:
        alloc.free_many(blks)
    cache.evict(10**9)
    assert cache.tree.num_blocks() == 0
    assert alloc.num_free_blocks == num_blocks
