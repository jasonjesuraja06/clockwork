import random

import pytest

from clockwork.kvcache import AllocatorOutOfMemory, BlockAllocator

NUM_BLOCKS = 16
BLOCK_SIZE = 4


@pytest.fixture
def alloc() -> BlockAllocator:
    return BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)


def test_allocate_returns_each_free_block_once(alloc):
    ids = [alloc.allocate() for _ in range(NUM_BLOCKS)]
    assert sorted(ids) == list(range(NUM_BLOCKS))
    assert alloc.num_free_blocks == 0
    for block_id in ids:
        assert alloc.refcount(block_id) == 1


def test_allocate_when_empty_raises(alloc):
    for _ in range(NUM_BLOCKS):
        alloc.allocate()
    with pytest.raises(AllocatorOutOfMemory):
        alloc.allocate()


def test_free_returns_block_to_pool(alloc):
    block_id = alloc.allocate()
    assert alloc.num_free_blocks == NUM_BLOCKS - 1
    alloc.free(block_id)
    assert alloc.num_free_blocks == NUM_BLOCKS
    assert alloc.refcount(block_id) == 0


def test_double_free_raises(alloc):
    block_id = alloc.allocate()
    alloc.free(block_id)
    with pytest.raises(ValueError):
        alloc.free(block_id)


def test_decref_below_zero_raises(alloc):
    block_id = alloc.allocate()
    assert alloc.decref(block_id) == 0
    with pytest.raises(ValueError):
        alloc.decref(block_id)


def test_incref_decref_on_unallocated_raise(alloc):
    with pytest.raises(ValueError):
        alloc.incref(3)
    with pytest.raises(ValueError):
        alloc.decref(3)
    with pytest.raises(ValueError):
        alloc.incref(NUM_BLOCKS)
    with pytest.raises(ValueError):
        alloc.decref(-1)


def test_refcount_out_of_range_raises(alloc):
    with pytest.raises(ValueError):
        alloc.refcount(NUM_BLOCKS)


def test_incref_and_decref_track_sharing(alloc):
    block_id = alloc.allocate()
    assert not alloc.is_shared(block_id)
    assert alloc.incref(block_id) == 2
    assert alloc.is_shared(block_id)
    assert alloc.decref(block_id) == 1
    assert not alloc.is_shared(block_id)
    assert alloc.num_free_blocks == NUM_BLOCKS - 1


def test_allocate_many(alloc):
    ids = alloc.allocate_many(5)
    assert len(ids) == len(set(ids)) == 5
    assert alloc.num_free_blocks == NUM_BLOCKS - 5
    for block_id in ids:
        assert alloc.refcount(block_id) == 1


def test_allocate_many_is_all_or_nothing(alloc):
    alloc.allocate_many(NUM_BLOCKS - 3)
    with pytest.raises(AllocatorOutOfMemory):
        alloc.allocate_many(4)
    assert alloc.num_free_blocks == 3
    ids = alloc.allocate_many(3)
    assert len(ids) == 3
    assert alloc.num_free_blocks == 0


def test_free_many(alloc):
    ids = alloc.allocate_many(6)
    alloc.free_many(ids)
    assert alloc.num_free_blocks == NUM_BLOCKS


def test_copy_on_write_on_shared_block(alloc):
    src = alloc.allocate()
    alloc.incref(src)
    free_before = alloc.num_free_blocks
    result = alloc.copy_on_write(src)
    assert result is not None
    got_src, dst = result
    assert got_src == src
    assert dst != src
    assert alloc.refcount(src) == 1
    assert alloc.refcount(dst) == 1
    assert alloc.num_free_blocks == free_before - 1


def test_copy_on_write_on_private_block_returns_none(alloc):
    block_id = alloc.allocate()
    free_before = alloc.num_free_blocks
    assert alloc.copy_on_write(block_id) is None
    assert alloc.refcount(block_id) == 1
    assert alloc.num_free_blocks == free_before


def test_copy_on_write_on_free_block_raises(alloc):
    with pytest.raises(ValueError):
        alloc.copy_on_write(0)
    block_id = alloc.allocate()
    alloc.free(block_id)
    with pytest.raises(ValueError):
        alloc.copy_on_write(block_id)


def test_copy_on_write_when_pool_exhausted_raises(alloc):
    ids = alloc.allocate_many(NUM_BLOCKS)
    alloc.incref(ids[0])
    with pytest.raises(AllocatorOutOfMemory):
        alloc.copy_on_write(ids[0])
    # OOM must leave the shared block untouched.
    assert alloc.refcount(ids[0]) == 2


def test_reset(alloc):
    ids = alloc.allocate_many(NUM_BLOCKS)
    alloc.incref(ids[0])
    alloc.reset()
    assert alloc.num_free_blocks == NUM_BLOCKS
    for block_id in range(NUM_BLOCKS):
        assert alloc.refcount(block_id) == 0


def test_fuzz_free_list_discipline():
    rng = random.Random(1234)
    num_blocks = 32
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    # Model of expected state: block_id -> refcount for live blocks.
    live: dict[int, int] = {}

    def check_invariants():
        assert alloc.num_free_blocks == num_blocks - len(live)
        for block_id, count in live.items():
            assert 0 <= block_id < num_blocks
            assert alloc.refcount(block_id) == count

    for _ in range(600):
        op = rng.choice(["alloc", "alloc_many", "free", "incref", "decref", "cow"])
        if op == "alloc":
            if len(live) == num_blocks:
                with pytest.raises(AllocatorOutOfMemory):
                    alloc.allocate()
            else:
                block_id = alloc.allocate()
                assert block_id not in live
                live[block_id] = 1
        elif op == "alloc_many":
            n = rng.randint(0, 6)
            free_before = alloc.num_free_blocks
            if n > free_before:
                with pytest.raises(AllocatorOutOfMemory):
                    alloc.allocate_many(n)
                assert alloc.num_free_blocks == free_before
            else:
                ids = alloc.allocate_many(n)
                assert len(ids) == n
                for block_id in ids:
                    assert block_id not in live
                    live[block_id] = 1
        elif op == "free" and live:
            block_id = rng.choice(sorted(live))
            alloc.free(block_id)
            live[block_id] -= 1
            if live[block_id] == 0:
                del live[block_id]
        elif op == "incref" and live:
            block_id = rng.choice(sorted(live))
            alloc.incref(block_id)
            live[block_id] += 1
        elif op == "decref" and live:
            block_id = rng.choice(sorted(live))
            alloc.decref(block_id)
            live[block_id] -= 1
            if live[block_id] == 0:
                del live[block_id]
        elif op == "cow" and live:
            block_id = rng.choice(sorted(live))
            shared = live[block_id] > 1
            if shared and len(live) == num_blocks:
                with pytest.raises(AllocatorOutOfMemory):
                    alloc.copy_on_write(block_id)
            else:
                result = alloc.copy_on_write(block_id)
                if shared:
                    src, dst = result
                    assert src == block_id
                    assert dst not in live
                    live[block_id] -= 1
                    live[dst] = 1
                else:
                    assert result is None
        check_invariants()
