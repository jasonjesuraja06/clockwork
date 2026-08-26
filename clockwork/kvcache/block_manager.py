from clockwork.engine.sequence import Sequence
from clockwork.kvcache.block import BlockAllocator
from clockwork.kvcache.kv_cache import PagedKVCache


class BlockManager:
    """Sequence-to-block mapping; owns admission, growth, copy-on-write, and prefix sharing."""

    def __init__(
        self,
        allocator: BlockAllocator,
        kv_cache: PagedKVCache | None = None,
        watermark: float = 0.01,
    ) -> None:
        if watermark < 0:
            raise ValueError(f"watermark must be non-negative, got {watermark}")
        self.allocator = allocator
        self.kv_cache = kv_cache
        self.block_size = allocator.block_size
        self.watermark_blocks = int(watermark * allocator.num_blocks)
        # Sequences that currently hold references through this manager. Guards
        # against double-free, which would corrupt refcounts of shared blocks.
        self._allocated: set[str] = set()

    def _blocks_needed(self, seq: Sequence) -> int:
        # A radix hit may have pre-populated seq.block_table (block-aligned);
        # only the tokens beyond that coverage need new blocks.
        total = (len(seq) + self.block_size - 1) // self.block_size
        return max(0, total - len(seq.block_table))

    def can_allocate(self, seq: Sequence) -> bool:
        # Admission invariant: admitting a request must leave watermark_blocks
        # free on top of the request's own need, so decode of already-running
        # sequences can still append slots. Without this headroom a greedy
        # admission drains the pool and forces immediate preemption.
        return self.allocator.num_free_blocks - self._blocks_needed(seq) >= self.watermark_blocks

    def allocate(self, seq: Sequence) -> None:
        needed = self._blocks_needed(seq)
        if needed:
            seq.block_table.extend(self.allocator.allocate_many(needed))
        self._allocated.add(seq.seq_id)

    def _append_position(self, seq: Sequence) -> int:
        # The next slot to materialize is the first uncomputed token position.
        return seq.num_computed_tokens

    def can_append(self, seq: Sequence) -> bool:
        logical = self._append_position(seq) // self.block_size
        if logical >= len(seq.block_table):
            return self.allocator.num_free_blocks >= 1
        if self.allocator.is_shared(seq.block_table[logical]):
            return self.allocator.num_free_blocks >= 1
        return True

    def append_slots(self, seq: Sequence) -> list[tuple[int, int]]:
        position = self._append_position(seq)
        logical = position // self.block_size
        table = seq.block_table
        if logical > len(table):
            raise ValueError(
                f"sequence {seq.seq_id}: position {position} is beyond the block table"
            )
        if logical == len(table):
            table.append(self.allocator.allocate())
            return []
        block_id = table[logical]
        cow = self.allocator.copy_on_write(block_id)
        if cow is None:
            return []
        src, dst = cow
        if self.kv_cache is not None:
            self.kv_cache.copy_block(src, dst)
        table[logical] = dst
        return [(src, dst)]

    def fork(self, parent: Sequence, child: Sequence) -> None:
        if child.block_table:
            raise ValueError(f"child {child.seq_id} already has blocks")
        for block_id in parent.block_table:
            self.allocator.incref(block_id)
        child.block_table = list(parent.block_table)
        self._allocated.add(child.seq_id)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self._allocated:
            raise ValueError(f"sequence {seq.seq_id} holds no blocks (double free?)")
        self._allocated.remove(seq.seq_id)
        self.allocator.free_many(seq.block_table)
        seq.block_table.clear()

    def slot_mapping(self, seq: Sequence, start: int, end: int) -> list[int]:
        covered = len(seq.block_table) * self.block_size
        if not 0 <= start <= end <= covered:
            raise ValueError(f"token range [{start}, {end}) outside covered range [0, {covered})")
        table = seq.block_table
        size = self.block_size
        return [table[pos // size] * size + pos % size for pos in range(start, end)]

    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks
