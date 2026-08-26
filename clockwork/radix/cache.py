from __future__ import annotations

from dataclasses import dataclass, field

from clockwork.kvcache.block import BlockAllocator
from clockwork.radix.tree import RadixTree


@dataclass
class RadixCacheStats:
    queries: int = 0
    hit_tokens: int = 0
    prompt_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.hit_tokens / self.prompt_tokens


@dataclass
class PrefixMatch:
    num_tokens: int
    block_ids: list[int] = field(default_factory=list)


class RadixPrefixCache:
    """Block-aligned prefix cache holding one allocator reference per stored KV block."""

    def __init__(self, allocator: BlockAllocator, block_size: int, enabled: bool = True) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.enabled = enabled
        self.stats = RadixCacheStats()
        self.tree = RadixTree(block_size)

    def match(self, token_ids: list[int]) -> PrefixMatch:
        self.stats.queries += 1
        self.stats.prompt_tokens += len(token_ids)
        if not token_ids:
            return PrefixMatch(0)
        # Cap at the largest block multiple strictly below len(token_ids): at least one
        # token is always left to prefill so the model has a query to score.
        cap = (len(token_ids) - 1) // self.block_size * self.block_size
        if not self.enabled or cap == 0:
            return PrefixMatch(0)
        matched, blocks = self.tree.match_prefix(token_ids[:cap])
        # Incref before returning so the blocks cannot be evicted to zero while the
        # sequence runs; the lock keeps the matched path out of LRU eviction entirely.
        for block_id in blocks:
            self.allocator.incref(block_id)
        if blocks:
            self.tree.lock(blocks[-1])
        self.stats.hit_tokens += matched
        return PrefixMatch(matched, blocks)

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None:
        if not self.enabled:
            return
        num_full = min(len(token_ids) // self.block_size, len(block_ids))
        if num_full == 0:
            return
        new_blocks = self.tree.insert(token_ids[: num_full * self.block_size], block_ids[:num_full])
        # Ownership rule: the tree adopts and increfs only the blocks it newly stored
        # (always the last new_blocks of the inserted span). For spans already cached
        # the tree keeps its existing blocks and the caller keeps sole ownership of its
        # duplicates, freeing them itself when it frees the sequence.
        for block_id in block_ids[num_full - new_blocks : num_full]:
            self.allocator.incref(block_id)

    def release(self, block_ids: list[int]) -> None:
        if not block_ids:
            return
        self.tree.unlock(block_ids[-1])
        for block_id in block_ids:
            self.allocator.decref(block_id)

    def evict(self, num_blocks: int) -> int:
        removed = self.tree.evict(num_blocks)
        for block_id in removed:
            self.allocator.decref(block_id)
        return len(removed)

    def reset(self) -> None:
        for block_id in self.tree.blocks():
            self.allocator.decref(block_id)
        self.tree = RadixTree(self.block_size)
        self.stats = RadixCacheStats()
