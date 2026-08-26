class AllocatorOutOfMemory(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation."""


class BlockAllocator:
    """Refcounted physical block pool with copy-on-write."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._refcounts = [0] * num_blocks
        # LIFO free list: recently freed blocks are reused first.
        self._free = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"block id {block_id} out of range [0, {self.num_blocks})")

    def _check_allocated(self, block_id: int) -> None:
        self._check_id(block_id)
        if self._refcounts[block_id] == 0:
            raise ValueError(f"block {block_id} is not allocated")

    def allocate(self) -> int:
        if not self._free:
            raise AllocatorOutOfMemory("no free blocks")
        block_id = self._free.pop()
        self._refcounts[block_id] = 1
        return block_id

    def allocate_many(self, n: int) -> list[int]:
        if n < 0:
            raise ValueError(f"cannot allocate {n} blocks")
        if n > len(self._free):
            # All-or-nothing: nothing is popped before this check.
            raise AllocatorOutOfMemory(f"requested {n} blocks, {len(self._free)} free")
        return [self.allocate() for _ in range(n)]

    def incref(self, block_id: int) -> int:
        self._check_allocated(block_id)
        self._refcounts[block_id] += 1
        return self._refcounts[block_id]

    def decref(self, block_id: int) -> int:
        self._check_allocated(block_id)
        self._refcounts[block_id] -= 1
        if self._refcounts[block_id] == 0:
            self._free.append(block_id)
        return self._refcounts[block_id]

    def free(self, block_id: int) -> None:
        self.decref(block_id)

    def free_many(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            self.free(block_id)

    def refcount(self, block_id: int) -> int:
        self._check_id(block_id)
        return self._refcounts[block_id]

    def is_shared(self, block_id: int) -> bool:
        self._check_id(block_id)
        return self._refcounts[block_id] > 1

    def copy_on_write(self, block_id: int) -> tuple[int, int] | None:
        self._check_allocated(block_id)
        if self._refcounts[block_id] == 1:
            return None
        # Allocate before decref so an OOM leaves the shared block untouched.
        dst = self.allocate()
        self.decref(block_id)
        return (block_id, dst)

    def reset(self) -> None:
        self._refcounts = [0] * self.num_blocks
        self._free = list(range(self.num_blocks - 1, -1, -1))
