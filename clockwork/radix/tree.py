from __future__ import annotations


class RadixNode:
    __slots__ = ("parent", "children", "token_ids", "block_ids", "last_access", "lock_count")

    def __init__(
        self,
        parent: RadixNode | None = None,
        token_ids: list[int] | None = None,
        block_ids: list[int] | None = None,
    ) -> None:
        self.parent = parent
        # Keyed by the tuple of the child's first block of tokens. Keying by the first
        # token alone cannot separate two branches that share a first token but diverge
        # inside the first block, and matching is per whole block anyway.
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.token_ids: list[int] = token_ids if token_ids is not None else []
        self.block_ids: list[int] = block_ids if block_ids is not None else []
        self.last_access = 0
        self.lock_count = 0

    def num_blocks(self) -> int:
        return len(self.block_ids)


class RadixTree:
    """Token-id radix tree over block-aligned KV spans."""

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self.root = RadixNode()
        self._clock = 0
        self._block_to_node: dict[int, RadixNode] = {}
        # Lock counts per locked block id: node.lock_count is the sum over the node's
        # blocks, and a split re-derives each half's count from this map.
        self._locks: dict[int, int] = {}

    def num_blocks(self) -> int:
        return len(self._block_to_node)

    def blocks(self) -> list[int]:
        return list(self._block_to_node)

    def match_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        self._clock += 1
        bs = self.block_size
        max_blocks = len(token_ids) // bs
        node = self.root
        blocks: list[int] = []
        while len(blocks) < max_blocks:
            pos = len(blocks) * bs
            child = node.children.get(tuple(token_ids[pos : pos + bs]))
            if child is None:
                break
            child.last_access = self._clock
            blocks.append(child.block_ids[0])
            k = 1
            while k < child.num_blocks() and len(blocks) < max_blocks:
                start = len(blocks) * bs
                if child.token_ids[k * bs : (k + 1) * bs] != list(token_ids[start : start + bs]):
                    break
                blocks.append(child.block_ids[k])
                k += 1
            if k < child.num_blocks():
                # The edge diverged (or the query ran out) mid-node: cannot descend.
                break
            node = child
        return len(blocks) * bs, blocks

    def insert(self, token_ids: list[int], block_ids: list[int]) -> int:
        self._clock += 1
        bs = self.block_size
        n = min(len(token_ids) // bs, len(block_ids))
        node = self.root
        bidx = 0
        new_blocks = 0
        while bidx < n:
            pos = bidx * bs
            key = tuple(token_ids[pos : pos + bs])
            child = node.children.get(key)
            if child is None:
                leaf = RadixNode(
                    parent=node,
                    token_ids=list(token_ids[pos : n * bs]),
                    block_ids=list(block_ids[bidx:n]),
                )
                leaf.last_access = self._clock
                node.children[key] = leaf
                for block_id in leaf.block_ids:
                    self._block_to_node[block_id] = leaf
                new_blocks += n - bidx
                break
            child.last_access = self._clock
            # The first block matched via the key; extend the whole-block match.
            k = 1
            while (
                k < child.num_blocks()
                and bidx + k < n
                and child.token_ids[k * bs : (k + 1) * bs]
                == list(token_ids[pos + k * bs : pos + (k + 1) * bs])
            ):
                k += 1
            node = self._split(child, k) if k < child.num_blocks() else child
            bidx += k
        return new_blocks

    def _split(self, node: RadixNode, k: int) -> RadixNode:
        # Split at the block boundary after k blocks. The upper node takes the shared
        # prefix; the existing node keeps the tail and its children. Locks follow the
        # blocks they were taken on.
        bs = self.block_size
        parent = node.parent
        assert parent is not None and 0 < k < node.num_blocks()
        upper = RadixNode(
            parent=parent,
            token_ids=node.token_ids[: k * bs],
            block_ids=node.block_ids[:k],
        )
        upper.last_access = node.last_access
        parent.children[tuple(upper.token_ids[:bs])] = upper
        node.token_ids = node.token_ids[k * bs :]
        node.block_ids = node.block_ids[k:]
        node.parent = upper
        upper.children[tuple(node.token_ids[:bs])] = node
        for block_id in upper.block_ids:
            self._block_to_node[block_id] = upper
        upper.lock_count = sum(self._locks.get(block_id, 0) for block_id in upper.block_ids)
        node.lock_count -= upper.lock_count
        return upper

    def lock(self, block_id: int) -> None:
        self._locks[block_id] = self._locks.get(block_id, 0) + 1
        self._block_to_node[block_id].lock_count += 1

    def unlock(self, block_id: int) -> None:
        count = self._locks.get(block_id)
        if count is None:
            return
        if count == 1:
            del self._locks[block_id]
        else:
            self._locks[block_id] = count - 1
        node = self._block_to_node.get(block_id)
        if node is not None:
            node.lock_count -= 1

    def evictable_blocks(self) -> int:
        # Blocks not on the path to any locked node; evict cascades leaf by leaf, so a
        # node is reclaimable exactly when neither it nor any descendant is locked.
        def walk(node: RadixNode) -> tuple[bool, int]:
            locked = node.lock_count > 0
            count = 0
            for child in node.children.values():
                child_locked, child_count = walk(child)
                locked = locked or child_locked
                count += child_count
            if not locked:
                count += node.num_blocks()
            return locked, count

        return walk(self.root)[1]

    def evict(self, num_blocks: int) -> list[int]:
        evicted: list[int] = []
        while len(evicted) < num_blocks:
            leaves: list[RadixNode] = []
            stack = [self.root]
            while stack:
                node = stack.pop()
                if node.children:
                    stack.extend(node.children.values())
                elif node is not self.root and node.lock_count == 0:
                    leaves.append(node)
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            parent = victim.parent
            assert parent is not None
            del parent.children[tuple(victim.token_ids[: self.block_size])]
            for block_id in victim.block_ids:
                del self._block_to_node[block_id]
            # Whole-node granularity: the returned list may exceed num_blocks.
            evicted.extend(victim.block_ids)
        return evicted
