import torch


class PagedKVCache:
    """Physical paged K/V tensors per layer, [num_blocks, block_size, num_kv_heads, head_dim]."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]

    def copy_block(self, src: int, dst: int) -> None:
        for layer in range(self.num_layers):
            self.k_cache[layer][dst].copy_(self.k_cache[layer][src])
            self.v_cache[layer][dst].copy_(self.v_cache[layer][src])

    def write(
        self, layer: int, slot_mapping: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        # Flat view [num_blocks * block_size, num_kv_heads, head_dim] shares storage
        # with the cache, so index_copy_ scatters in place.
        flat_k = self.k_cache[layer].view(-1, self.num_kv_heads, self.head_dim)
        flat_v = self.v_cache[layer].view(-1, self.num_kv_heads, self.head_dim)
        flat_k.index_copy_(0, slot_mapping, k.to(self.dtype))
        flat_v.index_copy_(0, slot_mapping, v.to(self.dtype))

    def gather(
        self, layer: int, block_table: list[int], ctx_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_blocks_used = (ctx_len + self.block_size - 1) // self.block_size
        table = torch.tensor(block_table[:num_blocks_used], dtype=torch.int64, device=self.device)
        offsets = torch.arange(self.block_size, dtype=torch.int64, device=self.device)
        slots = (table[:, None] * self.block_size + offsets[None, :]).reshape(-1)[:ctx_len]
        flat_k = self.k_cache[layer].view(-1, self.num_kv_heads, self.head_dim)
        flat_v = self.v_cache[layer].view(-1, self.num_kv_heads, self.head_dim)
        return flat_k.index_select(0, slots), flat_v.index_select(0, slots)
