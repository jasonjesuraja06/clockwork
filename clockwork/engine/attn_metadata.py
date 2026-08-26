"""Per-batch attention metadata passed from the model runner into the attention layers."""

from dataclasses import dataclass

import torch


@dataclass
class AttentionMetadata:
    is_prefill: bool
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor | None
    ctx_lens: torch.Tensor | None
    query_lens: list[int]
    seq_block_tables: list[list[int]]
    logits_indices: torch.Tensor
