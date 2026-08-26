"""Model execution over the paged KV cache: per-sequence prefill, batched decode, sampling."""

from __future__ import annotations

import torch

from clockwork.config import EngineConfig
from clockwork.engine.attn_metadata import AttentionMetadata
from clockwork.engine.sequence import SamplingParams, Sequence
from clockwork.kernels.attention import resolve_backend
from clockwork.kvcache.kv_cache import PagedKVCache


def _model_head_dim(hf_config) -> int:
    head_dim = getattr(hf_config, "head_dim", None)
    return head_dim or hf_config.hidden_size // hf_config.num_attention_heads


class ModelRunner:
    """Owns the model, the paged KV cache, and token sampling."""

    def __init__(self, cfg: EngineConfig, model=None, hf_config=None, tokenizer=None) -> None:
        self.cfg = cfg
        self.attention_backend = resolve_backend(cfg.attention_backend)
        if model is None:
            from clockwork.engine.loader import load_model

            model, hf_config, tokenizer = load_model(cfg.model)
        elif hf_config is None:
            raise ValueError("hf_config is required when a prebuilt model is injected")
        self.model = model
        self.hf_config = hf_config
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = torch.device(cfg.model.device)
        self.block_size = cfg.cache.block_size
        self.kv_cache = PagedKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=cfg.cache.num_blocks,
            block_size=cfg.cache.block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=_model_head_dim(hf_config),
            dtype=cfg.model.torch_dtype(),
            device=cfg.model.device,
        )
        self._generators: dict[str, torch.Generator] = {}

    def execute(
        self, seqs_to_prefill: list[Sequence], seqs_to_decode: list[Sequence]
    ) -> dict[str, int]:
        """Run scheduled prefills and decodes, append the sampled token to each sequence."""
        sampled: dict[str, int] = {}
        with torch.inference_mode():
            for seq in seqs_to_prefill:
                sampled[seq.seq_id] = self._prefill(seq)
            if seqs_to_decode:
                sampled.update(self._decode(list(seqs_to_decode)))
        return sampled

    def drop(self, seq_id: str) -> None:
        self._generators.pop(seq_id, None)

    def _slots(self, seq: Sequence, start: int, end: int) -> list[int]:
        table = seq.block_table
        size = self.block_size
        return [table[pos // size] * size + pos % size for pos in range(start, end)]

    def _prefill(self, seq: Sequence) -> int:
        # One forward per prefill sequence, no cross-request padding: positions
        # start at num_computed_tokens so a radix-hit prefix is reused from the
        # cache, and paged prefill attends over the full context via the table.
        start = seq.num_computed_tokens
        end = len(seq)
        tokens = seq.token_ids()[start:end]
        metadata = AttentionMetadata(
            is_prefill=True,
            slot_mapping=torch.tensor(
                self._slots(seq, start, end), dtype=torch.int64, device=self.device
            ),
            block_tables=None,
            ctx_lens=torch.tensor([end], dtype=torch.int32, device=self.device),
            query_lens=[end - start],
            seq_block_tables=[list(seq.block_table)],
            logits_indices=torch.tensor([end - start - 1], dtype=torch.int64, device=self.device),
        )
        logits = self.model(
            torch.tensor(tokens, dtype=torch.long, device=self.device),
            torch.arange(start, end, dtype=torch.long, device=self.device),
            self.kv_cache,
            metadata,
        )
        token = self._sample(seq, logits[0])
        seq.num_computed_tokens = end
        seq.append_token(token)
        return token

    def _decode(self, seqs: list[Sequence]) -> dict[str, int]:
        batch = len(seqs)
        input_ids = torch.tensor(
            [seq.token_ids()[-1] for seq in seqs], dtype=torch.long, device=self.device
        )
        positions = torch.tensor(
            [len(seq) - 1 for seq in seqs], dtype=torch.long, device=self.device
        )
        slot_mapping = torch.tensor(
            [self._slots(seq, len(seq) - 1, len(seq))[0] for seq in seqs],
            dtype=torch.int64,
            device=self.device,
        )
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.zeros(batch, max_blocks, dtype=torch.int32, device=self.device)
        for i, seq in enumerate(seqs):
            block_tables[i, : len(seq.block_table)] = torch.tensor(
                seq.block_table, dtype=torch.int32, device=self.device
            )
        metadata = AttentionMetadata(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            ctx_lens=torch.tensor([len(seq) for seq in seqs], dtype=torch.int32),
            query_lens=[1] * batch,
            seq_block_tables=[list(seq.block_table) for seq in seqs],
            logits_indices=torch.arange(batch, dtype=torch.int64, device=self.device),
        )
        logits = self.model(input_ids, positions, self.kv_cache, metadata)
        sampled: dict[str, int] = {}
        for i, seq in enumerate(seqs):
            token = self._sample(seq, logits[i])
            seq.num_computed_tokens += 1
            seq.append_token(token)
            sampled[seq.seq_id] = token
        return sampled

    def _sample(self, seq: Sequence, logits: torch.Tensor) -> int:
        params = seq.sampling_params
        if params.greedy:
            return int(logits.argmax())
        probs = self._probs(logits, params)
        generator = self._generators.get(seq.seq_id)
        if generator is None:
            generator = torch.Generator(device="cpu")
            seed = params.seed if params.seed is not None else self.cfg.model.seed
            generator.manual_seed(seed)
            self._generators[seq.seq_id] = generator
        return int(torch.multinomial(probs, num_samples=1, generator=generator).item())

    @staticmethod
    def _probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        logits = logits.to(torch.float32) / params.temperature
        if 0 < params.top_k < logits.shape[-1]:
            kth = torch.topk(logits, params.top_k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Keep the smallest set whose mass reaches top_p: shift the cutoff
            # one slot right so the token crossing the threshold stays in.
            sorted_removed = cumulative > params.top_p
            sorted_removed[1:] = sorted_removed[:-1].clone()
            sorted_removed[0] = False
            removed = torch.zeros_like(sorted_removed)
            removed[sorted_indices] = sorted_removed
            logits = logits.masked_fill(removed, float("-inf"))
        return torch.softmax(logits, dim=-1)
