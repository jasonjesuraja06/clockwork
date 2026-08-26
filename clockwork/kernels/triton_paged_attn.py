"""Triton paged-attention decode kernel seam; imports cleanly without Triton."""

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def triton_paged_attention_decode(q, k_cache, v_cache, block_tables, ctx_lens, scale):
    """Decode attention on the Triton backend; requires CUDA and an installed Triton."""
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed; use resolve_backend('torch') and paged_attention_decode"
        )
    raise NotImplementedError("Triton decode kernel lands in the kernel phase")
