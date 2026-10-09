import torch
import triton
import triton.language as tl


@triton.jit
def kv_append_kernel(
    k_new_ptr, v_new_ptr, k_cache_ptr, v_cache_ptr,
    pos, D,
    BLOCK_SIZE: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < D
    k_new_pos = tl.load(k_new_ptr + offset, mask=mask, other=0.0)
    v_new_pos = tl.load(v_new_ptr + offset, mask=mask)
    tl.store(k_cache_ptr + pos * D + offset, k_new_pos, mask=mask)
    tl.store(v_cache_ptr + pos * D + offset, v_new_pos, mask=mask)


def solve(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: int,
) -> None:
    """Write k_new and v_new into row `pos` of k_cache and v_cache in place."""
    D = k_new.numel()
    BLOCK_SIZE = 1024
    grid = ((D + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kv_append_kernel[grid](
        k_new, v_new, k_cache, v_cache,
        pos, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )