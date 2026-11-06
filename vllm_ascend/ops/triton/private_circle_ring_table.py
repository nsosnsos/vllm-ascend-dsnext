# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side private-circle ring block table construction.

Replaces the CPU-side ``MultiGroupBlockTable.set_private_circle_allocations``
table build (host numpy + per-step H2D) with a single Triton launch that reads
the already-corrected device ``seq_lens`` and the device-resident allocation
IDs, writing directly into the persistent block-table GPU buffer.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "num_cols",
        "window_size",
        "in_flight_tokens",
    ]
)
def _build_private_circle_ring_table_kernel(
    seq_lens_ptr,  # [num_reqs] int32 (device, corrected)
    alloc_ids_ptr,  # [num_reqs] int64 (device mirror)
    block_table_ptr,  # [num_reqs, num_cols] int32 (persistent GPU buffer)
    num_reqs,
    num_cols,
    stride_bt,
    window_size,
    in_flight_tokens,
    BLOCK_SIZE: tl.constexpr,  # physical block size
    BLOCKS_PER_ALLOC: tl.constexpr,  # ring blocks per allocation
    TILE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_reqs:
        return

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int64)
    alloc = tl.load(alloc_ids_ptr + row).to(tl.int64)

    # first_page = max(0, (seq_len - window) // BLOCK_SIZE)
    first_page = tl.maximum((seq_len - window_size) // BLOCK_SIZE, 0)
    # The right edge must cover the DSpark drafter's proposal query block
    # (``effective_seq_len + q_idx``, up to ``seq_len + in_flight - 1`` under
    # full acceptance), resolved against this step's table before it is
    # rebuilt: extend logical_count by in_flight without touching
    # first_page, otherwise the draft tail's slot lookups and win indices
    # resolve to the null block.
    # logical_count = max(1, (seq_len + in_flight - 1) // BLOCK_SIZE + 1)
    logical_count = tl.maximum(
        (seq_len + in_flight_tokens - 1) // BLOCK_SIZE + 1, 1
    )

    base = 1 + alloc * BLOCKS_PER_ALLOC

    for col_start in range(0, num_cols, TILE_SIZE):
        cols = col_start + tl.arange(0, TILE_SIZE)
        mask = cols < num_cols
        in_range = (cols >= first_page) & (cols < logical_count)
        ring_block = cols % BLOCKS_PER_ALLOC
        val = tl.where(in_range, base + ring_block, 0)
        tl.store(block_table_ptr + row * stride_bt + cols, val, mask=mask)


def build_private_circle_ring_table(
    seq_lens: torch.Tensor,
    alloc_ids: torch.Tensor,
    block_table_gpu: torch.Tensor,
    num_reqs: int,
    *,
    window_size: int,
    block_size: int,
    blocks_per_allocation: int,
    in_flight_tokens: int = 1,
) -> None:
    """Build the private-circle ring block table on device.

    ``window_size`` is the ring's visible history span (its capacity) and
    drives ``first_page``. ``in_flight_tokens`` extends ``logical_count``
    past the last scheduled token so a DSpark draft block drafted in the
    same step resolves to real ring blocks instead of the null block. The
    formula:
    ``ids[r,c] = (first_page <= c < logical_count) ? 1 + alloc*P + c%P : 0``
    with ``first_page = max(0, (seq_len - window) // bs)`` and
    ``logical_count = max(1, (seq_len + in_flight - 1) // bs + 1)``.
    Only ``[:num_reqs]`` rows are written; padding rows are handled by
    ``_build_attention_metadata``'s ``fill_(0)``.
    """
    num_cols = block_table_gpu.shape[1]
    if num_reqs == 0 or num_cols == 0:
        return

    tile_size = 128
    grid = (num_reqs,)
    _build_private_circle_ring_table_kernel[grid](
        seq_lens[:num_reqs],
        alloc_ids[:num_reqs],
        block_table_gpu,
        num_reqs,
        num_cols,
        block_table_gpu.stride(0),
        window_size,
        max(1, int(in_flight_tokens)),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ALLOC=blocks_per_allocation,
        TILE_SIZE=tile_size,
    )
