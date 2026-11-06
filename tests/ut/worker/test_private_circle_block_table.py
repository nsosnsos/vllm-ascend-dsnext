from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.block_table import MultiGroupBlockTable


class _FakeBlockTable:
    def __init__(self, *, is_private: bool, rows: int = 2, columns: int = 64):
        self.is_private_circle_group = is_private
        self.is_mamba_group = False
        self.physical_block_size = 32
        self.block_table = SimpleNamespace(
            np=np.zeros((rows, columns), dtype=np.int32),
            gpu=torch.zeros((rows, columns), dtype=torch.int32),
        )
        self.num_blocks_per_row = np.zeros(rows, dtype=np.int64)
        self.slot_mapping = SimpleNamespace(
            gpu=torch.arange(8, dtype=torch.int32)
        )


def _new_multi_group_table(
    *block_tables: _FakeBlockTable,
) -> MultiGroupBlockTable:
    table = MultiGroupBlockTable.__new__(MultiGroupBlockTable)
    table.block_tables = list(block_tables)
    table.private_circle_allocation_ids = torch.full(
        (2,), -1, dtype=torch.int64, device="cpu"
    )
    table.private_circle_blocks_per_allocation = 0
    table.private_circle_window_size = 0
    table.private_circle_ring_window_tokens = 0
    table.private_circle_ring_in_flight_tokens = 1
    table.private_circle_allocation_ids_gpu = None
    table._private_circle_installed_this_step = False
    return table


def test_private_circle_layout_accepts_highest_pool_handle():
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(private)

    table.set_private_circle_allocations(
        allocation_ids=np.array([7], dtype=np.int64),
        blocks_per_allocation=5,
        num_allocations=8,
        private_num_blocks=41,
        window_size=128,
        ring_window_tokens=133,
        ring_in_flight_tokens=6,
    )

    assert table.private_circle_blocks_per_allocation == 5
    assert table.private_circle_window_size == 128
    assert table.private_circle_ring_window_tokens == 133
    assert table.private_circle_ring_in_flight_tokens == 6
    assert table._private_circle_installed_this_step
    assert table.private_circle_allocation_ids.tolist() == [7, -1]
    assert table.private_circle_allocation_ids_gpu.tolist() == [7, -1]

    with pytest.raises(IndexError, match="handle is out of range"):
        table.set_private_circle_allocations(
            allocation_ids=np.array([8], dtype=np.int64),
            blocks_per_allocation=5,
            num_allocations=8,
            private_num_blocks=41,
        )


def test_private_circle_ring_window_tokens_defaults_to_window_size():
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(private)

    table.set_private_circle_allocations(
        allocation_ids=np.array([0], dtype=np.int64),
        blocks_per_allocation=2,
        num_allocations=4,
        private_num_blocks=9,
        window_size=128,
    )

    assert table.private_circle_ring_window_tokens == 128
    assert table.private_circle_ring_in_flight_tokens == 1


def test_private_circle_layout_rejects_inconsistent_pool_capacity():
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(private)

    with pytest.raises(ValueError, match="block count does not match"):
        table.set_private_circle_allocations(
            allocation_ids=np.array([0], dtype=np.int64),
            blocks_per_allocation=5,
            num_allocations=8,
            private_num_blocks=21,
        )


def _reference_ring_table(
    seq_len,
    ring_window_tokens,
    block_size,
    blocks_per_allocation,
    allocation,
    num_cols,
    in_flight=1,
):
    """Numpy mirror of ``_build_private_circle_ring_table_kernel``:
    ``ids[c] = (first_page <= c < logical_count) ? 1 + alloc*P + c%P : 0``
    with ``first_page = max(0, (seq_len - ring_window_tokens) // block_size)``
    and ``logical_count = max(1, (seq_len + in_flight - 1) // block_size + 1)``.
    The model runner installs ``ring_window_tokens = window + in_flight - 1``
    (the ring capacity) and ``in_flight = 1 + num_speculative_tokens``.
    """
    first_page = max(0, (seq_len - ring_window_tokens) // block_size)
    logical_count = max(1, (seq_len + in_flight - 1) // block_size + 1)
    base = 1 + allocation * blocks_per_allocation
    return [
        base + (c % blocks_per_allocation)
        if first_page <= c < logical_count
        else 0
        for c in range(num_cols)
    ]


def _reader_positions(seq_len, window, in_flight):
    """Absolute token positions any reader resolves through this step's
    ring table (``seq_len`` includes the ``in_flight`` verify tokens):

    - target causal windows ``[p - window + 1, p]`` for the verify queries
      ``p`` in ``[seq_len - in_flight, seq_len - 1]``;
    - DSpark proposal block windows ``[start - window, start + in_flight - 1]``
      for every acceptance outcome ``a`` with ``start = anchor + a + 1``:
      the proposal runs at the end of this step against this table
      (``_build_dspark_win_indices`` over the rewritten cad).
    """
    tokens = set()
    anchor = seq_len - in_flight
    for p in range(anchor, seq_len):
        tokens.update(range(p - window + 1, p + 1))
    for a in range(in_flight):
        start = anchor + a + 1
        tokens.update(range(start - window, start + in_flight))
    return tokens


@pytest.mark.parametrize(
    "window,in_flight,block_size,blocks_per_allocation",
    [
        (128, 6, 128, 2),
        (128, 1, 128, 2),
        (128, 6, 128, 3),
    ],
)
def test_ring_window_covers_all_reader_windows(
    window, in_flight, block_size, blocks_per_allocation
):
    """Every token any reader references must resolve to a non-nulled
    table column, for every seq_len residue."""
    ring_window_tokens = window + in_flight - 1
    for seq_len in range(0, 4 * window):
        num_cols = (seq_len + in_flight - 1) // block_size + 2
        ids = _reference_ring_table(
            seq_len,
            ring_window_tokens,
            block_size,
            blocks_per_allocation,
            allocation=0,
            num_cols=num_cols,
            in_flight=in_flight,
        )
        for token in _reader_positions(seq_len, window, in_flight):
            if token < 0:
                continue  # -1-masked before the kernel
            assert ids[token // block_size] != 0, (
                f"seq_len={seq_len} token={token} falls in a nulled column"
            )


def test_ring_window_one_short_nulls_reader_front_token():
    """Guards the left bound: with ``window + in_flight - 2`` the reader
    front token (``seq_len - window - in_flight + 1``, shared by the target
    causal window front and the worst-case proposal block front) is nulled
    whenever it is the last slot of its block; the runner's exact
    ``window + in_flight - 1`` covers it."""
    window, in_flight, block_size, blocks_per_allocation = 128, 6, 128, 2
    seq_len = 2 * block_size + 4
    num_cols = (seq_len + in_flight - 1) // block_size + 2
    front = seq_len - (window + in_flight - 1)
    assert front % block_size == block_size - 1

    short = _reference_ring_table(
        seq_len,
        window + in_flight - 2,
        block_size,
        blocks_per_allocation,
        allocation=0,
        num_cols=num_cols,
        in_flight=in_flight,
    )
    assert short[front // block_size] == 0

    exact = _reference_ring_table(
        seq_len,
        window + in_flight - 1,
        block_size,
        blocks_per_allocation,
        allocation=0,
        num_cols=num_cols,
        in_flight=in_flight,
    )
    assert exact[front // block_size] != 0


@pytest.mark.parametrize(
    "in_flight,block_size,blocks_per_allocation",
    [
        (6, 128, 2),
        (6, 128, 3),
        (1, 128, 2),
    ],
)
def test_ring_table_covers_draft_proposal_query_tail(
    in_flight, block_size, blocks_per_allocation
):
    """The drafter resolves its proposal query block
    (``effective_seq_len + q_idx``, worst case ``[seq_len, seq_len + q - 1]``
    under full acceptance) through this step's ring table before it is
    rebuilt: every tail position must resolve to a non-nulled column."""
    window = 128
    ring_window_tokens = window + in_flight - 1
    for seq_len in range(0, 4 * window):
        num_cols = (seq_len + in_flight - 1) // block_size + 2
        ids = _reference_ring_table(
            seq_len,
            ring_window_tokens,
            block_size,
            blocks_per_allocation,
            allocation=0,
            num_cols=num_cols,
            in_flight=in_flight,
        )
        for token in range(seq_len, seq_len + in_flight):
            assert ids[token // block_size] != 0, (
                f"seq_len={seq_len} tail token={token} falls in a nulled column"
            )


def test_ring_table_short_on_right_edge_nulls_draft_query_tail():
    """Guards the right-edge extension: without the ``in_flight`` term in
    ``logical_count`` the draft query tail is nulled whenever a block
    boundary falls inside ``[seq_len, seq_len + q - 1]``; the extended
    count covers it."""
    window, in_flight, block_size, blocks_per_allocation = 128, 6, 128, 2
    seq_len = 3 * block_size
    num_cols = (seq_len + in_flight - 1) // block_size + 2

    short = _reference_ring_table(
        seq_len,
        window + in_flight - 1,
        block_size,
        blocks_per_allocation,
        allocation=0,
        num_cols=num_cols,
        in_flight=0,
    )
    assert all(
        short[t // block_size] == 0 for t in range(seq_len, seq_len + in_flight)
    )

    fixed = _reference_ring_table(
        seq_len,
        window + in_flight - 1,
        block_size,
        blocks_per_allocation,
        allocation=0,
        num_cols=num_cols,
        in_flight=in_flight,
    )
    assert all(
        fixed[t // block_size] != 0 for t in range(seq_len, seq_len + in_flight)
    )


def test_bounded_replay_masks_shared_slots_but_keeps_private_ring_slots():
    shared = _FakeBlockTable(is_private=False)
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(shared, private)
    positions = torch.tensor([127, 128, 200, 31], dtype=torch.int64)
    request_indices = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    persistent_starts = torch.tensor([128, -1], dtype=torch.int64)
    private_before = private.slot_mapping.gpu.clone()

    table.mask_shared_slots_for_private_circle_bounded_replay(
        positions,
        request_indices,
        persistent_starts,
    )

    assert shared.slot_mapping.gpu[:4].tolist() == [-1, 1, 2, 3]
    assert torch.equal(private.slot_mapping.gpu, private_before)
