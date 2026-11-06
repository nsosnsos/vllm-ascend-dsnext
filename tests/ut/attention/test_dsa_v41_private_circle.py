# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards for the private-circle mixed-batch positional split."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41EagerAttentionImpl,
    _assert_decode_rows_first,
    _validate_batch_layout,
)


def _common(prefill_flags):
    return SimpleNamespace(
        is_prefilling=torch.tensor(prefill_flags, dtype=torch.bool),
    )


def test_guard_accepts_decode_rows_before_prefill_rows():
    _assert_decode_rows_first(_common([False, False, True]), num_reqs=3, num_decodes=2)


def test_guard_accepts_pure_decode_batch():
    _assert_decode_rows_first(_common([False, False]), num_reqs=2, num_decodes=2)


def test_guard_accepts_pure_prefill_batch():
    _assert_decode_rows_first(_common([True, True]), num_reqs=2, num_decodes=0)


def test_guard_rejects_prefill_row_before_decode_rows():
    with pytest.raises(RuntimeError, match="decode rows before prefill"):
        _assert_decode_rows_first(_common([True, False]), num_reqs=2, num_decodes=1)


def test_guard_rejects_interleaved_prefill_row():
    with pytest.raises(RuntimeError, match="decode rows before prefill"):
        _assert_decode_rows_first(_common([False, True, False]), num_reqs=3, num_decodes=2)


def test_guard_skips_when_is_prefilling_unavailable():
    _assert_decode_rows_first(SimpleNamespace(), num_reqs=2, num_decodes=1)
    _assert_decode_rows_first(
        SimpleNamespace(is_prefilling=None), num_reqs=2, num_decodes=1
    )


def test_guard_rejects_only_rows_below_num_decodes():
    # A prefill row at or after num_decodes is the expected layout.
    _assert_decode_rows_first(_common([False, True, True]), num_reqs=3, num_decodes=1)
    with pytest.raises(RuntimeError, match="decode rows before prefill"):
        _assert_decode_rows_first(_common([True, False, True]), num_reqs=3, num_decodes=1)


def _batch_layout_common(num_reqs=2, num_tokens=2):
    return SimpleNamespace(
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.full((num_tokens,), -1, dtype=torch.int32),
    )


def test_batch_layout_accepts_graph_padding():
    _validate_batch_layout(
        _batch_layout_common(num_reqs=4, num_tokens=4),
        num_reqs=4,
        num_actual_reqs=2,
        num_input_tokens=4,
        num_actual_tokens=2,
    )


@pytest.mark.parametrize(
    ("num_reqs", "num_actual_reqs", "num_input_tokens", "num_actual_tokens"),
    [
        (2, 3, 2, 2),
        (2, 2, 2, 3),
    ],
)
def test_batch_layout_rejects_actual_counts_above_padded_shape(
    num_reqs, num_actual_reqs, num_input_tokens, num_actual_tokens
):
    with pytest.raises(ValueError):
        _validate_batch_layout(
            _batch_layout_common(num_reqs=num_reqs, num_tokens=num_input_tokens),
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_input_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
        )


def test_batch_layout_requires_all_padded_request_rows():
    common = _batch_layout_common(num_reqs=2)
    common.block_table_tensor = common.block_table_tensor[:1]
    with pytest.raises(ValueError, match="block table row"):
        _validate_batch_layout(
            common,
            num_reqs=2,
            num_actual_reqs=0,
            num_input_tokens=2,
            num_actual_tokens=1,
        )


def test_batch_layout_allows_one_synthetic_padding_row():
    common = _batch_layout_common(num_reqs=2)
    common.block_table_tensor = common.block_table_tensor[:1]
    _validate_batch_layout(
        common,
        num_reqs=2,
        num_actual_reqs=1,
        num_input_tokens=2,
        num_actual_tokens=1,
    )


def test_query_projection_is_split_into_heads_consistently():
    query = torch.zeros((5, 12), dtype=torch.bfloat16)
    reshaped = DeepseekV41EagerAttentionImpl._reshape_query_heads(query, 4)
    assert reshaped.shape == (5, 3, 4)
    assert reshaped.data_ptr() == query.data_ptr()


def test_query_projection_rejects_partial_head():
    with pytest.raises(ValueError, match="divisible by head_dim"):
        DeepseekV41EagerAttentionImpl._reshape_query_heads(
            torch.zeros((2, 10)), 4
        )


def _ws_plan(query_lens, histories):
    qsl = [0]
    for query_len in query_lens:
        qsl.append(qsl[-1] + query_len)
    return {
        "prefill_query_start_loc": torch.tensor(qsl, dtype=torch.int32),
        "ws_local_start_pos": torch.tensor(histories, dtype=torch.int32),
    }


def _workspace_win_indices(plan, window_size=4):
    """Prefill sub-batch win indices: base helper + workspace start_pos shim."""
    qsl = plan["prefill_query_start_loc"]
    q = torch.zeros(int(qsl[-1].item()), 1, 8)
    shim = SimpleNamespace(start_pos=plan["ws_local_start_pos"])
    attn = SimpleNamespace(window_size=window_size)
    return DeepseekV41EagerAttentionImpl._get_window_topk_idxs(
        None, attn, q, shim, qsl.numel() - 1, qsl
    )


def _valid_rows(win_indices):
    return [tokens[tokens >= 0].tolist() for tokens in win_indices[:, 0]]


def test_workspace_window_rampup_without_history():
    plan = _ws_plan([5], [0])
    out = _workspace_win_indices(plan)
    assert out.shape == (5, 1, 4)
    assert out.dtype == torch.int32
    assert _valid_rows(out) == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4]]


def test_workspace_window_slides_with_history():
    plan = _ws_plan([3], [6])
    out = _workspace_win_indices(plan)
    assert _valid_rows(out) == [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]


def test_workspace_window_two_requests_keep_local_coordinates():
    plan = _ws_plan([6, 4], [3, 0])
    out = _workspace_win_indices(plan)
    assert _valid_rows(out) == [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
    ]


def test_workspace_slots_resolve_through_block_table():
    # Valid slots are local positions; ws_block_table translates them to the
    # request's workspace pages, so the resolved row must stay linear.
    block_size = 4
    plan = _ws_plan([6, 4], [3, 0])
    plan["ws_block_table"] = torch.tensor([[0, 1, 2], [3, 0, 0]], dtype=torch.int32)
    out = _workspace_win_indices(plan)
    request_of_token = [0] * 6 + [1] * 4
    page_base = {0: 0, 1: 3}
    for token in range(out.shape[0]):
        request = request_of_token[token]
        table = plan["ws_block_table"][request]
        for slot in out[token, 0].tolist():
            if slot < 0:
                continue
            resolved = int(table[slot // block_size]) * block_size + slot % block_size
            assert resolved == page_base[request] * block_size + slot


def test_mask_below_draft_start_math():
    from vllm_ascend.attention.dsa_v41 import _mask_below_draft_start

    row = torch.tensor([[393, 400, 520, 521, 526, -1]], dtype=torch.int32)
    # Strictly-below threshold masked; the threshold itself is kept (the
    # first anchor's KV exists: context precompute wrote it).
    masked = _mask_below_draft_start(row, torch.tensor([520], dtype=torch.int32))
    assert masked.tolist() == [[-1, -1, 520, 521, 526, -1]]
    # Threshold 0 (standalone: draft KV from position 0) masks nothing.
    zero = _mask_below_draft_start(row, torch.tensor([0], dtype=torch.int32))
    assert zero.tolist() == row.tolist()
    # Per-row thresholds; the threshold position itself is kept.
    two = torch.tensor([[1, 2, 3], [10, 20, 30]], dtype=torch.int32)
    starts = torch.tensor([2, 20], dtype=torch.int32)
    assert _mask_below_draft_start(two, starts).tolist() == [
        [-1, 2, 3],
        [-1, 20, 30],
    ]


def test_mask_below_draft_start_rejects_host_tensor_conversion():
    from vllm_ascend.attention.dsa_v41 import _mask_below_draft_start

    row = torch.tensor([[1, 2]], dtype=torch.int64)
    starts = torch.tensor([1], dtype=torch.int32)
    # A CPU test cannot construct a cross-device tensor, but the production
    # guard must reject one before PyTorch inserts an implicit aten::to copy.
    with pytest.raises(RuntimeError, match="must already be on the metadata device"):
        _mask_below_draft_start(row, starts.to("meta"))


def test_dsa_attention_lifetime_release_is_device_ordered_and_capture_safe(monkeypatch):
    from vllm_ascend.attention import dsa_v41

    class FakeEvent:
        def __init__(self):
            self.calls = []

        def record(self, stream):
            self.calls.append(("record", stream))

        def wait(self, stream):
            self.calls.append(("wait", stream))

    class FakeUtils:
        def __init__(self):
            self.erased = []

        def _erase_stream(self, tensor):
            self.erased.append(tensor)

    utils = FakeUtils()
    events = []
    fake_npu = SimpleNamespace(
        utils=utils,
        current_stream=lambda: "current",
        Event=lambda: events.append(FakeEvent()) or events[-1],
    )
    monkeypatch.setattr(dsa_v41.torch_npu, "npu", fake_npu)
    monkeypatch.setattr(dsa_v41.torch, "npu", SimpleNamespace(Event=fake_npu.Event))
    monkeypatch.setattr(dsa_v41, "get_forward_context", lambda: SimpleNamespace(capturing=False))

    q = object()
    output = object()
    dsa_v41._release_attention_stream_registrations(q, output)
    assert utils.erased == [q, output]
    assert events[0].calls == [("record", "current"), ("wait", "current")]

    utils.erased.clear()
    monkeypatch.setattr(dsa_v41, "get_forward_context", lambda: SimpleNamespace(capturing=True))
    dsa_v41._release_attention_stream_registrations(q, output)
    assert utils.erased == []


def _fake_dspark_builder(max_query_len: int, num_tokens: int, width: int):
    from vllm_ascend.attention.dsa_v41 import DeepseekV41MetadataBuilder

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder._dspark_max_query_len = max_query_len
    builder._dspark_win_indices_buf = torch.full(
        (num_tokens, 1, width), -1, dtype=torch.int32
    )
    return builder


def test_dspark_win_indices_mask_recycled_residual():
    """Positions below the request's first verify anchor are masked to -1:
    the draft planes hold no KV there (PD imports cover target planes only),
    so the proposal must not read ring residual from the previous owner."""
    window, query_len = 128, 6
    # Two requests: A starts drafting at 520 (import boundary), B at 700.
    starts = torch.tensor([521, 800], dtype=torch.int32)
    positions = torch.cat([
        starts[0] + torch.arange(query_len),
        starts[1] + torch.arange(query_len),
    ]).to(torch.int32)
    draft_write_start = torch.tensor([520, 700], dtype=torch.int32)
    builder = _fake_dspark_builder(
        max_query_len=query_len, num_tokens=positions.numel(),
        width=window + query_len,
    )

    win = builder._build_dspark_win_indices(
        positions, positions.numel(), 2, query_len, window,
        draft_write_start=draft_write_start,
    )
    assert win.shape == (positions.numel(), 1, window + query_len)

    def valid_set(row_idx):
        return {t for t in win[row_idx, 0].tolist() if t >= 0}

    # Block A: prefix [393, 520] keeps only [520, 520] plus its own queries.
    assert valid_set(0) == set(range(520, 527))
    # Block B: prefix [672, 799] keeps [700, 799] plus its own queries.
    assert valid_set(query_len) == set(range(700, 806))
    # No row references a position below its request's draft-write start.
    for row_idx, threshold in enumerate((520, 700)):
        assert all(t >= threshold for t in win[row_idx * query_len, 0].tolist() if t >= 0)

    # Without a threshold (pool off / standalone) the full window stays.
    unmasked = builder._build_dspark_win_indices(
        positions, positions.numel(), 2, query_len, window,
        draft_write_start=None,
    )
    assert {t for t in unmasked[0, 0].tolist() if t >= 0} == set(range(393, 527))


def test_dspark_win_indices_masks_short_threshold_tensor_safely():
    """Padding rows beyond the threshold tensor length mask nothing."""
    window, query_len = 128, 6
    positions = torch.arange(2 * query_len, dtype=torch.int32) + 1000
    builder = _fake_dspark_builder(
        max_query_len=query_len, num_tokens=positions.numel(),
        width=window + query_len,
    )
    # One entry for two blocks: the second (padding) row keeps its window.
    starts = torch.tensor([1200], dtype=torch.int32)
    win = builder._build_dspark_win_indices(
        positions, positions.numel(), 2, query_len, window,
        draft_write_start=starts,
    )
    assert {t for t in win[0, 0].tolist() if t >= 0} == set()
    assert {t for t in win[query_len, 0].tolist() if t >= 0} == set(
        range(1006 - window, 1006 + query_len)
    )
