# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.core.private_circle_pool import (
    PrivateCirclePool,
    PrivateCircleConfig,
    compute_prefix_bounded_replay_start,
    compute_private_circle_prefill_workspace_blocks,
    detect_new_prefix_hit_length,
)

def test_compact_sizing_and_allocation_lifecycle():
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    assert config.capacity_tokens == 132
    assert config.blocks_per_allocation == 5
    assert config.num_allocations == 16
    assert config.num_blocks == 81

    pool = PrivateCirclePool(config)
    allocation = pool.reserve("request-1")
    assert allocation is not None
    assert pool.reserve("request-1") == allocation
    assert list(pool.allocation_block_ids(allocation)) == list(
        range(1 + allocation * 5, 1 + allocation * 5 + 5)
    )
    assert pool.slot_for_position(allocation, 0) == (1 + allocation * 5) * 32
    assert pool.slot_for_position(allocation, 128) == (1 + allocation * 5 + 4) * 32
    assert pool.release("request-1")
    assert not pool.release("request-1")


def test_retained_allocation_survives_request_id_reuse():
    config = PrivateCircleConfig(
        block_size=128, window_size=128, in_flight_tokens=1,
        max_num_seqs=1,
    )
    pool = PrivateCirclePool(config)
    old = pool.reserve("request-1")
    assert old is not None
    pool.acquire_ref(old)
    assert pool.release("request-1")

    new = pool.reserve("request-1")
    assert new is not None and new != old
    assert pool.retained_allocations == 1
    assert pool.release_ref(old)
    assert pool.retained_allocations == 0
    assert pool.ref_count(old) == 0


def test_prefill_workspace_covers_block_aligned_bounded_replay_batch():
    max_num_batched_tokens = 2048
    max_num_seqs = 32
    window_size = 128
    block_size = 32
    workspace_blocks = compute_private_circle_prefill_workspace_blocks(
        max_num_batched_tokens,
        max_num_seqs,
        window_size,
        block_size,
    )

    # This valid batch exposes the old W-1 formula: every bounded replay request
    # starts a separately aligned workspace range and the query lengths sum to
    # max_num_batched_tokens.
    query_lens = [2017] + [1] * (max_num_seqs - 1)
    required_blocks = sum(
        (window_size + query_len + block_size - 1) // block_size
        for query_len in query_lens
    )
    old_workspace_blocks = (
        max_num_batched_tokens
        + max_num_seqs * (window_size - 1 + block_size - 1)
        + block_size
        - 1
    ) // block_size

    assert sum(query_lens) == max_num_batched_tokens
    assert old_workspace_blocks == 222
    assert required_blocks == workspace_blocks == 223


def test_prefill_workspace_size_rejects_invalid_inputs():
    invalid_args = (
        (-1, 1, 128, 32),
        (1, 0, 128, 32),
        (1, 1, 0, 32),
        (1, 1, 128, 0),
    )
    for args in invalid_args:
        try:
            compute_private_circle_prefill_workspace_blocks(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid workspace sizing input was accepted")


def test_common_scheduler_prefix_hit_detection():
    # A local or synchronous external hit leaves H cached while only P-H new
    # tokens are scheduled.
    assert detect_new_prefix_hit_length(0, 260, 4) == 256

    # Async KV loading may expose H without scheduling a model token, and some
    # scheduler versions set H before entering _update_after_schedule.
    assert detect_new_prefix_hit_length(0, 256, 0) == 256
    assert detect_new_prefix_hit_length(256, 256, 0, True) == 256

    # Ordinary chunk/decode work starts from an already computed prefix and
    # must not trigger one-shot SWA bounded replay.
    assert detect_new_prefix_hit_length(256, 260, 4) is None
    assert detect_new_prefix_hit_length(0, 4, 4) is None


def test_common_scheduler_prefix_hit_detection_rejects_negative_counts():
    for args in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
        try:
            detect_new_prefix_hit_length(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("negative scheduler token count was accepted")


def test_prefix_bounded_replay_start_is_exact_and_bounded():
    assert compute_prefix_bounded_replay_start(512, 128) == 384
    assert compute_prefix_bounded_replay_start(100, 128) == 0
    assert compute_prefix_bounded_replay_start(513, 0) == 513


def test_prefix_bounded_replay_start_floors_to_alignment():
    # Scheduler-level rollback: the reported hit must stay block-aligned.
    assert compute_prefix_bounded_replay_start(512, 128, alignment=128) == 384
    assert compute_prefix_bounded_replay_start(511, 128, alignment=128) == 256
    assert compute_prefix_bounded_replay_start(513, 128, alignment=128) == 384
    assert compute_prefix_bounded_replay_start(100, 128, alignment=128) == 0
    assert compute_prefix_bounded_replay_start(300, 128, alignment=64) == 128


def test_private_circle_spec_accepts_dspark_draft_planes():
    import torch

    from vllm_ascend.core.deepseek_v41 import (
        A5_WIN_ROW_BYTES,
        DeepseekV41A5DraftSWASpec,
        DeepseekV41DraftSWASpec,
        DeepseekV41SWASpec,
    )
    from vllm_ascend.core.private_circle_pool import is_private_circle_kv_cache_spec

    target_spec = DeepseekV41SWASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        sliding_window=128,
    )
    bf16_draft_spec = DeepseekV41DraftSWASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    a5_draft_spec = DeepseekV41A5DraftSWASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=A5_WIN_ROW_BYTES,
        dtype=torch.uint8,
        sliding_window=128,
    )
    assert is_private_circle_kv_cache_spec(target_spec)
    # Both draft SWA planes join the ring when the pool is enabled.
    assert is_private_circle_kv_cache_spec(bf16_draft_spec)
    assert is_private_circle_kv_cache_spec(a5_draft_spec)
    # An unrelated spec stays on the shared plane.
    assert not is_private_circle_kv_cache_spec(object())


def test_compact_ring_wraps_at_absolute_block_boundary():
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=1
    )
    pool = PrivateCirclePool(config)
    allocation = pool.reserve("ring")
    assert allocation is not None
    blocks = list(pool.allocation_block_ids(allocation))
    # 5 physical blocks hold 160 token slots; absolute positions alias by
    # physical block while preserving the intra-block offset.
    assert pool.slot_for_position(allocation, 0) == blocks[0] * 32
    assert pool.slot_for_position(allocation, 159) == blocks[4] * 32 + 31
    assert pool.slot_for_position(allocation, 160) == blocks[0] * 32
    assert pool.slot_for_position(allocation, 161) == blocks[0] * 32 + 1


def test_pool_exhaustion_and_reuse_after_release():
    config = PrivateCircleConfig(
        block_size=32, window_size=32, in_flight_tokens=1,
        max_num_seqs=2, delayed_free_slots=0,
    )
    pool = PrivateCirclePool(config)
    first = pool.reserve("first")
    second = pool.reserve("second")
    assert first is not None and second is not None and first != second
    assert pool.reserve("third") is None
    assert pool.release("first")
    assert pool.reserve("third") == first


def test_free_list_hands_out_head_and_queues_releases_at_tail():
    """FIFO reuse: a released allocation waits behind every free slot.

    The dirty-page zeroing window grows with the release-to-reuse distance,
    so a reservation must not take the just-released tail slot while an
    older free slot is still available.
    """
    config = PrivateCircleConfig(
        block_size=32, window_size=32, in_flight_tokens=1,
        max_num_seqs=1, delayed_free_slots=2,
    )
    pool = PrivateCirclePool(config)
    assert config.num_allocations == 3

    # Head-first handout: the first request takes allocation 0.
    first = pool.reserve("first")
    second = pool.reserve("second")
    assert first == 0
    assert second == 1

    # The released slot queues behind the still-free slot.
    assert pool.release("first")
    assert pool.reserve("third") == 2
    # Only after every other free slot is gone does the recycled slot return.
    assert pool.reserve("fourth") == first


def test_release_ref_queues_the_retained_slot_at_the_tail():
    config = PrivateCircleConfig(
        block_size=32, window_size=32, in_flight_tokens=1,
        max_num_seqs=1, delayed_free_slots=2,
    )
    pool = PrivateCirclePool(config)
    allocation = pool.reserve("xfer")
    assert allocation == 0
    pool.acquire_ref(allocation)
    assert pool.release("xfer")
    assert pool.retained_allocations == 1

    # A fresh reservation takes the next free slot, not the retained one.
    assert pool.reserve("next") == 1
    assert pool.release_ref(allocation)
    # The freed retained slot joins the tail; the last free slot goes first.
    assert pool.reserve("tail") == 2
    assert pool.reserve("last") == allocation


def test_pool_rejects_invalid_inputs():
    for kwargs in (
        dict(block_size=0, window_size=32, in_flight_tokens=1, max_num_seqs=1),
        dict(block_size=32, window_size=0, in_flight_tokens=1, max_num_seqs=1),
        dict(block_size=32, window_size=32, in_flight_tokens=0, max_num_seqs=1),
        dict(block_size=32, window_size=32, in_flight_tokens=1, max_num_seqs=0),
    ):
        try:
            PrivateCirclePool(PrivateCircleConfig(**kwargs))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid private circle sizing was accepted")

    config = PrivateCircleConfig(
        block_size=32, window_size=32, in_flight_tokens=1, max_num_seqs=1
    )
    pool = PrivateCirclePool(config)
    allocation = pool.reserve("invalid")
    assert allocation is not None
    try:
        pool.slot_for_position(allocation, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative absolute position was accepted")





def test_private_pool_keeps_block_zero_reserved():
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    highest_allocation = config.num_allocations - 1
    highest_block = list(
        PrivateCirclePool(config).allocation_block_ids(
            highest_allocation
        )
    )[-1]
    assert highest_block < config.num_blocks
    assert 0 not in list(
        PrivateCirclePool(config).allocation_block_ids(highest_allocation)
    )


def test_detach_and_deferred_return_lifecycle():
    """deferred-free path detaches ownership and returns after the fence."""
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    pool = PrivateCirclePool(config)
    total = config.num_allocations

    allocation = pool.reserve("request-deferred")
    assert allocation is not None
    assert pool.contains("request-deferred")

    # Deferred branch: ownership drops immediately; the slot is NOT recycled.
    detached = pool.detach("request-deferred")
    assert detached == allocation
    assert not pool.contains("request-deferred")
    assert pool.free_allocations == total - 1
    assert pool.detach("request-deferred") is None  # idempotent on second free

    # A preempted request must reserve a different slot while the fence pends.
    resumed = pool.reserve("request-deferred")
    assert resumed is not None and resumed != allocation

    # Fence processed: the detached slot returns to the free list exactly once.
    pool.return_detached_allocation(allocation, "request-deferred")
    assert pool.free_allocations == total - 1
    # A repeated drain must not double-free the slot.
    pool.return_detached_allocation(allocation, "request-deferred")
    assert pool.free_allocations == total - 1


def test_detached_allocation_with_pending_transfer_ref_is_retained():
    """A P/D transfer holding a detached slot defers its recycle, not its leak fix."""
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    pool = PrivateCirclePool(config)
    allocation = pool.reserve("request-xfer")
    pool.acquire_ref(allocation)
    pool.detach("request-xfer")

    pool.return_detached_allocation(allocation, "request-xfer")
    assert pool.retained_allocations == 1
    assert pool.free_allocations == config.num_allocations - 1

    assert pool.release_ref(allocation) is True
    assert pool.retained_allocations == 0
    assert pool.free_allocations == config.num_allocations


def test_recycled_allocation_dirty_lifecycle():
    """Release marks the ring pages dirty; peek keeps the mark until the
    worker consumes it after zeroing (an async step discarded before its
    input-prep callback must re-see the mark)."""
    config = PrivateCircleConfig(
        block_size=32, window_size=32, in_flight_tokens=1,
        max_num_seqs=2, delayed_free_slots=0,
    )
    pool = PrivateCirclePool(config)
    allocation = pool.reserve("owner")
    assert allocation is not None
    assert not pool.is_dirty(allocation)
    assert pool.release("owner")
    assert pool.is_dirty(allocation)
    assert pool.is_dirty(allocation)  # peek does not consume
    assert pool.consume_dirty(allocation) is True
    assert not pool.is_dirty(allocation)
    assert pool.consume_dirty(allocation) is False


def test_detached_return_also_marks_dirty():
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    pool = PrivateCirclePool(config)
    allocation = pool.reserve("request-deferred")
    assert allocation is not None
    pool.detach("request-deferred")
    assert not pool.is_dirty(allocation)
    pool.return_detached_allocation(allocation, "request-deferred")
    assert pool.is_dirty(allocation)


def test_module_singleton_clears_active_pool_dirty():
    from vllm_ascend.core import private_circle_pool as pool_module

    config = PrivateCircleConfig(
        block_size=32, window_size=32, in_flight_tokens=1,
        max_num_seqs=2, delayed_free_slots=0,
    )
    pool = PrivateCirclePool(config)
    allocation = pool.reserve("owner-2")
    assert allocation is not None
    pool.release("owner-2")
    assert pool_module.clear_private_circle_dirty(allocation) is True
    assert not pool.is_dirty(allocation)
    assert pool_module.clear_private_circle_dirty(allocation) is False
