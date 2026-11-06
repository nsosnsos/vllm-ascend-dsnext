# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-level bounded replay: rollback reporting and adoption restore."""

from types import SimpleNamespace

from vllm_ascend.patch.platform.patch_private_circle_pool import (
    _restore_full_hit_for_adoption,
)


def _request(hit_blocks=None, hit_length=None):
    return SimpleNamespace(
        private_circle_hit_blocks=hit_blocks,
        private_circle_hit_length=hit_length,
    )


def test_adoption_restore_noop_without_stash():
    request = _request()
    args, kwargs = _restore_full_hit_for_adoption(
        request, (8,), {"num_new_computed_tokens": 3}
    )
    assert args == (8,)
    assert kwargs == {"num_new_computed_tokens": 3}


def test_adoption_restore_rewrites_positional_new_tokens():
    request = _request(hit_blocks=["blocks"], hit_length=512)
    # Scheduler budgeted from replay_start=384 with 640 scheduled rows.
    args, kwargs = _restore_full_hit_for_adoption(
        request,
        (640,),
        {"num_new_computed_tokens": 384, "new_computed_blocks": "stale"},
    )
    assert args == (512,)
    assert kwargs["num_new_computed_tokens"] == 512
    assert kwargs["new_computed_blocks"] == "blocks"


def test_adoption_restore_rewrites_keyword_new_tokens():
    request = _request(hit_blocks=["blocks"], hit_length=256)
    args, kwargs = _restore_full_hit_for_adoption(
        request,
        (),
        {"num_new_tokens": 300, "num_new_computed_tokens": 128,
         "new_computed_blocks": "stale"},
    )
    assert args == ()
    assert kwargs["num_new_tokens"] == 172
    assert kwargs["num_new_computed_tokens"] == 256
    assert kwargs["new_computed_blocks"] == "blocks"


def test_adoption_restore_keeps_async_load_zero_tokens():
    # Consumer async load: num_new_tokens == 0 must stay 0 (the rollback is
    # absorbed by the external token count, not the new-token budget).
    request = _request(hit_blocks=["blocks"], hit_length=512)
    args, kwargs = _restore_full_hit_for_adoption(
        request,
        (0,),
        {"num_new_computed_tokens": 384, "new_computed_blocks": "stale"},
    )
    assert args == (0,)
    assert kwargs["num_new_computed_tokens"] == 512
    assert kwargs["new_computed_blocks"] == "blocks"


def test_adoption_restore_defers_when_chunk_cannot_cover_warmup():
    # Budget-tail admission: the chunk is smaller than the warm-up window.
    # Passing it through would commit cache blocks beyond the scheduled
    # range, so the restore must signal a deferral instead.
    request = _request(hit_blocks=["blocks"], hit_length=512)
    assert _restore_full_hit_for_adoption(
        request,
        (100,),
        {"num_new_computed_tokens": 384, "new_computed_blocks": "stale"},
    ) is None
    # Boundary: chunk == rollback would rewrite to zero; defer as well.
    assert _restore_full_hit_for_adoption(
        request,
        (128,),
        {"num_new_computed_tokens": 384, "new_computed_blocks": "stale"},
    ) is None
    # Keyword form defers too.
    assert _restore_full_hit_for_adoption(
        request,
        (),
        {"num_new_tokens": 50, "num_new_computed_tokens": 384,
         "new_computed_blocks": "stale"},
    ) is None


def test_adoption_restore_rewrites_just_above_warmup_boundary():
    request = _request(hit_blocks=["blocks"], hit_length=512)
    args, kwargs = _restore_full_hit_for_adoption(
        request,
        (129,),
        {"num_new_computed_tokens": 384, "new_computed_blocks": "stale"},
    )
    assert args == (1,)
    assert kwargs["num_new_computed_tokens"] == 512
    assert kwargs["new_computed_blocks"] == "blocks"


def test_allocate_slots_defers_and_releases_ring_reservation():
    from vllm_ascend.patch.platform.patch_private_circle_pool import (
        _patched_allocate_slots,
    )

    released = []

    class _Pool:
        def contains(self, request_id):
            return False

        def reserve(self, request_id):
            return 7

        def release(self, request_id):
            released.append(request_id)

    request = SimpleNamespace(
        request_id="req-1",
        private_circle_hit_blocks=["blocks"],
        private_circle_hit_length=512,
        private_circle_allocation=7,
    )
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(private_circle_pool=_Pool()),
    )
    result = _patched_allocate_slots(
        manager,
        request,
        100,
        num_new_computed_tokens=384,
        new_computed_blocks="stale",
    )
    assert result is None
    assert released == ["req-1"]
    assert request.private_circle_allocation is None
    # The stash is one-shot: the deferral clears it so the retried
    # admission re-stashes through get_computed_blocks.
    assert request.private_circle_hit_blocks is None
    assert request.private_circle_hit_length is None


def test_adoption_restore_does_not_mutate_caller_kwargs():
    request = _request(hit_blocks=["blocks"], hit_length=512)
    original = {"num_new_computed_tokens": 384, "new_computed_blocks": "stale"}
    _restore_full_hit_for_adoption(request, (640,), original)
    assert original == {"num_new_computed_tokens": 384,
                        "new_computed_blocks": "stale"}


def test_alignment_floor_keeps_rollback_at_least_window():
    from vllm_ascend.core.private_circle_pool import (
        compute_prefix_bounded_replay_start,
    )

    window, alignment = 128, 128
    for hit in range(alignment, 4096 + 1, alignment):
        start = compute_prefix_bounded_replay_start(hit, window, alignment)
        assert 0 <= start < hit
        assert hit - start >= window
        assert start % alignment == 0
