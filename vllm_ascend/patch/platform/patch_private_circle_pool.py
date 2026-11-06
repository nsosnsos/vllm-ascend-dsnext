# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported from vllm-ascend swa_private_pool (v0.23-era) via the v0.27.1rc
# private SWA pool; adapted for DSV4.1 on V27-dsv4.1 (PrivateCirclePool).
# vLLM v0.27 already tracks Request.num_in_flight_tokens, applies the
# in-flight adjustment inside remove_skipped_blocks(), and owns the
# admission-cap/max_memory_usage_bytes plumbing; only the request-private
# circle pool lifecycle is patched here.
from functools import wraps
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.request import Request

from vllm_ascend import envs
from vllm_ascend.core.deepseek_v41 import DeepseekV41SWASpec
from vllm_ascend.core.private_circle_pool import (
    PRIVATE_CIRCLE_TRANSFER_FAILED,
    PRIVATE_CIRCLE_TRANSFER_NONE,
    PRIVATE_CIRCLE_TRANSFER_READY,
    PRIVATE_CIRCLE_TRANSFER_RECEIVING,
    PrivateCircleConfig,
    PrivateCirclePool,
    compute_prefix_bounded_replay_start,
    detect_new_prefix_hit_length,
    imports_private_circle_from_peer,
)


def _find_private_circle_spec(spec: Any) -> SlidingWindowSpec | None:
    if isinstance(spec, DeepseekV41SWASpec):
        return spec
    nested = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested, dict):
        nested = list(nested.values())
    if isinstance(nested, (list, tuple, set)):
        for child in nested:
            found = _find_private_circle_spec(child)
            if found is not None:
                return found
    return None


_original_request_init = Request.__init__


@wraps(_original_request_init)
def _patched_request_init(self: Request, *args: Any, **kwargs: Any) -> None:
    _original_request_init(self, *args, **kwargs)
    self.private_circle_allocation = None
    self.private_circle_confirmed_length = 0
    self.private_circle_valid_length = 0
    self.private_circle_window_start = 0
    self.private_circle_pool_type = "private"
    # First prefix-cache hit only; None means no bounded replay is due.
    self.private_circle_original_hit_length = None
    self.private_circle_effective_start = None
    self.private_circle_transfer_allocation = None
    self.private_circle_transfer_state = PRIVATE_CIRCLE_TRANSFER_NONE
    self.private_circle_imported_window_start = None
    self.private_circle_imported_valid_length = None
    self.private_circle_remote_local_hit = None
    self.private_circle_transfer_shared_block_ids = frozenset()
    # Scheduler-level bounded replay: the full hit blocks/length stashed by
    # get_computed_blocks until allocate_slots adopts them.
    self.private_circle_hit_blocks = None
    self.private_circle_hit_length = None
    self.private_circle_store_plan = None
    self.private_circle_store_unverified = False
    self.private_circle_replay_config = None


def _record_private_circle_prefix_hit(
    request: Request,
    hit_length: int,
    replay_start: int,
    private_pool: PrivateCirclePool,
) -> None:
    """Record one-shot SWA bounded replay boundaries for a cache hit.

    ``replay_start`` is the block-aligned scheduler-visible hit after the
    bounded replay rollback; ``hit_length`` is the true local hit the shared
    planes adopt.
    """
    if not envs.VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL or hit_length <= 0:
        return
    previous_hit_length = getattr(request, "private_circle_original_hit_length", None)
    request.private_circle_original_hit_length = int(hit_length)
    request.private_circle_effective_start = int(replay_start)
    # Log only the first observation.
    if previous_hit_length != int(hit_length):
        total_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0))
        logger.info(
            "PRIVATE_CIRCLE_POOL prefix_hit request_id=%s total_prompt_tokens=%d "
            "hit_tokens=%d bounded_replay_start=%d bounded_replay_tokens=%d",
            request.request_id,
            total_prompt_tokens,
            int(hit_length),
            int(replay_start),
            int(hit_length) - int(replay_start),
        )


def _private_circle_import_covers(
    request: Request,
    confirmed_length: int,
    window_size: int,
) -> bool:
    imported_start = getattr(
        request, "private_circle_imported_window_start", None
    )
    imported_end = getattr(
        request, "private_circle_imported_valid_length", None
    )
    if imported_start is None or imported_end is None:
        return False
    required_start = max(0, confirmed_length - window_size + 1)
    return int(imported_start) <= required_start and int(imported_end) >= confirmed_length


def _latch_private_circle_draft_write_start(
    request: Request,
    request_id: str,
    allocation: int,
    confirmed: int,
    live_in_flight_tokens_before_schedule: int,
) -> None:
    """Latch the first draft-plane anchor for a private-circle allocation."""
    if getattr(request, "_private_circle_draft_alloc", None) == allocation:
        return
    if live_in_flight_tokens_before_schedule != 0:
        raise RuntimeError(
            "Private-circle draft_write_start cannot be latched while an "
            f"older output is in flight for request {request_id}: "
            f"live_in_flight_tokens_before_schedule={live_in_flight_tokens_before_schedule}, "
            f"allocation={allocation}, scheduler_visible_prefix={confirmed}"
        )
    request._private_circle_draft_alloc = allocation
    request.private_circle_draft_write_start = confirmed


_original_get_computed_blocks = KVCacheManager.get_computed_blocks


@wraps(_original_get_computed_blocks)
def _patched_get_computed_blocks(self: KVCacheManager, request: Request):
    """Scheduler-level bounded replay: report an aligned rollback start.

    Producer and standalone engines roll the reported hit back by one window
    so the scheduler schedules the warm-up rows inside the token budget; the
    full hit blocks are returned untouched and ``_patched_allocate_slots``
    restores the full hit length when they are adopted. Consumer engines keep
    the connector-facing hit semantics; a consumer-local hit that does not
    join a remote prefill is dropped (the decode-side token budget cannot
    hold the warm-up window, so the ring is rebuilt from scratch instead).
    """
    was_uncomputed = int(request.num_computed_tokens) == 0
    result = _original_get_computed_blocks(self, request)
    if was_uncomputed:
        private_pool = None
        if envs.VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL:
            private_pool = getattr(self.coordinator, "private_circle_pool", None)
        if private_pool is not None:
            alignment = int(
                getattr(self.coordinator, "scheduler_block_size", 0)
                or getattr(self.coordinator, "lcm_block_size", 0) or 1
            )
            if getattr(self.coordinator, "private_circle_scheduler_replay", False):
                request.private_circle_replay_config = (int(private_pool.config.window_size), alignment)
            hit_blocks, hit_length, boundary = result
            hit_length = int(hit_length)
            if hit_length > 0:
                scheduler_replay = getattr(
                    self.coordinator, "private_circle_scheduler_replay", False
                )
                if scheduler_replay:
                    alignment = int(
                        getattr(self.coordinator, "scheduler_block_size", 0)
                        or getattr(self.coordinator, "lcm_block_size", 0)
                        or 1
                    )
                    replay_start = compute_prefix_bounded_replay_start(
                        hit_length, int(private_pool.config.window_size), alignment
                    )
                    if replay_start >= hit_length:
                        replay_start = max(0, hit_length - alignment)
                    request.private_circle_hit_blocks = hit_blocks
                    request.private_circle_hit_length = hit_length
                    _record_private_circle_prefix_hit(
                        request, hit_length, replay_start, private_pool
                    )
                    return (hit_blocks, replay_start, boundary)
                params = getattr(request, "kv_transfer_params", None)
                remote_prefill = params is not None and params.get(
                    "do_remote_prefill", False
                )
                if not remote_prefill:
                    logger.info(
                        "PRIVATE_CIRCLE_POOL consumer_local_hit_dropped "
                        "request_id=%s hit_tokens=%d total_prompt_tokens=%d",
                        request.request_id,
                        hit_length,
                        int(getattr(request, "num_prompt_tokens", 0)),
                    )
                    return (self.empty_kv_cache_blocks, 0, boundary)
        else:
            # Keep prefix-cache observability when the private circle pool is
            # absent (feature disabled, or a non-V4.1 model with the env set).
            # No bounded replay state is created on this compatibility path.
            _, hit_length, _ = result
            if int(hit_length) > 0:
                total_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0))
                logger.info(
                    "PREFIX_CACHE hit request_id=%s total_prompt_tokens=%d "
                    "hit_tokens=%d private_circle_pool_enabled=False",
                    request.request_id,
                    total_prompt_tokens,
                    int(hit_length),
                )
    return result


KVCacheManager.get_computed_blocks = _patched_get_computed_blocks


_original_update_after_schedule = Scheduler._update_after_schedule


@wraps(_original_update_after_schedule)
def _patched_update_after_schedule(
    self: Scheduler, scheduler_output: SchedulerOutput
) -> None:
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_circle_pool", None
    )
    computed_before_schedule: dict[str, int] = {}
    live_in_flight_tokens_before_schedule: dict[str, int] = {}
    if private_pool is not None:
        # Scheduler-visible positions before upstream advances them; zero
        # identifies newly admitted requests whose prefix may come from an
        # external connector.
        computed_before_schedule = {
            request_id: int(request.num_computed_tokens)
            for request_id, request in self.requests.items()
        }
        live_in_flight_tokens_before_schedule = {
            request_id: max(
                0,
                int(getattr(request, "num_in_flight_tokens", 0))
                - int(getattr(request, "num_stale_output_tokens", 0)),
            )
            for request_id, request in self.requests.items()
        }
    _original_update_after_schedule(self, scheduler_output)
    private_metadata: dict[str, dict[str, Any]] = {}
    if private_pool is not None:
        # Reconcile the final hit after local and external matching; async
        # loads may have no scheduled tokens (WAITING_FOR_REMOTE_KVS).
        for request_id, computed_before in computed_before_schedule.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            status = getattr(request, "status", None)
            status_name = getattr(status, "name", str(status))
            num_scheduled = int(
                scheduler_output.num_scheduled_tokens.get(request_id, 0)
            )
            confirmed_length = max(
                0, int(request.num_computed_tokens) - num_scheduled
            )
            transfer_state = getattr(
                request, "private_circle_transfer_state",
                PRIVATE_CIRCLE_TRANSFER_NONE,
            )
            if transfer_state == PRIVATE_CIRCLE_TRANSFER_RECEIVING:
                continue
            if transfer_state == PRIVATE_CIRCLE_TRANSFER_FAILED:
                continue
            if transfer_state == PRIVATE_CIRCLE_TRANSFER_READY:
                if not _private_circle_import_covers(
                    request, confirmed_length, private_pool.config.window_size
                ):
                    request.private_circle_transfer_state = (
                        PRIVATE_CIRCLE_TRANSFER_FAILED
                    )
                    raise RuntimeError(
                        "imported private circle does not cover the confirmed "
                        f"prefix for request {request_id}"
                    )
                logger.debug(
                    "PRIVATE_CIRCLE_POOL import_cover request_id=%s "
                    "confirmed_length=%d imported_window=[%d, %d) "
                    "replay_cancelled=%s",
                    request_id,
                    confirmed_length,
                    int(request.private_circle_imported_window_start),
                    int(request.private_circle_imported_valid_length),
                    request.private_circle_effective_start is not None,
                )
                # The D node consumes P-generated SWA directly.
                request.private_circle_original_hit_length = None
                request.private_circle_effective_start = None
                continue
            # Local hits are recorded solely by _patched_get_computed_blocks
            # (scheduler-level rollback); external async loads reach the
            # READY branch above, which cancels any replay state.
            confirmed_from_schedule = detect_new_prefix_hit_length(
                computed_before,
                int(request.num_computed_tokens),
                num_scheduled,
                waiting_for_remote_kvs=status_name == "WAITING_FOR_REMOTE_KVS",
            )
            if confirmed_from_schedule is not None and confirmed_from_schedule > 0:
                logger.debug(
                    "PRIVATE_CIRCLE_POOL external prefix advance without "
                    "scheduler lookup: request_id=%s confirmed_length=%d",
                    request_id,
                    int(confirmed_from_schedule),
                )
    for request_id, num_scheduled_tokens in (
        scheduler_output.num_scheduled_tokens.items()
    ):
        request = self.requests.get(request_id)
        if request is None:
            # Upstream may drop an immediately finished request.
            continue
        if private_pool is None:
            continue
        allocation = private_pool.get_allocation(request_id)
        if allocation is None:
            raise RuntimeError(f"request {request_id} has no private circle allocation")
        # _update_after_schedule has already advanced num_computed_tokens to P.
        # Private metadata must describe the scheduler-visible prefix H.
        confirmed = max(
            0, int(request.num_computed_tokens) - int(num_scheduled_tokens)
        )
        original_hit = getattr(request, "private_circle_original_hit_length", None)
        effective_start = getattr(request, "private_circle_effective_start", None)
        valid = confirmed + int(num_scheduled_tokens)
        window = private_pool.config.window_size
        window_start = max(0, valid - window)
        # Draft KV accumulates on the ring only from this engine's first step
        # for the request (D-side: the first verify anchor position). Record
        # it once per (request, allocation): a request resuming on a fresh
        # allocation restarts draft accumulation at the new confirmed start.
        _latch_private_circle_draft_write_start(
            request,
            request_id,
            allocation,
            confirmed,
            live_in_flight_tokens_before_schedule.get(request_id, 0),
        )
        request.private_circle_allocation = allocation
        request.private_circle_confirmed_length = confirmed
        request.private_circle_valid_length = valid
        request.private_circle_window_start = window_start
        private_metadata[request_id] = {
            "layout_version": 1,
            "pool_type": "private",
            "allocation_handle": allocation,
            "confirmed_length": confirmed,
            "shared_cache_length": confirmed,
            "circle_compute_start": (
                effective_start if effective_start is not None else confirmed
            ),
            "bounded_replay_length": (
                confirmed - effective_start
                if effective_start is not None else 0
            ),
            # Compatibility aliases for the model metadata path.
            "original_prefix_hit_length": original_hit,
            "effective_compute_start": (
                effective_start if effective_start is not None else confirmed
            ),
            "valid_length": valid,
            "window_start": window_start,
            "physical_block_size": private_pool.config.block_size,
            "window_size": window,
            "in_flight_tokens": private_pool.config.in_flight_tokens,
            "blocks_per_allocation": private_pool.config.blocks_per_allocation,
            "num_allocations": private_pool.config.num_allocations,
            "private_num_blocks": private_pool.config.num_blocks,
            # Below this position the draft planes hold no KV for this
            # request: the win-indices build masks them (-1) so the
            # proposal never reads ring residual from the previous owner.
            "draft_write_start": int(getattr(
                request, "private_circle_draft_write_start", confirmed)),
            # Peek (not consume): the runner zeroes the recycled draft-plane
            # pages and only then clears the dirty mark.
            "draft_ring_dirty": private_pool.is_dirty(allocation),
        }
        # Bounded replay metadata and imported-SWA readiness are one-shot: later
        # chunks use the local ring as ordinary history.
        if (
            getattr(request, "private_circle_transfer_state", None)
            == PRIVATE_CIRCLE_TRANSFER_READY
        ):
            request.private_circle_transfer_state = PRIVATE_CIRCLE_TRANSFER_NONE
        request.private_circle_original_hit_length = None
        request.private_circle_effective_start = None
        request.private_circle_store_plan = None
    scheduler_output.private_circle_metadata = private_metadata


_original_handle_invalid_blocks = Scheduler._handle_invalid_blocks


@wraps(_original_handle_invalid_blocks)
def _patched_handle_invalid_blocks(
    self: Scheduler,
    invalid_block_ids: set[int],
    num_scheduled_tokens: dict[str, int],
) -> set[str]:
    """Map private-circle load failures without recomputation.

    The connector records one scheduler-visible shared group per receiving
    request and reports IDs from that group; a failed private-circle import is
    never eligible for recomputation.
    """
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_circle_pool", None
    )
    if private_pool is None:
        return _original_handle_invalid_blocks(
            self, invalid_block_ids, num_scheduled_tokens
        )

    # Native invalid-block recovery unpacks exactly one group. Private-circle
    # requests are hybrid; map failures across the shared groups explicitly.
    # This also catches requests sharing a synchronously loaded bad block.
    affected = set()
    evict = set(invalid_block_ids)
    requests = getattr(self, "requests", None)
    candidates = requests.values() if requests is not None else (*self.skipped_waiting, *self.running)
    for request in candidates:
        groups = self.kv_cache_manager.get_block_ids(request.request_id)
        shared = [
            ids for gid, ids in enumerate(groups)
            if gid not in self.kv_cache_manager.coordinator.private_circle_group_ids
        ]
        if any(invalid_block_ids.intersection(ids) for ids in shared):
            affected.add(request.request_id)
            evict.update(block for ids in shared for block in ids if block != 0)
    if affected:
        self.kv_cache_manager.evict_blocks(evict)
        logger.error("Private circle KV load failed; terminating requests %s", sorted(affected))
        return affected

    receiving_requests = [
        request
        for request in (*self.skipped_waiting, *self.running)
        if (
            getattr(request, "private_circle_transfer_state", None)
            == PRIVATE_CIRCLE_TRANSFER_RECEIVING
        )
    ]
    failed_request_ids = {
        request.request_id
        for request in receiving_requests
        if invalid_block_ids.intersection(
            getattr(
                request,
                "private_circle_transfer_shared_block_ids",
                (),
            )
        )
    }
    if failed_request_ids:
        return failed_request_ids

    if receiving_requests:
        # A private-circle load failure cannot be recovered safely. If a worker
        # reports IDs that cannot be mapped (for example due to a connector
        # version mismatch), fail all in-flight private imports rather than
        # accidentally promoting one with incomplete SWA.
        failed_request_ids = {
            request.request_id for request in receiving_requests
        }
        logger.error(
            "Private circle load errors %s did not match request block IDs; "
            "failing all receiving private-circle requests: %s",
            sorted(invalid_block_ids),
            sorted(failed_request_ids),
        )
        return failed_request_ids

    # Do not enter upstream's single-group fallback for a hybrid request.
    raise RuntimeError(f"Unmapped private circle invalid KV blocks: {sorted(invalid_block_ids)}")


_original_allocate_slots = KVCacheManager.allocate_slots


def _restore_full_hit_for_adoption(
    request: Request, args: tuple, kwargs: dict
) -> tuple[tuple, dict] | None:
    """Restore the full hit length when the stashed hit blocks are adopted.

    ``_patched_get_computed_blocks`` reported the rolled-back start to the
    scheduler (token budget), but the shared planes must adopt every hit
    block. Rewrite the allocate contract to the full hit: the warm-up rows
    before the original hit boundary need no new blocks (their shared-plane
    writes are masked by the model runner).

    Returns None when this step's chunk cannot cover the whole warm-up
    window: the caller defers the request, so the warm-up rows always land
    in one full-budget chunk (the one-shot replay mask/plan state and the
    ring commit invariant both rely on that).
    """
    stashed_blocks = getattr(request, "private_circle_hit_blocks", None)
    stashed_hit = getattr(request, "private_circle_hit_length", None)
    if stashed_blocks is None or stashed_hit is None:
        return args, kwargs
    replay_start = int(kwargs.get("num_new_computed_tokens") or 0)
    full_hit = int(stashed_hit)
    rollback = max(0, full_hit - replay_start)
    if args:
        num_new_tokens = args[0]
    else:
        num_new_tokens = kwargs.get("num_new_tokens")
    if isinstance(num_new_tokens, int) and 0 < num_new_tokens <= rollback:
        # A budget-tail chunk cannot hold the warm-up rows; deferring (rather
        # than adopting with an unrewritten token count) keeps the cache
        # commit inside the actually-scheduled range.
        return None
    kwargs = dict(kwargs)
    kwargs["new_computed_blocks"] = stashed_blocks
    kwargs["num_new_computed_tokens"] = full_hit
    if isinstance(num_new_tokens, int) and num_new_tokens > rollback:
        # Defense in depth: never reduce a positive allocation to zero (the
        # upstream allocator rejects that without external KV).
        if args:
            args = (num_new_tokens - rollback, *args[1:])
        else:
            kwargs["num_new_tokens"] = num_new_tokens - rollback
    return args, kwargs


@wraps(_original_allocate_slots)
def _patched_allocate_slots(self: KVCacheManager, request: Request, *args: Any, **kwargs: Any) -> Any:
    private_pool: PrivateCirclePool | None = getattr(
        self.coordinator, "private_circle_pool", None
    )
    reserved_here = False
    if private_pool is not None and not private_pool.contains(request.request_id):
        allocation = private_pool.reserve(request.request_id)
        if allocation is None:
            return None
        request.private_circle_allocation = allocation
        reserved_here = True
    try:
        plan = getattr(request, "private_circle_store_plan", None)
        if plan is not None:
            chunk = args[0] if args else kwargs.get("num_new_tokens", 0)
            warmup = plan.hit_tokens - plan.replay_start
            if 0 < chunk <= warmup:
                if reserved_here:
                    private_pool.release(request.request_id)
                    request.private_circle_allocation = None
                return None
            if not plan.loaded:
                kwargs = dict(kwargs)
                # Do not publish an unverified sync load for sibling requests
                # to adopt in this same step. Normal subsequent allocation
                # caches the prefix after the successful forward/receive.
                kwargs["delay_cache_blocks"] = True
                kwargs["num_new_computed_tokens"] = plan.local_hit_tokens
                kwargs["num_external_computed_tokens"] = plan.hit_tokens - plan.local_hit_tokens
                stashed = getattr(request, "private_circle_hit_blocks", None)
                if stashed is not None:
                    kwargs["new_computed_blocks"] = stashed
                if chunk:
                    if args:
                        args = (chunk - warmup, *args[1:])
                    else:
                        kwargs["num_new_tokens"] = chunk - warmup
            result = _original_allocate_slots(self, request, *args, **kwargs)
            if result is None:
                if reserved_here:
                    private_pool.release(request.request_id)
                    request.private_circle_allocation = None
                return None
            _record_private_circle_prefix_hit(request, plan.hit_tokens, plan.replay_start, private_pool)
            request.private_circle_store_unverified = not plan.loaded
            return result
        if getattr(request, "private_circle_hit_blocks", None) is not None:
            restored = _restore_full_hit_for_adoption(request, args, kwargs)
            if restored is None:
                # Defer: the scheduler retries with a fresh token budget
                # next step, where the chunk always covers the warm-up
                # rows. The stash is cleared below and re-stashed by
                # get_computed_blocks, so the retry is self-healing.
                if reserved_here:
                    private_pool.release(request.request_id)
                    request.private_circle_allocation = None
                chunk = args[0] if args else kwargs.get("num_new_tokens")
                hit_length = int(
                    getattr(request, "private_circle_hit_length", 0) or 0
                )
                replay_start = int(kwargs.get("num_new_computed_tokens") or 0)
                logger.info(
                    "PRIVATE_CIRCLE_POOL replay_deferred request_id=%s "
                    "chunk_tokens=%s warmup_tokens=%d: step budget cannot "
                    "hold the warm-up window, retrying next step",
                    request.request_id,
                    chunk,
                    hit_length - replay_start,
                )
                return None
            args, kwargs = restored
        if getattr(request, "private_circle_store_unverified", False):
            kwargs = dict(kwargs)
            kwargs["delay_cache_blocks"] = True
        if not reserved_here:
            return _original_allocate_slots(self, request, *args, **kwargs)
        result = _original_allocate_slots(self, request, *args, **kwargs)
        if result is None and private_pool is not None:
            private_pool.release(request.request_id)
            request.private_circle_allocation = None
        return result
    except Exception:
        if private_pool is not None and reserved_here:
            private_pool.release(request.request_id)
            request.private_circle_allocation = None
        raise
    finally:
        request.private_circle_hit_blocks = None
        request.private_circle_hit_length = None


_original_kv_cache_manager_free = KVCacheManager.free


@wraps(_original_kv_cache_manager_free)
def _patched_kv_cache_manager_free(self: KVCacheManager, request: Request) -> None:
    try:
        _original_kv_cache_manager_free(self, request)
    finally:
        private_pool: PrivateCirclePool | None = getattr(
            self.coordinator, "private_circle_pool", None
        )
        if private_pool is not None:
            private_pool.release(request.request_id)
        request.private_circle_allocation = None
        request.private_circle_transfer_state = PRIVATE_CIRCLE_TRANSFER_NONE
        request.private_circle_imported_window_start = None
        request.private_circle_imported_valid_length = None
        request.private_circle_remote_local_hit = None
        request.private_circle_store_plan = None
        request.private_circle_store_unverified = False
        request.private_circle_original_hit_length = None
        request.private_circle_effective_start = None
        request.private_circle_hit_blocks = None
        request.private_circle_hit_length = None


_original_preempt_request = Scheduler._preempt_request


@wraps(_original_preempt_request)
def _patched_preempt_request(self: Scheduler, request: Request, *args, **kwargs):
    """Fail fast when a PD decode node preempts: the forced local recompute
    mixes prefill rows into the decode batch and diverges DP MoE communication
    selection. Prefill nodes and standalone engines preempt normally.
    """
    kv_transfer = getattr(self.vllm_config, "kv_transfer_config", None)
    if kv_transfer is not None and kv_transfer.is_kv_consumer:
        raise RuntimeError(
            f"PD decode node preempted request {request.request_id} "
            f"(computed={request.num_computed_tokens}, "
            f"prompt={request.num_prompt_tokens}): D-side preemption forces a "
            "full local prefill recompute, which mixes prefill rows into the "
            "decode batch and diverges DP MoE communication selection. "
            "Prevent it by lowering max_num_seqs, enabling --watermark, or "
            "growing the KV budget."
        )
    return _original_preempt_request(self, request, *args, **kwargs)


_original_free_request_blocks = Scheduler._free_request_blocks


@wraps(_original_free_request_blocks)
def _patched_free_request_blocks(self: Scheduler, request: Request) -> None:
    """Release private allocations safely on the deferred-free path.

    Requests freed while their last step is in flight go through
    pop_blocks_for_free, not KVCacheManager.free, so the usual release hook
    never runs. Mirror the shared-block fence: detach ownership now, return
    the slot only after the owning step completes (an in-flight step may
    still write the ring pages, and a transfer-engine load into a recycled
    slot is not ordered against those writes).
    """
    request.private_circle_store_plan = None
    request.private_circle_store_unverified = False
    request.private_circle_original_hit_length = None
    request.private_circle_effective_start = None
    request.private_circle_hit_blocks = None
    request.private_circle_hit_length = None
    request._private_circle_draft_alloc = None
    request.private_circle_draft_write_start = None
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_circle_pool", None
    )
    if (
        private_pool is not None
        and self.defer_block_free
        and getattr(request, "last_sched_seq", 0) > self.processed_step_seq
    ):
        allocation = private_pool.detach(request.request_id)
        if allocation is not None:
            deferred_releases = getattr(
                self, "_private_circle_deferred_releases", None
            )
            if deferred_releases is None:
                deferred_releases = []
                self._private_circle_deferred_releases = deferred_releases
            deferred_releases.append(
                (getattr(request, "last_sched_seq", 0), request.request_id, allocation)
            )
            request.private_circle_allocation = None
            request.private_circle_transfer_state = PRIVATE_CIRCLE_TRANSFER_NONE
            request.private_circle_imported_window_start = None
            request.private_circle_imported_valid_length = None
            request.private_circle_remote_local_hit = None
    _original_free_request_blocks(self, request)


_original_update_from_output = Scheduler.update_from_output


@wraps(_original_update_from_output)
def _patched_update_from_output(
    self: Scheduler,
    scheduler_output: SchedulerOutput,
    model_runner_output: Any,
) -> Any:
    result = _original_update_from_output(self, scheduler_output, model_runner_output)
    deferred_releases = getattr(self, "_private_circle_deferred_releases", None)
    if deferred_releases:
        private_pool = getattr(
            self.kv_cache_manager.coordinator, "private_circle_pool", None
        )
        if private_pool is not None:
            remaining = []
            for fence, request_id, allocation in deferred_releases:
                if fence <= self.processed_step_seq:
                    # Every device write enqueued by the owning step has
                    # completed; the ring pages can be recycled safely.
                    private_pool.return_detached_allocation(allocation, request_id)
                else:
                    remaining.append((fence, request_id, allocation))
            self._private_circle_deferred_releases = remaining
    return result


_original_connector_finished = Scheduler._connector_finished


@wraps(_original_connector_finished)
def _patched_connector_finished(self: Scheduler, request: Request) -> tuple[bool, dict[str, Any] | None]:
    result = _original_connector_finished(self, request)
    delay_free, _ = result
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_circle_pool", None
    )
    allocation = getattr(request, "private_circle_allocation", None)
    if (
        delay_free
        and private_pool is not None
        and allocation is not None
        and request.private_circle_transfer_allocation is None
    ):
        private_pool.acquire_ref(int(allocation))
        request.private_circle_transfer_allocation = int(allocation)
    return result


_original_free_blocks = Scheduler._free_blocks


@wraps(_original_free_blocks)
def _patched_free_blocks(self: Scheduler, request: Request) -> None:
    allocation = getattr(request, "private_circle_transfer_allocation", None)
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_circle_pool", None
    )
    try:
        _original_free_blocks(self, request)
    finally:
        if allocation is not None and private_pool is not None:
            private_pool.release_ref(int(allocation))
        request.private_circle_transfer_allocation = None


_original_scheduler_init = Scheduler.__init__


_original_update_waiting_for_remote_kv = Scheduler._update_waiting_for_remote_kv


@wraps(_original_update_waiting_for_remote_kv)
def _patched_update_waiting_for_remote_kv(self: Scheduler, request: Request) -> None:
    plan = getattr(request, "private_circle_store_plan", None)
    if plan is None:
        return _original_update_waiting_for_remote_kv(self, request)
    # Publish the full restored prefix, then schedule the warm-up from R.
    # Store completion must never imply that a private SWA ring was imported.
    request.num_computed_tokens = plan.hit_tokens
    _original_update_waiting_for_remote_kv(self, request)
    plan.loaded = True
    request.private_circle_store_unverified = False
    request.num_computed_tokens = plan.replay_start


@wraps(_original_scheduler_init)
def _patched_scheduler_init(self: Scheduler, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
    _original_scheduler_init(self, vllm_config, *args, **kwargs)
    coordinator = self.kv_cache_manager.coordinator
    private_group_ids = (list(getattr(coordinator, "private_circle_group_ids", ()))
                           if envs.VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL else [])
    if private_group_ids:
        # Context-parallel variants route to their own impl classes and
        # bypass the private pool entirely (silent corruption); fail fast.
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.utils import enable_dsa_cp

        # The scheduler-side process initializes AscendConfig later than
        # Scheduler.__init__; ensure it exists before reading the CP flag
        # (idempotent: cached when already initialized with this config).
        init_ascend_config(vllm_config)

        parallel_config = vllm_config.parallel_config
        if (
            enable_dsa_cp()
            or parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size != 1
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 private circle pool does not support context "
                "parallelism (DSA-CP / PCP / DCP). Disable "
                "VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL or the context-parallel "
                "configuration."
            )
        private_spec = _find_private_circle_spec(
            coordinator.single_type_managers[private_group_ids[0]].kv_cache_spec)
        if private_spec is None:
            raise RuntimeError("DeepSeek-V4.1 private circle group has no SWA spec")
        speculative_config = vllm_config.speculative_config
        draft_tokens = (
            getattr(speculative_config, "num_speculative_tokens", 0)
            if speculative_config is not None
            else 0
        )
        private_config = PrivateCircleConfig(
            block_size=private_spec.block_size,
            window_size=private_spec.sliding_window,
            in_flight_tokens=1 + draft_tokens,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
        )
        # Producer and standalone engines schedule the bounded-replay warm-up
        # rows inside the token budget; consumers keep connector-facing hit
        # semantics (PD imports cover the window instead of replaying it).
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        is_consumer = imports_private_circle_from_peer(kv_transfer_config)
        coordinator.private_circle_scheduler_replay = not is_consumer
        if coordinator.private_circle_scheduler_replay:
            alignment = int(
                getattr(coordinator, "scheduler_block_size", 0)
                or getattr(coordinator, "lcm_block_size", 0)
                or int(private_config.block_size)
            )
            max_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            if max_batched_tokens < int(private_config.window_size) + alignment:
                # The rollback can span window + one alignment of tokens; a
                # smaller budget would strand a chunk entirely inside the
                # warm-up rows.
                raise RuntimeError(
                    "DeepSeek-V4.1 bounded replay requires "
                    f"max_num_batched_tokens ({max_batched_tokens}) >= "
                    f"sliding_window + block alignment "
                    f"({int(private_config.window_size)} + {alignment}); disable "
                    "VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL or raise the token "
                    "budget."
                )
            threshold = getattr(vllm_config.scheduler_config, "long_prefill_token_threshold", 0)
            if 0 < threshold < int(private_config.window_size) + alignment:
                raise RuntimeError(
                    "Private circle replay requires long_prefill_token_threshold=0 "
                    "or >= sliding_window + block alignment"
                )
        coordinator.private_circle_group_ids = frozenset(private_group_ids)
        coordinator.private_circle_config = private_config
        coordinator.private_circle_pool = PrivateCirclePool(private_config)
        # A failed private-circle import is terminal: recomputing the prefix
        # would recreate the overwrite bug this protocol prevents.
        self.recompute_kv_load_failures = False


Request.__init__ = _patched_request_init
Scheduler.__init__ = _patched_scheduler_init
Scheduler._update_after_schedule = _patched_update_after_schedule
Scheduler._handle_invalid_blocks = _patched_handle_invalid_blocks
Scheduler._connector_finished = _patched_connector_finished
Scheduler._free_blocks = _patched_free_blocks
Scheduler._preempt_request = _patched_preempt_request
Scheduler._free_request_blocks = _patched_free_request_blocks
Scheduler._update_waiting_for_remote_kv = _patched_update_waiting_for_remote_kv
Scheduler.update_from_output = _patched_update_from_output
KVCacheManager.allocate_slots = _patched_allocate_slots
KVCacheManager.free = _patched_kv_cache_manager_free
KVCacheManager.get_computed_blocks = _patched_get_computed_blocks

# Unused in the v0.27 port: vLLM core now owns num_in_flight_tokens
# bookkeeping, the in-flight-aware remove_skipped_blocks() signature, and the
# admission-cap/max_memory_usage_bytes plumbing that the v0.23-era patch had
# to install manually.
