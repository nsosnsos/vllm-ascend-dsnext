# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 DSA metadata and fused attention execution.

The model file owns the network topology and projection modules.  This module
owns the attention execution boundary: it gathers every cache plane's metadata
before running the compressor, indexer and sparse-attention operators without
moving cache or scheduler knowledge back into the model.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch_npu
import custom_ops
import cann_ops_transformer
import cann_ops_transformer_experimental
from cann_ops_transformer_experimental.ops.attention.mixed_quant_sparse_flash_mla_dsl.mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla_metadata,
)
from cann_ops_transformer.ops.ds41 import (
    quant_lightning_indexer_metadata,
    quant_sparse_lightning_indexer_metadata,
)
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.logger import logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41A5DraftSWASpec,
    DeepseekV41CompressorStateSpec,
    DeepseekV41DraftSWASpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
    is_v41_draft_swa_spec,
)
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.rope_dsv4 import (
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)
from vllm_ascend import envs
from vllm_ascend.utils import npu_stream_switch

_V41_PREFILL_WORKSPACES: dict = {}
_V41_PREFILL_WORKSPACE_BLOCKS_CACHE: dict = {}


def _release_attention_stream_registrations(q: torch.Tensor, output: torch.Tensor) -> None:
    """Retire redundant allocator stream registrations after a native DSA call.

    The native ``mixed_quant_sparse_flash_mla`` bridge records its tensor
    arguments on the current stream.  On affected torch_npu versions those
    registrations can keep Q and the output storage pending after the device
    work has been queued.  A same-stream event orders the retirement without
    synchronizing the host.  Graph capture must retain the registrations and
    tensor addresses for replay, so this helper is deliberately a no-op while
    capturing.
    """
    if torch.npu.is_current_stream_capturing():
        return
    stream = torch.npu.current_stream(q.device)
    submitted = stream.record_event()
    stream.wait_event(submitted)
    torch_npu.npu.utils._erase_stream(q, stream)
    torch_npu.npu.utils._erase_stream(output, stream)


def _prefill_workspace_blocks(vllm_config: VllmConfig, block_size: int, window_size: int) -> int:
    key = (id(vllm_config), block_size)
    entry = _V41_PREFILL_WORKSPACE_BLOCKS_CACHE.get(key)
    if entry is not None:
        return entry[1]
    from vllm_ascend.core.private_circle_pool import compute_private_circle_prefill_workspace_blocks

    blocks = compute_private_circle_prefill_workspace_blocks(
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
        window_size=window_size,
        block_size=block_size,
    )
    # Strong ref to the config keeps the id() key unique.
    _V41_PREFILL_WORKSPACE_BLOCKS_CACHE[key] = (vllm_config, blocks)
    return blocks


def _get_private_circle_workspace(
    vllm_config: VllmConfig, dtype: torch.dtype, device: torch.device, layer_shape: torch.Size,
    num_blocks: int,
) -> torch.Tensor:
    key = (id(vllm_config), str(device), dtype, tuple(layer_shape))
    entry = _V41_PREFILL_WORKSPACES.get(key)
    workspace = entry[1] if entry is not None else None
    expected_shape = (num_blocks, *layer_shape)
    if workspace is None:
        workspace = torch.empty(expected_shape, dtype=dtype, device=device)
        _V41_PREFILL_WORKSPACES[key] = (vllm_config, workspace)
    elif tuple(workspace.shape) != expected_shape:
        raise RuntimeError(
            "V4.1 prefill workspace layout changed after initialization: "
            f"expected={expected_shape}, actual={tuple(workspace.shape)}"
        )
    return workspace


def _format_private_circle_slots(flat: torch.Tensor, block_size: int) -> torch.Tensor:
    slots = torch.full((flat.numel(), 2), -1, dtype=torch.int32, device=flat.device)
    valid = flat >= 0
    slots[:, 0].copy_(torch.where(valid, torch.div(flat, block_size, rounding_mode="floor"), -1))
    slots[:, 1].copy_(torch.where(valid, flat.remainder(block_size), -1))
    return slots


# Tail-window prefill kill switch, owned by the V4.1 target model: set when
# spec-decode capture layers beyond the boundary need full-prompt hidden
# states (EAGLE3-style drafts). The plan builder must follow the same
# decision as the model's boundary gather, and both live in V4.1 code —
# the model flips this at load time, so no common-module plumbing is needed.
_tail_window_prefill_blocked = False


def disable_tail_window_prefill() -> None:
    """Force the private-circle plan builder onto the full-prefill path.

    Called by DeepseekV41Model when its aux-capture conflict check disables
    the boundary gather; the plan and the gather must stay in lockstep or
    the post-boundary layers would read full rows the model never gathered.
    """
    global _tail_window_prefill_blocked
    _tail_window_prefill_blocked = True


def _private_circle_batch_rows(
    common: Any, num_reqs: int, flags: torch.Tensor
) -> list[str]:
    """Describe every batch row for the decode-first violation log."""
    req_ids = getattr(common, "private_circle_req_ids", None) or []
    num_computed = getattr(common, "private_circle_num_computed", None)
    num_prompt = getattr(common, "private_circle_num_prompt", None)
    qsl = getattr(common, "query_start_loc_cpu", None)
    qsl_values = qsl.tolist() if qsl is not None else None
    desc = []
    for i in range(num_reqs):
        parts = [
            f"row={i}",
            f"req={req_ids[i] if i < len(req_ids) else '?'}",
            f"prefill={bool(flags[i])}",
        ]
        if qsl_values is not None and i + 1 < len(qsl_values):
            parts.append(f"q={qsl_values[i + 1] - qsl_values[i]}")
        if num_computed is not None and i < len(num_computed):
            parts.append(f"c={num_computed[i]}")
        if num_prompt is not None and i < len(num_prompt):
            parts.append(f"p={num_prompt[i]}")
        desc.append(":".join(parts))
    return desc


def _assert_decode_rows_first(common: Any, num_reqs: int, num_decodes: int) -> None:
    """The positional split assumes decode rows precede prefill rows.

    vLLM does not guarantee this ordering; it holds under the default
    scheduler's admission dynamics but can break with
    long_prefill_token_threshold or the reordering schedulers.
    """
    is_prefilling = getattr(common, "is_prefilling", None)
    if (
        is_prefilling is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return
    flags = is_prefilling[:num_reqs].bool()
    if bool(flags[:num_decodes].any()):
        rows = [i for i in range(num_decodes) if bool(flags[i])]
        logger.error(
            "PRIVATE_CIRCLE_POOL batch_composition num_reqs=%d num_decodes=%d "
            "rows=[%s]",
            num_reqs,
            num_decodes,
            ", ".join(_private_circle_batch_rows(common, num_reqs, flags)),
        )
        raise RuntimeError(
            "V4.1 private circle pool requires decode rows before prefill "
            f"rows in the batch; prefill rows {rows[:8]} precede decode rows "
            "(see PRIVATE_CIRCLE_POOL batch_composition above; q=1 marks "
            "last-token-recompute rows, large q marks multi-step local "
            "prefill, '?' rows lack runner metadata). Disable "
            "long_prefill_token_threshold and the reordering schedulers "
            "(short_request_first / dyntra_lb / batch_job_aware / "
            "profiling_chunk / recompute) when the private circle pool is "
            "enabled."
        )


def _tail_row_layout(
    qsl_values: Sequence[int],
    num_decodes: int,
    num_reqs: int,
    window_size: int,
) -> tuple[list[int], list[int], list[int]]:
    """Rows, cumulative query layout and per-row keeps of a tail-window step.

    Mirrors the keep rule of ``_prepare_private_circle_plan`` (decode rows run
    in full, each prefill row keeps its last ``min(window, query_len)`` query
    rows). Kept as a free function because the index-K metadata may be built
    before the SWA group's plan exists (group order is not guaranteed).
    """
    qsl = np.asarray(qsl_values[: num_reqs + 1], dtype=np.int64)
    query_lens = np.diff(qsl)
    keeps = query_lens.copy()
    keeps[num_decodes:] = np.minimum(window_size, query_lens[num_decodes:])
    tail_qsl = np.concatenate(([0], np.cumsum(keeps)))

    decode_rows = np.arange(qsl[num_decodes], dtype=np.int64)
    if num_reqs == num_decodes:
        rows = decode_rows
    else:
        prefill_ends = qsl[num_decodes + 1 :]
        prefill_keeps = keeps[num_decodes:]
        prefill_row_count = int(prefill_keeps.sum())
        prefix_counts = np.cumsum(prefill_keeps) - prefill_keeps
        row_offsets = np.arange(prefill_row_count, dtype=np.int64) - np.repeat(prefix_counts, prefill_keeps)
        prefill_rows = np.repeat(prefill_ends - prefill_keeps, prefill_keeps) + row_offsets
        rows = np.concatenate((decode_rows, prefill_rows))

    return rows.tolist(), tail_qsl.tolist(), keeps.tolist()


def _copy_private_circle_plan_tensors(
    tensors: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Upload host-built plan fields with one pinned asynchronous copy per dtype.

    Each step owns fresh staging storage: a later step cannot overwrite an
    in-flight DMA source. The pinned allocator tracks the non-blocking copy
    before recycling that storage. Device views retain their packed allocation
    for as long as any plan consumer needs it. No device data is read on host.
    """
    if device.type == "cpu":
        return tensors
    groups: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in tensors.items():
        groups.setdefault(tensor.dtype, []).append((name, tensor))

    result = {}
    for dtype, fields in groups.items():
        numel = sum(tensor.numel() for _, tensor in fields)
        staging = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        host_values = staging.numpy()
        offset = 0
        for _, tensor in fields:
            end = offset + tensor.numel()
            host_values[offset:end] = tensor.numpy().reshape(-1)
            offset = end
        packed = torch.empty(numel, dtype=dtype, device=device)
        packed.copy_(staging, non_blocking=True)
        offset = 0
        for name, tensor in fields:
            end = offset + tensor.numel()
            result[name] = packed[offset:end].view(tensor.shape)
            offset = end
    return result


def _prepare_private_circle_plan(
    common: Any,
    shared: dict,
    vllm_config: VllmConfig,
    storage_block_size: int,
    blocks_per_allocation: int,
    num_real_reqs: int | None = None,
) -> dict:
    """Step-local compact workspace plan for private-circle Prefill.

    Prefill chunks longer than the ring would overwrite ring pages before FA
    reads them, so prefill rows execute from a shared compact workspace:
    restore the trailing history, write this step's KV, run FA, then commit
    only the trailing window to the private ring. Computed once per step and
    shared by every SWA layer; restore and commit run per layer. The planner
    is host-side, so it consumes the CPU mirrors supplied by the runner; the
    resulting restore/commit tensors remain device tensors.
    """
    cached = shared.get("private_circle_plan")
    if cached is not None:
        return cached
    window_size = int(_config_value(vllm_config.model_config.hf_text_config, "sliding_window", 0))
    block_size = storage_block_size
    # The plan is host-side, but the tensors used by the kernels live on the
    # device.  Always consume the runner-provided CPU mirrors here; calling
    # ``tolist()`` on query_start_loc/seq_lens would otherwise synchronize the
    # NPU in the middle of input preparation, which is especially visible on
    # PD-prefill steps.
    qsl_values = _host_list(common.query_start_loc_cpu, "query_start_loc")
    seq_cpu = common._seq_lens_cpu
    seq_values = _host_list(seq_cpu, "seq_lens")
    num_reqs = len(seq_values)
    padded_num_reqs = num_reqs
    if num_real_reqs is not None:
        # common.seq_lens carries padded rows on graph/DP-aligned steps;
        # padding rows hold no allocations and must not enter the plan.
        num_reqs = min(num_reqs, int(num_real_reqs))
    query_lens = [qsl_values[i + 1] - qsl_values[i] for i in range(num_reqs)]
    num_decodes, num_decode_tokens, _num_prefills, _num_prefill_tokens = _request_counts(common, num_reqs)
    _assert_decode_rows_first(common, num_reqs, num_decodes)
    draft_tokens = (
        getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
        if vllm_config.speculative_config is not None
        else 0
    )
    capacity = window_size - 1 + 1 + draft_tokens
    if blocks_per_allocation != (capacity + block_size - 1) // block_size:
        raise RuntimeError(
            "private circle blocks_per_allocation disagrees with the pool "
            f"layout: impl={(capacity + block_size - 1) // block_size}, "
            f"pool={blocks_per_allocation}."
        )

    bounded_replay_rows: set[int] = set()
    bounded_replay_start = common.private_circle_bounded_replay_start
    persistent_start = common.private_circle_persistent_start
    bounded_replay_start_cpu = common.private_circle_bounded_replay_start_cpu
    persistent_start_cpu = common.private_circle_persistent_start_cpu
    if bounded_replay_start is not None and persistent_start is not None:
        bounded_replay_values = _host_list(
            bounded_replay_start_cpu
            if bounded_replay_start_cpu is not None
            else bounded_replay_start,
            "bounded_replay_start",
        )
        persistent_values = _host_list(
            persistent_start_cpu
            if persistent_start_cpu is not None
            else persistent_start,
            "persistent_start",
        )
        bounded_replay_rows = {
            i
            for i in range(min(len(bounded_replay_values), len(persistent_values)))
            if 0 <= bounded_replay_values[i] < persistent_values[i]
        }
    allocation_ids_cpu = common.private_circle_allocation_ids_cpu
    if allocation_ids_cpu is None:
        raise RuntimeError(
            "V4.1 private-circle metadata missing allocation IDs CPU mirror"
        )
    allocation_values = _host_list(
        allocation_ids_cpu[:num_reqs], "allocation_ids"
    )
    max_workspace_blocks = _prefill_workspace_blocks(vllm_config, block_size, window_size)
    # A non-prefilling request schedules multiple tokens only when it carries
    # draft tokens (spec verification, q = 1 + num_draft_tokens). Mixed
    # verify+prefill steps must keep the full path: the runner's verify
    # branch and the drafter's next first pass address the batch by
    # full-row coordinates (aux captures, sampled-token rows), which the
    # gathered tail layout would break. Pure prefill (P-side) steps never
    # schedule multi-token decode rows, so this guard is inert there.
    is_prefilling = getattr(common, "is_prefilling", None)
    spec_verify_step = bool(
        is_prefilling is not None
        and any(
            not bool(is_prefilling[i]) and query_lens[i] > 1
            for i in range(num_decodes)
        )
    )
    # Tail-window prefill bookkeeping: every prefill row keeps only its last
    # ``window_size`` query rows for the layers after the last KV source.
    # Decode rows always run in full and stay at the front of the batch.
    tail_window = (
        envs.VLLM_ASCEND_ENABLE_TAIL_WINDOW_PREFILL
        and not _tail_window_prefill_blocked
        and not spec_verify_step
    )
    # Tail rows / cumulative query layout / per-row keeps come from the shared
    # helper so this plan and the index-source metadata cannot drift apart.
    tail_rows_all, tail_qsl_all, keeps_all = (
        _tail_row_layout(qsl_values, num_decodes, num_reqs, window_size)
        if tail_window
        else ([], [0], [])
    )
    tail_row_values: list[int] = tail_rows_all
    tail_prefill_keeps: list[int] = keeps_all[num_decodes:]
    tail_local_starts: list[int] = []

    page_bases: list[int] = []
    history_lens: list[int] = []
    allocation_bases: list[int] = []
    workspace_seq_lens: list[int] = []
    confirmed_lens: list[int] = []
    workspace_rows: list[int] = []
    next_page = 0
    for row in range(num_decodes, num_reqs):
        query_len = int(query_lens[row])
        seq_len = int(seq_values[row])
        confirmed = seq_len - query_len
        if confirmed < 0:
            raise RuntimeError(
                f"inconsistent V4.1 prefill positions: row={row}, "
                f"confirmed={confirmed}, query_len={query_len}, seq_len={seq_len}"
            )
        history_len = 0 if row in bounded_replay_rows else min(window_size - 1, confirmed)
        local_len = history_len + query_len
        pages = max(1, (local_len + block_size - 1) // block_size)
        allocation = int(allocation_values[row])
        if allocation < 0:
            raise RuntimeError(
                f"private circle row {row} has no allocation metadata"
            )
        private_block = 1 + allocation * blocks_per_allocation
        allocation_base = 1 + ((private_block - 1) // blocks_per_allocation) * blocks_per_allocation
        if next_page + pages > max_workspace_blocks:
            raise RuntimeError(
                "V4.1 private circle prefill workspace exhausted; batch "
                f"exceeds profiled capacity ({next_page + pages} > "
                f"{max_workspace_blocks} blocks)"
            )
        page_bases.append(next_page)
        history_lens.append(history_len)
        allocation_bases.append(allocation_base)
        workspace_seq_lens.append(local_len)
        confirmed_lens.append(confirmed)
        workspace_rows.append(row)
        next_page += pages

    device = common.seq_lens.device
    restore_src: list[int] = []
    restore_dst: list[int] = []
    current_dst: list[int] = []
    commit_src: list[int] = []
    commit_dst: list[int] = []
    max_pages_per_req = max(
        1, max(((n + block_size - 1) // block_size for n in workspace_seq_lens), default=1)
    )
    workspace_block_table = np.zeros((len(workspace_rows), max_pages_per_req), dtype=np.int32)
    for sub_row, row in enumerate(workspace_rows):
        query_len = query_lens[row]
        seq_len = seq_values[row]
        confirmed = confirmed_lens[sub_row]
        history_len = history_lens[sub_row]
        page_base = page_bases[sub_row]
        allocation_base = allocation_bases[sub_row]
        local_len = workspace_seq_lens[sub_row]
        pages = max(1, (local_len + block_size - 1) // block_size)
        workspace_block_table[sub_row, :pages] = np.arange(page_base, page_base + pages, dtype=np.int32)
        if tail_window:
            # Keep length comes from the shared tail layout; the workspace-local
            # start marks where this request's kept rows begin (rows between
            # history_len and here belong to this chunk but are never written,
            # i.e. the tail-window garbage region).
            keep = keeps_all[num_decodes + sub_row]
            tail_local_starts.append(history_len + query_len - keep)
        if row not in bounded_replay_rows:
            for j, absolute_pos in enumerate(range(confirmed - history_len, confirmed)):
                ring_block = (absolute_pos // block_size) % blocks_per_allocation
                restore_src.append((allocation_base + ring_block) * block_size + absolute_pos % block_size)
                restore_dst.append(page_base * block_size + j)
        current_dst.extend(page_base * block_size + history_len + j for j in range(query_len))
        keep = min(window_size, local_len)
        absolute_start = seq_len - keep
        local_start = local_len - keep
        for j in range(keep):
            absolute_pos = absolute_start + j
            ring_block = (absolute_pos // block_size) % blocks_per_allocation
            commit_src.append(page_base * block_size + local_start + j)
            commit_dst.append((allocation_base + ring_block) * block_size + absolute_pos % block_size)

    integer_dtypes = {torch.int32: np.int32, torch.int64: np.int64}
    seq_dtype = integer_dtypes[common.seq_lens.dtype]
    qsl_dtype = integer_dtypes[common.query_start_loc.dtype]
    host_tensors = {
        "restore_src": torch.from_numpy(np.asarray(restore_src, dtype=np.int64)),
        "restore_dst": torch.from_numpy(np.asarray(restore_dst, dtype=np.int64)),
        "commit_src": torch.from_numpy(np.asarray(commit_src, dtype=np.int64)),
        "commit_dst": torch.from_numpy(np.asarray(commit_dst, dtype=np.int64)),
        "prefill_slots_flat": torch.from_numpy(np.asarray(current_dst, dtype=np.int32)),
        "ws_block_table": torch.from_numpy(workspace_block_table),
        "ws_seq_lens": torch.from_numpy(np.asarray(workspace_seq_lens, dtype=seq_dtype)),
        "prefill_query_start_loc": torch.from_numpy(
            np.asarray(qsl_values[num_decodes:], dtype=qsl_dtype) - int(qsl_values[num_decodes])
        ),
        "ws_local_start_pos": torch.from_numpy(np.asarray(history_lens, dtype=np.int32)),
    }
    host_tensors["prefill_slots"] = _format_private_circle_slots(host_tensors["prefill_slots_flat"], block_size)
    plan = {
        "vllm_config": vllm_config,
        "window_size": window_size,
        "block_size": block_size,
        "blocks_per_allocation": blocks_per_allocation,
        "num_decodes": num_decodes,
        "num_decode_tokens": int(num_decode_tokens),
        "workspace_num_blocks": max_workspace_blocks,
    }
    total_tokens = int(qsl_values[num_reqs]) if num_reqs else 0
    if tail_window and len(tail_row_values) != total_tokens:
        # Identity (every request fits its window) keeps the legacy full path.
        all_keeps = keeps_all
        tail_qsl_values = tail_qsl_all
        host_tensors.update({
            # Full-batch rows (decode rows in place + prefill tail rows) the
            # post-boundary layers compute; sorted, so index_select keeps the
            # decode-first ordering the positional splits rely on.
            "tail_row_idx": torch.from_numpy(np.asarray(tail_row_values, dtype=np.int64)),
            # Prefill-row-local view of the same rows, for tensors aligned to
            # the prefill slot mapping rather than the full batch.
            "tail_prefill_local_idx": torch.from_numpy(
                np.asarray(tail_row_values[int(num_decode_tokens):], dtype=np.int64) - int(num_decode_tokens)
            ),
            # Query start loc over the tail rows only: prefill sub-batch (for
            # the SWA workspace FA call) and the full tail batch (for the
            # indexer's cu_seqlens_q, which spans decode rows too).
            "tail_query_start_loc": torch.from_numpy(
                np.asarray(tail_qsl_values[num_decodes:], dtype=qsl_dtype) - tail_qsl_values[num_decodes]
            ),
            # The full tail batch spans the real requests only, but consumers
            # pair it with batch-dimension tensors that keep the padded
            # request count (cache_seq_lens for the QLI/QSLI builds, the
            # impl's replaced query_start_loc feeding select_quantized).
            # Pad with the trailing value so padding requests carry
            # zero-length spans, mirroring how the full-batch query_start_loc
            # stays consistent with batch_size on graph/DP-aligned steps.
            "tail_query_start_loc_full": torch.from_numpy(np.asarray(
                tail_qsl_values + [tail_qsl_values[-1]] * (padded_num_reqs - num_reqs), dtype=qsl_dtype
            )),
            # Workspace-local start of the kept rows per prefill request.
            "tail_local_start": torch.from_numpy(np.asarray(tail_local_starts, dtype=np.int32)),
            # Workspace rows in [garbage_lo, garbage_hi) belong to this chunk
            # but are never written: their content is the previous step's
            # residue in the shared workspace. Window entries landing there
            # must be masked out, not read.
            "tail_garbage_lo": host_tensors["ws_local_start_pos"],
            "tail_garbage_hi": torch.from_numpy(np.asarray(tail_local_starts, dtype=np.int32)),
        })
        plan["tail_max_query_len"] = max(all_keeps, default=1)
        if logger.isEnabledFor(logging.DEBUG):
            req_parts = [
                f"row{row}:q={query_lens[row]},keep={tail_prefill_keeps[i]}"
                f",hist={history_lens[i]},garbage=[{history_lens[i]},{tail_local_starts[i]})"
                for i, row in enumerate(workspace_rows)
            ]
            shown = ", ".join(req_parts[:8]) + (", ..." if len(req_parts) > 8 else "")
            logger.debug(
                "V4.1 tail-window prefill plan: %d rows -> %d tail rows "
                "(prefill=%d, decode=%d, window=%d) %s",
                total_tokens,
                len(tail_row_values),
                len(workspace_rows),
                num_decodes,
                window_size,
                shown,
            )
    elif tail_window:
        logger.debug(
            "V4.1 tail-window prefill inactive: every request fits its "
            "window, full chunk kept"
        )
    plan.update(_copy_private_circle_plan_tensors(host_tensors, device))
    shared["private_circle_plan"] = plan
    return plan
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

V41_METADATA_BUFFER_SIZE = 1024
V41_ATTENTION_METADATA_PRIORITY = -20
V41_COMPRESSOR_METADATA_PRIORITY = -10


@eager_break_during_capture
def dsa_v41_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Execute V4.1 attention behind an explicit graph side-effect boundary."""
    forward_context = get_forward_context()
    attn = forward_context.no_compile_layers[layer_name]
    attn.v41_impl.forward(attn, None, hidden_states, output)



def dsa_v41_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="dsa_v41_forward",
    op_func=dsa_v41_forward,
    mutates_args=["output"],
    fake_impl=dsa_v41_forward_fake,
    dispatch_key="PrivateUse1",
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an HF config object or a raw config dict."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _host_list(value: Any, field_name: str) -> Any:
    """Convert host metadata to Python values without synchronizing NPU."""
    if value is None:
        raise RuntimeError(
            f"V4.1 private-circle metadata missing {field_name} CPU mirror"
        )
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise RuntimeError(
            f"V4.1 private-circle metadata {field_name} must use a CPU mirror"
        )
    return value.tolist()


def operator_max_seqlen_q(max_query_len: int) -> int:
    """Normalize a scheduler query length into the indexer operators' domain.

    The QLI/QSLI scheduling operators treat ``-1`` as "any length"; ``0`` is a
    degenerate bound that would produce an empty schedule. The metadata build
    and the kernel call must agree, so both go through this helper.
    """
    return max_query_len if max_query_len > 0 else -1


def _candidate_plane_ratio(text_config: Any) -> int | None:
    """Compression ratio of the index-K plane that emits candidate blocks.

    Only that plane's QLI runs with candidate attributes and only that plane
    feeds QSLI consumers, mirroring the reference recipe's split between
    ``c2a_qli_metadata`` and ``c1a_qli_candidate_metadata``.
    """
    source = _config_value(text_config, "candidate_source_layer_id")
    ratios = _config_value(text_config, "compress_ratios")
    if source is None or ratios is None:
        return None
    source = int(source)
    if not 0 <= source < len(ratios):
        return None
    return int(ratios[source])


def _candidate_geometry(text_config: Any) -> tuple[int, int]:
    """Candidate block count and size the indexer operators are tiled with."""
    return (
        int(_config_value(text_config, "candidate_topk_blocks")),
        int(_config_value(text_config, "candidate_block_size")),
    )


def _v41_metadata_geometry(vllm_config: VllmConfig) -> tuple[int, int, int, int]:
    """Batch-independent ``(window_size, n_local_heads, head_dim, index_topk)``."""
    text_config = vllm_config.model_config.hf_text_config
    return (
        int(_config_value(text_config, "sliding_window", 0)),
        int(_config_value(text_config, "num_attention_heads"))
        // vllm_config.parallel_config.tensor_parallel_size,
        int(_config_value(text_config, "head_dim")),
        int(_config_value(text_config, "index_topk")),
    )


def _qli_static_attrs(text_config: Any) -> dict[str, Any]:
    """Batch-independent QLI operator args; :meth:`_build_qli_metadata` layers
    the per-call ``max_seqlen_q``/``cmp_ratio``/candidate fields on top."""
    candidate_topk_blocks, candidate_block_size = _candidate_geometry(text_config)
    return dict(
        num_heads_q=int(_config_value(text_config, "index_n_heads")),
        num_heads_k=1,
        head_dim=int(_config_value(text_config, "index_head_dim")),
        topk=int(_config_value(text_config, "index_topk")),
        seqused_q=None,
        max_seqlen_k=-1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


@dataclass
class DeepseekV41Metadata(AttentionMetadata):
    """Scheduler and cache-plane contract for one V4.1 cache resource.

    ``seq_lens``/``query_start_loc`` always stay in original-token
    coordinates, matching the common vLLM metadata. The ``cache_*`` fields
    describe the rows visible to the concrete cache plane. Keeping both
    coordinate systems here lets future fused kernels replace the eager path
    without rebuilding scheduling metadata in the model.
    """

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    positions: torch.Tensor | None = None
    # Builder-prepared 1-D flat slots (page * storage_block_size + offset,
    # -1 for skipped rows) for the A5 quantized cache writers. ``None`` keeps
    # the legacy [T, 2] contract and per-call 2D->1D conversion.
    slot_mapping_flat: torch.Tensor | None = None
    # A5 C2 direct-write slots in compressor_v2 compact-row order. Unlike
    # ``slot_mapping_flat``, one row corresponds to a completed compression
    # group rather than an original token.
    compact_slot_mapping_flat: torch.Tensor | None = None
    cos: Any = None
    sin: Any = None
    num_actual_tokens: int = 0
    num_input_tokens: int = 0
    num_reqs: int = 0
    num_actual_reqs: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    logical_block_size: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    start_pos: torch.Tensor | None = None
    cache_seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    attn_state: Any = None
    is_prefilling: torch.Tensor | None = None
    causal: bool | torch.Tensor = True
    ori_win_left: int = 0
    ori_win_right: int = 0
    qli_metadata: torch.Tensor | None = None
    qsli_metadata: torch.Tensor | None = None
    # Tail-window variants: the index sources above the KV-source boundary
    # schedule the gathered tail rows, so their QLI/QSLI metadata must be
    # built from the tail row layout instead of the full batch.
    qli_metadata_tail: torch.Tensor | None = None
    qsli_metadata_tail: torch.Tensor | None = None
    query_start_loc_tail: torch.Tensor | None = None
    cmp_residual: torch.Tensor | None = None
    c2_ring_metadata: torch.Tensor | None = None
    c2_complete_mask: torch.Tensor | None = None
    c2_source_positions: torch.Tensor | None = None
    c2_source_cos: torch.Tensor | None = None
    c2_source_sin: torch.Tensor | None = None
    c2_compact_source_positions: torch.Tensor | None = None
    c2_compact_source_cos: torch.Tensor | None = None
    c2_compact_source_sin: torch.Tensor | None = None
    c2_metadata_group_id: int | None = None
    private_circle_plan: dict | None = None
    win_indices: torch.Tensor | None = None
    win_topk_length: torch.Tensor | None = None
    cmp_topk_lengths: dict[int, torch.Tensor] | None = None
    fa_metadata: dict[str, torch.Tensor] | None = None
    fa_metadata_group_id: int | None = None


@dataclass(frozen=True)
class DeepseekV41CompressorMetadata:
    """V4-shaped cache/state bundle consumed by the compressor stage."""

    cache: DeepseekV41Metadata
    state: DeepseekV41Metadata | None = None


@dataclass(frozen=True)
class DeepseekV41IndexerMetadata:
    """V4-shaped source cache bundle consumed by the indexer stage."""

    cache: DeepseekV41Metadata


@dataclass(frozen=True)
class DeepseekV41LayerMetadata:
    """All metadata consumed by one V4.1 attention layer invocation."""

    attention: DeepseekV41Metadata | None
    swa: DeepseekV41Metadata
    compressor: DeepseekV41CompressorMetadata | None
    indexer: DeepseekV41IndexerMetadata | None

    @property
    def positions(self) -> torch.Tensor:
        if self.swa.positions is None:
            raise RuntimeError("V4.1 SWA metadata does not contain input positions")
        return self.swa.positions

    def rope(self, layer_name: str, num_tokens: int):
        if self.swa.cos is None or self.swa.sin is None:
            raise RuntimeError("V4.1 SWA metadata does not contain RoPE tensors")
        return self.swa.cos[layer_name][:num_tokens], self.swa.sin[layer_name][:num_tokens]


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    if ratio not in (1, 2):
        raise ValueError("V4.1 only supports ratio 1 or 2")
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def _c2_compact_scatter_index(complete: torch.Tensor, capacity: int) -> torch.Tensor:
    """Map token rows to compressor_v2 compact rows without dropping rows.

    ``complete`` mirrors the compressor's closed-group policy. Cache-slot
    validity must not participate in this rank: a completed group with a -1
    physical slot still occupies a compact row and must scatter -1, otherwise
    later compact rows would shift and be written to the wrong cache slot.
    """
    rank = complete.cumsum(0, dtype=torch.int64) - 1
    return torch.where(
        complete,
        rank,
        torch.full_like(rank, capacity),
    )


def _c2_compact_capacity(num_input_tokens: int, num_reqs: int) -> int:
    return min(num_input_tokens, num_input_tokens // 2 + num_reqs)


def _build_c2_complete_mask(
    common: Any,
    num_input_tokens: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    *,
    skip_ring_state_update: bool,
) -> torch.Tensor:
    positions = getattr(common, "positions", None)
    if positions is None:
        raise RuntimeError("V4.1 C2 metadata requires token positions")

    valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
    valid = torch.arange(num_input_tokens, device=positions.device) < valid_end
    complete = (positions[:num_input_tokens].remainder(2) == 1) & valid
    if skip_ring_state_update:
        return torch.zeros_like(complete)
    return complete


def _cache_coordinates(common: Any, ratio: int, compressed: bool):
    """Build the two coordinate vectors consumed by V4.1 cache operators.

    ``query_lens`` is derived locally only to compute ``start_pos``.  It is no
    longer part of the metadata contract: Indexer kernels use the cumulative
    query boundaries directly, avoiding a transient graph input tensor.
    """
    query_start_loc = common.query_start_loc[: common.num_reqs + 1]
    seq_lens = common.seq_lens[: common.num_reqs]
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    start_pos = seq_lens - query_lens
    plane_ratio = ratio if compressed else 1
    cache_seq_lens = torch.div(seq_lens, plane_ratio, rounding_mode="floor")
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)

    return dict(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens_cpu=seq_lens_cpu,
        start_pos=start_pos,
        cache_seq_lens=cache_seq_lens,
    )


def _request_counts(common: Any, num_reqs: int):
    """Return V4-shaped request counters without synchronizing the NPU."""
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return 0, 0, 0, 0
    flags = is_prefilling[:num_reqs].bool()
    query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
    num_prefills = int(flags.sum().item())
    num_decodes = num_reqs - num_prefills
    num_prefill_tokens = int(query_lens_cpu[flags].sum().item())
    num_decode_tokens = int(query_lens_cpu[~flags].sum().item())
    return num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens


def _prefill_rows_fit_ring_capacity(
    common: Any,
    num_reqs: int,
    max_query_len: int,
) -> bool:
    """Whether every prefill row's query fits the ring's in-flight capacity.

    Rows within the per-step in-flight capacity (a PD request's
    last-token recompute, optionally fused with the first speculative
    draft block) write ring slots directly and need no shared prefill
    workspace, so their position in the batch does not matter.
    """
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return False
    flags = is_prefilling[:num_reqs].bool()
    starts = query_start_loc_cpu[: num_reqs + 1]
    query_lens = starts[1:] - starts[:-1]
    prefill_query_lens = query_lens[flags]
    return bool((prefill_query_lens <= max_query_len).all())


def _mask_below_draft_start(
    row: torch.Tensor, starts: torch.Tensor
) -> torch.Tensor:
    """Mark window entries below each request's draft-write start as -1.

    The draft planes hold no KV below ``starts[b]`` (the first local verify
    anchor): those ring slots carry residual KV from the allocation's
    previous owner, and the compacted -1 tail keeps the fused FA op from
    reading them.
    """
    if starts.device != row.device:
        raise RuntimeError(
            "V4.1 private-circle draft_write_start must already be on the "
            f"metadata device: starts={starts.device}, row={row.device}"
        )
    # Both tensors are integer device tensors. Comparing without ``to`` keeps
    # this path free of an implicit CPU/NPU conversion during metadata build.
    return torch.where(
        row < starts.unsqueeze(1),
        torch.full_like(row, -1),
        row,
    )


def _dspark_query_block(common: Any) -> tuple[int, int]:
    """Return ``(num_blocks, query_len)`` for a uniform draft query block.

    The host ``query_start_loc_cpu`` mirror is authoritative for the draft's
    per-request block width. ``num_reqs``/``num_input_tokens`` cannot be used
    here: under DP or graph-bucket padding the former can be the real request
    count while the latter counts every padded token, so their quotient is not
    the block width. Returns ``(0, 0)`` for a non-uniform layout (e.g. a
    prompt-chunk prefill step), so the caller keeps the causal window instead.
    """
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if query_start_loc_cpu is None or len(query_start_loc_cpu) < 2:
        return 0, 0
    query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    query_len = int(query_lens[0])
    if query_len <= 0 or not bool((query_lens == query_len).all()):
        return 0, 0
    return len(query_start_loc_cpu) - 1, query_len


def _validate_batch_layout(
    common: Any,
    *,
    num_reqs: int,
    num_actual_reqs: int,
    num_input_tokens: int,
    num_actual_tokens: int,
) -> None:
    """Validate the shape contract shared by every V4.1 cache plane.

    The values themselves are device-side and must not be read back here. We
    only validate static lengths, preventing one plane from silently receiving
    a differently padded view of the same flattened batch.
    """
    if not 0 <= num_actual_reqs <= num_reqs:
        raise ValueError(
            "V4.1 metadata has invalid request counts: "
            f"actual={num_actual_reqs}, padded={num_reqs}"
        )
    if not 0 <= num_actual_tokens <= num_input_tokens:
        raise ValueError(
            "V4.1 metadata has invalid token counts: "
            f"actual={num_actual_tokens}, input={num_input_tokens}"
        )
    query_start_loc = getattr(common, "query_start_loc", None)
    block_table = getattr(common, "block_table_tensor", None)
    seq_lens = getattr(common, "seq_lens", None)
    slot_mapping = getattr(common, "slot_mapping", None)
    positions = getattr(common, "positions", None)
    if query_start_loc is None or query_start_loc.ndim != 1 or query_start_loc.numel() < num_reqs + 1:
        raise ValueError("V4.1 metadata requires query_start_loc with num_reqs + 1 entries")
    if block_table is None or block_table.ndim != 2:
        raise ValueError("V4.1 metadata requires a rank-2 block table")
    missing_block_rows = num_reqs - block_table.shape[0]
    if missing_block_rows > 1 or (missing_block_rows == 1 and num_actual_reqs != num_reqs - 1):
        raise ValueError("V4.1 metadata requires a block table row for every padded request")
    if seq_lens is None or seq_lens.ndim != 1 or seq_lens.numel() < num_reqs:
        raise ValueError("V4.1 metadata requires seq_lens for every padded request")
    if slot_mapping is None or slot_mapping.ndim != 1 or slot_mapping.numel() < num_input_tokens:
        raise ValueError("V4.1 metadata requires a slot for every input token")
    if positions is not None and (positions.ndim != 1 or positions.numel() < num_input_tokens):
        raise ValueError("V4.1 metadata requires a position for every input token")


def scatter_cache_v2(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Store rows using builder-prepared coordinates and V4's Ascend op.

    V4.1 cache planes can be views into a larger layer-outermost slot, so the
    physical page stride is not necessarily the contiguous stride implied by
    the plane shape. ``npu_scatter_nd_update_v2`` preserves that stride and
    treats the builder's ``[-1, -1]`` coordinates as skipped rows, matching V4.
    """
    if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
        raise ValueError(
            f"V4.1 fused cache store requires builder-prepared [T, 2] slot_mapping, got {tuple(slot_mapping.shape)}"
        )
    cache = cache.squeeze(-2)
    indices = slot_mapping[: values.shape[0]]
    updates = values.to(cache.dtype).contiguous()
    torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, indices, updates)


def _slots_for_cache_write(metadata: "DeepseekV41Metadata", cache: torch.Tensor) -> torch.Tensor:
    """Pick the slot mapping form the A5 quantized cache writers consume.

    Prefers the builder-prepared 1-D flat slots (one build per cache group per
    step instead of one conversion per layer) and validates that the flat
    values were produced with the target cache's page size. Falls back to the
    legacy [T, 2] mapping, which the writers convert per call, for
    non-quantized cache planes.
    """
    flat = metadata.slot_mapping_flat
    if flat is not None:
        if metadata.storage_block_size != cache.shape[1]:
            raise ValueError(
                "V4.1 flat slot mapping was built for a different page size: "
                f"{metadata.storage_block_size} != {cache.shape[1]}"
            )
        return flat
    return metadata.slot_mapping


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert V4.1's compact [T, K] selection into SMLA [T, 1, topk]."""
    if indices.ndim != 2:
        raise ValueError(f"V4.1 sparse indices must be rank 2, got {indices.shape}")
    if indices.shape[-1] > topk:
        raise ValueError(f"V4.1 sparse indices width {indices.shape[-1]} exceeds operator topk {topk}")
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices.unsqueeze(1).contiguous().int()


class DeepseekV41EagerAttentionImpl:
    """V4-shaped execution boundary backed by fused Ascend operators.

    Projection, compressor and indexer modules remain registered by the model,
    while this object resolves the complete per-layer metadata bundle and owns
    their invocation order.  That is the same separation used by ``dsa_v1``:
    model construction is independent from cache-aware attention execution.
    """

    def __init__(self, prefix, role, topology, long_kv_source_prefix,
                 index_k_source_prefix, index_k_folded_source_prefix=None):
        self.prefix = prefix
        self.layer_name = f"{prefix}.attn"
        self.role = role
        self.topology = topology
        self.swa_prefix = f"{prefix}.swa_cache"
        self.long_kv_source_prefix = long_kv_source_prefix
        self.index_k_source_prefix = index_k_source_prefix
        self.index_k_folded_source_prefix = index_k_folded_source_prefix
        self.compressor_state_prefix = (
            f"{prefix}.compressor.state_cache" if role.is_kv_source and role.compress_ratio == 2 else None
        )

    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        try:
            swa = metadata[self.swa_prefix]
            long_kv = metadata[self.long_kv_source_prefix] if self.long_kv_source_prefix is not None else None
            index_k = metadata[self.index_k_source_prefix] if self.index_k_source_prefix is not None else None
            compressor_state = (
                metadata[self.compressor_state_prefix] if self.compressor_state_prefix is not None else None
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing V4.1 cache metadata for {exc.args[0]}") from exc
        return DeepseekV41LayerMetadata(
            attention=long_kv,
            swa=swa,
            compressor=(
                DeepseekV41CompressorMetadata(long_kv, compressor_state)
                if self.role.is_kv_source and long_kv is not None
                else None
            ),
            indexer=(DeepseekV41IndexerMetadata(index_k) if index_k is not None else None),
        )

    @staticmethod
    def _reshape_query_heads(q: torch.Tensor, head_dim: int) -> torch.Tensor:
        if head_dim <= 0 or q.shape[-1] % head_dim:
            raise ValueError(
                "V4.1 query projection width must be divisible by head_dim: "
                f"width={q.shape[-1]}, head_dim={head_dim}"
            )
        return q.unflatten(-1, (q.shape[-1] // head_dim, head_dim))

    @staticmethod
    def _project_q_kv(attn, hidden_states, cos, sin):
        q_a = attn.wq_a(hidden_states)
        qr = attn.q_norm(q_a)
        q = attn.wq_b(qr)
        q = DeepseekV41EagerAttentionImpl._reshape_query_heads(q, attn.head_dim)
        kv = attn.kv_norm(attn.wkv(hidden_states))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        kv = kv.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr, kv.squeeze(1)


    def _to_attention_cache_order(self, x: torch.Tensor) -> torch.Tensor:
        """Convert model [NoPE|RoPE] rows to the A5 cache [RoPE|NoPE] ABI."""
        if x.shape[-1] != 448 + 64:
            raise ValueError(
                f"A5 attention cache requires logical width 512, got {x.shape[-1]}"
            )

        return torch.cat(
            (
                x[..., 448:],   # RoPE：最后64维
                x[..., :448],   # NoPE：前448维
            ),
            dim=-1,
        )
    def _scatter_rows(self, cache: torch.Tensor, slot_mapping: torch.Tensor, rows: torch.Tensor) -> None:
        """Scatter packed rows into a possibly page-strided A3 slot view.

        Production metadata supplies ``[T, 2]`` page/offset coordinates and uses
        ``[-1, -1]`` for padded rows.  Keep that fixed-shape representation on NPU
        and delegate the skip/update semantics to the same ScatterNd primitive as
        the A3 index writer.  In particular, do not boolean-index device tensors:
        that lowers to dynamic ``Nonzero`` and cannot be captured by ACLGraph.
        """
        target = cache.squeeze(-2)
        slots = slot_mapping[: rows.shape[0]]
        if slots.ndim == 1:
            valid = slots >= 0
            physical = slots.clamp_min(0).long()
            pages = torch.div(physical, target.shape[1], rounding_mode="floor")
            offsets = physical.remainder(target.shape[1])
            indices = torch.stack((pages, offsets), dim=-1)
        elif slots.ndim == 2 and slots.shape[-1] == 2:
            valid = (slots >= 0).all(-1)
            indices = slots
        else:
            raise ValueError(f"A5 slot_mapping must be [T] or [T,2], got {tuple(slots.shape)}")

        # Normalize partially invalid coordinates as well as flat -1 slots to the
        # sentinel consumed by ScatterNd.  All tensors retain their static T shape.
        # CANN ScatterNd silently stops updating when an INT32-indexed target view
        # reaches 2 GiB.  V4.1's layer-outermost slots legitimately exceed that
        # size (the FP32 state ring gives the first three slots a 128 KiB page
        # stride), so use the operator's INT64 index path.  The page/row values are
        # still small; INT64 is required for the internal byte-offset calculation.
        indices = (
            torch.where(valid.unsqueeze(-1), indices, torch.full_like(indices, -1))
            .to(torch.int64)
            .contiguous()
        )
        updates = rows.to(target.dtype).contiguous()
        if target.device.type == "npu":
            torch_npu.npu_scatter_nd_update_(target, indices, updates)
            return

        # CPU-only deterministic golden used by unit tests.  Device execution must
        # stay on the fixed-shape branch above.
        if bool(valid.any()):
            target[indices[valid, 0].long(), indices[valid, 1].long()] = updates[valid]

    def _native_pack_rows(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        import custom_ops  # noqa: F401  # Registers torch.ops.custom.* from the wheel.

        if kind == "cmp":
            row_bytes, row_dtype, group_size, quant_mode = (
                320,
                torch.uint8,
                16,
                "mxfp4_bf16",
            )
        elif kind == "win":
            # The Host ABI validates FP8 cache dtype even though vLLM stores this
            # physical plane as opaque bytes.  Reinterpret the completed row as
            # U8 before scattering it into the shared 544-byte cache slot.
            row_bytes, row_dtype, group_size, quant_mode = (
                544,
                torch.float8_e4m3fn,
                32,
                "mxfp8_bf16",
            )
        else:
            raise ValueError(f"Unsupported A5 packed cache kind: {kind}")
        # V2 addresses cache as a flat row table.  Use local flat slots here and
        # scatter into vLLM's possibly page-strided cache only after the operator
        # has produced rows.  The attention ABI stores RoPE before NoPE.
        rows = torch.empty((x.shape[0], row_bytes), dtype=row_dtype, device=x.device)
        slots = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
        torch.ops.custom.kv_compress_epilog_v2(
            rows,
            x,
            slots,
            quant_group_size=group_size,
            quant_mode=quant_mode,
            round_scale=True,
            x_scale=1.0,
        )
        return rows.view(torch.uint8)

    def _write_attention_cache(
            self,
            cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            values: torch.Tensor,
            *,
            kind: str,
            backend: str = "reference",
    ) -> str:
        """Write cmp/win rows and return the backend that actually executed."""
        if backend not in ("reference", "auto", "native"):
            raise ValueError(f"Unsupported A5 cache-writer backend: {backend}")
        if values.shape[0] == 0:
            return "reference" if backend == "reference" else backend
        if backend != "reference":
            try:
                # 小算子实现
                # rows = self._native_pack_rows(values, kind)
                # self._scatter_rows(cache, slot_mapping, rows)
                # 融合算子实现
                if slot_mapping.ndim == 2:
                    slot = (slot_mapping[:, 0]) * cache.shape[1] + slot_mapping[:, 1]
                    slot_mapping = slot.clamp(min=-1).to(torch.int32)
                elif slot_mapping.ndim != 1:
                    raise ValueError(
                        f"A5 cache-writer slot mapping must be [T, 2] or [T], got ndim={slot_mapping.ndim}"
                    )
                # 1-D builder-prepared flat slots are consumed as-is.
                if kind == "cmp":
                    group_size, quant_mode = (16, "mxfp4_bf16",)
                elif kind == "win":
                    group_size, quant_mode = (32, "mxfp8_bf16",)
                    # 算子校验要求
                    cache = cache.view(torch.float8_e4m3fn)
                else:
                    raise ValueError(f"Unsupported A5 packed cache kind: {kind}")
                torch.ops.custom.kv_compress_epilog_v2(
                cache,
                values,
                slot_mapping,
                quant_group_size=group_size,
                quant_mode=quant_mode,
                round_scale=True,
                x_scale=1.0,
                )
                return "native"
            except Exception:
                if backend == "native":
                    raise
        return "reference"

    def _get_private_circle_workspace(self, plan: dict, cache: torch.Tensor) -> torch.Tensor:
        return _get_private_circle_workspace(
            plan["vllm_config"],
            cache.dtype,
            cache.device,
            cache.shape[1:],
            plan["workspace_num_blocks"],
        )

    def _scatter_swa_kv(self, attn, kv: torch.Tensor, swa_metadata) -> None:
        """Write this step's KV: decode rows to the ring, prefill to workspace.

        The pool decides WHERE (ring/workspace); the base `_write_attention_cache`
        decides HOW (A5 quantized pack).
        """
        cache = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        plan = getattr(swa_metadata, "private_circle_plan", None)
        if plan is None:
            self._write_attention_cache(
                cache,
                _slots_for_cache_write(swa_metadata, cache),
                kv,
                kind="win",
                backend="native",
            )
            return
        workspace = self._get_private_circle_workspace(plan, cache)
        restore_src = plan["restore_src"]
        if restore_src.numel() > 0:
            # Byte views: lossless for low-precision caches.
            workspace.flatten(0, 1).view(torch.uint8).index_copy_(
                0,
                plan["restore_dst"],
                cache.flatten(0, 1).view(torch.uint8).index_select(0, restore_src),
            )
        num_decode_tokens = plan["num_decode_tokens"]
        if num_decode_tokens:
            self._write_attention_cache(
                cache,
                _slots_for_cache_write(swa_metadata, cache)[:num_decode_tokens],
                kv[:num_decode_tokens],
                kind="win",
                backend="native",
            )
        if kv.shape[0] > num_decode_tokens:
            prefill_slots = _slots_for_cache_write(swa_metadata, workspace)[num_decode_tokens:]
            # The shared state is the activation signal: it reflects what the
            # model actually gathered, not what the plan describes.
            tail_local_idx = (
                plan.get("tail_prefill_local_idx")
                if attn.shared_state.tail_row_idx is not None
                else None
            )
            if tail_local_idx is not None:
                prefill_slots = prefill_slots.index_select(0, tail_local_idx)
            self._write_attention_cache(
                workspace,
                prefill_slots,
                kv[num_decode_tokens:],
                kind="win",
                backend="native",
            )

    def _commit_private_circle_prefill(self, plan: dict, cache: torch.Tensor, workspace: torch.Tensor) -> None:
        """Commit only the post-chunk tail window to the private ring."""
        commit_src = plan["commit_src"]
        if commit_src.numel() == 0:
            return
        cache.flatten(0, 1).view(torch.uint8).index_copy_(
            0,
            plan["commit_dst"],
            workspace.flatten(0, 1).view(torch.uint8).index_select(0, commit_src),
        )

    def preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Project Q/KV and populate this layer's SWA cache on the current stream."""
        q, qr, kv = self._project_q_kv(attn, hidden_states, cos, sin)
        self._scatter_swa_kv(attn, kv, swa_metadata)
        return q, qr

    def multistream_preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Overlap Q Vector work with KV Cube work, then reverse their roles.

        Reuse V1's stream and projection wrappers. V4.1 keeps floating-point
        qr for its indexer and has no post-Wq_b Q RMSNorm. Stage events serialize
        the Cube matmuls; the final join makes SWA writes visible to attention.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        v1_impl = attn.dsa_attn.dsa_attn.impl
        wq_a, wkv, wq_b = v1_impl.cv_wq_a, v1_impl.cv_wkv, v1_impl.cv_wq_b
        share_quant = (
            type(wq_a._quant_method) is type(wkv._quant_method) and wq_a._has_communication == wkv._has_communication
        )

        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        q_quant, q_scale = wq_a.quantize(hidden_states)
        kv_quant_done = None
        if share_quant:
            kv_quant, kv_scale = q_quant, q_scale
        else:
            q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                aux_stream.wait_event(q_quant_done)
                kv_quant, kv_scale = wkv.quantize(hidden_states)
                kv_quant_done = aux_stream.record_event()
        q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)

        # Part 2: Q normalization/quantization (Vector) overlaps KV matmul (Cube).
        part2_start = main_stream.record_event()
        if kv_quant_done is not None:
            main_stream.wait_event(kv_quant_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part2_start)
            kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
            kv_matmul_done = aux_stream.record_event()
        qr = attn.q_norm(q_a)
        q_b_quant, q_b_scale = wq_b.quantize(qr)

        # Part 3: Q_b matmul (Cube) overlaps KV norm, RoPE and cache store (Vector).
        part3_start = main_stream.record_event()
        main_stream.wait_event(kv_matmul_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part3_start)
            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
            self._scatter_swa_kv(attn, kv.squeeze(1), swa_metadata)
        q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias)
        q = self._reshape_query_heads(q, attn.head_dim)
        main_stream.wait_stream(aux_stream)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        q = q.to(hidden_states.dtype)
        main_stream.wait_stream(aux_stream)
        return q, qr

    def _write_compressed_source(
        self,
        attn,
        hidden_states,
        positions,
        cos,
        sin,
        metadata,
    ):
        compressor = attn.compressor
        if compressor is None or metadata.compressor is None or metadata.indexer is None:
            raise RuntimeError("V4.1 KV source is missing compressor or source metadata")
        if attn.indexer is None:
            raise RuntimeError("V4.1 KV source is missing its indexer")
        compressor_metadata = metadata.compressor
        indexer_metadata = metadata.indexer
        # A5 quantized writers take the builder-prepared flat slots; other
        # planes keep the [T, 2] mapping. Both forms carry identical values.
        indexer_k_cache = attn.indexer.k_cache.kv_cache[0][0]
        long_kv_cache = attn.long_kv_cache.kv_cache[0]
        index_slots = _slots_for_cache_write(indexer_metadata.cache, indexer_k_cache)[: positions.shape[0]]
        long_slots = _slots_for_cache_write(compressor_metadata.cache, long_kv_cache)[: positions.shape[0]]
        ratio = self.role.compress_ratio
        if ratio == 1:
            latent = compressor(hidden_states)
            # C1 source positions are the current token positions. Reuse the
            # query RoPE selected by the SWA metadata builder instead of
            # indexing the global table a second time.
            source_cos = cos
            source_sin = sin
        else:
            if compressor_metadata.state is None:
                raise RuntimeError("V4.1 ratio-2 source is missing compressor-state metadata")
            state_metadata = compressor_metadata.state
            if state_metadata.c2_ring_metadata is None or state_metadata.c2_metadata_group_id is None:
                raise RuntimeError("V4.1 ring compressor metadata is missing")
            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)
            latent = compressor.pool_projected(hidden_states, state_metadata)
            if get_ascend_device_type() == AscendDeviceType.A5:
                compact_rows = latent.shape[0]
                index_slots = indexer_metadata.cache.compact_slot_mapping_flat
                long_slots = compressor_metadata.cache.compact_slot_mapping_flat
                source_cos = state_metadata.c2_compact_source_cos
                source_sin = state_metadata.c2_compact_source_sin
                if (
                    index_slots is None
                    or long_slots is None
                    or source_cos is None
                    or source_sin is None
                ):
                    raise RuntimeError("V4.1 A5 compact C2 metadata is incomplete")
                if (
                    index_slots.shape[0] != compact_rows
                    or long_slots.shape[0] != compact_rows
                    or source_cos.shape[0] != compact_rows
                    or source_sin.shape[0] != compact_rows
                ):
                    raise RuntimeError(
                        "V4.1 A5 compact C2 metadata rows do not match compressor output: "
                        f"latent={compact_rows}, index_slots={index_slots.shape[0]}, "
                        f"long_slots={long_slots.shape[0]}, "
                        f"source_cos={source_cos.shape[0]}, "
                        f"source_sin={source_sin.shape[0]}"
                    )
                index_slots = index_slots[:compact_rows]
                long_slots = long_slots[:compact_rows]
                source_cos = source_cos[:compact_rows]
                source_sin = source_sin[:compact_rows]
            else:
                source_cos = state_metadata.c2_source_cos
                source_sin = state_metadata.c2_source_sin
                if source_cos is None or source_sin is None:
                    fallback_cos, fallback_sin = get_cos_and_sin_dsa(
                        state_metadata.c2_source_positions
                    )
                    source_cos = fallback_cos[attn.rotary_emb.layername]
                    source_sin = fallback_sin[attn.rotary_emb.layername]
                source_cos = source_cos[: positions.shape[0]]
                source_sin = source_sin[: positions.shape[0]]

        attn.indexer.update_keys(
            latent,
            index_slots,
            source_cos,
            source_sin,
        )
        latent = latent.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            latent.unsqueeze(1),
            source_cos,
            source_sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        if long_kv_cache.dtype == torch.uint8:
            # A5 quantized path: the epilog op packs mxfp4 rows into the
            # flat uint8 view and expects the RoPE dims first.
            try:
                import custom_ops  # noqa: F401  registers torch.ops.custom.*
            except ImportError as exc:
                raise RuntimeError(
                    "DeepSeek V4.1 compressed-KV store requires the custom_ops module "
                    "registering torch.ops.custom.kv_compress_epilog_v2."
                ) from exc
            self._write_attention_cache(
                long_kv_cache,
                long_slots,
                latent.squeeze(1),
                kind="cmp",
                backend="native",
            )
        else:
            scatter_cache_v2(
                long_kv_cache,
                long_slots,
                latent.squeeze(1),
            )

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        # Tail-window steps address the shared buffers through the gathered
        # rows instead of the batch prefix: reads scatter-gather, writes land
        # back on the exact tail rows.
        tail_idx = shared.tail_row_idx
        if not self.role.is_index_source:
            if tail_idx is None:
                return shared.topk_indices[: hidden_states.shape[0]]
            return shared.topk_indices.index_select(0, tail_idx)
        if attn.indexer is None or metadata.indexer is None:
            raise RuntimeError("V4.1 index source is missing indexer metadata")

        source_metadata = metadata.indexer.cache
        candidate_indices = shared.candidate_indices[: hidden_states.shape[0]]
        candidate_lengths = shared.candidate_lengths[: hidden_states.shape[0]]
        if tail_idx is not None:
            plan = metadata.swa.private_circle_plan
            # The indexer's query layout must follow the tail rows; every
            # cache-side field (block table, seq lens) keeps its original
            # absolute-token coordinates.
            if (
                source_metadata.qli_metadata is not None
                and source_metadata.query_start_loc_tail is None
            ):
                raise RuntimeError(
                    "V4.1 tail-window index source is missing its tail QLI/QSLI "
                    "scheduling metadata; the full-batch metadata would mis-tile "
                    "the tail rows."
                )
            source_metadata = replace(
                source_metadata,
                query_start_loc=source_metadata.query_start_loc_tail,
                max_query_len=plan["tail_max_query_len"],
                qli_metadata=source_metadata.qli_metadata_tail,
                qsli_metadata=source_metadata.qsli_metadata_tail,
            )
            candidate_indices = shared.candidate_indices.index_select(0, tail_idx)
            candidate_lengths = shared.candidate_lengths.index_select(0, tail_idx)

        context = get_forward_context().no_compile_layers
        source_layer = context[self.index_k_source_prefix]
        folded_source = (
            context[self.index_k_folded_source_prefix]
            if self.role.uses_candidate_filter
            and self.index_k_folded_source_prefix is not None
            else None
        )
        selected, candidate_indices, candidate_lengths = attn.indexer.select(
            hidden_states,
            qr,
            positions,
            cos,
            sin,
            source_layer.kv_cache[0],
            source_metadata,
            is_candidate_source=self.role.is_candidate_source,
            uses_candidate_filter=self.role.uses_candidate_filter,
            candidate_topk_blocks=self.topology.candidate_topk_blocks,
            candidate_block_size=self.topology.candidate_block_size,
            candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths,
            source_cache_folded=(
                folded_source.kv_cache[0] if folded_source is not None else None
            ),
        )
        if tail_idx is None:
            shared.topk_indices[: selected.shape[0]].copy_(selected)
        else:
            shared.topk_indices.index_copy_(0, tail_idx, selected)
        if self.role.is_candidate_source:
            shared.candidate_indices[: candidate_indices.shape[0]].copy_(candidate_indices)
            shared.candidate_lengths[: candidate_indices.shape[0]].copy_(
                candidate_lengths.reshape(candidate_indices.shape[0], -1)[:, :1]
                .clamp_(max=self.topology.candidate_topk_blocks)
            )
        if tail_idx is None:
            return shared.topk_indices[: selected.shape[0]]
        return shared.topk_indices.index_select(0, tail_idx)

    def _attention(self, attn, q, metadata, compressed_indices):
        source_cache = None
        if self.role.has_long_context:
            source_cache = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache[0]
        return self._native_attention(
            attn,
            q,
            metadata,
            source_cache=source_cache,
            compressed_indices=compressed_indices,
        )


    def _get_window_topk_idxs(
            self,
            attn,
            q: torch.Tensor,
            swa_metadata,
            num_reqs: int,
            query_start_loc: torch.Tensor,
            *,
            positions: torch.Tensor | None = None,
            garbage_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute sliding-window KV slot indices in TNK format.

        When ``positions`` is provided (absolute token positions), a direct
        vectorised path mirroring the reference recipe's
        ``generate_win_topk_ids`` is used — no ``searchsorted``, no
        per-request index reconstruction.  When ``positions`` is ``None``
        (private-circle prefill with workspace-local coordinates), the
        original reconstruction from ``start_pos`` + ``query_start_loc`` is
        used instead.

        ``garbage_bounds`` (per-request ``[lo, hi)`` workspace rows) masks
        window entries that fall inside this chunk's never-written rows —
        tail-window prefill only writes the trailing rows, and the shared
        workspace holds the previous step's residue everywhere else.

        Returns a ``[total_tokens, 1, window_size]`` int32 tensor
        where rows are concatenated across requests (TNK layout).  ``-1``
        marks a slot that holds nothing or padding.
        """
        window_size = attn.window_size
        total_tokens = q.shape[0]
        device = q.device
        if total_tokens == 0:
            return torch.full(
                (total_tokens, 1, window_size),
                -1,
                dtype=torch.int32,
                device=device,
            )
        if positions is not None:
            positions = positions.to(torch.int32)
            cols = torch.arange(window_size, device=device, dtype=torch.int32)
            valid_len = torch.clamp(positions + 1, max=window_size)
            window_start = positions - valid_len + 1
            win_indices = window_start.unsqueeze(1) + cols.unsqueeze(0)
            valid_mask = cols.unsqueeze(0) < valid_len.unsqueeze(1)
            return torch.where(
                valid_mask, win_indices, torch.full_like(win_indices, -1)
            ).unsqueeze(1)
        start_pos = swa_metadata.start_pos
        if start_pos is None:
            return torch.full(
                (total_tokens, 1, window_size),
                -1,
                dtype=torch.int32,
                device=device,
            )
        global_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
        token_req_idx = torch.searchsorted(
            query_start_loc[1:num_reqs + 1], global_indices, right=True
        )
        token_pos = global_indices - query_start_loc[token_req_idx].to(torch.int32)
        token_sp = start_pos[token_req_idx].to(torch.int32)
        cols = torch.arange(window_size, device=device, dtype=torch.int32)
        token_abs_pos = token_sp + token_pos
        window_start = torch.clamp(token_abs_pos - window_size + 1, min=0)
        win_indices = window_start.unsqueeze(1) + cols.unsqueeze(0)
        win_indices = win_indices.masked_fill(
            win_indices > token_abs_pos.unsqueeze(1), -1
        )
        if garbage_bounds is not None:
            garbage_lo, garbage_hi = garbage_bounds
            lo = garbage_lo[token_req_idx].unsqueeze(1)
            hi = garbage_hi[token_req_idx].unsqueeze(1)
            win_indices = win_indices.masked_fill(
                (win_indices >= lo) & (win_indices < hi), -1
            )
            # The FA kernel consumes only the first ``win_topk_length`` entries
            # of each row, so valid slots must form an ascending row prefix.
            # Without the garbage region, -1s only ever appear as a suffix
            # (causal mask over ascending entries); masking the garbage region
            # punches -1s into the middle of the row, so compact the valid
            # slots back to the front before the operator reads them.
            sentinel = torch.iinfo(win_indices.dtype).max
            sort_key = torch.where(
                win_indices < 0,
                torch.full_like(win_indices, sentinel),
                win_indices,
            )
            win_indices = torch.sort(sort_key, dim=1).values
            win_indices = torch.where(
                win_indices == sentinel,
                torch.full_like(win_indices, -1),
                win_indices,
            )
        return win_indices.unsqueeze(1).to(torch.int32)

    def _get_cached_win_indices(
            self,
            attn,
            q: torch.Tensor,
            swa_metadata,
            num_reqs: int,
            query_start_loc: torch.Tensor,
            *,
            positions: torch.Tensor | None = None,
            garbage_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
            sub_batch: str = "full",
    ) -> torch.Tensor:
        """Resolve window indices through the step-local cache.

        The window indices depend only on step-level quantities
        (``positions``, ``start_pos``, ``query_start_loc``,
        ``window_size``), so — like the FA scheduling metadata — they are
        built once per ``(step, sub_batch)`` and reused by every layer
        instead of being recomputed per attention layer.
        """
        cache_lookup = getattr(getattr(attn, "shared_state", None),
                               "fa_metadata", None)
        if cache_lookup is None:
            return self._get_window_topk_idxs(
                attn, q, swa_metadata, num_reqs, query_start_loc,
                positions=positions,
                garbage_bounds=garbage_bounds,
            )
        return cache_lookup(
            get_forward_context(),
            ("win_indices", sub_batch),
            lambda: self._get_window_topk_idxs(
                attn, q, swa_metadata, num_reqs, query_start_loc,
                positions=positions,
                garbage_bounds=garbage_bounds,
            ),
        )

    def _prepare_cmp_attention(
        self,
        metadata,
        source_cache,
        compressed_indices,
        num_reqs: int,
    ):
        """Shared compressed-KV preparation for the sparse MLA stages."""
        if self.role.compress_ratio not in (1, 2):
            return None, None, None
        if source_cache is None or metadata.attention is None or compressed_indices is None:
            raise RuntimeError("V4.1 compressed attention is missing KV or TopK metadata")
        cmp_block_table = metadata.attention.block_table[:num_reqs]
        cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs]
        cmp_topk = self.topology.index_topk
        if cmp_topk not in (512, 1024):
            raise ValueError(f"SparseFlashMla only supports TopK 512 or 1024, got {cmp_topk}")
        return pad_sparse_indices(compressed_indices, cmp_topk), cmp_block_table, cmp_seq_lens

    def _private_circle_attention(
        self,
        attn,
        q,
        metadata,
        plan: dict,
        source_cache,
        compressed_indices,
    ):
        """Mixed batches split at num_decodes: decode rows run on the ring
        with absolute seq_lens, prefill rows on the compact workspace so long
        chunks never overwrite ring pages before FA reads them."""
        has_compressed = self.role.compress_ratio in (1, 2)
        num_reqs = metadata.swa.num_reqs
        num_decodes = plan["num_decodes"]
        num_decode_tokens = min(plan["num_decode_tokens"], q.shape[0])
        cache = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        workspace = self._get_private_circle_workspace(plan, cache)

        cmp_indices, cmp_block_table, cmp_seq_lens = self._prepare_cmp_attention(
            metadata, source_cache, compressed_indices, num_reqs
        )
        if cmp_indices is not None:
            decode_cmp_indices = cmp_indices[:num_decode_tokens]
            decode_cmp_block_table = cmp_block_table[:num_decodes]
            decode_cmp_seq_lens = cmp_seq_lens[:num_decodes]
            prefill_cmp_indices = cmp_indices[num_decode_tokens:]
            prefill_cmp_block_table = cmp_block_table[num_decodes:]
            prefill_cmp_seq_lens = cmp_seq_lens[num_decodes:]
        else:
            decode_cmp_indices = decode_cmp_block_table = decode_cmp_seq_lens = None
            prefill_cmp_indices = prefill_cmp_block_table = prefill_cmp_seq_lens = None

        outputs = []
        if num_decodes:
            if metadata.swa.start_pos is None:
                raise RuntimeError(
                    "V4.1 private circle decode is missing SWA start_pos metadata"
                )
            decode_qsl = metadata.swa.query_start_loc[: num_decodes + 1]
            decode_start_pos = metadata.swa.start_pos[:num_decodes]
            outputs.append(
                self._sparse_mla_attention(
                    attn,
                    q[:num_decode_tokens],
                    win_kv=cache,
                    win_block_table=metadata.swa.block_table[:num_decodes],
                    cu_seqlens_q=decode_qsl,
                    seqused_win_kv=metadata.swa.seq_lens[:num_decodes],
                    win_indices=self._get_cached_win_indices(
                        attn,
                        q[:num_decode_tokens],
                        SimpleNamespace(start_pos=decode_start_pos),
                        num_decodes,
                        decode_qsl,
                        positions=metadata.swa.positions[:num_decode_tokens]
                        if metadata.swa.positions is not None else None,
                        sub_batch="decode",
                    ),
                    cmp_indices=decode_cmp_indices,
                    cmp_block_table=decode_cmp_block_table,
                    seqused_cmp_kv=decode_cmp_seq_lens,
                    source_cache=source_cache,
                    has_compressed=has_compressed,
                    sub_batch="decode",
                )
            )
        if q.shape[0] > num_decode_tokens:
            prefill_q = q[num_decode_tokens:]
            tail_local_idx = (
                plan.get("tail_prefill_local_idx")
                if attn.shared_state.tail_row_idx is not None
                else None
            )
            if tail_local_idx is not None:
                # Tail-window step: the workspace geometry is unchanged (the
                # kept rows land on the same workspace slots a full chunk
                # would use), only the query rows and their window indices
                # shrink to the gathered tail rows.
                tail_sub_batch = "prefill_tail"
                prefill_qsl = plan["tail_query_start_loc"]
                prefill_num_reqs = prefill_qsl.shape[0] - 1
                win_indices = self._get_cached_win_indices(
                    attn,
                    prefill_q,
                    SimpleNamespace(start_pos=plan["tail_local_start"]),
                    prefill_num_reqs,
                    prefill_qsl,
                    garbage_bounds=(plan["tail_garbage_lo"], plan["tail_garbage_hi"]),
                    sub_batch=tail_sub_batch,
                )
            else:
                tail_sub_batch = "prefill"
                prefill_qsl = plan["prefill_query_start_loc"]
                win_indices = self._get_cached_win_indices(
                    attn,
                    prefill_q,
                    SimpleNamespace(start_pos=plan["ws_local_start_pos"]),
                    num_reqs - num_decodes,
                    prefill_qsl,
                    sub_batch=tail_sub_batch,
                )
            outputs.append(
                self._sparse_mla_attention(
                    attn,
                    prefill_q,
                    win_kv=workspace,
                    win_block_table=plan["ws_block_table"],
                    cu_seqlens_q=prefill_qsl,
                    seqused_win_kv=plan["ws_seq_lens"],
                    win_indices=win_indices,
                    cmp_indices=prefill_cmp_indices,
                    cmp_block_table=prefill_cmp_block_table,
                    seqused_cmp_kv=prefill_cmp_seq_lens,
                    source_cache=source_cache,
                    has_compressed=has_compressed,
                    sub_batch=tail_sub_batch,
                )
            )
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        self._commit_private_circle_prefill(plan, cache, workspace)
        return output

    def _build_fa_op_metadata(
            self,
            attn,
            q: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            win_topk_length: torch.Tensor,
            cmp_topk_length: torch.Tensor | None,
            has_compressed: bool,
    ):
        """Invoke the FA scheduling-metadata operator once.

        The metadata operator declares ``cmp_topk_length`` as a required
        positional input even on a window-only plane, so an absent compressed
        plane is described by a zero row of the window lengths.  The buffer is
        materialized here, inside the step-local cache-miss path, instead of on
        every layer: each attention layer re-enters ``_fa_op_metadata`` but only
        the first one runs this body.
        """
        if cmp_topk_length is None:
            cmp_topk_length = torch.zeros_like(win_topk_length)
        return mixed_quant_sparse_flash_mla_metadata(
            win_topk_length,
            cmp_topk_length,
            cu_seqlens_q=cu_seqlens_q,
            num_heads_q=q.shape[1],
            num_heads_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0].shape[2],
            head_dim=q.shape[-1],
            quant_mode=1,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=has_compressed,
        )

    def _fa_op_metadata(
            self,
            attn,
            q: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            win_topk_length: torch.Tensor,
            cmp_topk_length: torch.Tensor | None,
            has_compressed: bool,
            sub_batch: str,
    ):
        """Resolve the step-local FA scheduling metadata for one attention type.

        The metadata operator only consumes step-level quantities: the
        window/compressed topk lengths, ``cu_seqlens_q`` and the head layout
        shared by every layer.  Following the reference recipe's
        ``generate_kernel_metadata``, it is therefore invoked once per
        attention type per sub-batch each step -- a fixed three calls (win,
        c2a, c1a) on full-batch steps -- and every layer reuses the cached
        result instead of re-deriving it per attention layer.
        """
        cache_lookup = getattr(getattr(attn, "shared_state", None), "fa_metadata", None)
        if cache_lookup is None:
            return self._build_fa_op_metadata(
                attn, q, cu_seqlens_q, win_topk_length, cmp_topk_length, has_compressed
            )
        attention_type = "win" if not has_compressed else f"c{self.role.compress_ratio}a"
        return cache_lookup(
            get_forward_context(),
            (attention_type, sub_batch),
            lambda: self._build_fa_op_metadata(
                attn, q, cu_seqlens_q, win_topk_length, cmp_topk_length, has_compressed
            ),
        )

    def _sparse_mla_attention(
        self,
        attn,
        q: torch.Tensor,
        *,
        win_kv: torch.Tensor,
        win_block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_win_kv: torch.Tensor,
        win_indices: torch.Tensor,
        cmp_indices: torch.Tensor | None,
        cmp_block_table: torch.Tensor | None,
        seqused_cmp_kv: torch.Tensor | None,
        source_cache,
        has_compressed: bool,
        sub_batch: str = "full",
        precomputed_swa: DeepseekV41Metadata | None = None,
    ) -> torch.Tensor:
        """One mixed_quant_sparse_flash_mla call: shared by the base full-batch
        path and the private-circle decode/prefill sub-batches.  The FA
        scheduling metadata comes from the step-local cache keyed by
        ``(attention type, sub_batch)``, or from the builder-pre-computed
        metadata when available (full-batch path only)."""
        if (
                precomputed_swa is not None
                and sub_batch == "full"
                and precomputed_swa.fa_metadata is not None
                and precomputed_swa.win_topk_length is not None
        ):
            n = q.shape[0]
            win_topk_length = precomputed_swa.win_topk_length[:n]
            ratio = self.role.compress_ratio
            if has_compressed and precomputed_swa.cmp_topk_lengths is not None:
                cmp_topk_length = precomputed_swa.cmp_topk_lengths[ratio][:n]
            else:
                # A window-only plane has no compressed indices, so the
                # operator argument is dropped below. A compressed plane whose
                # builder did not publish its lengths keeps the zero row as a
                # defensive fallback.
                cmp_topk_length = None if not has_compressed else torch.zeros_like(win_topk_length)
            attention_type = "win" if ratio == 0 else f"c{ratio}a"
            op_metadata = precomputed_swa.fa_metadata[attention_type]
        else:
            win_topk_length = (win_indices >= 0).sum(dim=-1).to(torch.int32)
            cmp_topk_length = (
                None
                if cmp_indices is None
                else (cmp_indices >= 0).sum(dim=-1).to(torch.int32)
            )
            op_metadata = self._fa_op_metadata(
                attn,
                q,
                cu_seqlens_q,
                win_topk_length,
                cmp_topk_length,
                has_compressed,
                sub_batch,
            )
        cmp_topk_length = None if cmp_indices is None else cmp_topk_length
        output, _ = cann_ops_transformer.ops.ds41.mixed_quant_sparse_flash_mla(
            q,
            ori_kv=win_kv,
            cmp_kv=source_cache,
            ori_sparse_indices=win_indices,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=win_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_ori_kv=seqused_win_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            ori_topk_length=win_topk_length,
            cmp_topk_length=cmp_topk_length,
            sinks=attn.attn_sink.data,
            metadata=op_metadata,
            quant_mode=1,
            softmax_scale=attn.softmax_scale,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_softmax_lse=False,
        )
        _release_attention_stream_registrations(q, output)
        return output

    def _native_attention(
        self,
        attn,
        q,
        metadata,
        *,
        source_cache,
        compressed_indices,
    ):
        """Run SparseFlashMla with the same PA metadata for both operator stages."""
        if attn.head_dim != 512:
            raise ValueError(f"SparseFlashMla requires head_dim 512, got {attn.head_dim}")
        if attn.window_size != 128:
            raise ValueError(f"A2/A3 SparseFlashMla requires sliding_window 128, got {attn.window_size}")
        if not 1 <= attn.n_local_heads <= 128 or attn.n_local_heads & (attn.n_local_heads - 1):
            raise ValueError(
                "A2/A3 SparseFlashMla requires the local query-head count to be "
                f"a power of two in [1, 128], got {attn.n_local_heads}"
            )
        has_compressed = self.role.compress_ratio in (1, 2)
        private_circle_plan = getattr(metadata.swa, "private_circle_plan", None)
        if private_circle_plan is not None:
            return self._private_circle_attention(
                attn, q, metadata, private_circle_plan, source_cache, compressed_indices
            )
        num_reqs = metadata.swa.num_reqs
        query_start_loc = metadata.swa.query_start_loc[: num_reqs + 1]
        cmp_indices, cmp_block_table, cmp_seq_lens = self._prepare_cmp_attention(
            metadata, source_cache, compressed_indices, num_reqs
        )
        # Full-batch target steps precompute both win_indices and FA metadata;
        # DSpark draft steps precompute only win_indices (their FA metadata is
        # cheap and stays in-graph). Either way the graph reads a builder-owned
        # buffer instead of a per-step allocation.
        precomputed_swa = metadata.swa if (
                metadata.swa.fa_metadata is not None or metadata.swa.win_indices is not None
        ) else None
        if precomputed_swa is not None and precomputed_swa.win_indices is not None:
            if precomputed_swa.fa_metadata_group_id is not None:
                wait_for_device_metadata(
                    DeviceMetadataStage.ATTENTION, precomputed_swa.fa_metadata_group_id
                )
            win_indices = precomputed_swa.win_indices[:q.shape[0]]
        else:
            win_indices = self._get_cached_win_indices(
                attn, q, metadata.swa, num_reqs, query_start_loc,
                positions=metadata.swa.positions[:q.shape[0]]
                if metadata.swa.positions is not None else None,
            )
        return self._sparse_mla_attention(
            attn,
            q,
            win_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
            win_block_table=metadata.swa.block_table[:num_reqs],
            cu_seqlens_q=query_start_loc,
            seqused_win_kv=metadata.swa.seq_lens[:num_reqs],
            win_indices=win_indices,
            cmp_indices=cmp_indices,
            cmp_block_table=cmp_block_table,
            seqused_cmp_kv=cmp_seq_lens,
            source_cache=source_cache,
            has_compressed=has_compressed,
            precomputed_swa=precomputed_swa,
        )


    @staticmethod
    def update_graph_params(*args, **kwargs):
        """V4.1 owns stable metadata buffers; no backend pointer patch is needed."""
        return None

    def _project_output(self, attn, attention_output, hidden_states, *, projected):
        """Write the O-projection result into a caller-supplied buffer.
        When the attention output has fewer rows than ``hidden_states``
        (e.g. graph padding), zero-pad before the matmul so the stale tail
        of ``projected`` is never returned to the model.
        """
        padded = attention_output
        if attention_output.shape[0] != hidden_states.shape[0]:
            padded = attention_output.new_zeros(
                (hidden_states.shape[0], attention_output.shape[1], attention_output.shape[2])
            )
            padded[: attention_output.shape[0]] = attention_output
        attn.dsa_attn.dsa_attn.impl._forward_o_proj(padded, projected)
        return projected

    def forward(self, attn, positions, hidden_states, output: torch.Tensor | None = None):
        if output is None:
            output = torch.empty_like(hidden_states)
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            output.zero_()
            return output
        metadata = self._get_layer_metadata(forward_context.attn_metadata)
        shared = attn.shared_state
        if shared is not None and metadata.swa.private_circle_plan is not None:
            # The model reads the step plan at the head/tail boundary, after
            # the head layers have run. Plan-less layers (e.g. the draft SWA
            # group, which is excluded from the pool plan by design) must not
            # clear it.
            shared.step_private_circle_plan = metadata.swa.private_circle_plan
        tail_idx = shared.tail_row_idx if shared is not None else None
        if tail_idx is not None:
            if tail_idx.numel() != hidden_states.shape[0]:
                raise RuntimeError(
                    "V4.1 tail-window row count does not match the hidden "
                    f"states: tail={tail_idx.numel()}, hidden={hidden_states.shape[0]}"
                )
            # Tail-window layers address the batch through the gathered tail
            # rows. Positions keep their original absolute values; cos/sin
            # are batch-row-aligned tables and follow the same gather.
            positions = metadata.positions.index_select(0, tail_idx)
            full_cos, full_sin = metadata.rope(
                attn.rotary_emb.layername, metadata.positions.shape[0]
            )
            cos = full_cos.index_select(0, tail_idx)
            sin = full_sin.index_select(0, tail_idx)
        else:
            positions = metadata.positions[: hidden_states.shape[0]]
            cos, sin = metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
        v1_impl = attn.dsa_attn.dsa_attn.impl
        preprocess = self.multistream_preprocess if v1_impl.multistream_dsv4_dsa_overlap else self.preprocess
        q, qr = preprocess(attn, hidden_states, cos, sin, metadata.swa)
        if self.role.is_kv_source:
            self._write_compressed_source(
                attn,
                hidden_states,
                positions,
                cos,
                sin,
                metadata,
            )
        compressed_indices = self._select_sparse_indices(attn, hidden_states, qr, positions, cos, sin, metadata)
        attention_output = self._attention(attn, q, metadata, compressed_indices)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            attention_output.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        self._project_output(attn, attention_output, hidden_states, projected=output)
        return output


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
        max_reqs = getattr(vllm_config.scheduler_config, "max_num_seqs", 256)
        self._supports_device_ops = getattr(device, "type", "cpu") != "cpu"
        # Metadata consumed from inside an ACLGraph must not point at a
        # per-step cast/advanced-index result. Keep positions in a stable
        # builder-owned buffer just like the cache coordinates below.
        self._positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self._slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self._slot_mapping_2d = torch.full((max_tokens, 2), -1, dtype=torch.int32, device=device)
        self._slot_mapping_flat = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
        # Persistent private-circle prefill slot buffers.  The planner is
        # rebuilt per step, but the slot layout capacity is static; allocating
        # these tensors in every metadata build causes allocator/mapmem work
        # in the input-preparation path.
        if isinstance(kv_cache_spec, DeepseekV41SWASpec):
            self._private_circle_slots_2d = torch.empty(
                (max_tokens, 2), dtype=torch.int32, device=device
            )
            self._private_circle_slots_flat = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )
        else:
            self._private_circle_slots_2d = None
            self._private_circle_slots_flat = None
        self._block_table: torch.Tensor | None = None
        self._seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._start_pos = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cache_seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cmp_residual = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._qli_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._qsli_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._qli_metadata_tail = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._qsli_metadata_tail = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        is_compressor_state = isinstance(kv_cache_spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(kv_cache_spec, "compress_ratio", 1)
        is_compressed_cache = isinstance(
            kv_cache_spec, (DeepseekV41FullSpec, DeepseekV41IndexerSpec)
        )
        self._use_compact_c2_slots = (
            get_ascend_device_type() == AscendDeviceType.A5
            and is_compressed_cache
            and ratio == 2
        )
        self._use_compact_c2_rope = (
            get_ascend_device_type() == AscendDeviceType.A5 and is_compressor_state
        )
        self._c2_ring_metadata = (
            torch.zeros(5 * max_reqs, dtype=torch.int32, device=device)
            if is_compressor_state
            else None
        )
        self._c2_complete_mask = (
            torch.zeros(max_tokens, dtype=torch.bool, device=device)
            if is_compressor_state
            else None
        )
        self._compact_slot_mapping_flat = None
        self._c2_source_positions = (
            torch.zeros(max_tokens, dtype=torch.int64, device=device)
            if is_compressor_state and not self._use_compact_c2_rope
            else None
        )
        self._c2_compact_source_positions = (
            torch.zeros(max_tokens + 1, dtype=torch.int64, device=device)
            if self._use_compact_c2_rope
            else None
        )
        text_config = vllm_config.model_config.hf_text_config
        self._candidate_ratio = _candidate_plane_ratio(text_config)
        (
            self._window_size,
            self._n_local_heads,
            self._head_dim,
            self._index_topk,
        ) = _v41_metadata_geometry(vllm_config)
        self._qli_static_attrs = _qli_static_attrs(text_config)
        rope_dim = int(
            _config_value(
                text_config,
                "qk_rope_head_dim",
                _config_value(text_config, "head_dim"),
            )
        )
        c2_rope_rows = (
            max_tokens if self._supports_device_ops and isinstance(kv_cache_spec, DeepseekV41CompressorStateSpec) else 0
        )
        self._c2_source_cos = torch.ones(
            (c2_rope_rows, 1, 1, rope_dim),
            dtype=torch.float32,
            device=device,
        )
        self._c2_source_sin = torch.zeros_like(self._c2_source_cos)
        self._c2_rope_layer_names = tuple(
            name.removesuffix(".compressor.state_cache") + ".attn"
            for name in layer_names
            if name.endswith(".compressor.state_cache")
        )
        self._c2_full_source_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()
        self._fa_precomputed = isinstance(kv_cache_spec, DeepseekV41SWASpec) and self._supports_device_ops
        if self._fa_precomputed:
            window_size = self._window_size
            self._win_indices_buf = torch.full(
                (max_tokens, 1, window_size), -1, dtype=torch.int32, device=device,
            )
            self._win_topk_length_buf = torch.zeros(
                (max_tokens, 1), dtype=torch.int32, device=device,
            )
            self._cmp_topk_length_bufs: dict[int, torch.Tensor] = {
                r: torch.zeros((max_tokens, 1), dtype=torch.int32, device=device)
                for r in (1, 2)
            }
            self._fa_metadata_bufs: dict[str, torch.Tensor] = {}
            self._fa_metadata_sentinel = torch.zeros(1, dtype=torch.int32, device=device)
        # DSpark draft layers own a distinct win_indices layout: each draft
        # query reads a block-wide window (the window before the block plus the
        # block's own queries) instead of its causal prefix. Precompute it in
        # the metadata build, mirroring the reference ``build_sliding_topk_ids``,
        # so the draft graph replays a stable buffer address with no host
        # round-trip and no per-step allocation.
        self._dspark_precomputed = is_v41_draft_swa_spec(kv_cache_spec)
        self._dspark_max_query_len = 0
        if self._dspark_precomputed:
            spec_config = vllm_config.speculative_config
            num_speculative_tokens = (
                int(getattr(spec_config, "num_speculative_tokens", 0)) if spec_config is not None else 0
            )
            # The widest draft block is the anchor plus every speculative token;
            # shorter blocks pad the tail with -1 and stay graph-stable.
            self._dspark_max_query_len = num_speculative_tokens + 1
            window_size = self._window_size
            self._dspark_win_indices_buf = torch.full(
                (max_tokens, 1, window_size + self._dspark_max_query_len),
                -1,
                dtype=torch.int32,
                device=device,
            )
            # Cached model-static geometry so the per-step draft build
            # allocates nothing here.
            self._dspark_offsets = torch.arange(
                -window_size, 0, dtype=torch.int32, device=device
            )
            self._dspark_slots = torch.arange(
                window_size + self._dspark_max_query_len,
                dtype=torch.int64,
                device=device,
            )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ) -> DeepseekV41Metadata:
        # Direct callers of the capture hook do not necessarily pass the
        # runtime-mode keyword used by model_runner. Capture must nevertheless
        # receive stable RoPE buffers, exactly like FULL replay.
        kwargs.setdefault("full_graph_mode", True)
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if isinstance(self.kv_cache_spec, DeepseekV41CompressorStateSpec):
            if not self._c2_rope_layer_names:
                raise RuntimeError("V4.1 compressor-state builder has no source RoPE layer")
            source_rope = get_full_cos_and_sin_dsa_for_layer(self._c2_rope_layer_names[0])
            for rope_layer_name in self._c2_rope_layer_names[1:]:
                other_rope = get_full_cos_and_sin_dsa_for_layer(rope_layer_name)
                if any(other.data_ptr() != source.data_ptr() for other, source in zip(other_rope, source_rope)):
                    raise RuntimeError("V4.1 ratio-2 source layers must share one RoPE table")
            self._c2_full_source_rope = source_rope

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def _publish_task(
        self,
        shared: dict[str, Any],
        key: str,
        buffer: torch.Tensor,
        stage: DeviceMetadataStage,
        run,
        priority: int = 0,
    ) -> torch.Tensor:
        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer
        if self._device_metadata_enabled:
            self._device_metadata_tasks = (
                *self._device_metadata_tasks,
                DeviceMetadataTask(stage, run, id(buffer), priority=priority),
            )
        else:
            run()
        return buffer

    def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
        if not is_v41_draft_swa_spec(self.kv_cache_spec):
            raise TypeError("V4.1 drafting requires a draft SWA cache")
        # DSpark issues one eager block per step. Group-local tables and slots
        # remain independent; the builder owns the operator metadata buffers.
        # kwargs (shared step caches, spec-decode args) flow through so the
        # draft reuses the same step-scoped construction as the target.
        return self.build(0, common_attn_metadata, **kwargs)

    def _build_kernel_metadata(
            self,
            positions: torch.Tensor,
            num_tokens: int,
            num_actual_tokens: int,
            num_reqs: int,
            cu_seqlens_q: torch.Tensor,
            window_size: int,
            num_heads_q: int,
            head_dim: int,
            index_topk: int,
    ) -> None:
        """Pre-compute win_indices, topk lengths and FA metadata once per step.

        Mirrors the reference recipe's ``build_attn_metadata`` +
        ``generate_kernel_metadata`` flow: window indices and compressed
        top-k lengths are derived from positions alone, then the FA
        scheduling-metadata operator is invoked a fixed three times (win,
        c2a, c1a).  Results are written into the builder's persistent
        buffers (``_win_indices_buf``, ``_win_topk_length_buf``,
        ``_cmp_topk_length_bufs``, ``_fa_metadata_bufs``) so that every
        attention layer reuses them instead of re-deriving per layer.

        When ``_device_metadata_enabled`` is True the call is deferred via
        ``_publish_task`` and runs on a separate NPU stream; the layer
        forward must ``wait_for_device_metadata`` before consuming the
        buffers.

        Rows past ``num_actual_tokens`` belong to graph/DP padding requests:
        their positions are stale (the runner rewrites only the first
        ``total_num_scheduled_tokens`` entries each step) while the builder
        zeroes those requests' ``seq_lens``, so the indexer emits no valid
        sparse indices for them. Represent those rows as empty windows so
        win/cmp top-k lengths are zero and the FA kernel never dereferences
        their all-invalid (-1) index slots.
        """
        device = positions.device
        pos_i32 = positions[:num_tokens].to(torch.int32)
        if num_actual_tokens < num_tokens:
            pad_rows = torch.arange(num_tokens, device=device) >= num_actual_tokens
            pos_i32 = pos_i32.masked_fill(pad_rows, -1)
        cols = torch.arange(window_size, device=device, dtype=torch.int32)
        valid_len = torch.clamp(pos_i32 + 1, max=window_size)
        window_start = pos_i32 - valid_len + 1
        win_ids = window_start.unsqueeze(1) + cols.unsqueeze(0)
        valid_mask = cols.unsqueeze(0) < valid_len.unsqueeze(1)
        win_indices = self._win_indices_buf[:num_tokens]
        win_indices.copy_(
            torch.where(valid_mask, win_ids, torch.full_like(win_ids, -1)).unsqueeze(1)
        )
        win_topk_length = self._win_topk_length_buf[:num_tokens]
        win_topk_length.copy_(valid_len.unsqueeze(1).to(torch.int32))

        cmp_topk_lengths: dict[int, torch.Tensor] = {}
        for r in self._cmp_topk_length_bufs:
            buf = self._cmp_topk_length_bufs[r][:num_tokens]
            valid_length = (pos_i32 + 1) // r
            if index_topk > 0:
                valid_length = valid_length.clamp_max(index_topk)
            buf.copy_(valid_length.unsqueeze(1).to(torch.int32))
            cmp_topk_lengths[r] = buf

        for ratio, attn_type in ((0, "win"), (2, "c2a"), (1, "c1a")):
            has_cmp = ratio > 0
            cmp_tl = (
                cmp_topk_lengths[ratio] if has_cmp
                else torch.zeros_like(win_topk_length)
            )
            value = mixed_quant_sparse_flash_mla_metadata(
                win_topk_length,
                cmp_tl,
                cu_seqlens_q=cu_seqlens_q,
                num_heads_q=num_heads_q,
                num_heads_kv=1,
                head_dim=head_dim,
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=has_cmp,
            )
            buf = self._fa_metadata_bufs.get(attn_type)
            if buf is None or buf.shape != value.shape:
                buf = value.clone()
            else:
                buf.copy_(value)
            self._fa_metadata_bufs[attn_type] = buf

    def _build_dspark_win_indices(
            self,
            positions: torch.Tensor,
            num_tokens: int,
            num_blocks: int,
            query_len: int,
            draft_write_start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-compute the DSpark draft block window.

        Mirrors the reference recipe's ``build_sliding_topk_ids``: every draft
        query of a request reads one block-wide window -- the window of
        positions before the block's first query, followed by the block's own
        query positions. Negative entries are stably compacted to the tail and
        marked -1, so each row's leading valid prefix is exactly the window the
        fused FA op consumes.

        ``draft_write_start`` (per request, from the private-circle metadata)
        masks every position below the request's first local verify anchor:
        the draft planes hold no KV there (PD imports cover the target planes
        only), so an unmasked read would see ring residual from the previous
        owner of the allocation.

        Pure device arithmetic: no ``.item()`` and no host tensor, so the draft
        graph can replay it and the metadata build never triggers an H2D copy.
        """
        if num_blocks <= 0 or query_len <= 0 or num_tokens != num_blocks * query_len:
            raise ValueError(
                "DSpark draft metadata requires a uniform query block per "
                f"request: tokens={num_tokens}, blocks={num_blocks}, width={query_len}"
            )
        if query_len > self._dspark_max_query_len:
            raise ValueError(
                "DSpark draft query block exceeds the configured speculative "
                f"width: {query_len} > {self._dspark_max_query_len}"
            )
        if num_tokens > self._dspark_win_indices_buf.shape[0]:
            # A silent short slice would surface later as a confusing copy_
            # shape mismatch inside the captured draft graph.
            raise ValueError(
                "DSpark draft win_indices buffer is smaller than the padded "
                f"token count: tokens={num_tokens}, "
                f"capacity={self._dspark_win_indices_buf.shape[0]}"
            )
        pos = positions[:num_tokens].reshape(num_blocks, query_len).to(torch.int32)
        start_pos = pos[:, :1]
        prefix_pos = start_pos + self._dspark_offsets.unsqueeze(0)
        pad_len = self._dspark_max_query_len - query_len
        if pad_len:
            pos = F.pad(pos, (0, pad_len), value=-1)
        row = torch.cat((prefix_pos, pos), dim=1)
        if draft_write_start is not None:
            # Pad rows beyond the metadata length mask nothing (threshold 0);
            # their positions are padding and never dereference the ring.
            starts = draft_write_start[:num_blocks]
            if starts.numel() < num_blocks:
                starts = F.pad(starts, (0, num_blocks - starts.numel()))
            row = _mask_below_draft_start(row, starts)
        # Compact the negative entries to the tail by index arithmetic instead
        # of a width-``window + max_q`` stable argsort. ``dest`` (a bijection)
        # is each element's output slot: non-negatives keep place-order,
        # negatives land after all of them, reproducing the argsort layout.
        neg = row < 0
        rank_neg = neg.to(torch.int64).cumsum(dim=1) - 1
        n_keep = row.shape[1] - (rank_neg[:, -1:] + 1)
        dest = torch.where(neg, n_keep + rank_neg, self._dspark_slots - rank_neg - 1)
        compacted = torch.empty_like(row)
        compacted.scatter_(1, dest, row)
        compacted.masked_fill_(compacted < 0, -1)
        row = compacted
        win_indices = self._dspark_win_indices_buf[:num_tokens]
        # Broadcast the block rows across the queries instead of a
        # ``repeat_interleave`` copy; the explicit block axis keeps ``copy_``
        # from aligning it against the query dimension.
        win_indices.view(num_blocks, query_len, 1, row.shape[1]).copy_(
            row[:, None, None, :]
        )
        return win_indices

    def _tail_query_start_loc(
        self,
        common: Any,
        num_actual_reqs: int | None,
        window_size: int,
        cache_rows: int,
        dtype: torch.dtype,
        shared: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        """Tail-window query layout of this step, or ``None`` when inert.

        Locally derived through ``_tail_row_layout`` because this group may be
        built before the SWA group's plan exists. On success also stashes the
        gathered tail rows in ``self._tail_rows`` for the candidate plane.
        """
        self._tail_rows = None
        if (
            not envs.VLLM_ASCEND_ENABLE_TAIL_WINDOW_PREFILL
            or _tail_window_prefill_blocked
            or window_size <= 0
        ):
            return None
        qsl_cpu = getattr(common, "query_start_loc_cpu", None)
        if qsl_cpu is None:
            return None
        # Request count as the plan sees it (real requests of this step). The
        # CPU seq_lens mirror can be shorter than the device tensor, so it is
        # not used for the count.
        num_reqs = (
            int(num_actual_reqs)
            if num_actual_reqs is not None
            else len(qsl_cpu) - 1
        )
        qsl_values = _host_list(qsl_cpu[: num_reqs + 1], "query_start_loc")
        num_decodes, _, _, _ = _request_counts(common, num_reqs)
        is_prefilling = getattr(common, "is_prefilling", None)
        if is_prefilling is not None and any(
            not bool(is_prefilling[i])
            and (int(qsl_values[i + 1]) - int(qsl_values[i])) > 1
            for i in range(num_decodes)
        ):
            # Mixed verify+prefill steps keep the full path.
            return None
        tail_rows, tail_qsl_values, _ = _tail_row_layout(
            qsl_values, num_decodes, num_reqs, window_size
        )
        if len(tail_rows) == int(qsl_values[num_reqs]):
            # Every request already fits its window: no tail layout.
            return None
        if shared is not None:
            step_plan = shared.get("private_circle_plan")
            if isinstance(step_plan, dict):
                plan_qsl = step_plan.get("tail_query_start_loc_full")
                plan_rows = step_plan.get("tail_row_idx")
                if plan_qsl is not None and plan_rows is not None:
                    self._tail_rows = plan_rows
                    return plan_qsl
        layout_key = f"tail_index_layout:{dtype}:{cache_rows}:{num_actual_reqs}"
        if shared is not None:
            cached_layout = shared.get(layout_key)
            if cached_layout is not None:
                self._tail_rows = cached_layout["tail_rows"]
                return cached_layout["query_start_loc"]
        # The tail layout must cover real tokens only and stay strictly
        # ascending: the caller feeds it to ``index_select`` and passes its
        # length as both num_tokens and num_actual_tokens, so a padding row or
        # a duplicate would silently mis-address the gathered rows.
        real_tokens = int(qsl_values[num_reqs])
        tail_rows_arr = np.asarray(tail_rows, dtype=np.int64)
        if (
            tail_rows_arr.size == 0
            or tail_rows_arr[0] < 0
            or tail_rows_arr[-1] >= real_tokens
            or np.any(np.diff(tail_rows_arr) <= 0)
        ):
            raise RuntimeError(
                "V4.1 tail-window rows must be unique, ascending and inside "
                "the real token range"
            )
        device = common.seq_lens.device
        # Complete padding from scheduler-known host values. Using qsl[-1]
        # after H2D would read a device scalar back to the host; constructing
        # tensors directly on device would also use blocking copies.
        qsl_dtype = {torch.int32: np.int32, torch.int64: np.int64}[dtype]
        host_qsl = np.full(max(cache_rows + 1, len(tail_qsl_values)), tail_qsl_values[-1], dtype=qsl_dtype)
        host_qsl[: len(tail_qsl_values)] = tail_qsl_values
        uploaded = _copy_private_circle_plan_tensors(
            {
                "query_start_loc": torch.from_numpy(host_qsl),
                "tail_rows": torch.from_numpy(tail_rows_arr),
            },
            device,
        )
        self._tail_rows = uploaded["tail_rows"]
        if shared is not None:
            shared[layout_key] = uploaded
        return uploaded["query_start_loc"]

    def _build_qli_metadata(
        self,
        shared: dict[str, Any],
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        cmp_residual_k: torch.Tensor | None,
        cmp_ratio: int,
        max_seqlen_q: int,
        buffer: torch.Tensor | None = None,
        key_suffix: str = "",
    ) -> torch.Tensor:
        """Pre-compute the QLI scheduling metadata for one index-K cache plane.

        All inputs are scheduler-known geometry, so the build runs once per
        step and compression ratio and lands in the builder's stable buffer;
        ``_publish_task`` defers it to the INDEXER device-metadata stage when
        enabled, and the indexer waits on that stage before its QLI call.

        Candidate attributes describe the candidate-source plane only. Every
        other plane runs the plain position TopK and must request ``-1``, or
        the operator also schedules and emits unused candidate block tensors.
        """
        emits_candidates = cmp_ratio == self._candidate_ratio
        # Reference-recipe contract (cann-recipes deepseek_v4_1,
        # _generate_li_kernel_metadata): seqused_q=None, max_seqlen_q/-k = -1 and
        # the explicit batch size; anything else can mis-tile the scheduler.
        static_attrs = dict(
            self._qli_static_attrs,
            max_seqlen_q=max_seqlen_q,
            cmp_ratio=cmp_ratio,
            candidate_topk_blocks=(
                self._qli_static_attrs["candidate_topk_blocks"] if emits_candidates else -1
            ),
            candidate_block_size=(
                self._qli_static_attrs["candidate_block_size"] if emits_candidates else -1
            ),
        )

        target = buffer if buffer is not None else self._qli_metadata

        def build_qli() -> None:
            value = quant_lightning_indexer_metadata(
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                cmp_residual_k=cmp_residual_k,
                batch_size=int(seqused_k.shape[0]),
                **static_attrs,
            )
            target.copy_(value)

        return self._publish_task(
            shared,
            f"qli:c{cmp_ratio}{key_suffix}",
            target,
            DeviceMetadataStage.INDEXER,
            build_qli,
        )

    def _build_qsli_metadata(
        self,
        shared: dict[str, Any],
        positions: torch.Tensor,
        num_tokens: int,
        num_actual_tokens: int,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        cmp_ratio: int,
        max_seqlen_q: int,
        buffer: torch.Tensor | None = None,
        key_suffix: str = "",
    ) -> torch.Tensor:
        """Pre-compute the QSLI scheduling metadata of the candidate plane.

        How many candidate blocks a query row may score follows from its
        position alone, so this is scheduler-known geometry exactly like the
        QLI build. Deriving it here keeps the sparse indexer off the candidate
        source's critical path: consumers read one stable buffer instead of
        rebuilding it per layer from the source's runtime block lengths.

        Rows past ``num_actual_tokens`` belong to graph/DP padding requests:
        the runner rewrites only the first ``total_num_scheduled_tokens``
        position entries each step, so their positions are stale, while the
        builder zeroes those requests' ``seq_lens``. The reference recipe
        never feeds such rows to the scheduler (positions are re-derived from
        ``kv_len`` every step and padded rows use ``position_ids_pad_value``).
        Keep their candidate length at zero to match the QLI source output.
        """
        text_config = self.vllm_config.model_config.hf_text_config
        candidate_topk_blocks, candidate_block_size = _candidate_geometry(text_config)
        group_span = cmp_ratio * candidate_block_size
        # Same reference-recipe contract as the QLI build: seqused_q=None,
        # max_seqlen_q/-k = -1 and the explicit batch size.
        static_attrs = dict(
            num_heads_q=int(_config_value(text_config, "index_n_heads")),
            num_heads_k=1,
            head_dim=int(_config_value(text_config, "index_head_dim")),
            topk=int(_config_value(text_config, "index_topk")),
            quant_mode=1,
            candidate_block_size=candidate_block_size,
            seqused_q=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=cmp_ratio,
        )

        target = buffer if buffer is not None else self._qsli_metadata

        def build_qsli() -> None:
            rows = positions[:num_tokens].reshape(-1)
            candidate_block_length = (
                torch.div(rows + group_span, group_span, rounding_mode="floor")
                .clamp(1, candidate_topk_blocks)
                .to(torch.int32)
                .reshape(-1, 1)
            )
            if num_actual_tokens < num_tokens:
                pad_rows = (
                    torch.arange(num_tokens, device=rows.device)
                    >= num_actual_tokens
                )
                candidate_block_length = candidate_block_length.masked_fill(
                    pad_rows.unsqueeze(1), 0
                )
            value = quant_sparse_lightning_indexer_metadata(
                candidate_block_length=candidate_block_length,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                cmp_residual_k=None,
                batch_size=int(seqused_k.shape[0]),
                **static_attrs,
            )
            target.copy_(value)

        return self._publish_task(
            shared,
            f"qsli:c{cmp_ratio}{key_suffix}",
            target,
            DeviceMetadataStage.INDEXER,
            build_qsli,
        )

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        **kwargs,
    ):
        # Prefix blocks are already represented by block_table and seq_lens.
        # The V4.1 operators do not consume a separate common-prefix length,
        # so keep the same semantics as DSA v1 and ignore this hint.
        self._device_metadata_tasks = ()
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_compressor_state = isinstance(spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        if isinstance(spec, (DeepseekV41SWASpec, DeepseekV41DraftSWASpec, DeepseekV41A5DraftSWASpec)):
            cache_kind = "swa"
        elif isinstance(spec, DeepseekV41FullSpec):
            cache_kind = "long_kv"
        elif isinstance(spec, DeepseekV41IndexerSpec):
            cache_kind = "index_k"
        elif is_compressor_state:
            cache_kind = "compressor_state"
        else:
            raise TypeError(f"Unsupported V4.1 cache spec: {type(spec).__name__}")

        num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))
        num_actual_reqs = int(kwargs.get("num_actual_reqs", num_reqs))
        num_input_tokens = int(getattr(common, "num_input_tokens", common.slot_mapping.shape[0]))
        num_actual_tokens = int(getattr(common, "num_actual_tokens", num_input_tokens))
        _validate_batch_layout(
            common,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_input_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
        )
        shared = kwargs.get("common_v41_metadata")
        if shared is None:
            shared = {}

        # Keep a complete request-dimension table at a stable address. A
        # mixed full-graph batch can contain one synthetic padding request
        # whose row is not present in the runner-owned table.
        source_block_table = common.block_table_tensor
        if self._block_table is None or self._block_table.shape[1] != source_block_table.shape[1]:
            self._block_table = torch.zeros(
                (self._seq_lens.shape[0] + 1, source_block_table.shape[1]),
                dtype=source_block_table.dtype,
                device=source_block_table.device,
            )
        if num_reqs > self._block_table.shape[0]:
            raise ValueError(
                "V4.1 block-table buffer is smaller than the padded request count: "
                f"{num_reqs} > {self._block_table.shape[0]}"
            )
        self._block_table[:num_reqs].zero_()
        rows_to_copy = min(num_reqs, source_block_table.shape[0])
        if rows_to_copy:
            self._block_table[:rows_to_copy].copy_(source_block_table[:rows_to_copy])
        block_table = self._block_table[:num_reqs]

        # SWA uses original-token coordinates; circular state has no token slots.
        # Long KV and index K are addressed in completed compression groups.
        compressed = cache_kind in {"long_kv", "index_k"}
        raw_slots = (
            common.slot_mapping
            if cache_kind == "swa"
            else torch.full_like(common.slot_mapping, -1)
            if cache_kind == "compressor_state"
            else compressed_slot_mapping(common.slot_mapping, ratio)
        )
        compact_slot_mapping_flat = None
        if is_compressor_state:
            slot_mapping_flat = None
            if self._supports_device_ops:
                self._slot_mapping[:num_input_tokens].copy_(raw_slots[:num_input_tokens])
                slots = self._slot_mapping[:num_input_tokens]
            else:
                # State writes use ring ownership metadata; ordinary slots stay PAD.
                slots = raw_slots
        else:
            # Scope ``shared`` to one framework KV cache group in the model
            # runner. Long KV and Indexer builders with the same physical
            # layout then share one persistent [T, 2] mapping, while every SWA
            # group owns a distinct mapping buffer.
            mapping_scope = "swa" if cache_kind == "swa" else "compressed"
            slot_key = f"slot:{mapping_scope}:c{ratio}:b{spec.storage_block_size}"
            compact_slot_key = f"compact:{mapping_scope}:c{ratio}:b{spec.storage_block_size}"
            prepared_slots = shared.get(slot_key)
            prepared_compact_slots = shared.get(compact_slot_key)
            # A5 quantized writers consume 1-D flat slots. Preparing the flat
            # twin here lets every layer skip its per-call 2D->1D conversion;
            # non-quantized planes keep the [T, 2] contract unchanged.
            needs_flat = spec.dtype == torch.uint8
            flat_key = f"{slot_key}:flat"
            prepared_flat = shared.get(flat_key) if needs_flat else None
            use_compact_c2_slots = self._use_compact_c2_slots
            if use_compact_c2_slots and prepared_compact_slots is None:
                active_slots = raw_slots[:num_input_tokens]
                complete = _build_c2_complete_mask(
                    common,
                    num_input_tokens,
                    num_actual_reqs,
                    num_actual_tokens,
                    skip_ring_state_update=bool(
                        kwargs.get("skip_ring_state_update", False)
                    ),
                )
                capacity = _c2_compact_capacity(num_input_tokens, num_reqs)
                scatter_index = _c2_compact_scatter_index(complete, capacity)
                if self._compact_slot_mapping_flat is None:
                    self._compact_slot_mapping_flat = torch.full(
                        (self._slot_mapping_flat.shape[0] + 1,),
                        -1,
                        dtype=torch.int64,
                        device=raw_slots.device,
                    )
                compact_slots = self._compact_slot_mapping_flat[: capacity + 1]
                compact_slots.fill_(-1)
                compact_slots.scatter_(
                    0,
                    scatter_index,
                    active_slots.to(compact_slots.dtype),
                )
                prepared_compact_slots = self._compact_slot_mapping_flat[:capacity]
                shared[compact_slot_key] = prepared_compact_slots
            if use_compact_c2_slots:
                if prepared_compact_slots is None:
                    raise RuntimeError("V4.1 A5 compact slot mapping was not prepared")
                compact_slot_mapping_flat = prepared_compact_slots
                # The A5 C2 writers consume compact rows directly. The legacy
                # token-indexed 2D/flat mappings have no consumer on this path.
                slots = prepared_compact_slots
                slot_mapping_flat = prepared_compact_slots
            else:
                if prepared_slots is None or (needs_flat and prepared_flat is None):
                    active_slots = raw_slots[:num_input_tokens]
                    valid = active_slots >= 0
                    if compressed and ratio == 2:
                        # Prepare the C2 store mask once per cache group, before
                        # forward. Match the ring compressor's completion policy.
                        if kwargs.get("skip_ring_state_update", False):
                            valid.zero_()
                        else:
                            valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                            valid &= torch.arange(num_input_tokens, device=active_slots.device) < valid_end
                            if common.positions is not None:
                                valid &= common.positions[:num_input_tokens].remainder(2) == 1
                    physical = active_slots.clamp_min(0)
                    if prepared_slots is None:
                        self._slot_mapping_2d[:num_input_tokens, 0].copy_(
                            torch.where(
                                valid,
                                torch.div(
                                    physical,
                                    spec.storage_block_size,
                                    rounding_mode="floor",
                                ),
                                -1,
                            )
                        )
                        self._slot_mapping_2d[:num_input_tokens, 1].copy_(
                            torch.where(
                                valid,
                                physical.remainder(spec.storage_block_size),
                                -1,
                            )
                        )
                        prepared_slots = self._slot_mapping_2d[:num_input_tokens]
                        shared[slot_key] = prepared_slots
                    if needs_flat and prepared_flat is None:
                        # ``where(valid, physical, -1)`` is bitwise equal to the
                        # per-call reconstruction ``page * bs + offset`` clamped
                        # to -1 that the quantized writers used to run per layer.
                        self._slot_mapping_flat[:num_input_tokens].copy_(
                            torch.where(valid, physical, -1).to(torch.int32)
                        )
                        prepared_flat = self._slot_mapping_flat[:num_input_tokens]
                        shared[flat_key] = prepared_flat
                slots = prepared_slots
                slot_mapping_flat = prepared_flat
        coordinates = _cache_coordinates(common, ratio, compressed)
        self._seq_lens[:num_reqs].copy_(coordinates["seq_lens"])
        if num_actual_reqs < num_reqs:
            self._seq_lens[num_actual_reqs:num_reqs].zero_()
        # start_pos must live in a persistent buffer: the attention forward
        # reads it inside the ACLGraph capture, and a per-step allocation
        # would bake a stale address into the replayed graph.
        self._start_pos[:num_reqs].copy_(coordinates["start_pos"])
        if num_actual_reqs < num_reqs:
            self._start_pos[num_actual_reqs:num_reqs].zero_()
        # ``_cache_coordinates`` already floored-divided the same seq_lens by the
        # same plane ratio, so reuse its result instead of dividing again.
        self._cache_seq_lens[:num_reqs].copy_(coordinates["cache_seq_lens"])
        if num_actual_reqs < num_reqs:
            # The reused vector derives from the unzeroed rows, so re-zero the
            # padding tail (QLI/QSLI tiling reads every row up to num_reqs).
            self._cache_seq_lens[num_actual_reqs:num_reqs].zero_()
        coordinates["seq_lens"] = self._seq_lens[:num_reqs]
        coordinates["start_pos"] = self._start_pos[:num_reqs]
        coordinates["cache_seq_lens"] = self._cache_seq_lens[:num_reqs]
        cmp_residual_buffer = None
        if compressed and ratio == 2:
            self._cmp_residual[:num_reqs].copy_(self._seq_lens[:num_reqs].remainder(ratio))
            cmp_residual_buffer = self._cmp_residual[:num_reqs]
        positions = getattr(common, "positions", None)
        cos = sin = None
        if cache_kind == "swa" and positions is not None:
            self._positions[:num_input_tokens].copy_(positions[:num_input_tokens].to(torch.int64))
            positions = self._positions[:num_input_tokens]
        (
            num_decodes,
            num_decode_tokens,
            num_prefills,
            num_prefill_tokens,
        ) = _request_counts(common, num_actual_reqs)
        draft_tokens = (
            getattr(self.vllm_config.speculative_config, "num_speculative_tokens", 0)
            if self.vllm_config.speculative_config is not None
            else 0
        )
        if (
            cache_kind == "swa"
            and num_prefills > 0
            and not is_v41_draft_swa_spec(spec)
            and getattr(common, "private_circle_blocks_per_allocation", None)
            and _prefill_rows_fit_ring_capacity(
                common, num_actual_reqs, 1 + draft_tokens
            )
        ):
            # Every prefill row's query fits the ring's in-flight capacity:
            # a PD request's last-token recompute, optionally fused with the
            # first speculative draft block. The imported ring already covers
            # the window, so those rows run as decodes: no plan, and the step
            # stays graph-capturable regardless of row order.
            num_decodes += num_prefills
            num_decode_tokens += num_prefill_tokens
            num_prefills = 0
            num_prefill_tokens = 0
        if cache_kind == "swa" and positions is not None:
            # Single-token prefill rows (a PD request's last-token recompute)
            # appear in DecodeOnly graph steps. The fresh-indexing branch would
            # bake a per-step allocation into the replayed graph; those steps
            # must gather into the stable runtime buffers like pure decodes.
            # Their token count (num_reqs) is far below the buffer capacity.
            cos, sin = get_cos_and_sin_dsa(
                positions,
                use_cache=(
                    bool(kwargs.get("full_graph_mode", False))
                    or num_prefills == 0
                    or num_prefill_tokens == num_prefills
                ),
            )
        # Model-static geometry, derived once in __init__.
        window_size = self._window_size
        n_local_heads = self._n_local_heads
        head_dim = self._head_dim
        index_topk = self._index_topk
        qli_metadata = None
        qsli_metadata = None
        qli_metadata_tail = None
        qsli_metadata_tail = None
        query_start_loc_tail = None
        private_circle_plan = None
        # Draft SWA layers share the request-private ring but must not take the
        # plan's workspace / restore-commit path: their context writes are
        # clamped to each request's tail window (-1 = skip sentinel). Their
        # metadata therefore stays plan-less.
        if (
            cache_kind == "swa"
            and num_prefills > 0
            and not is_v41_draft_swa_spec(spec)
        ):
            blocks_per_allocation = getattr(common, "private_circle_blocks_per_allocation", None)
            if blocks_per_allocation:
                private_circle_plan = _prepare_private_circle_plan(
                    common,
                    shared,
                    self.vllm_config,
                    spec.storage_block_size,
                    int(blocks_per_allocation),
                    num_real_reqs=num_actual_reqs,
                )
                # Prefill tokens scatter into the shared compact workspace;
                # decode tokens keep the ring slots from the group table.
                # Padding rows beyond the real requests contribute no slots.
                if self._private_circle_slots_2d is None:
                    raise RuntimeError("V4.1 private-circle slot buffers are not initialized")
                if num_input_tokens > self._private_circle_slots_2d.shape[0]:
                    raise RuntimeError(
                        "V4.1 private-circle slot buffer is smaller than the "
                        f"step: {num_input_tokens} > "
                        f"{self._private_circle_slots_2d.shape[0]}"
                    )
                self._private_circle_slots_2d[:num_input_tokens].fill_(-1)
                if num_decode_tokens:
                    ring_slots = _format_private_circle_slots(
                        common.slot_mapping[:num_decode_tokens].long(),
                        spec.storage_block_size,
                    )
                    self._private_circle_slots_2d[:num_decode_tokens].copy_(ring_slots)
                prefill_end = num_decode_tokens + num_prefill_tokens
                self._private_circle_slots_2d[num_decode_tokens:prefill_end].copy_(
                    private_circle_plan["prefill_slots"]
                )
                self._private_circle_slots_2d[prefill_end:].fill_(-1)
                slots = self._private_circle_slots_2d[:num_input_tokens]
                if spec.dtype == torch.uint8:
                    # Flat twin of the ring/workspace mapping for the A5
                    # quantized writers: decode rows keep their ring slots,
                    # prefill rows take the plan's flat workspace slots, and
                    # padding stays -1 exactly like the 2D mapping.
                    if self._private_circle_slots_flat is None:
                        raise RuntimeError("V4.1 flat private-circle slot buffer is not initialized")
                    self._private_circle_slots_flat[:num_input_tokens].fill_(-1)
                    if num_decode_tokens:
                        self._private_circle_slots_flat[:num_decode_tokens].copy_(
                            common.slot_mapping[:num_decode_tokens].clamp(min=-1).to(torch.int32)
                        )
                    self._private_circle_slots_flat[num_decode_tokens:prefill_end].copy_(
                        private_circle_plan["prefill_slots_flat"]
                    )
                    self._private_circle_slots_flat[prefill_end:].fill_(-1)
                    slot_mapping_flat = self._private_circle_slots_flat[:num_input_tokens]
        if cache_kind == "index_k" and self._supports_device_ops:
            max_seqlen_q = -1
            qli_metadata = self._build_qli_metadata(
                shared,
                coordinates["query_start_loc"],
                coordinates["cache_seq_lens"],
                cmp_residual_buffer,
                ratio,
                max_seqlen_q,
            )
            if ratio == self._candidate_ratio and positions is not None:
                qsli_metadata = self._build_qsli_metadata(
                    shared,
                    positions,
                    num_input_tokens,
                    num_actual_tokens,
                    coordinates["query_start_loc"],
                    coordinates["cache_seq_lens"],
                    ratio,
                    max_seqlen_q,
                )
            # Tail-window variant, derived locally: the index-K group may be
            # built before the SWA group whose plan carries this layout. Same
            # keep rule as the plan; padding requests are appended as
            # zero-length rows so every seqused_k row has a matching request.
            cache_rows = int(coordinates["cache_seq_lens"].shape[0])
            # Index-K builders can live in multiple framework KV groups. Share
            # the immutable tail layout across those builders so one step does
            # one CPU plan and one H2D upload instead of repeating it per
            # index group. The group-local dict remains the owner of QLI/QSLI
            # buffers and private-circle state.
            tail_layout_shared = kwargs.get("fa_metadata_shared")
            if tail_layout_shared is None:
                tail_layout_shared = shared
            tail_qsl = self._tail_query_start_loc(
                common_attn_metadata,
                num_actual_reqs,
                window_size,
                cache_rows,
                coordinates["query_start_loc"].dtype,
                tail_layout_shared,
            )
            if tail_qsl is not None:
                query_start_loc_tail = tail_qsl
                qli_metadata_tail = self._build_qli_metadata(
                    shared,
                    tail_qsl,
                    coordinates["cache_seq_lens"],
                    cmp_residual_buffer,
                    ratio,
                    max_seqlen_q,
                    buffer=self._qli_metadata_tail,
                    key_suffix=":tail",
                )
                if ratio == self._candidate_ratio and positions is not None:
                    tail_rows = self._tail_rows
                    qsli_metadata_tail = self._build_qsli_metadata(
                        shared,
                        positions.index_select(0, tail_rows),
                        int(tail_rows.numel()),
                        int(tail_rows.numel()),
                        tail_qsl,
                        coordinates["cache_seq_lens"],
                        ratio,
                        max_seqlen_q,
                        buffer=self._qsli_metadata_tail,
                        key_suffix=":tail",
                    )

        c2_ring_metadata = None
        c2_complete_mask = None
        c2_source_positions = None
        c2_source_cos = None
        c2_source_sin = None
        c2_compact_source_positions = None
        c2_compact_source_cos = None
        c2_compact_source_sin = None
        c2_metadata_group_id = None
        if cache_kind == "compressor_state" and getattr(common, "positions", None) is not None:
            ring_meta = self._c2_ring_metadata[: 5 * num_reqs].view(5, num_reqs)
            input_positions = common.positions[:num_input_tokens].long()
            if self._supports_device_ops:
                if self._c2_full_source_rope is None:
                    raise RuntimeError("V4.1 source RoPE buffers were not initialized")
                full_source_cos, full_source_sin = self._c2_full_source_rope
            else:
                full_source_cos = full_source_sin = None
            use_compact_c2_rope = self._use_compact_c2_rope
            compact_capacity = _c2_compact_capacity(num_input_tokens, num_reqs)

            def build_c2_metadata() -> None:
                starts = common.query_start_loc[:num_reqs].int()
                ends = common.query_start_loc[1 : num_reqs + 1].int()
                query_lens = ends - starts
                live = torch.arange(num_reqs, device=starts.device) < num_actual_reqs
                used = (ends.clamp_max(num_actual_tokens) - starts).clamp_min(0)
                used = torch.where(live, used, 0)
                if kwargs.get("skip_ring_state_update", False):
                    used = torch.zeros_like(used)
                ring_meta[0].copy_((self._seq_lens[:num_reqs] - query_lens).clamp_min(0))
                ring_meta[1].copy_(used)
                ring_meta[2].copy_(starts)
                ring_meta[3].copy_(starts)
                ring_meta[4].copy_(torch.where(used > 0, block_table[:num_reqs, 0], 0))
                complete = _build_c2_complete_mask(
                    common,
                    num_input_tokens,
                    num_actual_reqs,
                    num_actual_tokens,
                    skip_ring_state_update=bool(
                        kwargs.get("skip_ring_state_update", False)
                    ),
                )
                self._c2_complete_mask[:num_input_tokens].copy_(complete)
                source_positions = torch.where(
                    complete,
                    input_positions - 1,
                    torch.zeros_like(input_positions),
                )
                if use_compact_c2_rope:
                    scatter_index = _c2_compact_scatter_index(
                        complete, compact_capacity
                    )
                    compact_positions = self._c2_compact_source_positions[
                        : compact_capacity + 1
                    ]
                    compact_positions.fill_(0)
                    compact_positions.scatter_(0, scatter_index, source_positions)
                    rope_positions = self._c2_compact_source_positions[
                        :compact_capacity
                    ]
                    rope_rows = compact_capacity
                else:
                    self._c2_source_positions[:num_input_tokens].copy_(
                        source_positions
                    )
                    rope_positions = self._c2_source_positions[:num_input_tokens]
                    rope_rows = num_input_tokens
                if full_source_cos is not None and full_source_sin is not None:
                    gather_idx = (
                        rope_positions.reshape(-1, 1, 1, 1)
                        .expand(
                            rope_rows,
                            1,
                            1,
                            full_source_cos.shape[-1],
                        )
                    )
                    torch.gather(
                        full_source_cos,
                        0,
                        gather_idx,
                        out=self._c2_source_cos[:rope_rows],
                    )
                    torch.gather(
                        full_source_sin,
                        0,
                        gather_idx,
                        out=self._c2_source_sin[:rope_rows],
                    )

            compressor_group = self._publish_task(
                shared,
                "c2:compressor",
                self._c2_complete_mask,
                DeviceMetadataStage.COMPRESSOR,
                build_c2_metadata,
                priority=V41_COMPRESSOR_METADATA_PRIORITY,
            )
            if compressor_group is not self._c2_complete_mask:
                raise RuntimeError("V4.1 compressor metadata must have one owner")
            c2_complete_mask = self._c2_complete_mask[:num_input_tokens]
            c2_ring_metadata = ring_meta
            if use_compact_c2_rope:
                c2_compact_source_positions = self._c2_compact_source_positions[
                    :compact_capacity
                ]
                if self._supports_device_ops:
                    c2_compact_source_cos = self._c2_source_cos[:compact_capacity]
                    c2_compact_source_sin = self._c2_source_sin[:compact_capacity]
            else:
                c2_source_positions = self._c2_source_positions[:num_input_tokens]
                if self._supports_device_ops:
                    c2_source_cos = self._c2_source_cos[:num_input_tokens]
                    c2_source_sin = self._c2_source_sin[:num_input_tokens]
            c2_metadata_group_id = id(self._c2_complete_mask)
        win_indices_pre = None
        win_topk_length_pre = None
        cmp_topk_lengths_pre = None
        fa_metadata_pre = None
        fa_metadata_group_id = None
        if (
                self._fa_precomputed
                and positions is not None
                and private_circle_plan is None
                and num_actual_tokens > 0
        ):
            fa_shared = kwargs.get("fa_metadata_shared")
            if fa_shared is None:
                fa_shared = shared
            fa_key = "fa:kernel_metadata"
            fa_cache = fa_shared.get(fa_key)
            if fa_cache is not None:
                n_tokens = num_input_tokens
                win_indices_pre = fa_cache["win_indices"][:n_tokens]
                win_topk_length_pre = fa_cache["win_topk_length"][:n_tokens]
                cmp_topk_lengths_pre = {
                    r: buf[:n_tokens] for r, buf in fa_cache["cmp_topk_lengths"].items()
                }
                fa_metadata_pre = fa_cache["fa_metadata"]
                fa_metadata_group_id = fa_cache["group_id"]
            else:
                cu_seqlens_q = common.query_start_loc[: num_reqs + 1]
                pos_for_build = positions
                n_tokens = num_input_tokens
                n_actual = num_actual_tokens
                n_reqs = num_reqs
                ws = window_size
                nhq = n_local_heads
                hd = head_dim
                itk = index_topk

                def build_fa_metadata() -> None:
                    self._build_kernel_metadata(
                        pos_for_build, n_tokens, n_actual, n_reqs, cu_seqlens_q,
                        ws, nhq, hd, itk,
                    )

                if self._device_metadata_enabled:
                    self._device_metadata_tasks = (
                        *self._device_metadata_tasks,
                        DeviceMetadataTask(
                            DeviceMetadataStage.ATTENTION,
                            build_fa_metadata,
                            id(self._fa_metadata_sentinel),
                            priority=V41_ATTENTION_METADATA_PRIORITY,
                        ),
                    )
                else:
                    build_fa_metadata()
                fa_metadata_group_id = id(self._fa_metadata_sentinel)
                fa_shared[fa_key] = {
                    "win_indices": self._win_indices_buf,
                    "win_topk_length": self._win_topk_length_buf,
                    "cmp_topk_lengths": self._cmp_topk_length_bufs,
                    "fa_metadata": self._fa_metadata_bufs,
                    "group_id": fa_metadata_group_id,
                }
                win_indices_pre = self._win_indices_buf[:n_tokens]
                win_topk_length_pre = self._win_topk_length_buf[:n_tokens]
                cmp_topk_lengths_pre = {
                    r: buf[:n_tokens] for r, buf in self._cmp_topk_length_bufs.items()
                }
                fa_metadata_pre = self._fa_metadata_bufs
        if self._dspark_precomputed and positions is not None and num_actual_tokens > 0:
            # The draft query block width is authoritative only in the host
            # query_start_loc mirror; num_reqs/num_input_tokens disagree under
            # DP or graph-bucket padding. A non-uniform or over-wide block is a
            # prompt-chunk prefill step: it attends causally within the prompt,
            # so leave win_indices unset and let _native_attention fall back to
            # _get_cached_win_indices. Only the decode proposal's uniform
            # speculative block takes the DSpark block window.
            qsl_blocks, query_len = _dspark_query_block(common)
            padded_blocks = num_input_tokens // query_len if query_len > 0 else 0
            if (
                    query_len > 0
                    and query_len <= self._dspark_max_query_len
                    and num_input_tokens == padded_blocks * query_len
                    and qsl_blocks == padded_blocks
            ):
                # Mask draft-plane positions below the request's first local
                # verify anchor: the ring holds the previous owner's residual
                # KV there (private circle only; absent when the pool is off).
                draft_write_start = getattr(
                    common, "private_circle_draft_write_start", None
                )
                win_indices_pre = self._build_dspark_win_indices(
                    positions, num_input_tokens, padded_blocks, query_len,
                    draft_write_start=draft_write_start,
                )
            else:
                logger.info_once(
                    "DSpark draft step has no uniform speculative query block "
                    "(prompt-chunk prefill); using the causal sliding window."
                )
        return DeepseekV41Metadata(
            block_table=block_table,
            slot_mapping=slots,
            slot_mapping_flat=slot_mapping_flat,
            compact_slot_mapping_flat=compact_slot_mapping_flat,
            compress_ratio=ratio,
            storage_block_size=spec.storage_block_size,
            positions=positions,
            cos=cos,
            sin=sin,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            logical_block_size=spec.block_size,
            max_query_len=int(getattr(common, "max_query_len", 0)),
            max_seq_len=int(getattr(common, "max_seq_len", 0)),
            attn_state=getattr(common, "attn_state", None),
            is_prefilling=getattr(common, "is_prefilling", None),
            causal=getattr(common, "causal", True),
            ori_win_left=max(0, window_size - 1),
            ori_win_right=0,
            qli_metadata=qli_metadata,
            qsli_metadata=qsli_metadata,
            qli_metadata_tail=qli_metadata_tail,
            qsli_metadata_tail=qsli_metadata_tail,
            query_start_loc_tail=query_start_loc_tail,
            cmp_residual=cmp_residual_buffer,
            c2_ring_metadata=c2_ring_metadata,
            c2_complete_mask=c2_complete_mask,
            c2_source_positions=c2_source_positions,
            c2_source_cos=c2_source_cos,
            c2_source_sin=c2_source_sin,
            c2_compact_source_positions=c2_compact_source_positions,
            c2_compact_source_cos=c2_compact_source_cos,
            c2_compact_source_sin=c2_compact_source_sin,
            c2_metadata_group_id=c2_metadata_group_id,
            private_circle_plan=private_circle_plan,
            win_indices=win_indices_pre,
            win_topk_length=win_topk_length_pre,
            cmp_topk_lengths=cmp_topk_lengths_pre,
            fa_metadata=fa_metadata_pre,
            fa_metadata_group_id=fa_metadata_group_id,
            **coordinates,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @staticmethod
    def get_impl_cls():
        return DeepseekV41EagerAttentionImpl

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = [torch.empty(0)]
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache prefix: {prefix}")
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return DeepseekV41CacheBackend
