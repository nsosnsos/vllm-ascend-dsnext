# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache specs and layer-outermost hybrid allocation."""

from dataclasses import dataclass, replace

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.circular_buffer import AscendCircularBufferSpec
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.speculative_mode import SpeculativeMode, classify_speculative_mode

STATE_RING_ROWS = 16

A5_WIN_LOGICAL_DIM = 512
A5_WIN_QUANT_GROUP_SIZE = 32
A5_WIN_DATA_BYTES = A5_WIN_LOGICAL_DIM
A5_WIN_SCALE_COUNT = A5_WIN_LOGICAL_DIM // A5_WIN_QUANT_GROUP_SIZE
A5_WIN_SCALE_OFFSET_BYTES = A5_WIN_DATA_BYTES
A5_WIN_ROW_BYTES = A5_WIN_DATA_BYTES + A5_WIN_SCALE_COUNT * torch.bfloat16.itemsize


@dataclass(frozen=True, kw_only=True)
class DeepseekV41FullSpec(AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, _INDEXER_SLOT_FAMILY)
            and s.block_size == self.block_size
            and (
                s.compress_ratio in (1, 2)
                or isinstance(s, DeepseekV41IndexerFoldedSpec)
            )
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41IndexerSpec(AscendMLAAttentionSpec):
    """Byte-addressable FP4 index keys followed by E8M0 scales."""

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, _INDEXER_SLOT_FAMILY)
            and s.block_size == self.block_size
            and (
                s.compress_ratio in (1, 2)
                or isinstance(s, DeepseekV41IndexerFoldedSpec)
            )
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41IndexerFoldedSpec(AscendMLAAttentionSpec):
    """Candidate-block folded twin of a candidate-source indexer cache.

    One row per candidate block of ``compress_ratio`` source rows: the
    quantized keys and E8M0 scales of those rows are concatenated in place,
    so the flat row width is compress_ratio * (payload + scale) bytes. The
    row-count semantics differ from token-indexed planes. Producers write the
    folded representation via ``indexer_prologue_k`` and Mooncake transfers
    it as part of the containing physical slot; consumers read the folded
    view directly without repacking the token-indexed twin.
    """

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41IndexerFoldedSpec)
            and s.block_size == self.block_size
            for s in specs.values()
        )


_INDEXER_SLOT_FAMILY = (
    DeepseekV41FullSpec, DeepseekV41IndexerSpec, DeepseekV41IndexerFoldedSpec
)


@dataclass(frozen=True, kw_only=True)
class DeepseekV41SWASpec(AscendSlidingWindowMLASpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41SWASpec) and s.sliding_window == self.sliding_window for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41DraftSWASpec(AscendSlidingWindowMLASpec):
    """DSpark SWA owned by G12, aliasing target slots at distinct block IDs."""

    def __post_init__(self):
        if self.dtype != torch.bfloat16 or self.num_kv_heads != 1 or self.compress_ratio != 1:
            raise ValueError("Aurora DSpark requires one uncompressed BF16 KV plane")

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41DraftSWASpec, DeepseekV41A5DraftSWASpec))
            and s.block_size == self.block_size
            and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41A5DraftSWASpec(AscendSlidingWindowMLASpec):
    """DSpark SWA in the A5 MQSMLA FP8/32 plus BF16-scale row format."""

    logical_head_size: int = A5_WIN_LOGICAL_DIM
    quant_group_size: int = A5_WIN_QUANT_GROUP_SIZE
    scale_dtype: torch.dtype = torch.bfloat16
    scale_offset_bytes: int = A5_WIN_SCALE_OFFSET_BYTES

    def __post_init__(self):
        if (
            self.dtype != torch.uint8
            or self.num_kv_heads != 1
            or self.head_size != A5_WIN_ROW_BYTES
            or self.logical_head_size != A5_WIN_LOGICAL_DIM
            or self.quant_group_size != A5_WIN_QUANT_GROUP_SIZE
            or self.scale_dtype != torch.bfloat16
            or self.scale_offset_bytes != A5_WIN_SCALE_OFFSET_BYTES
            or self.compress_ratio != 1
        ):
            raise ValueError("A5 DSpark win_kv requires one U8[544] row: FP8/32 payload plus 16 embedded BF16 scales")

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41A5DraftSWASpec)
            and s.block_size == self.block_size
            and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


def is_v41_draft_swa_spec(spec):
    return isinstance(spec, (DeepseekV41DraftSWASpec, DeepseekV41A5DraftSWASpec))


@dataclass(frozen=True, kw_only=True)
class DeepseekV41CompressorStateSpec(AscendCircularBufferSpec):
    """One private 16-row FP32 KV/score ring page per active request."""

    compress_ratio: int = 1

    def __post_init__(self):
        if self.dtype != torch.float32 or self.block_size != STATE_RING_ROWS or self.compress_ratio != 1:
            raise ValueError(
                f"Aurora state requires a {STATE_RING_ROWS}-row FP32 uncompressed ring"
            )
        if self.num_kv_heads != 1:
            raise ValueError("Aurora state requires one packed KV/score plane")


def is_v41_spec(spec):
    return isinstance(
        spec,
        (
            DeepseekV41FullSpec,
            DeepseekV41IndexerSpec,
            DeepseekV41IndexerFoldedSpec,
            DeepseekV41SWASpec,
            DeepseekV41DraftSWASpec,
            DeepseekV41A5DraftSWASpec,
            DeepseekV41CompressorStateSpec,
        ),
    )


def _resolve_divert(specs, divert_swa):
    """Private-circle SWA diversion: explicit flag, else env + spec presence."""
    if divert_swa is not None:
        return divert_swa
    from vllm_ascend import envs

    if not envs.VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL:
        return False
    return any(isinstance(s, DeepseekV41SWASpec) for s in specs.values())


def _uniform(members, label):
    if not members:
        raise ValueError(f"V4.1 cache group {label} is empty")
    uniform = UniformTypeKVCacheSpecs.from_specs(members)
    if uniform is None:
        raise ValueError(f"Incompatible V4.1 resource layouts in {label}")
    return uniform


@dataclass(frozen=True)
class CachePlacement:
    name: str
    offset: int
    page_size_bytes: int


@dataclass(frozen=True)
class CacheSlot:
    page_size_bytes: int
    placements: tuple[CachePlacement, ...]


MXFP4_QUANT_GROUP_SIZE = 16


def mxfp4_row_bytes(width: int, group_size: int = MXFP4_QUANT_GROUP_SIZE) -> int:
    """Packed uint8 row width of the mxfp4 compressed KV plane.

    Matches the kv_compress_epilog_v2 op layout: width/2 data bytes (two fp4
    nibbles per byte) + 2 * (width/group) bf16 scale bytes, padded up to a
    32-byte boundary. width=512 -> 256 + 64 -> 320.
    """
    data = width // 2
    scales = 2 * (width // group_size)
    return (data + scales + 31) // 32 * 32


def _layer_number(name):
    try:
        return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid V4.1 cache resource name: {name}") from exc


def _cache_plane_sizes(spec):
    rows = spec.storage_block_size * spec.num_kv_heads
    key_bytes = rows * spec.head_size * spec.dtype.itemsize
    if isinstance(spec, DeepseekV41IndexerSpec):
        return key_bytes, rows * spec.scale_dim * spec.scale_dtype.itemsize
    return (key_bytes,)


def _draft_layer_number(name):
    try:
        return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid Aurora DSpark cache resource name: {name}") from exc


def plan_cache_slots(specs, *, divert_swa: bool | None = None):
    """Place source KV/index tuples, state and SWA in four shared layer slots.

    Sizes come from payloads, never previously padded specs. Different groups
    overlay a slot at distinct live block IDs; a source's KV and index share
    the same ID at disjoint offsets within its page.

    When ``divert_swa`` is set, SWA layers live in the request-private circle
    ring instead and receive no shared slot placement.
    """
    if not all(is_v41_spec(spec) for spec in specs.values()):
        raise ValueError(
            "V4.1 requires explicit target or Aurora DSpark cache specs; foreign resources are unsupported"
        )
    divert = _resolve_divert(specs, divert_swa)
    full = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41FullSpec)), key=_layer_number)
    state = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41CompressorStateSpec)), key=_layer_number)
    swa = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    draft = sorted((n for n, s in specs.items() if is_v41_draft_swa_spec(s)), key=_draft_layer_number)
    if draft and list(map(_draft_layer_number, draft)) != [0, 1, 2]:
        raise ValueError("Aurora DSpark requires exactly three ordered draft layers: mtp.0, mtp.1, mtp.2")
    if list(map(_layer_number, full)) != [2, 8, 14, 20]:
        raise ValueError("V4.1 requires KV source layers 2, 8, 14, 20")
    if list(map(_layer_number, state)) != [2, 8, 14]:
        raise ValueError("V4.1 requires state source layers 2, 8, 14")
    if list(map(_layer_number, swa)) != list(range(40)):
        raise ValueError("V4.1 requires exactly 40 ordered SWA resources")

    slots = []
    folded = sorted(
        (n for n, s in specs.items() if isinstance(s, DeepseekV41IndexerFoldedSpec)),
        key=_layer_number,
    )
    for slot_idx, kv_name in enumerate(full):
        prefix, suffix = kv_name.rsplit(".", 1)
        index_name = prefix + ".indexer.k_cache"
        index_spec = specs.get(index_name)
        kv_spec = specs[kv_name]
        ratio = 2 if slot_idx < len(state) else 1
        if (
            suffix != "long_kv_cache"
            or not isinstance(index_spec, DeepseekV41IndexerSpec)
            or kv_spec.compress_ratio != ratio
            or index_spec.compress_ratio != ratio
            or kv_spec.block_size != index_spec.block_size
        ):
            raise ValueError(f"V4.1 source {prefix} has incompatible KV/index specs")
        slot_layer = _layer_number(kv_name)
        folded_names = [n for n in folded if _layer_number(n) == slot_layer]
        for n in folded_names:
            if _layer_number(n) != _layer_number(index_name):
                raise ValueError(f"folded twin {n} does not belong to {prefix}")
        folded_bytes = sum(sum(_cache_plane_sizes(specs[n])) for n in folded_names)
        aliases = [state[slot_idx]] if slot_idx < len(state) else []
        if not divert:
            aliases += swa[slot_idx :: len(full)]
        kv_bytes = sum(_cache_plane_sizes(kv_spec))
        index_bytes = sum(_cache_plane_sizes(index_spec))
        capacity = max(
            [kv_bytes + index_bytes + folded_bytes,
             *(sum(_cache_plane_sizes(specs[n])) for n in aliases)]
        )
        if slot_idx < len(draft):
            draft_name = draft[slot_idx]
            draft_spec = specs[draft_name]
            if divert:
                # The draft joins the request-private ring tensors and never
                # aliases the shared slot; ring geometry is validated by the
                # pool's single-layout check instead.
                pass
            else:
                draft_bytes = sum(_cache_plane_sizes(draft_spec))
                swa_spec = specs[swa[slot_idx]]
                # Hardware-specific target and draft row encodings may differ.
                # Aliased groups own distinct block IDs in the shared slot, so
                # only their block geometry must agree.
                if (
                    draft_spec.block_size != swa_spec.block_size
                    or draft_spec.sliding_window != swa_spec.sliding_window
                ):
                    raise ValueError("Aurora DSpark block geometry must match target SWA")
                            # The draft is a slot member like any other alias, and on A5
                # its plane (128 quantized rows) can exceed every target plane
                # of a ratio-2 source slot. Size the slot by its largest
                # member, draft included.
                capacity = max(capacity, draft_bytes)
                aliases.append(draft_name)
        placements = [
            CachePlacement(kv_name, 0, kv_bytes),
            CachePlacement(index_name, kv_bytes, capacity - kv_bytes - folded_bytes),
            *(CachePlacement(n, kv_bytes + index_bytes, folded_bytes)
              for n in folded_names),
            *(CachePlacement(name, 0, capacity) for name in aliases),
        ]
        slots.append(CacheSlot(capacity, tuple(placements)))
    names = [p.name for slot in slots for p in slot.placements]
    uncovered = set(specs) - set(names) - ((set(swa) | set(draft)) if divert else set())
    if len(names) != len(set(names)) or uncovered:
        raise ValueError("V4.1 slot placement must cover each resource exactly once")
    return tuple(slots)


def group_cache_specs(specs, *, divert_swa: bool | None = None):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if not any(is_v41_spec(s) for s in specs.values()):
        return None
    divert = _resolve_divert(specs, divert_swa)
    slots = plan_cache_slots(specs, divert_swa=divert)
    padded = {
        p.name: replace(specs[p.name], page_size_padded=p.page_size_bytes) for slot in slots for p in slot.placements
    }
    full = {
        n: s
        for n, s in padded.items()
        if isinstance(s, _INDEXER_SLOT_FAMILY)
    }
    state = {n: s for n, s in padded.items() if isinstance(s, DeepseekV41CompressorStateSpec)}
    groups = [_uniform(full, "full"), _uniform(state, "state")]
    swa = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    swa_source = specs if divert else padded
    if divert:
        # Diverted SWA layers share one ring allocation per request and never
        # touch the shared slots, so the slot-aligned split below is redundant;
        # one group keeps a single identical ring table per step.
        groups.append(_uniform({n: swa_source[n] for n in swa}, "swa"))
    else:
        groups.extend(
            _uniform({n: swa_source[n] for n in swa[start : start + len(slots)]}, f"swa{start}")
            for start in range(0, len(swa), len(slots))
        )
    draft = sorted((n for n, s in specs.items() if is_v41_draft_swa_spec(s)), key=_draft_layer_number)
    if draft:
        # The draft leaves the shared slots under divert (it joins the
        # request-private ring tensors), so its group must read the raw
        # spec instead of the slot-padded one.
        draft_source = specs if divert else padded
        groups.append(_uniform({n: draft_source[n] for n in draft}, "dspark"))
    return groups


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def has_v41_groups(groups):
    return any(
        is_v41_spec(s)
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        for s in g.kv_cache_spec.kv_cache_specs.values()
    )


def _group_specs(groups):
    specs = {}
    for group in groups:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            raise ValueError("V4.1 requires uniform-type cache groups")
        for name in group.layer_names:
            if name in specs:
                raise ValueError(f"V4.1 resource belongs to multiple cache groups: {name}")
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return specs


def cache_slots_from_groups(groups, *, divert_swa: bool | None = None):
    return plan_cache_slots(_group_specs(groups), divert_swa=divert_swa)


def pool_bytes_per_block(groups, *, divert_swa: bool | None = None):
    return sum(slot.page_size_bytes for slot in cache_slots_from_groups(groups, divert_swa=divert_swa))


def request_blocks(vllm_config, groups, *, divert_swa: bool | None = None):
    # Different logical groups consume different IDs in one global block pool.
    # Diverted private-circle groups own ring allocations, not global IDs.
    from vllm_ascend.core.private_circle_pool import is_private_circle_kv_cache_spec

    divert = _resolve_divert(_group_specs(groups), divert_swa)
    total = 0
    for g in groups:
        if divert and all(
            is_private_circle_kv_cache_spec(s)
            for s in g.kv_cache_spec.kv_cache_specs.values()
        ):
            continue
        total += max(
            (s.max_memory_usage_bytes(vllm_config) + s.page_size_bytes - 1) // s.page_size_bytes
            for s in g.kv_cache_spec.kv_cache_specs.values()
        )
    return total


def allocate_cache_config(vllm_config, groups, available_memory, *, divert_swa: bool | None = None, reserved_private_bytes: int = 0):
    """Allocate four independent layer slots backed by one global block-ID pool."""
    divert = _resolve_divert(_group_specs(groups), divert_swa)
    slots = cache_slots_from_groups(groups, divert_swa=divert)
    budget = available_memory - reserved_private_bytes
    capacity = budget // sum(slot.page_size_bytes for slot in slots)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")
    return num_blocks, [
        KVCacheTensor(
            size=num_blocks * slot.page_size_bytes,
            shared_by=[p.name for p in slot.placements],
            block_stride=slot.page_size_bytes,
        )
        for slot in slots
    ]


def reshape_cache(raw: torch.Tensor, spec, *, num_blocks, offset, block_stride):
    """Create typed per-page views using the containing slot's physical stride."""
    if raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError("V4.1 cache requires contiguous one-dimensional uint8 storage")
    if num_blocks <= 0 or block_stride <= 0 or raw.numel() != num_blocks * block_stride:
        raise ValueError("V4.1 cache backing does not match its declared layout")
    plane_sizes = _cache_plane_sizes(spec)
    if offset < 0 or offset + sum(plane_sizes) > block_stride:
        raise ValueError("V4.1 cache component exceeds its slot page")
    if isinstance(spec, DeepseekV41CompressorStateSpec) and sum(plane_sizes) > block_stride:
        raise ValueError("Aurora circular state exceeds its slot")

    def view(dtype, width, byte_offset):
        dtype_size = dtype.itemsize
        storage_offset = raw.storage_offset() + byte_offset
        if storage_offset % dtype_size or block_stride % dtype_size or raw.numel() % dtype_size:
            raise ValueError("V4.1 cache offset/stride is not dtype aligned")
        return torch.as_strided(
            raw.view(dtype),
            size=(num_blocks, spec.storage_block_size, spec.num_kv_heads, width),
            stride=(block_stride // dtype_size, spec.num_kv_heads * width, width, 1),
            storage_offset=storage_offset // dtype_size,
        )

    key = view(spec.dtype, spec.head_size, offset)
    if isinstance(spec, DeepseekV41IndexerSpec):
        return key, view(spec.scale_dtype, spec.scale_dim, offset + plane_sizes[0])
    return key


def validate_cache_runtime(vllm_config):
    if vllm_config.use_v2_model_runner:
        raise NotImplementedError("V4.1 cache initialization currently requires model runner V1")
    cudagraph_mode = getattr(
        vllm_config.compilation_config,
        "cudagraph_mode",
        CUDAGraphMode.NONE if vllm_config.model_config.enforce_eager else CUDAGraphMode.FULL,
    )
    if cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        raise NotImplementedError("V4.1 currently supports only eager or FULL_DECODE_ONLY graph mode")
    speculative = vllm_config.speculative_config
    if speculative is not None:
        speculative_mode = classify_speculative_mode(speculative)
        if speculative_mode is SpeculativeMode.MTP:
            raise NotImplementedError(
                "Aurora MTP requires a V4.1-compatible MTP draft adapter and "
                "checkpoint contract; this runtime currently provides only "
                "the Aurora DSpark draft implementation"
            )
        if speculative_mode is not SpeculativeMode.DSPARK:
            raise NotImplementedError("Aurora supports only DSpark speculative decoding")
        # Verification writes the anchor and up to S speculative rows. After
        # rejection, the earliest needed residual is the verified anchor.
        # It must survive the final ring write: S must be strictly below the
        # configured ring capacity.
        if not 0 < speculative.num_speculative_tokens < STATE_RING_ROWS:
            raise ValueError(
                f"Aurora DSpark requires 1..{STATE_RING_ROWS - 1} speculative "
                "tokens to preserve FP32 ring residuals"
            )
        per_batch = getattr(speculative, "num_speculative_tokens_per_batch_size", None) or ()
        if any(not 0 <= count <= speculative.num_speculative_tokens for _, _, count in per_batch):
            raise ValueError("Aurora DSpark per-batch speculation must stay within the configured ring-safe maximum")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise NotImplementedError("V4.1 initial runtime requires PP=DCP=PCP=1")
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("V4.1 requires the hybrid KV cache manager")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("V4.1 cache dtype selector must be auto or bfloat16")
