# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Folded indexer twin: spec formulas and slot placement.

The folded twin re-packs a candidate-source indexer cache into
candidate-block rows ([N, block_size//candidate_block_size, 16*68]); this
verifies the spec arithmetic and the slot placement bookkeeping.  The folded
plane uses a different byte layout from the token-indexed plane, so the
Mooncake path transfers the complete physical slot and preserves both views
without a device-side re-pack.
"""

import torch

from vllm_ascend.core.deepseek_v41 import (
    STATE_RING_ROWS,
    DeepseekV41CompressorStateSpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerFoldedSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
    _cache_plane_sizes,
    group_cache_specs,
    plan_cache_slots,
    reshape_cache,
)

WIDTH = 128
BLOCK = 128
FOLDED_PAGE = 16 * 544


def _specs(with_folded: bool) -> dict:
    specs = {}
    for layer in (2, 8, 14, 20):
        ratio = 2 if layer < 20 else 1
        specs[f"model.layers.{layer}.self_attn.long_kv_cache"] = DeepseekV41FullSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=320,
            dtype=torch.uint8,
            compress_ratio=ratio,
        )
        specs[f"model.layers.{layer}.indexer.k_cache"] = DeepseekV41IndexerSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=WIDTH // 2,
            dtype=torch.uint8,
            compress_ratio=ratio,
            scale_dim=WIDTH // 32,
            scale_dtype=torch.uint8,
        )
    for layer in (2, 8, 14):
        specs[f"model.layers.{layer}.self_attn.state_cache"] = (
            DeepseekV41CompressorStateSpec(
                block_size=STATE_RING_ROWS,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
            )
        )
    for layer in range(40):
        specs[f"model.layers.{layer}.self_attn.swa_cache"] = DeepseekV41SWASpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=544,
            dtype=torch.uint8,
            sliding_window=128,
        )
    if with_folded:
        specs["model.layers.20.self_attn.indexer.k_cache_folded"] = (
            _folded_spec()
        )
    return specs


def _folded_spec() -> DeepseekV41IndexerFoldedSpec:
    return DeepseekV41IndexerFoldedSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=544,
        dtype=torch.uint8,
        compress_ratio=8,
    )


def test_folded_spec_formulas():
    spec = _folded_spec()
    assert spec.storage_block_size == 16
    assert sum(_cache_plane_sizes(spec)) == FOLDED_PAGE == 8704


def test_folded_slot_placement_grows_only_candidate_slot():
    base = [s.page_size_bytes for s in plan_cache_slots(_specs(False))]
    folded = [s.page_size_bytes for s in plan_cache_slots(_specs(True))]
    assert folded[:3] == base[:3]
    assert folded[3] == base[3] + FOLDED_PAGE
    slot3 = plan_cache_slots(_specs(True))[3]
    names = [p.name for p in slot3.placements]
    assert "model.layers.20.self_attn.indexer.k_cache_folded" in names
    kv = next(p for p in slot3.placements if p.name.endswith("long_kv_cache"))
    index = next(p for p in slot3.placements if p.name.endswith("indexer.k_cache"))
    twin = next(p for p in slot3.placements if p.name.endswith("k_cache_folded"))
    index_plane = 128 * 68
    assert twin.offset == kv.page_size_bytes + index_plane
    assert (
        kv.page_size_bytes + index.page_size_bytes + twin.page_size_bytes
        == slot3.page_size_bytes
    )


def test_folded_joins_full_group():
    groups = group_cache_specs(_specs(True), divert_swa=True)
    assert len(groups) == 4
    full_names = list(groups[0].kv_cache_specs)
    assert any(n.endswith("k_cache_folded") for n in full_names)


def test_folded_slot_views_use_distinct_physical_regions():
    specs = _specs(True)
    slots = plan_cache_slots(specs)
    slot = slots[3]
    placements = {placement.name: placement for placement in slot.placements}
    index_name = "model.layers.20.self_attn.indexer.k_cache"
    folded_name = "model.layers.20.self_attn.indexer.k_cache_folded"
    num_blocks = 2
    raw = torch.zeros(num_blocks * slot.page_size_bytes, dtype=torch.uint8)

    index_src = reshape_cache(
        raw,
        specs[index_name],
        num_blocks=num_blocks,
        offset=placements[index_name].offset,
        block_stride=slot.page_size_bytes,
    )
    folded_src = reshape_cache(
        raw,
        specs[folded_name],
        num_blocks=num_blocks,
        offset=placements[folded_name].offset,
        block_stride=slot.page_size_bytes,
    )
    index_src[0].fill_(0x11)
    index_src[1].fill_(0x12)
    folded_src.fill_(0x22)

    index_offset = placements[index_name].offset
    index_key_bytes, index_scale_bytes = _cache_plane_sizes(specs[index_name])
    folded_offset = placements[folded_name].offset
    folded_bytes = sum(_cache_plane_sizes(specs[folded_name]))
    assert torch.all(raw[index_offset : index_offset + index_key_bytes] == 0x11)
    assert torch.all(
        raw[
            index_offset + index_key_bytes : index_offset + index_key_bytes + index_scale_bytes
        ]
        == 0x12
    )
    assert torch.all(raw[folded_offset : folded_offset + folded_bytes] == 0x22)
