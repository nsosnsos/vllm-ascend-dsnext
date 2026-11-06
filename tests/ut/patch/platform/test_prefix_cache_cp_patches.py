# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.deepseek_v41 import DeepseekV41SWASpec
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
    _is_deepseek_v4_kv_cache_spec,
    get_kv_cache_coordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _PRIVATE_CIRCLE_TENSOR_ATTR,
    _append_private_circle_tensors,
    _ascend_resolve_kv_cache_block_sizes,
    _get_kimi_k3_dspark_mixed_kv_cache_groups,
    _get_kv_cache_config_deepseek_v4,
    group_and_unify_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager
from vllm_ascend.worker.v2.attn_utils import _validate_kv_cache_num_blocks


def _make_hybrid_kv_cache_config(
    full_block_size: int = 16,
    mamba_block_size: int = 16,
) -> KVCacheConfig:
    full_spec = FullAttentionSpec(
        block_size=full_block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=mamba_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=full_spec.page_size_bytes * 10, shared_by=["attn"]),
            KVCacheTensor(size=mamba_spec.page_size_bytes * 10, shared_by=["mamba"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
        ],
    )


def _make_kimi_k3_dspark_kv_cache_specs(
    *,
    block_size: int = 384,
    page_size: int = 488448,
    target_layer_count: int = 24,
    draft_layer_count: int = 5,
    mamba_layer_count: int = 69,
    draft_uses_mla: bool = False,
) -> dict:
    target_mla_spec = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
        cache_dtype_str="auto",
    )
    if draft_uses_mla:
        draft_attention_spec = AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            page_size_padded=page_size,
            cache_dtype_str="auto",
            non_causal_multi_token_decode=True,
        )
    else:
        draft_attention_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.bfloat16,
            page_size_padded=page_size,
        )
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((10, 2304), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size,
        mamba_cache_mode="align",
        num_speculative_blocks=7,
    )
    specs = {
        f"language_model.model.layers.{layer_idx}.self_attn.attn": target_mla_spec
        for layer_idx in range(target_layer_count)
    }
    specs.update(
        {
            f"model.layers.{layer_idx}.self_attn.attn": draft_attention_spec
            for layer_idx in range(93, 93 + draft_layer_count)
        }
    )
    specs.update(
        {f"language_model.model.layers.{layer_idx}.self_attn": mamba_spec for layer_idx in range(mamba_layer_count)}
    )
    return specs


def _make_deepseek_v4_kv_cache_config() -> KVCacheConfig:
    c4_spec = MLAAttentionSpec(
        block_size=128 * 4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    c128_spec = MLAAttentionSpec(
        block_size=128 * 128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    c4_group_spec = UniformTypeKVCacheSpecs.from_specs({"c4_attn": c4_spec})
    c128_group_spec = UniformTypeKVCacheSpecs.from_specs({"c128_attn": c128_spec})
    assert c4_group_spec is not None
    assert c128_group_spec is not None
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=c4_spec.page_size_bytes * 10, shared_by=["c4_attn"]),
            KVCacheTensor(size=c128_spec.page_size_bytes * 10, shared_by=["c128_attn"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["c4_attn"], kv_cache_spec=c4_group_spec),
            KVCacheGroupSpec(layer_names=["c128_attn"], kv_cache_spec=c128_group_spec),
        ],
    )


def _make_vllm_config(
    *,
    enable_prefix_caching: bool,
    dcp: int,
    block_size: int = 16,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
            mamba_cache_mode="align",
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
        ),
        kv_transfer_config=None,
    )


def _make_coordinator_for_effective_block_size(
    *,
    dcp_world_size: int,
    enable_caching: bool,
) -> AscendHybridKVCacheCoordinator:
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.dcp_world_size = dcp_world_size
    coordinator.enable_caching = enable_caching
    return coordinator


def test_ascend_mla_page_size_includes_scale_storage() -> None:
    spec = AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        scale_dim=1,
        scale_dtype=torch.float16,
    )

    expected_page_size = 16 * (128 * 2 + 2)
    assert spec.unpadded_page_size_bytes == expected_page_size
    assert spec.real_page_size_bytes == expected_page_size
    assert spec.page_size_bytes == expected_page_size


def test_ascend_mla_merge_preserves_upstream_layout_fields() -> None:
    spec = AscendMLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        cache_dtype_str="fp8_ds_mla",
        page_size_padded=(512 // 4) * (128 * 2 + 2) + 128,
        compress_ratio=4,
        model_version="deepseek_v4",
        indexes_kv_by_block_stride=True,
        scale_dim=1,
        scale_dtype=torch.float16,
    )

    merged = AscendMLAAttentionSpec.merge([spec, replace(spec)])

    assert merged.block_size == spec.block_size
    assert merged.real_page_size_bytes == (512 // 4) * (128 * 2 + 2)
    assert merged.page_size_bytes == spec.page_size_padded
    assert merged.compress_ratio == spec.compress_ratio
    assert merged.model_version == spec.model_version
    assert merged.indexes_kv_by_block_stride == spec.indexes_kv_by_block_stride
    assert merged.scale_dim == spec.scale_dim
    assert merged.scale_dtype == spec.scale_dtype


@pytest.mark.parametrize(
    ("enable_prefix_caching", "expected_hash_block_size"),
    [
        pytest.param(False, math.lcm(16, 32) * 2, id="dcp-without-prefix-caching"),
        pytest.param(True, math.gcd(16, 32), id="dcp-with-prefix-caching"),
    ],
)
def test_resolve_kv_cache_block_sizes_with_cp_hybrid_groups(
    enable_prefix_caching: bool,
    expected_hash_block_size: int,
) -> None:
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=32)
    vllm_config = _make_vllm_config(
        enable_prefix_caching=enable_prefix_caching,
        dcp=2,
    )

    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        kv_cache_config,
        vllm_config,
    )

    expected_scheduler_block_size = math.lcm(16, 32) * 2
    assert scheduler_block_size == expected_scheduler_block_size
    assert hash_block_size == expected_hash_block_size


def test_deepseek_v4_groups_use_logical_sizes_and_full_attention_manager() -> None:
    c128_spec = MLAAttentionSpec(
        block_size=128 * 128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    c4_spec = MLAAttentionSpec(
        block_size=128 * 4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        sliding_window=512,
    )

    grouped_specs = group_and_unify_kv_cache_specs(
        {
            "c128": c128_spec,
            "swa": swa_spec,
            "c4": c4_spec,
        }
    )

    assert grouped_specs is not None
    assert [group.block_size for group in grouped_specs[:2]] == [512, 16384]
    for group in grouped_specs[:2]:
        spec = next(iter(group.kv_cache_specs.values()))
        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager


@pytest.mark.parametrize(
    ("block_size", "page_size", "draft_uses_mla"),
    [
        pytest.param(384, 488448, False, id="gqa-tp16"),
        pytest.param(768, 976896, False, id="gqa-tp8"),
        pytest.param(384, 488448, True, id="mla-tp16"),
    ],
)
def test_kimi_k3_dspark_uses_four_mixed_kv_groups(block_size, page_size, draft_uses_mla) -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(
        _make_kimi_k3_dspark_kv_cache_specs(
            block_size=block_size,
            page_size=page_size,
            draft_uses_mla=draft_uses_mla,
        )
    )

    assert groups is not None
    assert [len(group.layer_names) for group in groups] == [29, 23, 23, 23]
    assert all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in groups)
    mixed_specs = groups[0].kv_cache_spec.kv_cache_specs
    expected_mla_count = 29 if draft_uses_mla else 24
    assert sum(isinstance(spec, MLAAttentionSpec) for spec in mixed_specs.values()) == expected_mla_count
    assert all(isinstance(spec, FullAttentionSpec) for spec in mixed_specs.values())


def test_kimi_k3_gqa_mixed_groups_preserve_scheduler_and_mamba_contracts() -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(_make_kimi_k3_dspark_kv_cache_specs())
    assert groups is not None
    max_model_len = 133120
    vllm_config = _make_vllm_config(enable_prefix_caching=True, dcp=1, block_size=384)
    worker_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )

    assert worker_config.has_mamba_layers
    assert worker_config.needs_kv_cache_zeroing
    worker_mamba_widths = {
        group.kv_cache_spec.max_num_blocks_per_req(vllm_config, max_model_len) for group in groups[1:]
    }
    assert worker_mamba_widths == {354}

    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    assert isinstance(scheduler_config.kv_cache_groups[0].kv_cache_spec, MLAAttentionSpec)
    assert all(isinstance(group.kv_cache_spec, MambaSpec) for group in scheduler_config.kv_cache_groups[1:])
    scheduler_mamba_widths = {
        group.kv_cache_spec.max_num_blocks_per_req(vllm_config, max_model_len)
        for group in scheduler_config.kv_cache_groups[1:]
    }
    assert scheduler_mamba_widths == worker_mamba_widths
    assert scheduler_config.needs_kv_cache_zeroing


def test_kimi_k3_gqa_mixed_groups_use_expected_physical_layout(monkeypatch) -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(_make_kimi_k3_dspark_kv_cache_specs())
    assert groups is not None
    page_size = 488448
    expected_num_blocks = 100
    available_memory = page_size * 29 * expected_num_blocks
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils.may_override_num_blocks",
        lambda _config, num_blocks: num_blocks,
    )

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        SimpleNamespace(),
        groups,
        available_memory,
    )

    assert num_blocks == expected_num_blocks
    assert len(tensors) == 29
    assert [len(tensor.shared_by) for tensor in tensors] == [4] * 23 + [1] * 6
    assert all(tensor.size == page_size * expected_num_blocks for tensor in tensors)
    assert sum(tensor.size for tensor in tensors) == available_memory


def test_kimi_k3_gqa_mixed_grouping_falls_back_on_unrecognized_layer() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    specs["unrecognized.layer"] = next(iter(specs.values()))

    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_kimi_k3_dspark_group_count_is_derived_from_layer_ratio() -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(
        _make_kimi_k3_dspark_kv_cache_specs(
            target_layer_count=20,
            draft_layer_count=4,
            mamba_layer_count=70,
        )
    )

    assert groups is not None
    assert [len(group.layer_names) for group in groups] == [24, 24, 23, 23]


def test_kimi_k3_dspark_mixed_grouping_falls_back_on_unaligned_pages() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    draft_layer = "model.layers.93.self_attn.attn"
    specs[draft_layer] = replace(specs[draft_layer], page_size_padded=976896)

    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_deepseek_v4_scheduler_lcm_uses_logical_group_sizes() -> None:
    kv_cache_config = _make_deepseek_v4_kv_cache_config()
    vllm_config = _make_vllm_config(
        enable_prefix_caching=True,
        dcp=1,
        block_size=128,
    )

    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        kv_cache_config,
        vllm_config,
    )

    assert scheduler_block_size == 16384
    assert hash_block_size == 512


@pytest.mark.parametrize(
    ("spec_factory", "dcp", "enable_caching", "expected"),
    [
        pytest.param(
            lambda: FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.float16,
            ),
            2,
            True,
            32,
            id="full-attention-scales-with-dcp",
        ),
        pytest.param(
            lambda: MambaSpec(
                block_size=16,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="none",
            ),
            2,
            True,
            16,
            id="mamba-keeps-physical-block-size-with-prefix-caching",
        ),
        pytest.param(
            lambda: FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.float16,
            ),
            1,
            True,
            16,
            id="full-attention-no-cp",
        ),
    ],
)
def test_get_effective_block_size(
    spec_factory,
    dcp: int,
    enable_caching: bool,
    expected: int,
) -> None:
    coordinator = _make_coordinator_for_effective_block_size(
        dcp_world_size=dcp,
        enable_caching=enable_caching,
    )

    assert coordinator._get_effective_block_size(spec_factory()) == expected


def test_get_kv_cache_coordinator_delegates_single_group(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=16)
    single_group_config = KVCacheConfig(
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_tensors=kv_cache_config.kv_cache_tensors[:1],
        kv_cache_groups=kv_cache_config.kv_cache_groups[:1],
    )

    def _fake_orig(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )

    coordinator = get_kv_cache_coordinator(
        single_group_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=16,
    )

    assert coordinator is sentinel


def test_get_kv_cache_coordinator_delegates_hybrid_without_caching(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=16)

    def _fake_orig(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=2,
        pcp_world_size=1,
        hash_block_size=16,
    )

    assert coordinator is sentinel


def test_get_kv_cache_coordinator_uses_ascend_for_deepseek_v4(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_deepseek_v4_kv_cache_config()

    def _fake_orig(*args, **kwargs):
        raise AssertionError("DeepSeek V4 should use AscendHybridKVCacheCoordinator")

    def _fake_ascend_coordinator(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator.AscendHybridKVCacheCoordinator",
        _fake_ascend_coordinator,
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=128,
    )

    assert coordinator is sentinel


class _FakeEagleManager:
    def __init__(self) -> None:
        self.use_eagle = False


def test_verify_and_split_propagates_eagle_to_managers() -> None:
    """Regression for DeepSeek-V4 prefix-cache hit rate 0% with MTP/EAGLE.

    The eagle bit must reach each single-type manager: the SWA write path
    (``cache_blocks`` -> ``reachable_block_mask``) keys the retained checkpoint
    tail on ``manager.use_eagle``, while the read path
    (``find_longest_cache_hit``) applies ``drop_eagle_block`` to the same
    groups. If the manager keeps the default ``use_eagle=False`` the retained
    tail is one block short of the eagle peek boundary, the SWA group never
    hits, and the min-over-groups hybrid hit collapses to 0%.
    """
    kv_cache_config = _make_deepseek_v4_kv_cache_config()

    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = kv_cache_config
    coordinator.dcp_world_size = 1
    coordinator.enable_caching = True
    # The c128 group (index 1) carries the EAGLE/MTP layers.
    coordinator.eagle_group_ids = {1}

    coordinator.single_type_managers = (_FakeEagleManager(), _FakeEagleManager())

    coordinator.verify_and_split_kv_cache_groups()

    assert coordinator.single_type_managers[1].use_eagle is True
    assert coordinator.single_type_managers[0].use_eagle is False


def test_verify_and_split_propagates_eagle_to_merged_spec_siblings() -> None:
    """Upstream ``_annotate_eagle_groups_deepseek_v4`` flags only the single
    group holding the MTP layer, but the read path merges same-spec groups and
    applies ``drop_eagle_block`` to the whole merged group. So every sibling
    sharing that spec must also get ``use_eagle=True`` on the write path, else
    ``get_cached_block`` (which needs the block cached for *all* group ids)
    misses and the hit collapses to 0%.
    """
    base_config = _make_deepseek_v4_kv_cache_config()
    # Reuse the c128 spec object so the two c128 groups compare equal and merge
    # into one attention group in verify_and_split.
    c128_group_spec = base_config.kv_cache_groups[1].kv_cache_spec
    kv_cache_config = KVCacheConfig(
        num_blocks=base_config.num_blocks,
        kv_cache_tensors=base_config.kv_cache_tensors,
        kv_cache_groups=[
            base_config.kv_cache_groups[0],  # c4   -> gid 0 (distinct spec)
            base_config.kv_cache_groups[1],  # c128 -> gid 1
            KVCacheGroupSpec(layer_names=["c128_attn_mtp"], kv_cache_spec=c128_group_spec),  # gid 2
        ],
    )

    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = kv_cache_config
    coordinator.dcp_world_size = 1
    coordinator.enable_caching = True
    # Only the MTP sibling (gid 2) is flagged, exactly as upstream does.
    coordinator.eagle_group_ids = {2}

    coordinator.single_type_managers = (
        _FakeEagleManager(),
        _FakeEagleManager(),
        _FakeEagleManager(),
    )

    coordinator.verify_and_split_kv_cache_groups()

    # Both gid 1 and gid 2 share the c128 spec and merge, so both must be eagle.
    assert coordinator.single_type_managers[1].use_eagle is True
    assert coordinator.single_type_managers[2].use_eagle is True
    assert coordinator.single_type_managers[0].use_eagle is False


def test_deepseek_v4_detection_handles_non_mapping_nested_specs() -> None:
    kv_cache_spec = SimpleNamespace(
        kv_cache_specs=[
            SimpleNamespace(model_version="deepseek_v4"),
        ]
    )
    unknown_spec = SimpleNamespace(kv_cache_specs=object())

    assert _is_deepseek_v4_kv_cache_spec(kv_cache_spec)
    assert not _is_deepseek_v4_kv_cache_spec(unknown_spec)


def test_get_num_blocks_to_allocate_matches_upstream_interface() -> None:
    coordinator = AscendHybridKVCacheCoordinator.__new__(
        AscendHybridKVCacheCoordinator
    )
    manager = MagicMock()
    manager.get_num_blocks_to_allocate.return_value = 3
    coordinator.single_type_managers = (manager,)
    coordinator.private_circle_group_ids = set()
    computed_blocks = ([MagicMock()],)

    result = coordinator.get_num_blocks_to_allocate(
        request_id="request-0",
        num_tokens=12,
        new_computed_blocks=computed_blocks,
        num_encoder_tokens=0,
        total_computed_tokens=8,
        num_local_computed_tokens=6,
        num_tokens_main_model=10,
        apply_admission_cap=True,
    )

    assert result == 3
    manager.get_num_blocks_to_allocate.assert_called_once_with(
        "request-0",
        12,
        computed_blocks[0],
        8,
        6,
        10,
        apply_admission_cap=True,
    )


def test_remove_skipped_blocks_matches_upstream_interface() -> None:
    coordinator = AscendHybridKVCacheCoordinator.__new__(
        AscendHybridKVCacheCoordinator
    )
    manager = MagicMock()
    coordinator.single_type_managers = (manager,)
    coordinator.private_circle_group_ids = set()

    coordinator.remove_skipped_blocks("request-0", 8)

    manager.remove_skipped_blocks.assert_called_once_with("request-0", 8)


def test_allocate_new_computed_blocks_uses_two_phase_upstream_interface() -> None:
    """Shared groups use the single upstream API for local and external KV."""
    coordinator = AscendHybridKVCacheCoordinator.__new__(
        AscendHybridKVCacheCoordinator
    )
    shared_manager = MagicMock()
    private_circle_manager = MagicMock()
    coordinator.single_type_managers = (shared_manager, private_circle_manager)
    coordinator.private_circle_group_ids = {1}
    shared_blocks = [MagicMock()]
    private_blocks = [MagicMock()]

    coordinator.allocate_new_computed_blocks(
        "request-0",
        (shared_blocks, private_blocks),
        num_local_computed_tokens=128,
        num_external_computed_tokens=64,
    )

    shared_manager.add_local_computed_blocks.assert_called_once_with(
        "request-0", shared_blocks, 128, 64
    )
    shared_manager.allocate_external_computed_blocks.assert_called_once_with(
        "request-0", 128, 64
    )
    private_circle_manager.add_local_computed_blocks.assert_not_called()
    private_circle_manager.allocate_external_computed_blocks.assert_not_called()

def test_ascend_mamba_manager_uses_logical_block_size_with_prefix_caching() -> None:
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    block_pool = BlockPool(
        10,
        True,
        16,
        False,
        MagicMock(),
    )

    manager_kwargs = dict(
        kv_cache_spec=mamba_spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=1,
        dcp_world_size=2,
        pcp_world_size=1,
    )
    manager_kwargs["scheduler_block_size"] = mamba_spec.block_size
    manager = AscendMambaManager(**manager_kwargs)

    assert manager.block_size == mamba_spec.block_size


def test_private_circle_tensors_are_appended_after_shared_rank_alignment() -> None:
    shared_spec = MLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    private_spec = DeepseekV41SWASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        sliding_window=128,
        compress_ratio=1,
        model_version="deepseek_v4",
    )
    shared_group = UniformTypeKVCacheSpecs.from_specs({"shared": shared_spec})
    private_group = UniformTypeKVCacheSpecs.from_specs({"private": private_spec})
    assert shared_group is not None
    assert private_group is not None
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(size=shared_spec.page_size_bytes * 8, shared_by=["shared"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["shared"], kv_cache_spec=shared_group),
            KVCacheGroupSpec(layer_names=["private"], kv_cache_spec=private_group),
        ],
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        speculative_config=None,
    )

    _append_private_circle_tensors(vllm_config, kv_cache_config)

    private_tensor = next(
        tensor for tensor in kv_cache_config.kv_cache_tensors
        if tensor.shared_by == ["private"]
    )
    # W=128, B=32, no speculation: 2 × max_num_seqs allocations × 4 blocks + null block.
    assert private_tensor.size == private_spec.page_size_bytes * 17
    assert getattr(private_tensor, _PRIVATE_CIRCLE_TENSOR_ATTR) is True
    # The shared count is unchanged and a repeated post-alignment call is idempotent.
    assert kv_cache_config.num_blocks == 8
    _append_private_circle_tensors(vllm_config, kv_cache_config)
    assert len([t for t in kv_cache_config.kv_cache_tensors if t.shared_by == ["private"]]) == 1


def test_private_circle_worker_accepts_fixed_ring_capacity(monkeypatch) -> None:
    private_spec = DeepseekV41SWASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        sliding_window=128,
        compress_ratio=1,
        model_version="deepseek_v4",
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.attn_utils.envs.VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL",
        True,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        speculative_config=None,
    )
    kv_cache_config = SimpleNamespace(num_blocks=8)

    _validate_kv_cache_num_blocks(17, kv_cache_config, private_spec, vllm_config)
    with pytest.raises(AssertionError, match="fixed ring layout"):
        _validate_kv_cache_num_blocks(8, kv_cache_config, private_spec, vllm_config)


def test_swa_reachable_block_mask_sparse_with_lcm_alignment() -> None:
    """Regression: when ``scheduler_block_size`` is aligned to ``lcm_block_size``
    (instead of the raw-block-size LCM), ``SlidingWindowManager.reachable_block_mask``
    must produce a sparse mask rather than returning ``None``.

    Before the fix, ``alignment_tokens`` was the LCM of raw block_sizes (e.g. 32),
    making ``need >= per_segment`` always true for Ascend's SWA configuration and
    the mask returned ``None`` (cache everything). After the fix the alignment is
    ``lcm_block_size`` (e.g. 4096), which is large enough that only the tail
    blocks within each segment need caching.
    """
    spec = SlidingWindowMLASpec(
        block_size=32,  # Ascend SWA block_size (--block-size 32)
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
        sliding_window=128,  # DeepSeek V4 window
        compress_ratio=1,
    )
    alignment_tokens = 4096  # lcm_block_size

    mask = SlidingWindowManager.reachable_block_mask(
        start_block=0,
        end_block=256,  # 256 × 32 = 8192 tokens (2 × alignment_tokens)
        alignment_tokens=alignment_tokens,
        kv_cache_spec=spec,
        use_eagle=False,
        retention_interval=None,
    )

    # Must produce a sparse mask, not None.
    assert mask is not None, "should produce sparse mask with lcm alignment"

    true_blocks = sum(mask)

    # need = cdiv(window−1, block_size) = cdiv(127, 32) = 4
    # per_segment = alignment_tokens // block_size = 4096 // 32 = 128
    # Each 128-block segment caches the last 4 blocks (= 0 % sparse padding).
    total_blocks = len(mask)
    expected = 4 * (total_blocks // 128)
    assert true_blocks == expected, (
        f"expected {expected} cached blocks ({4}/{128} per segment), got {true_blocks}/{total_blocks}"
    )
    assert true_blocks > 0 and true_blocks < total_blocks, f"mask should be sparse, got {true_blocks}/{total_blocks}"


def _make_deferred_free_scheduler(
    private_pool,
    *,
    defer_block_free: bool = True,
    processed_step_seq: int = 0,
):
    scheduler = SimpleNamespace(
        defer_block_free=defer_block_free,
        processed_step_seq=processed_step_seq,
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(private_circle_pool=private_pool)
        ),
    )
    calls = []
    scheduler._orig_calls = calls
    return scheduler


def _install_noop_originals(monkeypatch):
    """Stub the wrapped upstream methods the deferred-free hooks call into."""
    import vllm_ascend.patch.platform.patch_private_circle_pool as lifetime

    free_calls = []

    def _noop_free_request_blocks(self, request):
        free_calls.append(request.request_id)

    def _noop_update_from_output(self, scheduler_output, model_runner_output):
        return None

    monkeypatch.setattr(
        lifetime, "_original_free_request_blocks", _noop_free_request_blocks
    )
    monkeypatch.setattr(
        lifetime, "_original_update_from_output", _noop_update_from_output
    )
    return lifetime, free_calls


def test_deferred_free_path_defers_private_allocation_return(monkeypatch):
    """abort/preempt on the deferred path must not leak the allocation.

    Ownership detaches immediately (a resumed request reserves a different
    slot), and the slot only re-enters the free list once the owning step's
    fence is processed.
    """
    from vllm_ascend.core.private_circle_pool import (
        PrivateCirclePool,
        PrivateCircleConfig,
    )

    lifetime, free_calls = _install_noop_originals(monkeypatch)
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    pool = PrivateCirclePool(config)
    total = config.num_allocations
    allocation = pool.reserve("request-d1")
    assert allocation is not None

    request = SimpleNamespace(
        request_id="request-d1",
        last_sched_seq=5,
        private_circle_allocation=allocation,
        private_circle_transfer_state=lifetime.PRIVATE_CIRCLE_TRANSFER_RECEIVING,
        private_circle_imported_window_start=1,
        private_circle_imported_valid_length=9,
    )
    scheduler = _make_deferred_free_scheduler(pool, processed_step_seq=3)

    lifetime._patched_free_request_blocks(scheduler, request)
    # Original upstream path still ran exactly once.
    assert free_calls == ["request-d1"]
    # Bookkeeping dropped, slot not yet recycled, request fields cleaned.
    assert not pool.contains("request-d1")
    assert request.private_circle_allocation is None
    assert request.private_circle_transfer_state == lifetime.PRIVATE_CIRCLE_TRANSFER_NONE
    assert pool.free_allocations == total - 1

    # A resumed/preempted request cannot observe the detached slot.
    assert pool.reserve("request-d1") != allocation

    # Fence not yet processed: drain keeps the entry pending.
    lifetime._patched_update_from_output(scheduler, SimpleNamespace(), SimpleNamespace())
    assert pool.free_allocations == total - 2  # resumed request still holds one

    # Fence processed: the detached slot returns to the free list once.
    scheduler.processed_step_seq = 5
    lifetime._patched_update_from_output(scheduler, SimpleNamespace(), SimpleNamespace())
    assert pool.free_allocations == total - 1
    assert scheduler._private_circle_deferred_releases == []


def test_deferred_free_disabled_path_defers_nothing(monkeypatch):
    """Non-deferred finishes keep going through the KVCacheManager.free hook."""
    from vllm_ascend.core.private_circle_pool import (
        PrivateCirclePool,
        PrivateCircleConfig,
    )

    lifetime, free_calls = _install_noop_originals(monkeypatch)
    config = PrivateCircleConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    pool = PrivateCirclePool(config)
    total = config.num_allocations
    pool.reserve("request-normal")

    # last step already processed -> upstream takes the immediate-free branch.
    scheduler = _make_deferred_free_scheduler(pool, processed_step_seq=7)
    request = SimpleNamespace(
        request_id="request-normal",
        last_sched_seq=7,
        private_circle_allocation=0,
        private_circle_transfer_state=lifetime.PRIVATE_CIRCLE_TRANSFER_NONE,
        private_circle_imported_window_start=None,
        private_circle_imported_valid_length=None,
    )
    lifetime._patched_free_request_blocks(scheduler, request)
    assert free_calls == ["request-normal"]
    # The patch hook did not detach; the release happens inside
    # KVCacheManager.free (asserted by ownership still present here).
    assert pool.contains("request-normal")
    assert not hasattr(scheduler, "_private_circle_deferred_releases")
    pool.release("request-normal")
    assert pool.free_allocations == total


def _make_private_pool_scheduler_env(monkeypatch, *, enable_dsa_cp: bool):
    """Wire a scheduler whose original __init__ already saw private groups."""
    import vllm_ascend.patch.platform.patch_private_circle_pool as lifetime
    from vllm_ascend.core.deepseek_v41 import DeepseekV41SWASpec

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL", "1")

    def _noop_original_init(scheduler, vllm_config, *args, **kwargs):
        private_spec = DeepseekV41SWASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
            model_version="deepseek_v4",
        )
        scheduler.kv_cache_manager = SimpleNamespace(
            coordinator=SimpleNamespace(
                private_circle_group_ids=frozenset({1}),
                single_type_managers={
                    1: SimpleNamespace(kv_cache_spec=private_spec)
                },
            )
        )

    monkeypatch.setattr(
        lifetime, "_original_scheduler_init", _noop_original_init
    )
    monkeypatch.setattr(
        "vllm_ascend.utils.enable_dsa_cp", lambda: enable_dsa_cp
    )
    scheduler = SimpleNamespace()
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=None,
    )
    return lifetime, scheduler, vllm_config


def test_private_pool_rejects_dsa_cp_at_startup(monkeypatch):
    """pool + DSA-CP must fail fast, not silently mis-write the ring."""
    lifetime, scheduler, vllm_config = _make_private_pool_scheduler_env(
        monkeypatch, enable_dsa_cp=True
    )
    with pytest.raises(RuntimeError, match="context parallelism"):
        lifetime._patched_scheduler_init(scheduler, vllm_config)


@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_private_pool_rejects_pcp_and_dcp_at_startup(
    monkeypatch, parallel_kwargs
):
    """PCP / DCP route away from AscendDSAImpl the same way; fail fast too."""
    lifetime, scheduler, vllm_config = _make_private_pool_scheduler_env(
        monkeypatch, enable_dsa_cp=False
    )
    for key, value in parallel_kwargs.items():
        setattr(vllm_config.parallel_config, key, value)
    with pytest.raises(RuntimeError, match="context parallelism"):
        lifetime._patched_scheduler_init(scheduler, vllm_config)


def test_private_pool_startup_succeeds_without_cp(monkeypatch):
    lifetime, scheduler, vllm_config = _make_private_pool_scheduler_env(
        monkeypatch, enable_dsa_cp=False
    )
    lifetime._patched_scheduler_init(scheduler, vllm_config)
    coordinator = scheduler.kv_cache_manager.coordinator
    assert coordinator.private_circle_pool is not None
    assert coordinator.private_circle_config.window_size == 128
    assert coordinator.private_circle_group_ids == frozenset({1})
    # D nodes must never recompute after a private-circle load failure.
    assert scheduler.recompute_kv_load_failures is False
