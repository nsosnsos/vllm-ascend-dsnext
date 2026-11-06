# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.fused_moe import moe_comm_method
from vllm_ascend.utils import AscendDeviceType


def _vllm_config(role, draft_topology):
    kv_config = None
    if role is not None:
        kv_config = SimpleNamespace(
            kv_role=role,
            is_kv_consumer=role in ("kv_consumer", "kv_both"),
            is_kv_producer=role in ("kv_producer", "kv_both"),
        )
    spec_config = None
    if draft_topology is not None:
        spec_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    n_routed_experts=draft_topology[0], num_experts_per_tok=draft_topology[1]
                )
            )
        )
    return SimpleNamespace(
        kv_transfer_config=kv_config,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(n_routed_experts=384, num_experts_per_tok=6)),
        speculative_config=spec_config,
    )


@pytest.fixture
def mc2_environment(monkeypatch):
    for name in ("_MC2", "_MC2_DRAFT", "_MEGA_MOE", "_P_TP", "_OTP", "_LMTP", "_EMBED_TP", "_MLP_TP", "_DYNAMIC_EPLB"):
        monkeypatch.setattr(parallel_state, name, None)
    parallel_config = SimpleNamespace(
        data_parallel_size=8,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    ascend_config = SimpleNamespace(
        pd_tp_ratio=1,
        pd_head_ratio=1,
        enable_fused_mc2=1,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=0,
            lmhead_tensor_parallel_size=0,
            embedding_tensor_parallel_size=0,
            mlp_tensor_parallel_size=0,
        ),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "hccl")
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: SimpleNamespace(local_rank=0, device_group=object()))
    monkeypatch.setattr(parallel_state, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(parallel_state, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setattr(parallel_state, "_initialize_mega_moe_communicator", MagicMock())

    def create_group(ranks, local_rank, backend, group_name):
        group = MagicMock(name=group_name)
        group.unique_name = f"{group_name}:0"
        group.ranks = ranks[0]
        group.rank_in_group = local_rank
        group.world_size = len(ranks[0])
        return group

    factory = MagicMock(side_effect=create_group)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", factory)

    def configure(config):
        monkeypatch.setattr(parallel_state, "get_current_vllm_config_or_none", lambda: config)
        monkeypatch.setattr(moe_comm_method, "get_current_vllm_config_or_none", lambda: config)
        return parallel_config, factory

    return configure


@pytest.mark.parametrize("role", [None, "kv_consumer", "kv_producer", "kv_both"])
@pytest.mark.parametrize("draft_topology", [(128, 3), (384, 3), (128, 6), (384, 6), None])
def test_mc2_domains_follow_topology_in_every_deployment_mode(mc2_environment, role, draft_topology):
    config = _vllm_config(role, draft_topology)
    parallel_config, factory = mc2_environment(config)
    parallel_state.init_ascend_model_parallel(parallel_config)

    secondary = draft_topology is not None and draft_topology != (384, 6)
    expected_names = ["mc2", "mc2_draft", "mega_moe"] if secondary else ["mc2", "mega_moe"]
    assert [call.kwargs["group_name"] for call in factory.call_args_list] == expected_names
    # The logical domains use identical ranks but distinct physical group
    # requests, in one config-derived order on every participating rank.
    for call in factory.call_args_list:
        assert call.args == ([list(range(8))], 0, "hccl")
    target = parallel_state.get_mc2_group()
    draft = parallel_state.get_mc2_draft_group()
    mega = parallel_state.get_mega_moe_group()
    assert (
        moe_comm_method.resolve_mc2_group_for_topology(SimpleNamespace(num_experts=384, experts_per_token=6)) is target
    )
    if secondary:
        assert draft is not target and draft is not mega
        draft_moe = SimpleNamespace(num_experts=draft_topology[0], experts_per_token=draft_topology[1])
        assert moe_comm_method.resolve_mc2_group_for_topology(draft_moe) is draft
    else:
        assert draft is None

    # Repeated initialization must not add collective calls. Teardown owns
    # each domain once, including the new mixed-mode draft communicator.
    parallel_state.init_ascend_model_parallel(parallel_config)
    assert factory.call_count == len(expected_names)
    parallel_state.destroy_ascend_model_parallel()
    parallel_state.destroy_ascend_model_parallel()
    target.destroy.assert_called_once()
    mega.destroy.assert_called_once()
    if draft is not None:
        draft.destroy.assert_called_once()
    assert parallel_state.get_mc2_draft_group() is None
    assert not parallel_state.model_parallel_initialized()


def test_no_config_keeps_primary_domain(mc2_environment):
    parallel_config, factory = mc2_environment(None)
    parallel_state.init_ascend_model_parallel(parallel_config)
    assert [call.kwargs["group_name"] for call in factory.call_args_list] == ["mc2", "mega_moe"]
    primary = parallel_state.get_mc2_group()
    assert (
        moe_comm_method.resolve_mc2_group_for_topology(SimpleNamespace(num_experts=384, experts_per_token=6)) is primary
    )


def test_secondary_topology_rejects_missing_draft_domain(mc2_environment, monkeypatch):
    parallel_config, _ = mc2_environment(_vllm_config(None, (128, 3)))
    parallel_state.init_ascend_model_parallel(parallel_config)
    monkeypatch.setattr(parallel_state, "_MC2_DRAFT", None)
    with pytest.raises(RuntimeError, match="dedicated draft MC2 communication group"):
        moe_comm_method.resolve_mc2_group_for_topology(SimpleNamespace(num_experts=128, experts_per_token=3))
