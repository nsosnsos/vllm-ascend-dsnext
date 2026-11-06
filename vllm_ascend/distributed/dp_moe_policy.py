# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-local communication contracts for DP ranks with unequal inputs."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.ascend_forward_context import MoECommType


@dataclass(frozen=True)
class DPMoEPolicy:
    """An algorithm and capacity agreed at startup, never from a local batch."""

    comm_type: "MoECommType"
    max_tokens_per_rank: int
    topology: tuple[int, int, int]
    runtime_only: bool = False

    def check_num_tokens(self, num_tokens: int) -> None:
        if not 0 < num_tokens <= self.max_tokens_per_rank:
            raise ValueError(
                "Input violates the fixed DP MoE communication contract: "
                f"tokens={num_tokens}, capacity={self.max_tokens_per_rank}, "
                f"communication={self.comm_type.name}."
            )

    @property
    def ep_size(self) -> int:
        return self.topology[2]

    @property
    def global_bs(self) -> int:
        return self.max_tokens_per_rank * self.ep_size


def _pd_connector_types(name: str, role: str, extra_config: dict[str, Any]) -> frozenset[str] | None:
    """Resolve supported leaves without loading connector implementations.

    All children must have the parent's P/D role. An unknown sibling can
    change loading/recompute semantics, so finding Mooncake alone is not
    enough to establish a fixed communication policy.
    """
    if role not in ("kv_producer", "kv_consumer"):
        return None
    if name == "AscendStoreConnector" and role == "kv_consumer" and extra_config.get("consumer_is_to_load", False):
        # A loading Store consumer can schedule local prompt replay. Its
        # execution bound is not the decode-only max_seqs * (K + 1) contract.
        return None
    if name in ("MooncakeHybridConnector", "AscendStoreConnector"):
        return frozenset((name,))
    if name not in ("MultiConnector", "AscendMultiConnector"):
        return None
    children = extra_config.get("connectors", [])
    if not children:
        return None
    types: set[str] = set()
    for child in children:
        if child["kv_role"] != role:
            return None
        child_types = _pd_connector_types(child["kv_connector"], role, child.get("kv_connector_extra_config", {}))
        if child_types is None:
            return None
        types.update(child_types)
    return frozenset(types)


def build_dp_moe_policies(vllm_config: "VllmConfig", target_model, drafter) -> tuple[DPMoEPolicy, DPMoEPolicy] | None:
    """Resolve the independent P/D execution bounds after loading both models.

    Called by the V4.1 runner after both models have loaded, before profiling.
    D policies apply to decode execution and capture, while memory profiling
    retains the original eager communication selection and capacities.
    """
    # Worker-only imports, also avoiding the forward-context/MoE import cycle.
    from vllm.config import CUDAGraphMode
    from vllm.logger import logger

    from vllm_ascend import envs
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ascend_forward_context import (
        _MC2_TOKENS_PER_RANK_LIMIT,
        MoECommType,
        _get_a5_moe_group_size,
        _select_a5_moe_comm_method,
        get_a5_mega_moe_buffer_tokens_per_rank,
        get_mc2_tokens_capacity,
    )
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.moe_comm_method import make_moe_topology_key
    from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, get_secondary_moe_topology

    parallel = vllm_config.parallel_config
    transfer = vllm_config.kv_transfer_config
    ascend = get_ascend_config()
    finegrained = ascend.finegrained_tp_config
    spec = vllm_config.speculative_config
    connector_types = (
        _pd_connector_types(transfer.kv_connector, transfer.kv_role, transfer.kv_connector_extra_config)
        if transfer is not None
        else None
    )
    if (
        get_ascend_device_type() != AscendDeviceType.A5
        or transfer is None
        or connector_types is None
        or "MooncakeHybridConnector" not in connector_types
        or parallel.data_parallel_size <= 1
        or not parallel.enable_expert_parallel
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.enable_dbo
        or vllm_config.lora_config is not None
        or ascend.get_mc2_comm_alg() == "hierarchy"
        or any(
            (
                finegrained.oproj_tensor_parallel_size,
                finegrained.lmhead_tensor_parallel_size,
                finegrained.embedding_tensor_parallel_size,
                finegrained.mlp_tensor_parallel_size,
                finegrained.olora_tensor_parallel_size,
            )
        )
        or not isinstance(drafter, AscendDSparkProposer)
        or drafter.dynamic_spec is not None
        or spec.num_speculative_tokens_per_batch_size is not None
        or spec.disable_padded_drafter_batch
        or spec.num_speculative_tokens <= 0
    ):
        return None

    is_decode = transfer.is_kv_consumer and not transfer.is_kv_producer
    if is_decode:
        if not (envs.VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL or ascend.scheduler_config.recompute_scheduler_enable):
            return None
        if vllm_config.compilation_config.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
            return None
    elif not (
        transfer.is_kv_producer
        and not transfer.is_kv_consumer
        and vllm_config.model_config.enforce_eager
        and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        and not drafter.use_cuda_graph
    ):
        return None

    secondary = get_secondary_moe_topology(vllm_config)
    if secondary is None:
        return None

    target_layers = [layer for layer in target_model.modules() if isinstance(layer, AscendMoERunner)]
    draft_layers = [layer for layer in drafter.model.modules() if isinstance(layer, AscendMoERunner)]
    if not target_layers or not draft_layers:
        return None

    mc2_capacity = get_mc2_tokens_capacity()
    if is_decode:
        target_tokens = vllm_config.scheduler_config.max_num_seqs * (1 + spec.num_speculative_tokens)
        if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            target_tokens = max(target_tokens, vllm_config.compilation_config.max_cudagraph_capture_size or 0)
        # Runtime dispatch capacity is independent of the scheduler's total
        # token budget and of the legacy capacity used during profiling.
        mc2_capacity = _MC2_TOKENS_PER_RANK_LIMIT
    else:
        target_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    draft_tokens = drafter.max_query_tokens
    ep_size = parallel.world_size_across_dp
    target_topology = make_moe_topology_key(
        vllm_config.model_config.get_num_experts(),
        vllm_config.model_config.hf_text_config.num_experts_per_tok,
        ep_size,
    )
    draft_topology = make_moe_topology_key(*secondary, ep_size)

    # Inspect every actual layer, rather than the process-wide quantization
    # cache, which can also be populated while constructing the draft model.
    for layers, topology, tokens, secondary_topology, expected in (
        (target_layers, target_topology, target_tokens, False, MoECommType.MC2 if is_decode else MoECommType.FUSED_MC2),
        (draft_layers, draft_topology, draft_tokens, True, MoECommType.MC2),
    ):
        for layer in layers:
            config = layer.moe_config
            if config.is_sequence_parallel:
                return None
            if make_moe_topology_key(config.num_experts, config.experts_per_token, config.ep_size) != topology:
                return None
            method = _select_a5_moe_comm_method(
                tokens,
                vllm_config,
                mc2_capacity,
                layer.quant_type,
                config.activation.value,
                _get_a5_moe_group_size(vllm_config),
                is_secondary_moe_topology=secondary_topology,
            )
            if method != expected:
                logger.info(
                    "PD node keeps DP metadata synchronization: topology=%s, tokens=%d, communication=%s",
                    topology,
                    tokens,
                    method.name,
                )
                return None

    target_policy = DPMoEPolicy(
        MoECommType.MC2 if is_decode else MoECommType.FUSED_MC2,
        target_tokens if is_decode else get_a5_mega_moe_buffer_tokens_per_rank(vllm_config),
        target_topology,
        runtime_only=is_decode,
    )
    draft_policy = DPMoEPolicy(MoECommType.MC2, draft_tokens, draft_topology, runtime_only=is_decode)
    return target_policy, draft_policy


def configure_dp_moe_policies(
    vllm_config: "VllmConfig", target_model, drafter
) -> tuple[DPMoEPolicy, DPMoEPolicy] | None:
    """Agree once at startup; execution passes each policy through its context."""
    # Worker-only imports; MoE imports the forward context and this module.
    import torch.distributed as dist
    from vllm.distributed import get_dp_group
    from vllm.logger import logger

    policies = build_dp_moe_policies(vllm_config, target_model, drafter)
    signature = (
        None
        if policies is None
        else tuple(
            (policy.comm_type.name, policy.max_tokens_per_rank, policy.topology, policy.runtime_only)
            for policy in policies
        )
    )
    if vllm_config.parallel_config.data_parallel_size > 1:
        signatures = [None] * vllm_config.parallel_config.data_parallel_size
        dist.all_gather_object(signatures, signature, group=get_dp_group().cpu_group)
        if any(other != signature for other in signatures):
            raise RuntimeError(
                "PD communication policies differ across ranks: "
                f"{signatures}. Refusing to execute mismatched communication domains."
            )
    if policies is None:
        return None
    target_policy, draft_policy = policies
    logger.info(
        "PD execution skips DP metadata synchronization: target=%s capacity=%d global_bs=%d, "
        "draft=%s capacity=%d global_bs=%d; runtime_only=%s.",
        target_policy.comm_type.name,
        target_policy.max_tokens_per_rank,
        target_policy.global_bs,
        draft_policy.comm_type.name,
        draft_policy.max_tokens_per_rank,
        draft_policy.global_bs,
        target_policy.runtime_only,
    )
    return target_policy, draft_policy
