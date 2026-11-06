# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Engram projection and gate."""

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.utils.torch_utils import direct_register_custom_op

from .engram_gate import engram_gate


def engram_begin(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    local_positions: torch.Tensor,
    lookback_ids: torch.Tensor,
    lookback_dead: torch.Tensor,
    dead_mask: torch.Tensor,
    layer_name: str,
    num_layers: int,
    hash_columns: int,
    single_token_per_request: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = get_forward_context()
    if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        raise RuntimeError(
            "Engram multi-stream requires eager or FULL graph execution. "
            "Use FULL_DECODE_ONLY; fetch handles cannot cross independent PIECEWISE replays."
        )
    engram = context.no_compile_layers[layer_name]
    return engram.executor.begin(
        input_ids,
        positions,
        request_ids,
        local_positions,
        lookback_ids,
        lookback_dead,
        dead_mask,
        single_token_per_request=single_token_per_request,
    )


def engram_begin_fake(
    input_ids,
    positions,
    request_ids,
    local_positions,
    lookback_ids,
    lookback_dead,
    dead_mask,
    layer_name,
    num_layers,
    hash_columns,
    single_token_per_request=False,
):
    tokens = input_ids.shape[0]
    return (
        input_ids.new_empty((tokens, num_layers, hash_columns), dtype=torch.int64),
        input_ids.new_empty((tokens,), dtype=torch.bool),
    )


def engram_precompute(
    source: torch.Tensor,
    dependency: torch.Tensor,
    layer_name: str,
    hc_mult: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # source links this stage to engram_begin in FX; dependency keeps it after
    # the chosen backbone layer. Handles exist only while executing/capturing
    # this forward. FULL replay uses the captured native fetch/wait tensors.
    # Neither argument adds a backbone-completion barrier on the side stream.
    engram = get_forward_context().no_compile_layers[layer_name]
    return engram.executor.project(engram, source)


def engram_precompute_fake(source, dependency, layer_name, hc_mult, hidden_size):
    # V is a view into packed K/V storage; its stride is part of the op contract.
    kv = source.new_empty((source.shape[0], (hc_mult + 1) * hidden_size), dtype=torch.bfloat16)
    key, value = kv.split([hc_mult * hidden_size, hidden_size], dim=-1)
    return (
        key.float().view(source.shape[0], hc_mult, hidden_size),
        value,
        source.new_empty((hc_mult, hidden_size), dtype=torch.float32),
    )


def engram_apply_precomputed(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    engram = get_forward_context().no_compile_layers[layer_name]
    return engram.executor.consume(engram, hidden, key, value, weight, mask)


def engram_apply_precomputed_fake(hidden, key, value, weight, mask, layer_name):
    return torch.empty_like(hidden)


# Python orchestration is opaque to Dynamo; its device ops and native stream
# edges are captured together by FULL ACLGraph. These are not eager breaks.
direct_register_custom_op(
    op_name="engram_begin",
    op_func=engram_begin,
    mutates_args=[],
    fake_impl=engram_begin_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="engram_precompute",
    op_func=engram_precompute,
    mutates_args=[],
    fake_impl=engram_precompute_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="engram_apply_precomputed",
    op_func=engram_apply_precomputed,
    mutates_args=[],
    fake_impl=engram_apply_precomputed_fake,
    dispatch_key="PrivateUse1",
)


class AscendEngram(nn.Module):
    """Checkpoint Engram lookup, Wkv precompute, and hidden-dependent gate."""

    def __init__(self, config, quant_config, prefix, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = tuple(config.engram_layer_ids).index(layer_id)
        self.layer_name = f"{prefix}.engram"
        self.norm_eps = config.rms_norm_eps
        self.executor = None
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.embed = None
        hash_width = (
            (config.engram_max_ngram_size - 1)
            * config.engram_n_heads
            * config.engram_head_dim
        )
        self.wkv = ReplicatedLinear(
            hash_width,
            (config.hc_mult + 1) * config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.engram.wkv",
            return_bias=False,
        )
        self.q_weight = nn.Parameter(
            torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
        )
        self.k_weight = nn.Parameter(
            torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
        )
        logger.debug(
            "FOR-ENGRAM gate module initialized: prefix=%s hash_width=%d "
            "hidden_size=%d hc_mult=%d basis=checkpoint-native",
            prefix,
            hash_width,
            self.hidden_size,
            self.hc_mult,
        )

    def lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if self.embed is None:
            raise RuntimeError("Engram embedding was not initialized")
        return self.embed(hash_ids).flatten(1)

    def apply_lookup(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        active_mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        key, value, weight = self.precompute_kv(embedding)
        return engram_gate(hidden_states, key, value, weight, active_mask, eps)

    def start_prefetch(self, hash_ids: torch.Tensor):
        if self.embed is None:
            raise RuntimeError("Engram embedding was not initialized")
        return self.embed.start_fetch(hash_ids)

    def finish_prefetch(self, handle) -> torch.Tensor:
        return self.embed.finish_fetch(handle).flatten(1)

    def precompute_kv(self, embedding: torch.Tensor):
        kv = self.wkv(embedding)
        key, value = kv.split(
            [self.hc_mult * self.hidden_size, self.hidden_size],
            dim=-1,
        )
        return (
            key.float().view(embedding.shape[0], self.hc_mult, self.hidden_size),
            value,
            self.q_weight.float() * self.k_weight.float(),
        )

    def apply_precomputed(self, hidden_states, key, value, weight, active_mask):
        return engram_gate(hidden_states, key, value, weight, active_mask, self.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        active_mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return self.apply_lookup(
            hidden_states,
            self.lookup(hash_ids),
            active_mask,
            eps,
        )
