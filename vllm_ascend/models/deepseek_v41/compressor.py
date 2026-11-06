# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C1 projection path, C2 fused-compressor pooling with token-aligned bridge,
and fused RMS normalization."""

from typing import Any

import torch
import torch_npu
from torch import nn

from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
from vllm_ascend.core.deepseek_v41 import STATE_RING_ROWS, DeepseekV41CompressorStateSpec
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import maybe_trans_nz


def _compact_to_token_aligned(
    compact: torch.Tensor,
    complete: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Spread compact closed-group rows back onto their closing tokens.

    ``compact`` holds one row per completed group in token order (the
    compressor_v2 output contract, padded at the tail); ``complete[t]``
    marks the closing token of each group. Non-closing rows of ``out`` are
    zeroed and never consumed downstream (their cache slots are -1).
    """
    num_tokens = complete.shape[0]
    if compact.shape[0] == 0 or num_tokens == 0:
        out.zero_()
        return out
    rank = complete.long().cumsum(0) - 1
    rank = rank.clamp_(min=0, max=compact.shape[0] - 1)
    gathered = compact[rank]
    # Masked boolean indexing lowers to dynamic Nonzero and cannot be
    # captured by ACLGraph; broadcast the mask instead.
    out.copy_(
        torch.where(
            complete.unsqueeze(-1),
            gathered,
            torch.zeros_like(gathered),
        ).to(out.dtype)
    )
    return out


def _read(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        try:
            return config[name]
        except KeyError as exc:
            raise ValueError(f"DeepSeek V4.1 config is missing {name!r}") from exc
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ValueError(f"DeepSeek V4.1 config is missing {name!r}") from exc


def text_config_of(config: Any) -> Any:
    if isinstance(config, dict):
        return config.get("text_config", config)
    return getattr(config, "text_config", config)


class DeepseekV41CompressorStateCache(DeepseekV41CacheLayer):
    """State-cache module owning one packed FP32 circular page per request.

    Pass kv_cache[0].squeeze(-2) and the state's block table to the compressor.
    The V4 constructor itself cannot be reused: it asserts ratio in (4, 128).
    """

    def __init__(self, vllm_config, prefix, spec):
        if spec.dtype != torch.float32 or spec.compress_ratio != 1 or spec.block_size != STATE_RING_ROWS:
            raise ValueError(
                f"V4.1 compressor state requires a {STATE_RING_ROWS}-row FP32 ring"
            )
        super().__init__(vllm_config, prefix, spec)
        self.state_dim = spec.head_size
        self.dtype = spec.dtype
        self.compress_ratio = 2  # Pooling ratio; spec storage ratio remains one.
        self.block_size = spec.block_size


class DeepseekV41RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config, ratio, vllm_config=None, prefix="compressor"):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError("V4.1 compressor requires ratio 1 or 2")
        self.ratio = ratio
        self.width = _read(config, "head_dim")
        dim = _read(config, "hidden_size")
        self.wkv = nn.Linear(dim, self.width, bias=False, dtype=torch.float32 if ratio == 2 else torch.bfloat16)
        self.norm = DeepseekV41RMSNorm(self.width, _read(config, "rms_norm_eps"))
        if ratio == 2:
            self.wgate = nn.Linear(dim, self.width, bias=False, dtype=torch.float32)
            # Non-A5 paths align compact rows back to token rows. Allocate that
            # persistent output before memory profiling, so its footprint is
            # included in the cache budget rather than added after allocation.
            if vllm_config is not None and get_ascend_device_type() != AscendDeviceType.A5:
                capacity = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
                self.register_buffer(
                    "_ring_pooled",
                    torch.empty(capacity, self.width, dtype=torch.bfloat16, device=self.wkv.weight.device),
                    persistent=False,
                )
            # Standalone unfused-reference tests may supply pages explicitly.
            if vllm_config is not None:
                self.state_cache = DeepseekV41CompressorStateCache(
                    vllm_config,
                    f"{prefix}.state_cache",
                    DeepseekV41CompressorStateSpec(
                        block_size=STATE_RING_ROWS,
                        num_kv_heads=1,
                        head_size=2 * self.width,
                        dtype=torch.float32,
                    ),
                )

    def process_weights_after_loading(self):
        if self.ratio == 1:
            self.wkv.weight.data = maybe_trans_nz(self.wkv.weight.data)
        elif self.ratio == 2:
            self.wkv.weight.data = self.wkv.weight.data.to(torch.bfloat16)
            self.wgate.weight.data = self.wgate.weight.data.to(torch.bfloat16)

    def prepare_ring_compressor(self, max_tokens, device):
        """Check the profiled per-source output buffer before capture."""
        if get_ascend_device_type() == AscendDeviceType.A5:
            return
        actual_device = self._ring_pooled.device
        compatible_device = actual_device.type == device.type and (
            device.index is None or actual_device.index == device.index
        )
        if self._ring_pooled.shape[0] < max_tokens or not compatible_device:
            raise ValueError("Ring output capacity/device must be established before memory profiling")

    def pool_projected(self, hidden_states, metadata):
        """Fused compressor_v2 path: project + gated-pool inside the op.

        The op (cann_ops_transformer.ops.ds41.compressor) takes the raw
        hidden_states plus the wkv/wgate weights, pools closed groups with a
        per-channel softmax gate over the FP32 ring state, and returns
        compact closed-group rows; norm/RoPE/scatter stay outside per the
        op contract.
        """
        try:
            from cann_ops_transformer.ops.ds41 import compressor as compressor_v2
        except ImportError as exc:
            raise RuntimeError(
                "DeepSeek V4.1 C2 compression requires cann_ops_transformer "
                "(CANN 9.2) with the ds41 compressor op."
            ) from exc
        use_compact_output = get_ascend_device_type() == AscendDeviceType.A5
        if not use_compact_output:
            if not hasattr(self, "_ring_pooled"):
                raise RuntimeError("Ring compressor must be initialized before graph capture")
            if hidden_states.shape[0] > self._ring_pooled.shape[0]:
                raise ValueError("Compressor batch exceeds its prepared output capacity")
        ring_meta = metadata.c2_ring_metadata
        compact = compressor_v2(
            hidden_states,
            self.wkv.weight,
            self.wgate.weight,
            self.state_cache.kv_cache[0].squeeze(-2),
            state_block_table=ring_meta[4],  # ring page id per request
            cu_seqlens=metadata.query_start_loc.int(),
            seqused=ring_meta[1],  # per-request valid tokens in this chunk
            start_pos=ring_meta[0],  # chunk-first-token absolute position
            cmp_ratio=2,
        )
        if use_compact_output:
            latent = compact
        else:
            num_tokens = hidden_states.shape[0]
            latent = _compact_to_token_aligned(
                compact,
                metadata.c2_complete_mask[:num_tokens],
                self._ring_pooled[:num_tokens],
            )
        latent = self.norm(latent)
        return latent

    def forward(self, x):
        """Project an uncompressed source; ratio-2 uses ``pool_projected``."""
        if x.ndim != 2:
            raise ValueError("Expected [tokens, hidden] input")
        if self.ratio != 1:
            raise RuntimeError("Ratio-2 compression must use pool_projected")
        return self.norm(self.wkv(x))
