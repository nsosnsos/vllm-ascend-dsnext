# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Engram gating in the checkpoint-native residual basis.

    ``hidden`` and ``key`` have shape [tokens, hc_mult, hidden_size].
    """
    dim = hidden.shape[-1]
    hidden_float = hidden.float()
    key = key.float()
    rstd = torch.rsqrt(hidden_float.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    dot = (hidden_float * channel_weight.float() * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, magnitude, -magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden_float + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
