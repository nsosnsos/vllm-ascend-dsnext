# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend import envs
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    scatter_cache_v2,
)
from vllm_ascend.core.deepseek_v41 import DeepseekV41IndexerSpec
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)

from .compressor import DeepseekV41RMSNorm, _read
from cann_ops_transformer.ops.ds41 import indexer_prologue_k
from cann_ops_transformer.ops.ds41 import quant_lightning_indexer
from cann_ops_transformer.ops.ds41 import quant_sparse_lightning_indexer
from vllm_ascend.core.deepseek_v41 import DeepseekV41IndexerFoldedSpec

# MXFP4 layout of the indexer query and key operands: two 4-bit values per
# byte, one E8M0 scale per 32 values, and scales travelling in pairs.
MXFP4_VALUES_PER_BYTE = 2
MXFP4_SCALE_PAIR_SIZE = 64
MXFP4_SCALE_PAIR_GROUPS = 2

# ``indexer_prologue_qw`` consumes its MXFP8 operands as paired E8M0 scales:
# one scale per 32 values, with the groups packed two at a time.
INDEXER_PROLOGUE_SCALE_PAIR_SIZE = 64
INDEXER_PROLOGUE_SCALE_PAIR_GROUPS = 2


class DeepseekV41Indexer(nn.Module):
    """Small side attention that selects compressed KV positions.

    All index heads are replicated on each TP rank for the correctness path,
    so every rank produces identical sparse indices without an all-reduce.
    """

    # FP32 copy of the K RMSNorm weight for ``indexer_prologue_k``; stays
    # ``None`` until ``process_weights_after_loading`` runs.
    _prologue_norm_weight: torch.Tensor | None = None

    def __init__(
        self,
        config,
        owns_k,
        vllm_config,
        prefix,
        compress_ratio,
        quant_config=None,
        is_candidate_source=False,
    ):
        super().__init__()
        self.owns_k = owns_k
        self.compress_ratio = compress_ratio
        self.k_cache_folded = None
        self.dim = int(_read(config, "hidden_size"))
        self.q_lora_rank = int(_read(config, "q_lora_rank"))
        # Populated by ``process_weights_after_loading`` when the fused
        # ``indexer_prologue_qw`` prologue is enabled.
        self._prologue_qw_fused = False
        self._prologue_qw_wqb: torch.Tensor | None = None
        self._prologue_qw_ww: torch.Tensor | None = None
        self._prologue_qw_wqb_scale: torch.Tensor | None = None
        self.n_heads = int(_read(config, "index_n_heads"))
        self.width = int(_read(config, "index_head_dim"))
        self.rope_width = int(_read(config, "qk_rope_head_dim"))
        self.index_topk = int(_read(config, "index_topk"))
        self.candidate_block_size = int(_read(config, "candidate_block_size"))
        self.softmax_scale = self.width**-0.5
        self.weights_scale = self.softmax_scale * self.n_heads**-0.5
        self.wq_b = ReplicatedLinear(
            _read(config, "q_lora_rank"),
            self.n_heads * self.width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.weights_proj = ReplicatedLinear(
            _read(config, "hidden_size"),
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        if owns_k:
            self.wk = nn.Linear(
                _read(config, "head_dim"),
                self.width,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = DeepseekV41RMSNorm(self.width, _read(config, "rms_norm_eps"))
            use_a5_quantized_cache = get_ascend_device_type() == AscendDeviceType.A5
            self.use_a5_quantized_cache = use_a5_quantized_cache
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                DeepseekV41IndexerSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.width // 2 if use_a5_quantized_cache else self.width,
                    dtype=torch.uint8 if use_a5_quantized_cache else torch.int8,
                    compress_ratio=compress_ratio,
                    scale_dim=self.width // 32 if use_a5_quantized_cache else 1,
                    scale_dtype=(torch.uint8 if use_a5_quantized_cache else torch.float16),
                ),
            )
            if use_a5_quantized_cache and is_candidate_source:
                groups = self.candidate_block_size
                self.k_cache_folded = DeepseekV41CacheLayer(
                    vllm_config,
                    f"{prefix}.k_cache_folded",
                    DeepseekV41IndexerFoldedSpec(
                        block_size=vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=groups * (self.width // 2 + self.width // 32),
                        dtype=torch.uint8,
                        compress_ratio=groups,
                    ),
                )
            else:
                self.k_cache_folded = None

    @staticmethod
    def _output(linear, value):
        output = linear(value)
        return output[0] if isinstance(output, tuple) else output

    def process_weights_after_loading(self):
        if self.owns_k:
            weight = maybe_trans_nz(self.wk.weight.data)
            if (
                self.use_a5_quantized_cache
                and torch_npu.get_npu_format(weight) != ACL_FORMAT_FRACTAL_NZ
            ):
                raise RuntimeError(
                    "V4.1 indexer_prologue_k requires an Nz wk weight; start "
                    "with --additional-config '{\"weight_nz_mode\": 2}' (A5 "
                    "defaults to mode 1, which leaves BF16 weights in ND)"
                )
            self.wk.weight.data = weight
            self._prologue_norm_weight = self.k_norm.weight.detach().to(torch.float32)
        if envs.VLLM_ASCEND_ENABLE_INDEXER_PROLOGUE_QW:
            self._prepare_prologue_qw_operands()

    def _prepare_prologue_qw_operands(self) -> None:
        """Stage the fused ``indexer_prologue_qw`` operands once after loading.

        ``wq_b`` is a DS block-wise MXFP8 linear: when the fused prologue is
        enabled the scheme leaves it output-major (``[N, K]``) and Nz, which is
        exactly what the kernel DMA expects, so the weight is used as-is and
        only re-viewed as bytes. ``weights_proj`` is an unquantized bf16 linear,
        so the Ascend unquantized method already left it ``[N, dim]`` and Nz;
        both are consumed as-is rather than converted a second time.
        """
        wqb = self.wq_b.weight.data
        ww = self.weights_proj.weight.data
        if torch_npu.get_npu_format(wqb) != ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError(
                "indexer_prologue_qw requires an Nz indexer.wq_b weight; start "
                "with --additional-config '{\"weight_nz_mode\": 2}'"
            )
        if torch_npu.get_npu_format(ww) != ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError(
                "indexer_prologue_qw requires an Nz indexer.weights_proj weight; "
                "start with --additional-config '{\"weight_nz_mode\": 2}'"
            )
        self._prologue_qw_wqb = wqb.view(torch.uint8)
        scale_pair_dim = -(-self.q_lora_rank // INDEXER_PROLOGUE_SCALE_PAIR_SIZE)
        self._prologue_qw_wqb_scale = self.wq_b.weight_scale.data.contiguous()
        if self._prologue_qw_wqb_scale.shape != (
            self.n_heads * self.width,
            scale_pair_dim,
            INDEXER_PROLOGUE_SCALE_PAIR_GROUPS,
        ):
            raise RuntimeError(
                "indexer_prologue_qw expects wq_b scales in the output-major "
                f"paired layout {(self.n_heads * self.width, scale_pair_dim, INDEXER_PROLOGUE_SCALE_PAIR_GROUPS)}, "
                f"got {tuple(self._prologue_qw_wqb_scale.shape)}"
            )
        self._prologue_qw_ww = ww
        self._prologue_qw_fused = True

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        if not self.owns_k or latent.shape[0] == 0:
            return
        k_cache, scale_cache = self.k_cache.kv_cache[0]
        if slots.ndim == 1:
            # Builder-prepared flat slots; identical to converting the [T, 2]
            # mapping per call, computed once per cache group per step.
            slot_mapping = slots
        elif slots.ndim == 2:
            slot_mapping = (slots[:, 0]) * k_cache.shape[1] + slots[:, 1]
            slot_mapping = slot_mapping.clamp(min=-1).to(torch.int32)
        else:
            raise ValueError(
                f"indexer update_keys slot mapping must be [T, 2] or [T], got ndim={slots.ndim}"
            )

        if self.use_a5_quantized_cache:
            norm_weight = self._prologue_norm_weight
            if norm_weight is None:
                norm_weight = self.k_norm.weight.float()
            prologue_common = dict(
                latent=latent,
                wk=self.wk.weight.detach(),
                norm_weight=norm_weight,
                rope_sin=sin.reshape(-1, self.rope_width),
                rope_cos=cos.reshape(-1, self.rope_width),
                norm_eps=self.k_norm.eps,
            )
            cache_index = slot_mapping.to(torch.int64)
            indexer_prologue_k(
                k_cache=k_cache,
                k_scale_cache=scale_cache,
                cache_index=cache_index,
                storage_mode=0,
                combined_block_size=-1,
                **prologue_common,
            )
            if self.k_cache_folded is not None:
                indexer_prologue_k(
                    k_cache=self.k_cache_folded.kv_cache[0],
                    k_scale_cache=None,
                    cache_index=cache_index,
                    storage_mode=1,
                    combined_block_size=self.candidate_block_size,
                    **prologue_common,
                )
        else:
            # Non-quantized cache planes keep the eager chain: the fused
            # prologue ABI only covers mxfp4 uint8 layouts.
            key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                key.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.width - self.rope_width, self.width],
            )
            key = key.squeeze(1)
            torch.ops.cann_ops_transformer.indexer_quant_cache(
                cache=k_cache,
                cache_scale=scale_cache,
                x=key,
                slot_mapping=slot_mapping,
                quant_mode="mxfp4",
            )

    def select(
        self,
        hidden_states,
        qr,
        positions,
        cos,
        sin,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidate_indices,
        candidate_lengths,
        source_cache_folded=None,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        if hidden_states.shape[0] == 0:
            return self._empty_selection(
                self.index_topk,
                (0, 1, candidate_topk_blocks),
                is_candidate_source,
                hidden_states.device,
                candidate_indices,
                candidate_lengths,
            )
        quantized_query, query_scale, weights = self._prologue_qw(
            hidden_states, qr, cos, sin
        )
        return self.select_quantized(
            quantized_query,
            query_scale,
            weights,
            positions,
            source_cache,
            source_metadata,
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths,
            source_cache_folded=source_cache_folded,
        )

    def _prologue_qw(self, hidden_states, qr, cos, sin):
        """Project the index query and its head weights, then quantize the query.

        Returns the packed MXFP4 query, its E8M0 scale pairs and the
        ``[T, n_heads]`` head-aggregation weights the lightning indexer
        consumes.
        """
        if self._prologue_qw_fused:
            return self._prologue_qw_fused_op(hidden_states, qr, cos, sin)
        weights = self._output(self.weights_proj, hidden_states) * self.weights_scale
        query = self._output(self.wq_b, qr).unflatten(-1, (self.n_heads, self.width))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        quantized_query, query_scale = torch_npu.npu_dynamic_mx_quant(
            query, dst_type=torch_npu.float4_e2m1fn_x2
        )
        return quantized_query.view(torch.uint8), query_scale, weights

    def _prologue_qw_fused_op(self, hidden_states, qr, cos, sin):
        """Fused ``indexer_prologue_qw`` prologue.

        One operator call replaces the whole eager chain: the MXFP8 ``wq_b``
        projection, the partial RoPE on Q's tail and its MXFP4 quantization,
        plus the ``weights_proj`` GEMM already scaled by ``weights_scale``.
        """
        from cann_ops_transformer.ops.ds41 import indexer_prologue_qw

        qr_quant, qr_scale = torch_npu.npu_dynamic_mx_quant(
            qr.view(-1, self.q_lora_rank),
            dst_type=torch.float8_e4m3fn,
        )
        qr_quant = qr_quant.view(torch.uint8)
        scale_pair_dim = -(-self.q_lora_rank // INDEXER_PROLOGUE_SCALE_PAIR_SIZE)
        qr_scale = qr_scale.view(torch.uint8).reshape(
            qr_quant.shape[0],
            scale_pair_dim,
            INDEXER_PROLOGUE_SCALE_PAIR_GROUPS,
        )
        query, query_scale, weights = indexer_prologue_qw(
            hidden_states.view(-1, self.dim),
            qr_quant,
            self._prologue_qw_wqb,
            self._prologue_qw_ww,
            qr_scale,
            self._prologue_qw_wqb_scale,
            sin.view(-1, self.rope_width).float(),
            cos.view(-1, self.rope_width).float(),
            softmax_scale=self.weights_scale,
        )
        return query, query_scale, weights

    @staticmethod
    def _validate_selection(
        num_tokens,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_indices,
        candidate_lengths,
        source_cache_folded,
    ):
        if is_candidate_source and uses_candidate_filter:
            raise ValueError("A candidate source must use the unfiltered position TopK")
        if uses_candidate_filter and (candidate_indices is None or candidate_lengths is None):
            raise RuntimeError("V4.1 candidate-filtering indexer ran before its source")
        if uses_candidate_filter and source_cache_folded is None:
            raise RuntimeError("candidate indexer is missing the folded twin")
        candidate_shape = (num_tokens, 1, candidate_topk_blocks)
        if uses_candidate_filter and (candidate_indices.shape != candidate_shape or candidate_indices.dtype != torch.int32):
            raise ValueError("Candidate consumer requires INT32 block IDs with matching query rows")
        return candidate_shape

    @staticmethod
    def _empty_selection(
        topk,
        candidate_shape,
        is_candidate_source,
        device,
        candidate_indices,
        candidate_lengths,
    ):
        selected = torch.full(
            (0, topk), -1, dtype=torch.int32, device=device
        )
        if is_candidate_source:
            candidate_indices = torch.full(
                candidate_shape, -1, dtype=torch.int32, device=device
            )
            candidate_lengths = torch.zeros(
                (0, 1), dtype=torch.int32, device=device
            )
        return selected, candidate_indices, candidate_lengths

    def select_quantized(
        self,
        quantized_query,
        query_scale,
        weights,
        positions,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidate_indices,
        candidate_lengths,
        source_cache_folded=None,
    ):
        """Run the lightning indexer on an already-quantized query.

        ``quantized_query``/``query_scale`` are the MXFP4 pair produced by
        ``_prologue_qw``; both are canonicalized here into the packed layouts
        the indexer operators consume. ``weights`` is the ``[T, n_heads]``
        head-aggregation input.
        """
        candidate_shape = self._validate_selection(
            quantized_query.shape[0],
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths,
            source_cache_folded=source_cache_folded,
        )
        topk = self.index_topk
        if quantized_query.shape[0] == 0:
            return self._empty_selection(
                topk,
                candidate_shape,
                is_candidate_source,
                quantized_query.device,
                candidate_indices,
                candidate_lengths,
            )

        quantized_query = quantized_query.reshape(
            -1, self.n_heads, self.width // MXFP4_VALUES_PER_BYTE
        )
        query_scale = query_scale.contiguous().view(torch.uint8).reshape(
            quantized_query.shape[0],
            self.n_heads,
            self.width // MXFP4_SCALE_PAIR_SIZE,
            MXFP4_SCALE_PAIR_GROUPS,
        )
        weights = weights.reshape(-1, self.n_heads).float()

        key, key_scale = source_cache
        key_scale = key_scale.view(*key_scale.shape[:-1], 2, 2)

        common = dict(
            cu_seqlens_q=source_metadata.query_start_loc,
            seqused_q=None,
            seqused_k=source_metadata.cache_seq_lens,
            block_table=source_metadata.block_table,
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=False,
        )
        if uses_candidate_filter:
            qsli_metadata = source_metadata.qsli_metadata
            if qsli_metadata is None:
                raise RuntimeError(
                    "V4.1 candidate-filtering indexer has no builder QSLI metadata"
                )
            wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(qsli_metadata))
            selected, _ = quant_sparse_lightning_indexer(
                q=quantized_query,
                k=source_cache_folded.squeeze(2),
                w=weights,
                descale_q=query_scale,
                candidate_block_indices=candidate_indices,
                candidate_block_length=candidate_lengths,
                topk=topk,
                quant_mode=1,  # mxfp4
                candidate_block_size=candidate_block_size,
                metadata=qsli_metadata,
                # QSLI is the candidate consumer: cmp_ratio is 1 here, so the
                # operator contract requires cmp_residual_k=None.
                cmp_residual_k=None,
                **common,
            )
        else:
            qli_metadata = source_metadata.qli_metadata
            if qli_metadata is not None:
                wait_for_device_metadata(
                    DeviceMetadataStage.INDEXER, id(qli_metadata)
                )
            selected, _, cand_indices, cand_lengths = quant_lightning_indexer(
                q=quantized_query,
                k=key,
                w=weights,
                descale_q=query_scale,
                descale_k=key_scale,
                topk=topk,
                quant_mode=1,  # mxfp4
                candidate_topk_blocks=(
                    candidate_topk_blocks if is_candidate_source else -1
                ),
                candidate_block_size=(
                    candidate_block_size if is_candidate_source else -1
                ),
                metadata=qli_metadata,
                # Only the compressed (ratio != 1) plane carries a remainder.
                cmp_residual_k=(
                    None if self.compress_ratio == 1 else source_metadata.cmp_residual
                ),
                **common,
            )
            if is_candidate_source:
                candidate_indices = cand_indices
                candidate_lengths = cand_lengths

        selected = prepare_indexer_indices(selected.squeeze(1), positions, self.compress_ratio)
        return selected, candidate_indices, candidate_lengths
