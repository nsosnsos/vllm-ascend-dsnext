# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enable_pcp
from vllm_ascend.core.private_circle_pool import is_private_circle_kv_cache_spec
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer, _compute_num_programs
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's dedicated ``dspark`` method in user config. Its execution
    shape is closer to DFlash: target hidden states prepopulate draft K/V, then
    one anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        hf_config = self.draft_model_config.hf_config
        hf_config = getattr(hf_config, "text_config", hf_config)
        self.sample_from_anchor = getattr(hf_config, "sample_from_anchor", True)
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_tokens
        else:
            self.num_query_per_req = 1 + self.num_speculative_tokens

        dspark_noise_token_id = getattr(hf_config, "dspark_noise_token_id", None)
        if isinstance(dspark_noise_token_id, int) and dspark_noise_token_id >= 0:
            self.parallel_drafting_token_id = dspark_noise_token_id

        blk = 1 + self.num_speculative_tokens
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # Replace the target-sized DFlash buffers with the draft model's hidden
        # size. Assignment releases the old tensors without an explicit del.
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        dynamic_spec_config = get_ascend_config().dynamic_spec_config
        self.dynamic_spec = None

        if dynamic_spec_config.method == "dspark":
            self.dynamic_spec = DynamicSpecScheduler(
                method="dspark",
                method_params=dynamic_spec_config.method_params,
                max_batch_size=self.max_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
                device=device,
            )
        # DSpark draft ACL graphs support the deterministic greedy path only:
        # probabilistic sampling carries replay-frozen RNG and host sampling
        # flags, and the dynamic verify-length scheduler syncs to host inside
        # the draft loop. Fall back to eager drafting for those configs.
        if self.use_cuda_graph:
            unsafe_reason = None
            if self.speculative_config.draft_sample_method == "probabilistic":
                unsafe_reason = "draft_sample_method='probabilistic'"
            elif self.dynamic_spec is not None:
                unsafe_reason = "dynamic spec decoding"
            if unsafe_reason is not None:
                logger.warning_once(
                    "%s is not supported by DSpark draft ACL graphs; falling back to eager drafting.",
                    unsafe_reason,
                )
                self.use_cuda_graph = False
        # Max query tokens depend on whether sampling from anchor or not.
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
        # Lazily built per-step reuse buffers (see _draft_query_start_loc and
        # set_inputs_first_pass); the tables are built on first use.
        self._dspark_qsl_table_k: int | None = None
        self._dspark_qsl_gpu_table: torch.Tensor | None = None
        self._dspark_qsl_cpu_table: torch.Tensor | None = None
        self._dspark_token_indices_buf: torch.Tensor | None = None
        self._dspark_seq_lens_cpu_buf: torch.Tensor | None = None
        # Position ids for the draft query block [max_query_tokens].
        # Overrides dflash:49; v2 uses input_buffers.positions.
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Primary-group query slot mapping buffer [max_query_tokens].
        # Overrides dflash:37; v2 uses BlockTables.slot_mappings. Per-non-
        # primary-gid buffers live in _per_group_query_slot_mapping_buffers.
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        # The v1 runner owns block tables and slot mappings. Keep per-group
        # references here because K3 draft layers can span multiple cache
        # groups with different logical block sizes.
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # Per-gid logical block size used to expand slot mappings. The KV
        # manager's physical page can be larger when hybrid cache groups share
        # one allocation, so kv_cache_spec.block_size is not interchangeable
        # with the attention kernel's block size.
        self._per_group_kernel_block_sizes: dict[int, int] = {}

        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None
        # Draft groups whose SWA plane joined the request-private ring
        # (pool enabled): gid -> ring window size for the context clamp.
        self._private_ring_window: dict[int, int] = {}

    def _compute_confidence(
        self,
        last_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        num_tokens = num_reqs * self.num_speculative_tokens
        flat_hidden = last_hidden_states.reshape(num_tokens, last_hidden_states.shape[-1])
        # Markov embeddings of the draft input tokens (cheap lookup, so they
        # are recomputed here instead of being captured in the drafting loop).
        markov_embs = self.model.markov_embed(draft_token_ids[:, : self.num_speculative_tokens])
        # The confidence head concatenates both inputs, so their dtypes must
        # match; it upcasts to float32 internally.
        flat_markov = markov_embs.reshape(num_tokens, markov_embs.shape[-1]).to(flat_hidden.dtype)
        conf_raw = self.model.compute_confidence(flat_hidden, flat_markov)
        confidence = self._dspark_confidence_logits_buffer[:num_reqs]
        confidence.copy_(conf_raw.reshape(num_reqs, self.num_speculative_tokens))
        return confidence

    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))
        self._per_group_kernel_block_sizes = {}
        self.draft_attn_groups: list[AttentionGroup] = []

        for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            draft_layer_names_in_group = set(kv_cache_group_spec.layer_names) & self._draft_attn_layer_names
            if not draft_layer_names_in_group:
                continue

            attention_groups: dict[tuple[str, Any], AttentionGroup] = {}
            # iterate in a way like vllm's llm_base_proposer
            for layer_name in draft_layer_names_in_group:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                if is_private_circle_kv_cache_spec(layer_kv_cache_spec):
                    self._private_ring_window[kv_cache_gid] = int(
                        layer_kv_cache_spec.sliding_window
                    )
                key = (attn_backend.full_cls_name(), layer_kv_cache_spec)

                if key not in attention_groups:
                    kernel_block_size = int(
                        kernel_block_sizes[kv_cache_gid]
                        if kernel_block_sizes is not None and kv_cache_gid < len(kernel_block_sizes)
                        else layer_kv_cache_spec.block_size
                    )
                    attn_group = AttentionGroup(
                        attn_backend,
                        [layer_name],
                        layer_kv_cache_spec,
                        kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    self._per_group_kernel_block_sizes[kv_cache_gid] = kernel_block_size
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

            self.draft_attn_groups.extend(attention_groups.values())

        if (
            getattr(self.runner, "device_metadata_executor", None) is not None
            and self.dcp_size == 1
            and not enable_pcp()
        ):
            for attn_group in self.draft_attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, AscendDSAMetadataBuilder):
                    builder.enable_dspark_device_metadata(self.max_query_tokens)

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]

        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names
            if ln in self.attn_layer_names
        }
        self._layer_group_idx = [name_to_gid[name] for name in self.attn_layer_names]

        # some buffers need information of groups
        self._per_group_query_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_query_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        self._per_group_context_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping
        self._ensure_padded_block_table(gid, block_table)

    def _ensure_padded_block_table(
        self,
        gid: int,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        max_rows = max(
            int(self.max_batch_size),
            int(getattr(self.runner, "max_num_reqs", 0)),
            int(block_table.shape[0]),
        )
        buffer = self._per_group_block_table_buffers.get(gid)
        if (
            buffer is None
            or buffer.shape[0] < max_rows
            or buffer.shape[1] != block_table.shape[1]
            or buffer.dtype != block_table.dtype
            or buffer.device != block_table.device
        ):
            buffer = torch.zeros(
                (max_rows, block_table.shape[1]),
                dtype=block_table.dtype,
                device=block_table.device,
            )
            self._per_group_block_table_buffers[gid] = buffer
        return buffer

    def _refresh_padded_block_table(
        self,
        gid: int,
        block_table: torch.Tensor,
    ) -> None:
        """Refresh a persistent, zero-padded per-group block table buffer."""
        buffer = self._ensure_padded_block_table(gid, block_table)
        rows_to_copy = min(block_table.shape[0], buffer.shape[0])
        buffer[:rows_to_copy].copy_(block_table[:rows_to_copy])
        if buffer.shape[0] > rows_to_copy:
            buffer[rows_to_copy:].zero_()

    def _clamp_private_context_slots_to_tail_window(
        self, cad: CommonAttentionMetadata, batch_size: int
    ) -> None:
        """Keep only each request's tail-window context writes on the ring.

        The draft context precompute would otherwise scatter the whole
        context span into the ring; a span longer than the ring capacity
        wraps onto itself and the surviving rows would depend on scatter
        ordering. Skipping the rows below each request's tail window mirrors
        the target model's "commit only the trailing window" discipline;
        the skipped rows are outside the protocol-A table's exposed range
        and are never read.
        """
        qsl = cad.query_start_loc[: batch_size + 1]
        num_context = self._dflash_num_context
        spans = qsl[1:] - qsl[:-1]
        keeps = spans.clamp_max(max(self._private_ring_window.values()))
        boundary = torch.repeat_interleave(
            qsl[1:] - keeps, spans, output_size=num_context
        )
        head_rows = torch.arange(num_context, device=qsl.device) < boundary
        for gid in self._private_ring_window:
            buffer = self._per_group_context_slot_mapping_buffers.get(gid)
            if buffer is not None:
                buffer[:num_context].masked_fill_(head_rows, -1)

    def _draft_query_start_loc(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(device, cpu) ``arange[:batch_size + 1] * num_query_per_req`` views.

        The tables equal the retired per-step recomputations bit for bit and
        are rebuilt only when the per-request query budget changes (dynamic
        spec decoding). The graph-bucket dispatch rebinds both fields to the
        proposer's persistent buffers, so no consumer writes them in place.
        """
        k = self.num_query_per_req
        if self._dspark_qsl_table_k != k or self._dspark_qsl_gpu_table is None:
            self._dspark_qsl_gpu_table = self.arange_dflash * k
            self._dspark_qsl_cpu_table = torch.from_numpy(self.token_arange_np).clone() * k
            self._dspark_qsl_table_k = k
        return self._dspark_qsl_gpu_table[: batch_size + 1], self._dspark_qsl_cpu_table[: batch_size + 1]

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            self._refresh_padded_block_table(
                gid,
                self._per_group_block_tables[gid],
            )
        self._context_slot_mapping_buffers = None
        # Tail-window prefill step: the context rows are the model's gathered
        # tail rows. The draft is pure-SWA on the private ring and its first
        # pass stores context KV only inside each request's tail window (the
        # full-path clamp discards the head rows), so the gathered rows carry
        # every feature this pass consumes; only the row layout changes. The
        # target model has already gathered the aux captures to this layout.
        get_step_plan = getattr(self.runner, "get_step_private_circle_plan", None)
        step_plan = get_step_plan() if get_step_plan is not None else None
        plan_has_tail = step_plan is not None and "tail_row_idx" in step_plan
        tail_plan = step_plan if plan_has_tail else None
        if tail_plan is not None:
            tail_idx = tail_plan["tail_row_idx"]
            num_context = int(tail_idx.numel())
            target_positions = target_positions.index_select(0, tail_idx)
            draft_gids = {g.kv_cache_group_id for g in self.draft_attn_groups}
            context_slot_mappings = {
                gid: self._per_group_slot_mappings[gid].index_select(0, tail_idx)
                for gid in draft_gids
            }
            # Query start locations over the tail rows, padded to the
            # (possibly DP-padded) request count: padded rows produce
            # phantom query rows that are discarded downstream.
            kernel_qsl = tail_plan["tail_query_start_loc_full"]
            if kernel_qsl.shape[0] < batch_size + 1:
                kernel_qsl = torch.cat(
                    [
                        kernel_qsl,
                        kernel_qsl.new_full(
                            (batch_size + 1 - kernel_qsl.shape[0],), num_context
                        ),
                    ]
                )
            logger.debug(
                "DSpark first pass follows tail-window layout: %d -> %d context rows",
                int(cad.query_start_loc_cpu[batch_size]),
                num_context,
            )
        else:
            num_context = int(cad.query_start_loc_cpu[batch_size])
            context_slot_mappings = self._per_group_slot_mappings
            kernel_qsl = cad.query_start_loc
        if num_context != target_hidden_states.shape[0]:
            raise RuntimeError(
                "DSpark context rows disagree with the target hidden states: "
                f"num_context={num_context}, hidden_rows="
                f"{target_hidden_states.shape[0]}; the tail-window plan was "
                "not visible to the drafter."
            )
        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states[:num_context]

        # Persistent buffer: the inputs kernel overwrites all of
        # [0, num_sample_total) each launch, so only growth allocates.
        if self._dspark_token_indices_buf is None or self._dspark_token_indices_buf.shape[0] < num_sample_total:
            self._dspark_token_indices_buf = torch.empty(num_sample_total, dtype=torch.int32, device=self.device)
        token_indices_to_sample = self._dspark_token_indices_buf[:num_sample_total]

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices.
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers[gid]
            kernel_block_size = self._per_group_kernel_block_sizes[gid]
            copy_and_expand_dflash_and_dspark_inputs_kernel[
                (_compute_num_programs(self._dflash_num_context, num_query_total),)
            ](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=context_slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=kernel_qsl,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kernel_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
            )
        # to compute self._context_slot_mapping_buffers from dict to list
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
        ]
        # Clamp ring context writes to each request's tail window before the
        # cad is rewritten to the draft's query layout below. Skipped on
        # tail-window steps: every context row is already a window row, and
        # the clamp's spans come from the full-batch query_start_loc.
        if self._private_ring_window and self._dflash_num_context > 0 and tail_plan is None:
            self._clamp_private_context_slots_to_tail_window(cad, batch_size)

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc, cad.query_start_loc_cpu = self._draft_query_start_loc(batch_size)
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        # The model runner has already corrected this canonical host mirror
        # with the accepted-token count. Extend it on CPU alongside the device
        # lengths, without another reject D2H copy or attention-side wait.
        if cad._seq_lens_cpu is not None:
            # copy_ + add_ over a persistent buffer equals the retired
            # clone + add_ bit for bit; the padding tail keeps the source
            # values, following the seq_lens_group aliasing discipline.
            seq_width = cad._seq_lens_cpu.shape[0]
            seq_buf = self._dspark_seq_lens_cpu_buf
            if (
                seq_buf is None
                or seq_buf.shape[0] < seq_width
                or seq_buf.dtype != cad._seq_lens_cpu.dtype
                or seq_buf.device != cad._seq_lens_cpu.device
            ):
                seq_buf = torch.empty_like(cad._seq_lens_cpu)
                self._dspark_seq_lens_cpu_buf = seq_buf
            draft_seq_lens_cpu = seq_buf[:seq_width]
            draft_seq_lens_cpu.copy_(cad._seq_lens_cpu)
            draft_seq_lens_cpu[:batch_size].add_(self.num_query_per_req)
            cad._seq_lens_cpu = draft_seq_lens_cpu
            if getattr(cad, "seq_lens_cpu", None) is not None:
                cad.seq_lens_cpu = draft_seq_lens_cpu

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = self.num_query_per_req
        cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions  # this would be sliced in attention backend
        if hasattr(self.model, "get_draft_attn_causal"):
            # Currently, attention causality across draft layers are uniform.
            cad.causal = self.model.get_draft_attn_causal()[0]
        else:
            cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        use_runtime_dp_policy: bool = True,
        **kwargs,
    ) -> None:
        num_query_total = num_reqs * self.num_query_per_req
        num_query_tokens = min(num_query_total if num_reqs > 0 else num_tokens, self.max_query_tokens)
        use_runtime_dp_policy = use_runtime_dp_policy and not is_profile

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(
            num_query_tokens,
            is_draft_model=True,
            cudagraph_mode=aclgraph_runtime_mode,
            is_dummy_run=True,
            use_runtime_dp_policy=use_runtime_dp_policy,
        )

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        self.token_indices_to_sample.fill_(0)
        # ``_pad_draft_buffers`` uses this boundary when clearing every
        # per-group context slot-mapping tail.  DP-synchronized dummy runs can
        # reach that helper before the non-profile runnable below, including
        # idle ranks whose local request count is zero.  Publish the current
        # context length for both profile and non-profile paths first.
        self._dflash_num_context = num_input_tokens
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        # Graph capture must run the real drafting metadata path so the
        # captured attention ops read the builders' persistent buffers (which
        # build_draft_attn_metadata refreshes before every replay). Capturing
        # with empty metadata would bake the zero-output branch of
        # DeepseekV41EagerAttentionImpl.forward into the draft graph.
        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.draft_attn_groups) > 0:
            multi_steps_attn_metadata, _ = self._build_capture_metadata(
                num_reqs, num_input_tokens, num_query_total
            )

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
            dp_moe_policy=self.runner.get_dp_moe_policy(
                is_draft_model=True, use_runtime_dp_policy=use_runtime_dp_policy
            ),
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

    def _build_capture_metadata(self, num_reqs, num_input_tokens, num_query_total):
        """Build drafting attention metadata for ACL graph capture.

        Mirrors the runtime path: set_inputs_first_pass rewrites the common
        metadata to num_query_per_req-wide query blocks, then
        build_draft_attn_metadata fills the per-group builders. Capturing
        through the same builders makes the graph bake their persistent
        buffer addresses, which the per-step rebuild refreshes before each
        replay.
        """
        num_query_per_req = self.num_query_per_req
        # The FA ops receive common_attn_metadata.query_start_loc directly
        # (metadata.swa.query_start_loc), unlike the other request-dimension
        # fields that the builder copies into its own persistent buffers.
        # Binding a capture-time temp would bake the temp's address into the
        # graph, and the temp is freed when this method returns: replays
        # would then read whatever tensor later reused that allocator block
        # (garbage per-request segmentation for multi-request batches).
        # Write the dummy layout into the same persistent buffer the runtime
        # path refreshes before every replay and hand the graph that address.
        qsl = self.arange_dflash[: num_reqs + 1] * num_query_per_req
        qsl_cpu = torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * num_query_per_req
        self.query_start_loc.gpu[: num_reqs + 1].copy_(qsl)
        self.query_start_loc.cpu[: num_reqs + 1].copy_(qsl_cpu)
        self.query_start_loc_group[0][: num_reqs + 1].copy_(qsl)
        self.seq_lens_group[0][:num_reqs].copy_(self.runner.seq_lens[:num_reqs])
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc_group[0][: num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
            seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
            seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
            seq_lens=self.seq_lens_group[0][:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_query_total,
            num_input_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=0,
            slot_mapping=self._slot_mapping_buffer[:num_query_total],
            positions=self.positions,
            actual_seq_lengths_q=[num_query_per_req] * num_reqs,
            decode_token_per_req=num_query_per_req,
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=(
                self.model.get_draft_attn_causal()[0]
                if hasattr(self.model, "get_draft_attn_causal")
                else False
            ),
            is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
            block_table_tensor=self.runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor()[
                :num_reqs
            ],
        )
        # The runtime path assembles per-group block tables in
        # set_inputs_first_pass; provide the runner-owned dummy tables here so
        # build_draft_attn_metadata sees the same structure at capture time.
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            dummy_block_table = self._per_group_block_tables.get(
                gid,
                self.runner.input_batch.block_table[gid].get_device_tensor(),
            )[:num_reqs]
            self._refresh_padded_block_table(gid, dummy_block_table)
        return self.build_draft_attn_metadata(common_attn_metadata, num_input_tokens, num_query_total)
