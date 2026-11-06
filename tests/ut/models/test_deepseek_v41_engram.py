# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from vllm_ascend.models.deepseek_v41.engram_gate import engram_gate
from vllm_ascend.models.deepseek_v41.engram_hash import (
    EngramTokenSequence,
    NgramHashState,
    build_lookback_token_ids,
)
from vllm_ascend.models.deepseek_v41.engram_offload import (
    ElasticEngramEmbedding,
    EngramFetchHandle,
    EngramTableState,
    create_engram_process_group,
)
from vllm_ascend.models.deepseek_v41.model import (
    AscendDeepseekV41ForCausalLM,
    DeepseekV41Model,
)
from vllm_ascend.models.deepseek_v41.vl_model import (
    AscendDeepseekV41ForConditionalGeneration,
)


class _IndexCounter(TorchDispatchMode):
    def __init__(self):
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.index.Tensor:
            self.calls += 1
        return func(*args, **(kwargs or {}))


class _BackendTokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return self.tokens[token_ids[0]]

    def id_to_token(self, token_id):
        return self.tokens[token_id]


class _Tokenizer:
    def __init__(self):
        self.tokens = [
            "<pad>",
            "alpha",
            "beta",
            "gamma",
            "<image>",
            "<image-pad>",
            "delta",
        ]
        self.backend_tokenizer = _BackendTokenizer(self.tokens)

    def __len__(self):
        return len(self.tokens)


def _make_hash_state():
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[112, 252],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=4,
        engram_vocab_size=11,
        engram_pad_token_id=0,
        engram_compressed_vocab_size=7,
        image_token_id=4,
        image_pad_token_id=5,
    )
    return NgramHashState(config, _Tokenizer())


def test_build_lookback_is_newest_first_and_marks_image_history_dead():
    token_ids = np.array(
        [
            [10, 11, 4, 13, 14],
            [20, 21, 22, 23, 24],
        ],
        dtype=np.int32,
    )

    lookback, dead = build_lookback_token_ids(
        token_ids,
        np.array([4, 0], dtype=np.int32),
        depth=3,
        image_token_id=4,
        image_pad_token_id=5,
    )

    np.testing.assert_array_equal(lookback, [[13, 4, 11], [-1, -1, -1]])
    np.testing.assert_array_equal(dead, [[False, True, False], [True, True, True]])


def test_model_parent_keeps_scheduler_token_budget():
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=256,
        max_num_seqs=1,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
    )
    parent_capacities = []

    def fail_parent_init(_self, *, vllm_config, prefix):
        del prefix
        parent_capacities.append(
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        raise RuntimeError("stop after observing parent capacity")

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.model.get_ascend_config",
            return_value=SimpleNamespace(enable_engram=False),
        ),
        patch.object(DeepseekV4Model, "__init__", fail_parent_init),
        pytest.raises(RuntimeError, match="stop after observing parent capacity"),
    ):
        DeepseekV41Model.__init__(model, vllm_config=vllm_config)

    assert parent_capacities == [256]
    assert scheduler_config.max_num_batched_tokens == 256


def test_build_lookback_fails_when_authoritative_history_is_missing():
    histories = [EngramTokenSequence([10, 11], [12])]

    with pytest.raises(ValueError, match="missing authoritative text token history"):
        build_lookback_token_ids(
            histories,
            np.array([4], dtype=np.int32),
            depth=3,
            image_token_id=4,
            image_pad_token_id=5,
        )


def test_build_lookback_marks_explicit_image_spans_dead():
    histories = [EngramTokenSequence([10, 11, 12, 13], [])]

    lookback, dead = build_lookback_token_ids(
        histories,
        np.array([4], dtype=np.int32),
        depth=3,
        image_token_id=4,
        image_pad_token_id=5,
        image_spans=[[(1, 2)]],
    )

    np.testing.assert_array_equal(lookback, [[13, 12, 11]])
    np.testing.assert_array_equal(dead, [[False, True, True]])


def test_chunk_hash_matches_same_positions_in_full_prefill():
    state = _make_hash_state()
    all_ids = torch.tensor([1, 2, 3, 6, 1, 2])
    full_lookback = torch.full((1, 3), -1)
    full_dead = torch.ones((1, 3), dtype=torch.bool)
    full_hash, full_mask = state(
        all_ids,
        torch.arange(6),
        torch.tensor([0, 6]),
        lookback_token_ids=full_lookback,
        lookback_dead_mask=full_dead,
    )

    chunk_ids = all_ids[3:]
    chunk_lookback = torch.tensor([[3, 2, 1]])
    chunk_hash, chunk_mask = state(
        chunk_ids,
        torch.arange(3, 6),
        torch.tensor([0, 3]),
        lookback_token_ids=chunk_lookback,
        lookback_dead_mask=torch.zeros_like(chunk_lookback, dtype=torch.bool),
    )

    torch.testing.assert_close(chunk_hash, full_hash[3:])
    torch.testing.assert_close(chunk_mask, full_mask[3:])


def test_hash_compresses_current_and_history_once(monkeypatch):
    state = _make_hash_state()
    input_ids = torch.tensor([1, 2, 3, 6, 1])
    positions = torch.tensor([3, 4, 5, 7, 8])
    request_ids = torch.tensor([0, 0, 0, 1, 1])
    local_positions = torch.tensor([0, 1, 2, 0, 1])
    lookback = torch.tensor([[6, 1, 2], [3, 2, 1]])
    lookback_dead = torch.zeros_like(lookback, dtype=torch.bool)
    current_dead = torch.zeros_like(input_ids, dtype=torch.bool)
    compressed_shapes = []
    original_compress = state._compress

    def counted_compress(token_ids):
        compressed_shapes.append(tuple(token_ids.shape))
        return original_compress(token_ids)

    monkeypatch.setattr(state, "_compress", counted_compress)

    hashes, mask = state.forward_packed(
        input_ids,
        positions,
        request_ids,
        local_positions,
        lookback,
        lookback_dead,
        current_dead,
    )

    assert hashes.shape == (5, 2, 6)
    assert mask.all()
    assert compressed_shapes == [(5,), (2, 3)]


def test_packed_hash_selects_encoded_source_once_per_shift():
    state = _make_hash_state()
    input_ids = torch.tensor([1, 2, 3, 6, 1])
    positions = torch.tensor([3, 4, 5, 7, 8])
    request_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    local_positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
    lookback = torch.tensor([[6, 1, 2], [3, 2, 1]])
    lookback_dead = torch.zeros_like(lookback, dtype=torch.bool)
    current_dead = torch.zeros_like(input_ids, dtype=torch.bool)
    counter = _IndexCounter()

    with counter:
        state.forward_packed(
            input_ids,
            positions,
            request_ids,
            local_positions,
            lookback,
            lookback_dead,
            current_dead,
        )

    token_map_indexes = 2
    source_indexes_per_shift = 2
    assert counter.calls == token_map_indexes + source_indexes_per_shift * (
        state.max_ngram_size - 1
    )


def test_packed_hash_single_token_path_uses_history_rows_without_source_indexes():
    state = _make_hash_state()
    input_ids = torch.tensor([3, 6])
    positions = torch.tensor([5, 7])
    request_ids = torch.tensor([0, 1], dtype=torch.int32)
    local_positions = torch.zeros(2, dtype=torch.int32)
    lookback = torch.tensor([[2, 1, 0], [5, 4, 3]])
    lookback_dead = torch.tensor(
        [[False, False, True], [False, True, False]]
    )
    current_dead = torch.zeros_like(input_ids, dtype=torch.bool)
    inputs = (
        input_ids,
        positions,
        request_ids,
        local_positions,
        lookback,
        lookback_dead,
        current_dead,
    )

    reference = state.forward_packed(*inputs)
    counter = _IndexCounter()
    with counter:
        actual = state.forward_packed(
            *inputs,
            single_token_per_request=True,
        )

    torch.testing.assert_close(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1])
    assert counter.calls == 2


def test_packed_hash_trusts_runner_dead_masks(monkeypatch):
    state = _make_hash_state()
    input_ids = torch.tensor([3])
    positions = torch.tensor([3])
    request_ids = torch.tensor([0])
    local_positions = torch.tensor([0])
    lookback = torch.tensor([[4, -1, 2]])
    lookback_dead = torch.tensor([[True, True, False]])
    current_dead = torch.zeros_like(input_ids, dtype=torch.bool)
    checked_shapes = []
    original_dead_mask = state._dead_mask

    def counted_dead_mask(token_ids):
        checked_shapes.append(tuple(token_ids.shape))
        return original_dead_mask(token_ids)

    monkeypatch.setattr(state, "_dead_mask", counted_dead_mask)

    state.forward_packed(
        input_ids,
        positions,
        request_ids,
        local_positions,
        lookback,
        lookback_dead,
        current_dead,
    )

    assert checked_shapes == []


def test_regular_hash_still_validates_supplied_history_dead_mask():
    state = _make_hash_state()
    input_ids = torch.tensor([3])
    positions = torch.tensor([3])
    query_start_loc = torch.tensor([0, 1])
    lookback = torch.tensor([[4, -1, 2]])
    implicit = state(
        input_ids,
        positions,
        query_start_loc,
        lookback_token_ids=lookback,
    )
    explicit = state(
        input_ids,
        positions,
        query_start_loc,
        lookback_token_ids=lookback,
        lookback_dead_mask=torch.zeros_like(lookback, dtype=torch.bool),
    )

    torch.testing.assert_close(implicit[0], explicit[0])
    torch.testing.assert_close(implicit[1], explicit[1])


def test_image_token_is_a_barrier_and_disables_its_own_gate():
    state = _make_hash_state()
    first_hash, first_mask = state(
        torch.tensor([1, 2, 4, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
    )
    second_hash, second_mask = state(
        torch.tensor([6, 1, 4, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
    )

    assert not first_mask[2]
    assert not second_mask[2]
    torch.testing.assert_close(first_hash[3], second_hash[3])


def test_engram_gate_matches_recipes_checkpoint_native_formula():
    torch.manual_seed(0)
    hidden = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    key = torch.randn_like(hidden)
    value = torch.randn(3, 8, dtype=torch.bfloat16)
    channel_weight = torch.randn(2, 8, dtype=torch.float32)
    token_mask = torch.tensor([True, False, True])
    eps = 1e-6

    hidden_float = hidden.float()
    key_float = key.float()
    rstd = torch.rsqrt(hidden_float.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key_float.square().mean(-1) + eps)
    dot = (hidden_float * channel_weight * key_float).sum(-1) * rstd * hidden.shape[-1] ** -0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, magnitude, -magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    expected = (hidden_float + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)

    actual = engram_gate(
        hidden,
        key,
        value,
        channel_weight,
        token_mask,
        eps,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], hidden[1], rtol=0, atol=0)


def test_engram_precompute_fake_matches_packed_value_layout():
    from vllm_ascend.models.deepseek_v41.engram import AscendEngram, engram_precompute_fake

    engram = SimpleNamespace(
        hc_mult=2,
        hidden_size=4,
        wkv=lambda embedding: torch.arange(embedding.shape[0] * 12, dtype=torch.bfloat16).view(-1, 12),
        q_weight=torch.ones(2, 4, dtype=torch.bfloat16),
        k_weight=torch.ones(2, 4, dtype=torch.bfloat16),
    )
    for tokens in (1, 4):
        embedding = torch.empty(tokens, 8, dtype=torch.bfloat16)
        actual = AscendEngram.precompute_kv(engram, embedding)
        hashes = torch.empty(tokens, 2, 2, dtype=torch.int64)
        fake = engram_precompute_fake(hashes, embedding, "test", 2, 4)
        for a, f in zip(actual, fake):
            assert (a.shape, a.dtype, a.stride()) == (f.shape, f.dtype, f.stride())


def test_model_forward_sync_fallback_bypasses_multistream_executor():
    calls = []

    class _Engram:
        layer_name = "layer.engram"

        def apply_lookup(self, hidden_states, embedding, active_mask, eps):
            del embedding, active_mask, eps
            calls.append("apply")
            return hidden_states + 1

    class _Layer:
        layer_idx = 0
        engram = _Engram()

        def __call__(self, positions, hidden_states, pre_mix, residual, input_ids=None):
            del positions, residual, input_ids
            return hidden_states, pre_mix

        def hc_pre_mix(self, hidden_states, pre_mix):
            del pre_mix
            return hidden_states.mean(dim=1)

    inputs_embeds = torch.randn(2, 4, dtype=torch.bfloat16)
    lookups = {0: torch.randn(2, 4, dtype=torch.bfloat16)}
    token_mask = torch.ones(2, dtype=torch.bool)
    model = SimpleNamespace(
        _engram_executor=object(),
        config=SimpleNamespace(rms_norm_eps=1e-6),
        hc_mult=2,
        prepare_engram=MagicMock(return_value=(lookups, token_mask)),
        shared_attention_state=SimpleNamespace(reset=MagicMock()),
        aux_hidden_state_layers=(),
        needs_moe_input_ids=False,
        layers=[_Layer()],
        norm=torch.nn.Identity(),
    )
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    with patch(
        "vllm_ascend.models.deepseek_v41.model.get_pp_group",
        return_value=pp_group,
    ):
        output = DeepseekV41Model.forward(
            model,
            input_ids=torch.arange(2),
            positions=torch.arange(2),
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )

    assert calls == ["apply"]
    model.prepare_engram.assert_called_once()
    torch.testing.assert_close(output, inputs_embeds + 1, rtol=0, atol=0)


def test_model_forward_prepares_eager_engram_inputs_inside_forward():
    class _Engram:
        layer_name = "layer.engram"

        def apply_lookup(self, hidden_states, embedding, active_mask, eps):
            del embedding, active_mask, eps
            return hidden_states + 1

    class _Layer:
        layer_idx = 0
        engram = _Engram()

        def __call__(self, positions, hidden_states, pre_mix, residual, input_ids=None):
            del positions, residual, input_ids
            return hidden_states, pre_mix

        def hc_pre_mix(self, hidden_states, pre_mix):
            del pre_mix
            return hidden_states.mean(dim=1)

    inputs_embeds = torch.randn(2, 4, dtype=torch.bfloat16)
    lookups = {0: torch.randn(2, 4, dtype=torch.bfloat16)}
    token_mask = torch.ones(2, dtype=torch.bool)
    calls = []
    prepare_engram_inputs = MagicMock(
        side_effect=lambda *args, **kwargs: (
            calls.append("prepare")
            or {"engram_lookups": lookups, "engram_mask": token_mask}
        )
    )
    embed_input_ids = MagicMock(
        side_effect=lambda _input_ids: calls.append("embed") or inputs_embeds
    )
    model = SimpleNamespace(
        _engram_executor=None,
        config=SimpleNamespace(rms_norm_eps=1e-6),
        hc_mult=2,
        prepare_engram_inputs=prepare_engram_inputs,
        prepare_engram=MagicMock(),
        embed_input_ids=embed_input_ids,
        shared_attention_state=SimpleNamespace(reset=MagicMock()),
        aux_hidden_state_layers=(),
        needs_moe_input_ids=False,
        layers=[_Layer()],
        norm=torch.nn.Identity(),
    )
    input_ids = torch.arange(2)
    positions = torch.arange(2)
    query_start_loc = torch.tensor([0, 2])
    lookback_token_ids = torch.full((1, 3), -1)
    lookback_dead_mask = torch.ones((1, 3), dtype=torch.bool)
    current_dead_mask = torch.zeros(2, dtype=torch.bool)

    with patch(
        "vllm_ascend.models.deepseek_v41.model.get_pp_group",
        return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
    ):
        output = DeepseekV41Model.forward(
            model,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,
            engram_padded_tokens=2,
            engram_query_start_loc=query_start_loc,
            engram_lookback_token_ids=lookback_token_ids,
            engram_lookback_dead_mask=lookback_dead_mask,
            engram_current_dead_mask=current_dead_mask,
        )

    prepare_engram_inputs.assert_called_once_with(
        input_ids,
        positions,
        2,
        query_start_loc=query_start_loc,
        lookback_token_ids=lookback_token_ids,
        lookback_dead_mask=lookback_dead_mask,
        current_dead_mask=current_dead_mask,
    )
    model.prepare_engram.assert_not_called()
    assert calls[:2] == ["prepare", "embed"]
    torch.testing.assert_close(output, inputs_embeds + 1, rtol=0, atol=0)


def test_release_engram_does_not_require_capture_or_replay_host_state():
    executor = SimpleNamespace(release=MagicMock())
    DeepseekV41Model.release_engram(SimpleNamespace(_engram_executor=executor))
    executor.release.assert_called_once_with()
    DeepseekV41Model.release_engram(SimpleNamespace(_engram_executor=None))


def test_current_placeholder_mask_is_a_hash_barrier_after_sanitization():
    state = _make_hash_state()
    dead_mask = torch.tensor([False, True, False, False])
    first_hash, first_mask = state(
        torch.tensor([1, 0, 2, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
        dead_mask=dead_mask,
    )
    second_hash, second_mask = state(
        torch.tensor([6, 0, 2, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
        dead_mask=dead_mask,
    )

    assert not first_mask[1]
    assert not second_mask[1]
    torch.testing.assert_close(first_hash[2:], second_hash[2:])


def test_tp1_creates_a_dedicated_non_null_process_group():
    process_group = object()
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
            return_value=0,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_backend",
            return_value="hccl",
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.new_group",
            return_value=process_group,
        ) as new_group,
    ):
        group, owns_group = create_engram_process_group(
            1,
            expected_world_size=1,
        )

    assert group is process_group
    assert owns_group
    new_group.assert_called_once_with(ranks=[0], backend="hccl")


def test_process_group_rejects_runtime_world_mismatch_before_creation():
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_world_size",
            return_value=4,
        ),
        patch("vllm_ascend.models.deepseek_v41.engram_offload.dist.new_group") as new_group,
        pytest.raises(RuntimeError, match="runtime=4, configured=8"),
    ):
        create_engram_process_group(4, expected_world_size=8)

    new_group.assert_not_called()


def test_process_group_selects_contiguous_group_from_global_world():
    first_group = object()
    second_group = object()
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_world_size",
            return_value=8,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
            return_value=5,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_backend",
            return_value="hccl",
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.new_group",
            side_effect=[first_group, second_group],
        ) as new_group,
    ):
        group, owns_group = create_engram_process_group(
            4,
            expected_world_size=8,
        )

    assert group is second_group
    assert owns_group
    assert [entry.kwargs["ranks"] for entry in new_group.call_args_list] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_vl_wrapper_forwards_engram_inputs_and_lifecycle():
    language_model = MagicMock()
    wrapper = SimpleNamespace(language_model=language_model)
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    query_start_loc = torch.tensor([0, 2])
    lookback_token_ids = torch.tensor([[-1, -1, -1]])
    lookback_dead_mask = torch.ones_like(lookback_token_ids, dtype=torch.bool)
    current_dead_mask = torch.tensor([False, True])

    AscendDeepseekV41ForConditionalGeneration.prepare_engram_inputs(
        wrapper,
        input_ids,
        positions,
        8,
        query_start_loc=query_start_loc,
        lookback_token_ids=lookback_token_ids,
        lookback_dead_mask=lookback_dead_mask,
        current_dead_mask=current_dead_mask,
    )
    AscendDeepseekV41ForConditionalGeneration.offload_weights(wrapper)
    AscendDeepseekV41ForConditionalGeneration.release_engram(wrapper)
    AscendDeepseekV41ForConditionalGeneration.abort_engram(wrapper)
    AscendDeepseekV41ForConditionalGeneration.destroy_engram(wrapper)

    language_model.prepare_engram_inputs.assert_called_once_with(
        input_ids,
        positions,
        8,
        query_start_loc=query_start_loc,
        lookback_token_ids=lookback_token_ids,
        lookback_dead_mask=lookback_dead_mask,
        current_dead_mask=current_dead_mask,
    )
    language_model.offload_weights.assert_called_once_with()
    language_model.release_engram.assert_called_once_with()
    language_model.abort_engram.assert_called_once_with()
    language_model.destroy_engram.assert_called_once_with()


def test_prepare_engram_inputs_keeps_sync_forward_inputs_at_fixed_capacity():
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.bfloat16,
    )
    mask = torch.tensor([True, False])
    lookup_buffer = torch.full((4, 2), 9.0, dtype=torch.bfloat16)
    mask_buffer = torch.ones(4, dtype=torch.bool)
    model = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1]),
        _engram_executor=None,
        _engram_max_tokens=4,
        _engram_input_buffers=({1: lookup_buffer}, mask_buffer),
        prepare_engram=MagicMock(return_value=({1: values}, mask)),
    )

    with patch(
        "vllm_ascend.models.deepseek_v41.model.get_ascend_config",
        return_value=SimpleNamespace(enable_engram=True),
    ):
        result = DeepseekV41Model.prepare_engram_inputs(
            model,
            torch.tensor([10, 11]),
            torch.tensor([0, 1]),
            padded_tokens=3,
            query_start_loc=torch.tensor([0, 2]),
        )

    assert result["engram_lookups"][1].shape == (4, 2)
    assert result["engram_lookups"][1].data_ptr() == lookup_buffer.data_ptr()
    torch.testing.assert_close(result["engram_lookups"][1][:2], values)
    torch.testing.assert_close(result["engram_lookups"][1][2], torch.zeros(2, dtype=torch.bfloat16))
    torch.testing.assert_close(result["engram_lookups"][1][3], torch.full((2,), 9.0, dtype=torch.bfloat16))
    assert torch.equal(result["engram_mask"], torch.tensor([True, False, False, True]))


def test_engram_checkpoint_collectives_are_deferred_until_post_load():
    wrapper = AscendDeepseekV41ForCausalLM.__new__(AscendDeepseekV41ForCausalLM)
    model = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1, 14]),
        destroy_engram=MagicMock(),
    )
    object.__setattr__(wrapper, "model", model)
    consumed = []

    def load_base(_self, weights):
        consumed.extend(name for name, _ in weights)
        return set(consumed)

    weights = [
        ("model.layers.14.engram.embed.weight", torch.empty(0)),
        ("model.layers.0.self_attn.weight", torch.empty(0)),
        ("model.layers.1.engram.embed.scale", torch.empty(0)),
        ("model.layers.1.engram.embed.weight", torch.empty(0)),
        ("model.layers.14.engram.embed.scale", torch.empty(0)),
    ]
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.model.get_ascend_config",
            return_value=SimpleNamespace(enable_engram=True),
        ),
        patch.object(
            AscendDeepseekV4ForCausalLM,
            "load_weights",
            new=load_base,
        ),
    ):
        loaded = wrapper.load_weights(weights)

    assert consumed == ["model.layers.0.self_attn.weight"]
    assert loaded == {"model.layers.0.self_attn.weight"}
    assert wrapper._engram_checkpoint_keys == {
        1: "model.layers.1.engram.embed.weight",
        14: "model.layers.14.engram.embed.weight",
    }
    model.destroy_engram.assert_not_called()


def test_engram_post_load_offload_uses_configured_layer_order():
    events = []

    class _Embedding:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.state = EngramTableState.EMPTY

        def load_checkpoint(self, root, key):
            events.append(("load", self.layer_id, root, key))
            self.state = EngramTableState.STAGED

        def offload_weights(self):
            events.append(("offload", self.layer_id))
            self.state = EngramTableState.READY

    layers = [SimpleNamespace(engram=None) for _ in range(15)]
    layers[1].engram = SimpleNamespace(embed=_Embedding(1))
    layers[14].engram = SimpleNamespace(embed=_Embedding(14))
    model = SimpleNamespace(
        engram_hash=object(),
        config=SimpleNamespace(engram_layer_ids=[1, 14]),
        layers=layers,
        engram_root="checkpoint",
    )

    DeepseekV41Model.offload_weights(
        model,
        {
            14: "model.layers.14.engram.embed.weight",
            1: "model.layers.1.engram.embed.weight",
        },
    )
    DeepseekV41Model.offload_weights(model, {})

    assert events == [
        ("load", 1, "checkpoint", "model.layers.1.engram.embed.weight"),
        ("offload", 1),
        ("load", 14, "checkpoint", "model.layers.14.engram.embed.weight"),
        ("offload", 14),
    ]


def test_model_destroy_attempts_every_engram_table():
    events = []

    class _Embedding:
        def __init__(self, layer_id, fails=False):
            self.layer_id = layer_id
            self.fails = fails

        def destroy(self):
            events.append(self.layer_id)
            if self.fails:
                raise ValueError(f"destroy {self.layer_id}")

    layers = [SimpleNamespace(engram=None) for _ in range(15)]
    layers[1].engram = SimpleNamespace(embed=_Embedding(1))
    layers[14].engram = SimpleNamespace(embed=_Embedding(14, fails=True))
    model = SimpleNamespace(
        engram_hash=object(),
        config=SimpleNamespace(engram_layer_ids=[1, 14]),
        layers=layers,
    )

    with pytest.raises(RuntimeError, match="failed to destroy") as exc_info:
        DeepseekV41Model.destroy_engram(model)

    assert events == [14, 1]
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_offload_preserves_primary_error_when_cleanup_fails():
    class _Embedding:
        state = EngramTableState.EMPTY

        def load_checkpoint(self, root, key):
            del root, key
            raise ValueError("primary load failure")

        def offload_weights(self):
            raise AssertionError("offload must not run after load failure")

        def destroy(self):
            raise RuntimeError("cleanup failure")

    model = SimpleNamespace(
        engram_hash=object(),
        config=SimpleNamespace(engram_layer_ids=[1]),
        layers=[SimpleNamespace(engram=None), SimpleNamespace(engram=SimpleNamespace(embed=_Embedding()))],
        engram_root="checkpoint",
    )
    model.destroy_engram = MethodType(DeepseekV41Model.destroy_engram, model)

    with pytest.raises(ValueError, match="primary load failure"):
        DeepseekV41Model.offload_weights(
            model,
            {1: "model.layers.1.engram.embed.weight"},
        )


def test_checkpoint_metadata_rejects_empty_scale_columns():
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(5, 4, 1, group=object())

    with pytest.raises(ValueError, match="incompatible scale shape"):
        table._validate_checkpoint_metadata(
            "weight",
            [5, 4],
            "F8_E4M3FN",
            "scale",
            [5, 0],
            "F8_E8M0",
        )


def test_checkpoint_metadata_rejects_rows_below_hash_coverage():
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(
            6,
            32,
            1,
            group=object(),
            minimum_rows=5,
        )

    with pytest.raises(ValueError, match="checkpoint rows must be in"):
        table._validate_checkpoint_metadata(
            "weight",
            [4, 32],
            "F8_E4M3FN",
            "scale",
            [4, 1],
            "F8_E8M0",
        )


def test_elastic_fetch_waits_once_and_destroy_is_idempotent():
    class _Buffer:
        def __init__(self):
            self.fetches = 0
            self.waits = 0
            self.destroys = 0

        def engram_fetch(self, ids):
            self.fetches += 1

            def wait():
                self.waits += 1
                codes = torch.zeros((ids.numel(), 4), dtype=torch.uint8).view(torch.float8_e4m3fn)
                scales = torch.full((ids.numel(), 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
                return codes, scales

            return wait

        def destroy(self):
            self.destroys += 1

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(8, 4, 1, group=object())
    buffer = _Buffer()
    table.engram_buffer = buffer
    table._state = EngramTableState.READY

    table._scale_block_size = 2
    output = table(torch.tensor([[0, 1], [2, 3]], dtype=torch.int64))
    table.destroy()
    table.destroy()

    assert output.shape == (2, 2, 4)
    assert buffer.fetches == 1
    assert buffer.waits == 1
    assert buffer.destroys == 1


def test_fp8_fetch_handle_defers_wait_and_is_single_use():
    class _Buffer:
        def __init__(self):
            self.waits = 0

        def engram_fetch(self, ids):
            def wait():
                self.waits += 1
                codes = torch.zeros((ids.numel(), 4), dtype=torch.uint8).view(torch.float8_e4m3fn)
                scales = torch.full((ids.numel(), 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
                return codes, scales

            return wait

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(8, 4, 1, group=object())
    buffer = _Buffer()
    table.engram_buffer = buffer
    table._scale_block_size = 2
    table._state = EngramTableState.READY

    handle = table.start_fetch(torch.tensor([[0, 1]], dtype=torch.int64))

    assert isinstance(handle, EngramFetchHandle)
    assert buffer.waits == 0
    assert table._fetch_in_flight
    output = table.finish_fetch(handle)
    assert output.shape == (1, 2, 4)
    assert buffer.waits == 1
    assert not table._fetch_in_flight
    with pytest.raises(RuntimeError, match="already been completed"):
        table.finish_fetch(handle)


def test_fp8_fetch_handle_cannot_cross_tables():
    class _Buffer:
        def engram_fetch(self, ids):
            def wait():
                codes = torch.zeros((ids.numel(), 4), dtype=torch.uint8).view(torch.float8_e4m3fn)
                scales = torch.full((ids.numel(), 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
                return codes, scales

            return wait

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        first = ElasticEngramEmbedding(8, 4, 1, group=object())
        second = ElasticEngramEmbedding(8, 4, 1, group=object())
    first.engram_buffer = _Buffer()
    second.engram_buffer = _Buffer()
    first._scale_block_size = 2
    second._scale_block_size = 2
    first._state = EngramTableState.READY
    second._state = EngramTableState.READY

    handle = first.start_fetch(torch.tensor([0], dtype=torch.int32))
    with pytest.raises(ValueError, match="another table"):
        second.finish_fetch(handle)
    first.finish_fetch(handle)


def test_failed_wait_poisons_table_until_destroy():
    class _Buffer:
        def __init__(self):
            self.destroys = 0

        def engram_fetch(self, ids):
            del ids

            def wait():
                raise RuntimeError("wait failed")

            return wait

        def destroy(self):
            self.destroys += 1

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(2, 4, 1, group=object())
    buffer = _Buffer()
    table.engram_buffer = buffer
    table._state = EngramTableState.READY

    with pytest.raises(RuntimeError, match="fetch/wait failed") as error:
        table(torch.tensor([0]))
    assert str(error.value.__cause__) == "wait failed"
    assert table.state is EngramTableState.POISONED
    with pytest.raises(RuntimeError, match="not ready"):
        table(torch.tensor([0]))

    table.destroy()
    assert table.state is EngramTableState.DESTROYED
    assert buffer.destroys == 1


def test_distributed_preflight_reports_remote_failure():
    def fail_one_rank(status, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        assert group is process_group
        status.zero_()

    process_group = object()
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(
            4,
            4,
            2,
            group=process_group,
            device="cpu",
        )
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.all_reduce",
        side_effect=fail_one_rank,
    ):
        assert not table._sync_phase_status(True, "test")
