# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram_executor import EngramMultiStreamExecutor


class _Stream:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def wait_event(self, event):
        self.calls.append((self.name, "wait", event.name))

    def synchronize(self):
        self.calls.append((self.name, "synchronize"))


class _Event:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def record(self, stream):
        self.calls.append((stream.name, "record", self.name))


@pytest.fixture
def executor_env(monkeypatch):
    calls = []
    main = _Stream("main", calls)
    current = main
    stream_names = iter(("engram",))
    names = iter(("inputs", "ready-1", "ready-14"))

    @contextmanager
    def stream_context(stream):
        nonlocal current
        previous, current = current, stream
        try:
            yield
        finally:
            current = previous

    monkeypatch.setattr(torch.npu, "Stream", lambda: _Stream(next(stream_names), calls))
    monkeypatch.setattr(torch.npu, "Event", lambda: _Event(next(names), calls))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current)
    monkeypatch.setattr(torch.npu, "stream", stream_context)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, stream: calls.append(("tensor", "use", stream.name))
    )

    def hash_inputs(tokens):
        calls.append((current.name, "hash"))
        return tokens[:, None, None].expand(-1, 2, 1), tokens >= 0

    executor = EngramMultiStreamExecutor(
        (1, 14),
        hash_fn=SimpleNamespace(forward_packed=hash_inputs),
        engram_layers={layer: _Engram(layer, calls) for layer in (1, 14)},
    )
    return executor, calls


class _Engram:
    def __init__(self, layer_id, calls):
        self.layer_id = layer_id
        self.layer_hash_index = (1, 14).index(layer_id)
        self.calls = calls

        self.pending = None

    def start_prefetch(self, ids):
        assert self.pending is None
        self.calls.append((torch.npu.current_stream().name, "start", self.layer_id))
        self.pending = SimpleNamespace(ids=ids)
        return self.pending

    def finish_prefetch(self, handle):
        assert handle is self.pending
        self.calls.append((torch.npu.current_stream().name, "finish", self.layer_id))
        self.pending = None
        return handle.ids.float() + self.layer_id

    def precompute_kv(self, embedding):
        self.calls.append((torch.npu.current_stream().name, "wkv", self.layer_id))
        return embedding * 2, embedding * 3, torch.ones(1)

    def apply_precomputed(self, hidden, key, value, weight, mask):
        self.calls.append((torch.npu.current_stream().name, "gate", self.layer_id))
        return hidden + (key + value) * weight * mask[:, None]


def test_serial_side_stream_joins_only_at_consumers_and_tracks_tensor_lifetimes(executor_env):
    executor, calls = executor_env
    early, late = (executor.engram_layers[layer] for layer in (1, 14))
    for tokens in (torch.tensor([1, -1, 3]), torch.tensor([7, 4, -1])):
        calls.clear()
        hashes, mask = executor.begin(tokens)
        calls.append(("main", "layer-0"))
        early_kv = executor.project(early, hashes)
        hidden = executor.consume(early, torch.zeros(3, 1), *early_kv, mask)
        calls.append(("main", "layer-1"))
        late_kv = executor.project(late, hashes)
        calls.append(("main", "layers-2-13"))
        hidden = executor.consume(late, hidden, *late_kv, mask)
        executor.release()
        expected = (tokens[:, None].float() * 2 + 15) * 5 * mask[:, None]
        torch.testing.assert_close(hidden, expected)
        stages = [call for call in calls if call[1] != "use"]
        assert stages == [
            ("main", "record", "inputs"),
            ("engram", "wait", "inputs"),
            ("engram", "hash"),
            ("engram", "start", 1),
            ("main", "layer-0"),
            ("engram", "finish", 1),
            ("engram", "wkv", 1),
            ("engram", "record", "ready-1"),
            ("engram", "start", 14),
            ("main", "wait", "ready-1"),
            ("main", "gate", 1),
            ("main", "layer-1"),
            ("engram", "finish", 14),
            ("engram", "wkv", 14),
            ("engram", "record", "ready-14"),
            ("main", "layers-2-13"),
            ("main", "wait", "ready-14"),
            ("main", "gate", 14),
        ]
        # K/V/weight/mask cross to main; hash never crosses to another side stream.
        assert calls.count(("tensor", "use", "main")) == 8
        assert calls.count(("tensor", "use", "engram")) == 1
        assert not executor._pending_fetches
        assert early.pending is late.pending is None

        # Reconstruct device dependencies from stream FIFO and event edges.
        # Serial L14 work must not become a predecessor of the L1 consumer.
        ancestors, stream_tail, records = {}, {}, {}
        for index, (stream, action, *detail) in enumerate(stages):
            parents = [stream_tail[stream]] if stream in stream_tail else []
            if action == "wait":
                parents.append(records[detail[0]])
            ancestors[index] = set(parents)
            for parent in parents:
                ancestors[index].update(ancestors[parent])
            stream_tail[stream] = index
            if action == "record":
                records[detail[0]] = index
        hash_op = stages.index(("engram", "hash"))
        late_start = stages.index(("engram", "start", 14))
        assert hash_op in ancestors[late_start]
        assert records["ready-1"] in ancestors[late_start]
        assert late_start not in ancestors[records["ready-1"]]
        early_gate = stages.index(("main", "gate", 1))
        assert late_start not in ancestors[early_gate]
        assert early_gate not in ancestors[late_start]
        assert early_gate not in ancestors[records["ready-14"]]
        for layer in (1, 14):
            gate = stages.index(("main", "gate", layer))
            wkv = stages.index(("engram", "wkv", layer))
            assert wkv in ancestors[records[f"ready-{layer}"]]
            assert wkv in ancestors[gate]
            assert records[f"ready-{layer}"] in ancestors[gate]


@pytest.mark.parametrize("failure", ["wkv", "gate"])
def test_failure_blocks_reuse_and_abort_drains_pending_fetch(executor_env, failure):
    executor, calls = executor_env
    engram = executor.engram_layers[1]
    hashes, mask = executor.begin(torch.ones(2))
    method = "precompute_kv" if failure == "wkv" else "apply_precomputed"
    setattr(engram, method, lambda *args: (_ for _ in ()).throw(RuntimeError("Engram failed")))
    with pytest.raises(RuntimeError, match="Engram failed"):
        kv = executor.project(engram, hashes)
        executor.consume(engram, torch.zeros(2, 1), *kv, mask)
    with pytest.raises(RuntimeError, match="closed or failed"):
        executor.begin(torch.ones(2))
    executor.abort()
    assert calls.count(("engram", "finish", 14)) == (1 if failure == "gate" else 0)
    assert ("engram", "wkv", 14) not in calls
    assert executor.engram_layers[14].pending is None
    assert not executor._pending_fetches
    assert calls[-2:] == [("engram", "synchronize"), ("main", "synchronize")]
    executor.close()
    before = len(calls)
    executor.close()
    assert len(calls) == before


def test_abort_attempts_main_drain_even_if_side_drain_fails(executor_env):
    executor, calls = executor_env
    executor._stream.synchronize = lambda: (_ for _ in ()).throw(RuntimeError("side failed"))
    with pytest.raises(RuntimeError, match="Failed to drain Engram streams"):
        executor.abort()
    assert calls[-1] == ("main", "synchronize")


def test_unknown_consumer_fails_before_submission(executor_env):
    executor, calls = executor_env
    with pytest.raises(ValueError, match="Unknown Engram layer"):
        executor.wait(9)
    assert not calls


def test_one_layer_needs_no_hash_fork_event(executor_env, monkeypatch):
    _, calls = executor_env
    names = iter(("inputs", "ready-1"))
    monkeypatch.setattr(torch.npu, "Stream", lambda: _Stream("engram", calls))
    monkeypatch.setattr(torch.npu, "Event", lambda: _Event(next(names), calls))
    early = _Engram(1, calls)
    executor = EngramMultiStreamExecutor(
        (1,),
        hash_fn=SimpleNamespace(forward_packed=lambda ids: (ids[:, None, None], ids >= 0)),
        engram_layers={1: early},
    )
    hashes, mask = executor.begin(torch.tensor([1, 2]))
    kv = executor.project(early, hashes)
    executor.consume(early, torch.zeros(2, 1), *kv, mask)
    executor.release()
    assert not executor.next_layer_ids
    assert [call for call in calls if call[1] == "wait"] == [
        ("engram", "wait", "inputs"),
        ("main", "wait", "ready-1"),
    ]


def test_late_launch_failure_does_not_repeat_completed_early_fetch(executor_env):
    executor, calls = executor_env
    executor.engram_layers[14].start_prefetch = lambda _: (_ for _ in ()).throw(RuntimeError("launch failed"))
    hashes, _ = executor.begin(torch.ones(2))
    with pytest.raises(RuntimeError, match="launch failed"):
        executor.project(executor.engram_layers[1], hashes)
    executor.abort()
    assert calls.count(("engram", "finish", 1)) == 1
    assert not executor._pending_fetches
    assert executor.engram_layers[1].pending is None
    assert calls.count(("engram", "wkv", 1)) == 1


@pytest.mark.parametrize("failed_layer", [1, 14])
def test_abort_drains_streams_even_when_pending_wait_fails(executor_env, failed_layer):
    executor, calls = executor_env
    hashes, _ = executor.begin(torch.ones(2))
    if failed_layer == 14:
        executor.project(executor.engram_layers[1], hashes)
    executor.engram_layers[failed_layer].finish_prefetch = lambda _: (_ for _ in ()).throw(RuntimeError("wait failed"))
    with pytest.raises(RuntimeError, match="Failed to drain Engram streams") as exc:
        executor.abort()
    assert str(exc.value.__cause__) == "wait failed"
    assert not executor._pending_fetches
    assert calls[-2:] == [("engram", "synchronize"), ("main", "synchronize")]


def test_unfinished_forward_cannot_be_released_or_overwritten(executor_env):
    executor, calls = executor_env
    hashes, _ = executor.begin(torch.ones(2))
    before = len(calls)
    with pytest.raises(RuntimeError, match="unfinished fetches"):
        executor.release()
    with pytest.raises(RuntimeError, match="unfinished fetches"):
        executor.begin(torch.ones(2))
    with pytest.raises(ValueError, match="does not own"):
        executor.project(_Engram(1, calls), hashes)
    assert len(calls) == before
    with pytest.raises(RuntimeError, match="No pending"):
        executor.project(executor.engram_layers[14], hashes)
    executor.project(executor.engram_layers[1], hashes)
    with pytest.raises(RuntimeError, match="No pending"):
        executor.project(executor.engram_layers[1], hashes)
    executor.abort()


def test_piecewise_rejected_before_launch(executor_env, monkeypatch):
    from vllm.config import CUDAGraphMode

    import vllm_ascend.models.deepseek_v41.engram as ops

    executor, calls = executor_env
    monkeypatch.setattr(
        ops, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
    )
    tokens = torch.ones(2, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        ops.engram_begin(*([tokens] * 7), "layers.1.engram", 2, 1)
    assert not executor._pending_fetches
    assert not calls


@pytest.mark.parametrize("failure", [None, (1, "wait"), (1, "wkv"), (14, "wait"), (14, "wkv")])
def test_real_table_handles_follow_serial_order_and_recover(executor_env, monkeypatch, failure):
    from vllm_ascend.models.deepseek_v41.engram import AscendEngram
    from vllm_ascend.models.deepseek_v41.engram_offload import ElasticEngramEmbedding, EngramTableState

    executor, calls = executor_env
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _: 0)

    class Buffer:
        def __init__(self, layer):
            self.layer = layer

        def engram_fetch(self, ids):
            calls.append(("fetch", self.layer))
            launch_stream = torch.npu.current_stream()
            codes = ids[:, None].expand(-1, 4).to(torch.float8_e4m3fn)
            scales = torch.full((ids.numel(), 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

            def wait():
                assert torch.npu.current_stream() is launch_stream
                calls.append(("wait", self.layer))
                if failure == (self.layer, "wait"):
                    raise RuntimeError("wait failed")
                return codes, scales

            return wait

        def destroy(self):
            calls.append(("destroy", self.layer))

    class Engram(_Engram):
        start_prefetch = AscendEngram.start_prefetch
        finish_prefetch = AscendEngram.finish_prefetch

    for layer in executor.layer_ids:
        engram = Engram(layer, calls)
        engram.embed = ElasticEngramEmbedding(8, 4, 1, group=object(), layer_id=layer)
        engram.embed.engram_buffer = Buffer(layer)
        engram.embed._state = EngramTableState.READY
        engram.embed._scale_block_size = 2
        executor.engram_layers[layer] = engram

    early, late = (executor.engram_layers[layer] for layer in (1, 14))
    hashes, mask = executor.begin(torch.tensor([1, 2]))
    assert [call for call in calls if call[0] in ("fetch", "wait")] == [("fetch", 1)]
    assert early.embed._fetch_in_flight and not late.embed._fetch_in_flight
    if failure is not None and failure[1] == "wkv":
        executor.engram_layers[failure[0]].precompute_kv = lambda _: (_ for _ in ()).throw(RuntimeError("Wkv failed"))

    def consume_layers():
        for engram in (early, late):
            kv = executor.project(engram, hashes)
            torch.testing.assert_close(kv[1], torch.tensor([[3] * 4, [6] * 4], dtype=torch.bfloat16))
            actual = executor.consume(engram, torch.zeros(2, 4), *kv, mask)
            torch.testing.assert_close(actual, torch.tensor([[5] * 4, [10] * 4], dtype=torch.float32))

    if failure is not None:
        with pytest.raises(RuntimeError):
            consume_layers()
        executor.abort()
    else:
        consume_layers()
        executor.release()
    assert not executor._pending_fetches
    for engram in (early, late):
        if failure == (engram.layer_id, "wait"):
            # Failed completion cannot prove the table is idle. It stays
            # poisoned until teardown rather than allowing another fetch.
            assert engram.embed.state is EngramTableState.POISONED
            assert engram.embed._fetch_in_flight
        else:
            assert not engram.embed._fetch_in_flight
    expected_layers = (1,) if failure is not None and failure[0] == 1 else (1, 14)
    assert [call for call in calls if call[0] in ("fetch", "wait")] == [
        (action, layer) for layer in expected_layers for action in ("fetch", "wait")
    ]
    executor.close()
    early.embed.destroy()
    late.embed.destroy()
    assert early.embed.state is late.embed.state is EngramTableState.DESTROYED


@pytest.mark.parametrize(
    "layers, hash_fn, engram_layers, error",
    [
        ((), object(), {}, "unique layer IDs"),
        ((1, 1), object(), {1: object()}, "unique layer IDs"),
        ((1,), None, {1: object()}, "hash module"),
        ((1, 14), object(), {1: object()}, "layers were not attached"),
    ],
)
def test_invalid_configuration_fails_before_stream_creation(layers, hash_fn, engram_layers, error):
    with pytest.raises(ValueError, match=error):
        EngramMultiStreamExecutor(layers, hash_fn=hash_fn, engram_layers=engram_layers)


def _hash_state(layer_count):
    from vllm_ascend.models.deepseek_v41.engram_hash import NgramHashState

    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.max_ngram_size, state.lookback_depth = 3, 2
    state.pad_id, state.image_token_id, state.image_pad_token_id = 0, 14, 15
    state.register_buffer("token_map", torch.arange(17))
    state.register_buffer("primes", torch.tensor([[[11], [13]]]).repeat(layer_count, 1, 1))
    state.register_buffer("multipliers", torch.tensor([[3, 5, 7]]).repeat(layer_count, 1))
    state.register_buffer("offsets", torch.tensor([[0, 11]]).repeat(layer_count, 1))
    return state


def test_packed_hash_handles_reordered_requests_images_history_and_padding():
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoHostRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert "_local_scalar_dense" not in str(func)
            assert "repeat_interleave" not in str(func)
            return func(*args, **(kwargs or {}))

    state = _hash_state(2)
    ids = torch.tensor([2, 3, 6, 14, 7, 0])
    positions = torch.tensor([8, 9, 1, 2, 3, 0])
    requests = torch.tensor([1, 1, 0, 0, 0, 0, 99, 99])
    local = torch.tensor([0, 1, 0, 1, 2, 0, 99, 99])
    history = torch.tensor([[1, -1], [8, 9], [99, 99], [99, 99]])
    history_dead = torch.tensor([[False, True], [False, True], [True, True], [True, True]])
    dead = torch.tensor([False, False, False, True, False, True, True, True])
    with NoHostRead():
        hashes, mask = state.forward_packed(ids, positions, requests, local, history, history_dead, dead)
    # Build an independent scalar reference for the packed history semantics.
    expected = []
    for row in range(len(ids)):
        blocked, window = False, []
        for shift in range(3):
            if local[row] >= shift:
                token = int(ids[max(0, row - shift)])
                is_dead = bool(dead[max(0, row - shift)])
            else:
                h = shift - int(local[row]) - 1
                token = int(history[requests[row], h])
                is_dead = bool(history_dead[requests[row], h])
            blocked |= is_dead or token < 0 or token >= 17 or token in (14, 15) or positions[row] < shift
            window.append(0 if blocked else token)
        rolling = window[0] * 3
        first = (rolling ^ (window[1] * 5)) % 11
        second = (rolling ^ (window[1] * 5) ^ (window[2] * 7)) % 13 + 11
        expected.append([first, second])
    torch.testing.assert_close(hashes, torch.tensor(expected)[:, None, :].expand(-1, 2, -1))
    assert mask.tolist() == [True, True, True, False, True, False]


@pytest.mark.parametrize("layer_ids", [(1,), (1, 3), (3, 1), (0, 2, 3), (1, 14)])
def test_real_model_forward_custom_ops_compile_without_breaks(executor_env, monkeypatch, layer_ids, request):
    from vllm.config import CUDAGraphMode

    import vllm_ascend.models.deepseek_v41.engram as ops
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model

    executor, _ = executor_env
    # Construct the requested event layout, including nonstandard checkpoint order.
    calls = []
    executor.layer_ids = tuple(sorted(layer_ids))
    executor._stream = _Stream("engram", calls)
    executor.next_layer_ids = dict(zip(executor.layer_ids, executor.layer_ids[1:]))
    executor._layer_ready = {layer: _Event(f"ready-{layer}", calls) for layer in layer_ids}
    executor.hash_fn = _hash_state(len(layer_ids))

    class Engram(torch.nn.Module):
        apply_precomputed = ops.AscendEngram.apply_precomputed

        def __init__(self, layer):
            super().__init__()
            self.layer_id, self.layer_hash_index = layer, layer_ids.index(layer)
            self.layer_name = f"layers.{layer}.engram"
            self.hidden_size, self.hc_mult, self.norm_eps = 4, 2, 1e-6
            self.executor = executor
            self.wkv = torch.nn.Linear(2, 12, bias=False, dtype=torch.bfloat16)
            self.q_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
            self.k_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))

        def precompute_embeddings(self, ids):
            return ids.to(torch.bfloat16) / 32

        def start_prefetch(self, ids):
            calls.append(("start", self.layer_id))
            return ids.clone()

        def finish_prefetch(self, handle):
            calls.append(("finish", self.layer_id))
            return self.precompute_embeddings(handle)

        def precompute_kv(self, embedding):
            calls.append(("wkv", self.layer_id, torch.npu.current_stream().name))
            return ops.AscendEngram.precompute_kv(self, embedding)

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.layer_idx = index
            self.engram = Engram(index) if index in layer_ids else None

        def forward(self, positions, hidden, pre_mix, residual, input_ids=None):
            return hidden + (self.layer_idx + 1) / 32, pre_mix

        def hc_pre_mix(self, hidden, pre_mix):
            return hidden.mean(dim=1)

    class Model(torch.nn.Module):
        forward = DeepseekV41Model.forward

        def __init__(self):
            super().__init__()
            self._engram_executor = executor
            self._engram_layer_ids = executor.layer_ids
            self._engram_next_layer_ids = executor.next_layer_ids
            self.config = SimpleNamespace(
                engram_layer_ids=layer_ids,
                engram_max_ngram_size=3,
                engram_n_heads=1,
                engram_head_dim=1,
                rms_norm_eps=1e-6,
            )
            self.hc_mult = 2
            self.layers = torch.nn.ModuleList(Layer(i) for i in range(max(4, max(layer_ids) + 1)))
            self.shared_attention_state = SimpleNamespace(reset=lambda: None)
            self.aux_hidden_state_layers = ()
            self.needs_moe_input_ids = False
            self.norm = torch.nn.Identity()

    model = Model()
    layers = {layer.engram.layer_name: layer.engram for layer in model.layers if layer.engram is not None}
    executor.engram_layers = {layer.layer_id: layer for layer in layers.values()}
    context = SimpleNamespace(no_compile_layers=layers, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(ops, "get_forward_context", lambda: context)
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr("vllm_ascend.models.deepseek_v41.model.get_pp_group", lambda: pp_group)
    # Production registers PrivateUse1 kernels. Register their same Python bodies
    # on CPU for this numerical / Dynamo contract test, retaining the fake impls.
    library = torch.library.Library("vllm", "IMPL", "CPU")
    request.addfinalizer(library._destroy)
    for name in ("engram_begin", "engram_precompute", "engram_apply_precomputed"):
        library.impl(name, getattr(ops, name))
    compiled_graphs = []

    def backend(gm, inputs):
        compiled_graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    history = torch.tensor([[1, 2], [8, 9], [-1, -1], [-1, -1]])
    history_dead = history < 0
    request_ids = torch.zeros(8, dtype=torch.int32)
    local_positions = torch.zeros(8, dtype=torch.int32)
    dead = torch.ones(8, dtype=torch.bool)
    with torch.inference_mode():
        for step, tokens in enumerate((4, 6, 4)):
            real = tokens - 1
            ids = (torch.arange(tokens) + step + 1) % 14
            positions = torch.arange(tokens) + 2
            hidden = torch.randn(tokens, 4, dtype=torch.bfloat16)
            request_ids[:real].fill_(step % 2)
            local_positions[:real] = torch.arange(real, dtype=torch.int32)
            request_ids[real:] = 0
            local_positions[real:] = 0
            dead[:real] = False
            dead[real:] = True
            history[step % 2] = torch.tensor([step + 3, step + 4])
            kwargs = dict(
                input_ids=ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=hidden,
                engram_request_ids=request_ids,
                engram_local_positions=local_positions,
                engram_lookback_token_ids=history,
                engram_lookback_dead_mask=history_dead,
                engram_current_dead_mask=dead,
            )
            # Serial math reference uses the actual gate with the same weights.
            hashes, mask = executor.hash_fn.forward_packed(
                ids,
                positions,
                request_ids,
                local_positions,
                history,
                history_dead,
                dead,
            )
            expected = hidden[:, None].repeat(1, 2, 1)
            for layer in model.layers:
                if layer.engram is not None:
                    e = layer.engram
                    kv = e.precompute_kv(e.precompute_embeddings(hashes[:, e.layer_hash_index]))
                    expected = e.apply_precomputed(expected, *kv, mask)
                expected = expected + (layer.layer_idx + 1) / 32
            for tensor in (ids, positions, hidden):
                torch._dynamo.mark_dynamic(tensor, 0)
            calls.clear()
            context.cudagraph_runtime_mode = CUDAGraphMode.FULL if step == 1 else CUDAGraphMode.NONE
            actual = compiled(**kwargs)
            torch.testing.assert_close(actual, expected.mean(dim=1), rtol=0, atol=0)
            fetch_calls = [call for call in calls if call[0] in ("start", "finish")]
            assert fetch_calls == [(action, layer) for layer in sorted(layer_ids) for action in ("start", "finish")]
            assert [call for call in calls if call[0] == "wkv"] == [
                ("wkv", layer, "engram") for layer in sorted(layer_ids)
            ]
            executor.release()
            assert not executor._pending_fetches
    assert len(compiled_graphs) == 1
    targets = [str(n.target) for n in compiled_graphs[0].graph.nodes if n.op == "call_function"]
    assert sum("engram_begin" in target for target in targets) == 1
    assert sum("engram_precompute" in target for target in targets) == len(layer_ids)
    assert sum("engram_apply_precomputed" in target for target in targets) == len(layer_ids)
