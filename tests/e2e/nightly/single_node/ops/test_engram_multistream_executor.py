# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU forward-local hash/Wkv/gate and native event capture/replay smoke.

Run directly with Python on one NPU for synthetic fetches. The multi-rank FP8
smoke also calls run_smoke with real ElasticBuffer tables. Both use production
custom ops and plain BF16 Wkv; checkpoint-specific quantized Wkv needs a model run.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch


def make_hash(device, rows=None):
    from vllm_ascend.models.deepseek_v41.engram_hash import NgramHashState

    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.max_ngram_size, state.lookback_depth = 3, 2
    state.pad_id, state.image_token_id, state.image_pad_token_id = 0, 30, 31
    state.register_buffer("token_map", torch.arange(32))
    if rows is None:
        primes, multipliers, offsets = [[[11], [13]], [[17], [19]]], [[3, 5, 7], [5, 7, 9]], [[0, 11], [0, 17]]
    else:
        # Put the second hash column in the second half of the table to exercise
        # remote-rank fetches even for small smoke inputs.
        half = rows // 2
        primes = [[[half - 1], [half - 1]], [[half - 1], [half - 1]]]
        multipliers, offsets = [[307, 509, 701], [503, 701, 907]], [[0, half], [0, half]]
    state.register_buffer("primes", torch.tensor(primes))
    state.register_buffer("multipliers", torch.tensor(multipliers))
    state.register_buffer("offsets", torch.tensor(offsets))
    return state.to(device)


def make_engram(layer, device, table=None, reference=None):
    from vllm_ascend.models.deepseek_v41.engram import AscendEngram

    class Engram(torch.nn.Module):
        apply_precomputed = AscendEngram.apply_precomputed

        def __init__(self):
            super().__init__()
            self.layer_id, self.layer_hash_index = layer, (1, 14).index(layer)
            self.layer_name = f"layers.{layer}.engram"
            self.hc_mult, self.hidden_size, self.norm_eps = 2, 4, 1e-6
            self.embed, self.table_reference, self.fetch_calls = table, reference, []
            width = table.width if table is not None else (reference.shape[1] if reference is not None else 1)
            self.wkv = torch.nn.Linear(2 * width, 12, bias=False, dtype=torch.bfloat16)
            self.wkv.weight.data.fill_(1 / (16 * width))
            self.q_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
            self.k_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))

        def precompute_embeddings(self, ids):
            if self.table_reference is not None:
                return self.table_reference[ids].flatten(1)
            return ids.to(torch.bfloat16) / 32

        def start_prefetch(self, ids):
            assert torch.npu.current_stream() == self.executor._stream
            self.fetch_calls.append("start")
            if self.embed is not None:
                return AscendEngram.start_prefetch(self, ids)
            return ids.clone()

        def finish_prefetch(self, handle):
            assert torch.npu.current_stream() == self.executor._stream
            self.fetch_calls.append("finish")
            if self.embed is not None:
                return AscendEngram.finish_prefetch(self, handle)
            return self.precompute_embeddings(handle)

        def precompute_kv(self, embedding):
            if hasattr(self, "executor"):
                assert torch.npu.current_stream() == self.executor._stream
            return AscendEngram.precompute_kv(self, embedding)

    return Engram().to(device)


@torch.inference_mode()
def run_smoke(tables=None, reference_tables=None):
    import torch_npu  # noqa: F401
    from vllm.config import CUDAGraphMode

    import vllm_ascend.models.deepseek_v41.engram as ops
    from vllm_ascend.models.deepseek_v41.engram_executor import EngramMultiStreamExecutor

    if tables is None:
        torch.npu.set_device(0)
    tables, reference_tables = tables or {}, reference_tables or {}
    rows = reference_tables[1].shape[0] if reference_tables else None
    capacity = 16
    early, late = (make_engram(layer, "npu", table=tables.get(layer)) for layer in (1, 14))
    executor = EngramMultiStreamExecutor(
        (1, 14),
        hash_fn=make_hash("npu", rows),
        engram_layers={layer.layer_id: layer for layer in (early, late)},
    )
    for layer in (early, late):
        layer.executor = executor
    context = SimpleNamespace(
        no_compile_layers={layer.layer_name: layer for layer in (early, late)},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    reference_hash = make_hash("cpu", rows)
    reference_layers = [make_engram(layer, "cpu", reference=reference_tables.get(layer)) for layer in (1, 14)]
    ids = torch.zeros(capacity, dtype=torch.int64, device="npu")
    positions = torch.arange(capacity, device="npu") + 2
    requests = torch.zeros_like(ids)
    local = torch.arange(capacity, device="npu")
    history = torch.ones((4, 2), dtype=torch.int64, device="npu")
    history_dead = torch.zeros_like(history, dtype=torch.bool)
    dead = torch.ones(capacity, dtype=torch.bool, device="npu")
    main = torch.npu.Stream()
    main.wait_stream(torch.npu.current_stream())
    graphs, snapshots = {}, []

    def forward(gear):
        hashes, mask = torch.ops.vllm.engram_begin(
            ids[:gear],
            positions[:gear],
            requests,
            local,
            history,
            history_dead,
            dead,
            early.layer_name,
            2,
            2,
        )
        hidden = ids[:gear, None, None].to(torch.bfloat16).expand(-1, 2, 4) / 32
        hidden = hidden + 1 / 32  # Backbone layer 0.
        kv = torch.ops.vllm.engram_precompute(hashes, hidden, early.layer_name, 2, 4)
        hidden = torch.ops.vllm.engram_apply_precomputed(hidden, *kv, mask, early.layer_name)
        hidden = hidden + 2 / 32  # Backbone layer 1.
        kv = torch.ops.vllm.engram_precompute(hashes, hidden, late.layer_name, 2, 4)
        hidden = hidden + 3 / 32  # Independent backbone layers.
        result = torch.ops.vllm.engram_apply_precomputed(hidden, *kv, mask, late.layer_name)
        executor.release()
        return result

    try:
        with patch.object(ops, "get_forward_context", return_value=context), torch.npu.stream(main):
            for gear in (8, 4, 16, 2, 8):
                forward(gear)
                main.synchronize()
                graph = torch.npu.NPUGraph()
                context.cudagraph_runtime_mode = CUDAGraphMode.FULL
                with torch.npu.graph(graph, stream=main):
                    output = forward(gear)
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                graphs[gear] = (graph, output)

            for use_graph in (False, True, True, False):
                for step, gear in enumerate((8, 4, 16, 8, 2)):
                    # Change tokens, history, request mapping and padding. All
                    # refreshes are device ops on main, with no readback/wait.
                    real = (0, gear, gear - 1)[step % 3]
                    value = len(snapshots) + 1
                    ids.fill_(value % 29)
                    history.fill_(value % 13)
                    requests[:real].fill_(step % 2)
                    requests[real:].zero_()
                    local.copy_(torch.arange(capacity, device="npu"))
                    local[real:].zero_()
                    dead[:real].fill_(False)
                    dead[real:].fill_(True)
                    if use_graph:
                        graph, output = graphs[gear]
                        call_counts = [len(layer.fetch_calls) for layer in (early, late)]
                        graph.replay()
                        assert [len(layer.fetch_calls) for layer in (early, late)] == call_counts
                    else:
                        output = forward(gear)
                    snapshots.append((output.clone(), gear, real, value))
                    executor.release()
                    assert not executor._pending_fetches
        main.synchronize()
        for actual, gear, real, value in snapshots:
            cpu_ids = torch.full((gear,), value % 29, dtype=torch.int64)
            cpu_local = torch.arange(gear)
            cpu_local[real:] = 0
            hashes, mask = reference_hash.forward_packed(
                cpu_ids,
                torch.arange(gear) + 2,
                torch.zeros(gear, dtype=torch.int64),
                cpu_local,
                torch.full((4, 2), value % 13, dtype=torch.int64),
                torch.zeros((4, 2), dtype=torch.bool),
                torch.arange(gear) >= real,
            )
            expected = cpu_ids[:, None, None].to(torch.bfloat16).expand(-1, 2, 4) / 32 + 1 / 32
            for slot, layer in enumerate(reference_layers):
                kv = layer.precompute_kv(layer.precompute_embeddings(hashes[:, slot]))
                expected = layer.apply_precomputed(expected, *kv, mask)
                if slot == 0:
                    expected = expected + 2 / 32
                    expected = expected + 3 / 32
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
    finally:
        executor.close()


def test_engram_multistream_executor():
    run_smoke()


if __name__ == "__main__":
    run_smoke()
    print("Engram forward-local eager/FULL multi-gear asynchronous replay smoke passed")
