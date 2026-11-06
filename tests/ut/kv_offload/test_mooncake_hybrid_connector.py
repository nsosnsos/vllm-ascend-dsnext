import sys
import threading
import time
import types
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import torch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm.v1.request import RequestStatus  # noqa: E402

from vllm_ascend.core.private_circle_pool import (  # noqa: E402
    PRIVATE_CIRCLE_TRANSFER_FAILED,
    PRIVATE_CIRCLE_TRANSFER_READY,
    PRIVATE_CIRCLE_TRANSFER_RECEIVING,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (  # noqa: E402
    MAX_REQUESTS_PER_PEER_HANDLER,
    KVCacheRecvingThread,
    MooncakeConnectorScheduler,
)


class MockRequest:
    def __init__(
        self,
        request_id,
        prompt_token_ids,
        kv_transfer_params,
        status,
        num_prompt_tokens=None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        if num_prompt_tokens is None:
            num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0
        self.num_prompt_tokens = num_prompt_tokens
        self.kv_transfer_params = kv_transfer_params
        self.status = status
        self.output_token_ids = [101]


class TestHybridKVCacheRecvingThreadDispatch(unittest.TestCase):
    def _make_thread(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.executor = ThreadPoolExecutor(max_workers=2)
        thread.peer_request_queues = defaultdict(deque)
        thread.active_peer_request_handlers = set()
        thread.peer_request_queues_lock = threading.Lock()
        thread.request_task_counts = defaultdict(int)
        thread.finished_request_markers = set()
        thread.request_task_counts_lock = threading.Lock()
        thread.failed_recv_requests = set()
        thread.invalid_block_ids = set()
        thread.failed_recv_requests_lock = threading.Lock()
        thread.private_circle_group_layouts = {"1": {}}
        thread.task_tracker = MagicMock()
        thread.proc_not_transfer_request = {}
        thread.proc_not_transfer_request_lock = threading.Lock()
        thread.request_queue = MagicMock()
        thread.use_hybrid = True
        thread._send_done_signal_to_free_remote_port = MagicMock()
        thread._send_done_recv_signal = MagicMock()
        return thread

    @staticmethod
    def _recv_meta(all_task_done: bool) -> dict[str, Any]:
        return {
            "request_id": "request-1",
            "remote_request_id": "remote-request-1",
            "remote_host": "host-a",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "local_block_ids": ([10, 11], [200, 201]),
            "all_task_done": all_task_done,
        }

    def test_executor_workers_bind_kv_cache_device_before_handling_requests(self):
        expected_device_index = 5
        kv_cache = MagicMock(device=expected_device_index)
        model_config = types.SimpleNamespace(
            is_deepseek_mla=False,
            hf_config=types.SimpleNamespace(compress_ratios=[1]),
            hf_text_config=types.SimpleNamespace(
                num_hidden_layers=1,
                head_dim=64,
                num_key_value_heads=8,
            ),
        )
        vllm_config = types.SimpleNamespace(
            model_config=model_config,
            cache_config=types.SimpleNamespace(block_size=16),
        )
        kv_cache_config = types.SimpleNamespace(kv_cache_groups=[])
        worker_events: defaultdict[int, list[tuple[str, int | str]]] = defaultdict(list)
        events_lock = threading.Lock()
        both_workers_started = threading.Event()
        release_workers = threading.Event()

        def record_set_device(device):
            device_index = device if isinstance(device, int) else torch.device(device).index
            with events_lock:
                worker_events[threading.get_ident()].append(("set_device", device_index))

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.torch.npu.set_device",
                side_effect=record_set_device,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.is_vl_model",
                return_value=False,
            ),
        ):
            thread = KVCacheRecvingThread(
                tp_rank=1,
                tp_size=2,
                _prefill_pp_size=1,
                engine=MagicMock(),
                local_engine_id="local_engine",
                local_handshake_port=5555,
                side_channel_port=30000,
                local_kv_caches_base_addr=[0x1000],
                block_len_per_addr=[1024],
                block_stride_per_addr=[1024],
                addr_group_idx=[0],
                mamba_ssm_size=(0, 0),
                use_hybrid=False,
                has_mamba=False,
                hma_group_size=1,
                ready_event=threading.Event(),
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                kv_caches={"layer.0": (kv_cache, kv_cache)},
                private_circle_group_layouts={},
            )

            def handle_request(req_meta: dict[str, Any]):
                with events_lock:
                    worker_events[threading.get_ident()].append(("handle", req_meta["request_id"]))
                    handled_worker_count = sum(
                        any(event == "handle" for event, _ in events) for events in worker_events.values()
                    )
                    if handled_worker_count == 2:
                        both_workers_started.set()
                release_workers.wait()

            thread._handle_request = handle_request  # type: ignore[method-assign]
            try:
                for index in range(2):
                    thread._submit_request(
                        {
                            "request_id": f"req-{index}",
                            "remote_host": f"host-{index}",
                            "remote_handshake_port": 6000 + index,
                            "all_task_done": True,
                        }
                    )
                self.assertTrue(both_workers_started.wait(timeout=5.0), "executor did not start two workers")
            finally:
                release_workers.set()
                thread.executor.shutdown(wait=True, cancel_futures=True)

        handled_worker_events = [events for events in worker_events.values() if any(e == "handle" for e, _ in events)]
        self.assertEqual(len(handled_worker_events), 2)
        for events in handled_worker_events:
            self.assertEqual(events[0], ("set_device", expected_device_index))
            self.assertEqual(events[1][0], "handle")

    def test_init_initializes_runtime_state(self):
        """__init__ must fully initialize its runtime state.

        The receiving thread owns the executor, request queues, locks, codecs,
        and configuration references used by the pull path. Construct it for
        real and pin that contract.
        """
        kv_cache = MagicMock(device=0)
        model_config = types.SimpleNamespace(
            is_deepseek_mla=False,
            hf_config=types.SimpleNamespace(compress_ratios=[1]),
            hf_text_config=types.SimpleNamespace(
                num_hidden_layers=1,
                head_dim=64,
                num_key_value_heads=8,
            ),
        )
        vllm_config = types.SimpleNamespace(
            model_config=model_config,
            cache_config=types.SimpleNamespace(block_size=16),
        )
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.torch.npu.set_device",
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.is_vl_model",
                return_value=False,
            ),
        ):
            thread = KVCacheRecvingThread(
                tp_rank=1,
                tp_size=2,
                _prefill_pp_size=1,
                engine=MagicMock(),
                local_engine_id="local_engine",
                local_handshake_port=5555,
                side_channel_port=30000,
                local_kv_caches_base_addr=[0x1000],
                block_len_per_addr=[1024],
                block_stride_per_addr=[1024],
                addr_group_idx=[0],
                mamba_ssm_size=(0, 0),
                use_hybrid=False,
                has_mamba=False,
                hma_group_size=1,
                ready_event=threading.Event(),
                vllm_config=vllm_config,
                kv_cache_config=types.SimpleNamespace(kv_cache_groups=[]),
                kv_caches={"layer.0": (kv_cache, kv_cache)},
                private_circle_group_layouts={},
            )
        try:
            for attr in (
                "failed_recv_requests",
                "failed_recv_requests_lock",
                "invalid_block_ids",
                "proc_not_transfer_request",
                "proc_not_transfer_request_lock",
                "encoder",
                "decoder",
                "remote_sockets_lock",
                "vllm_config",
                "kv_cache_config",
                "model_config",
                "block_size",
                "num_layers",
                "timeout",
                "_folded_sync_state",
            ):
                self.assertTrue(hasattr(thread, attr), f"__init__ did not initialize {attr}")
        finally:
            thread.executor.shutdown(wait=True)

    def test_submit_request_serializes_same_peer_fifo(self):
        thread = self._make_thread()
        release_first_request = threading.Event()
        first_request_started = threading.Event()
        other_peer_started = threading.Event()
        handled_requests: list[str] = []
        active_by_peer: defaultdict[tuple[str, int], int] = defaultdict(int)
        max_active_by_peer: defaultdict[tuple[str, int], int] = defaultdict(int)
        state_lock = threading.Lock()

        def handle_request(req_meta: dict[str, Any]):
            peer_key = (req_meta["remote_host"], req_meta["remote_handshake_port"])
            with state_lock:
                active_by_peer[peer_key] += 1
                max_active_by_peer[peer_key] = max(max_active_by_peer[peer_key], active_by_peer[peer_key])
                handled_requests.append(req_meta["request_id"])

            if req_meta["request_id"] == "same-peer-1":
                first_request_started.set()
                self.assertTrue(release_first_request.wait(timeout=2.0))
            elif req_meta["request_id"] == "other-peer-1":
                other_peer_started.set()

            time.sleep(0.01)
            with state_lock:
                active_by_peer[peer_key] -= 1

        thread._handle_request = handle_request  # type: ignore[method-assign]
        same_peer_1 = {
            "request_id": "same-peer-1",
            "remote_host": "host-a",
            "remote_handshake_port": 6000,
            "all_task_done": False,
        }
        same_peer_2 = {
            "request_id": "same-peer-2",
            "remote_host": "host-a",
            "remote_handshake_port": 6000,
            "all_task_done": True,
        }
        other_peer = {
            "request_id": "other-peer-1",
            "remote_host": "host-b",
            "remote_handshake_port": 6001,
            "all_task_done": True,
        }

        try:
            thread._submit_request(same_peer_1)
            self.assertTrue(first_request_started.wait(timeout=1.0))
            thread._submit_request(same_peer_2)
            thread._submit_request(other_peer)

            self.assertTrue(other_peer_started.wait(timeout=1.0))
            time.sleep(0.05)
            self.assertNotIn("same-peer-2", handled_requests)
        finally:
            release_first_request.set()
            thread.executor.shutdown(wait=True, cancel_futures=True)

        self.assertLess(handled_requests.index("same-peer-1"), handled_requests.index("same-peer-2"))
        self.assertEqual(max_active_by_peer[("host-a", 6000)], 1)
        self.assertEqual(max_active_by_peer[("host-b", 6001)], 1)

    def test_peer_handler_yields_after_batch_limit(self):
        thread = self._make_thread()
        peer_key = ("host-a", 6000)
        requests = [
            {
                "request_id": f"req-{idx}",
                "remote_host": peer_key[0],
                "remote_handshake_port": peer_key[1],
            }
            for idx in range(MAX_REQUESTS_PER_PEER_HANDLER + 1)
        ]
        handled_requests: list[str] = []
        thread.peer_request_queues[peer_key].extend(requests)
        thread.active_peer_request_handlers.add(peer_key)
        thread.executor = MagicMock()

        def handle_request(req_meta: dict[str, Any]):
            handled_requests.append(req_meta["request_id"])

        thread._handle_request = handle_request  # type: ignore[method-assign]

        thread._handle_peer_requests(peer_key)

        self.assertEqual(handled_requests, [f"req-{idx}" for idx in range(MAX_REQUESTS_PER_PEER_HANDLER)])
        self.assertEqual(
            [req["request_id"] for req in thread.peer_request_queues[peer_key]],
            [f"req-{MAX_REQUESTS_PER_PEER_HANDLER}"],
        )
        self.assertIn(peer_key, thread.active_peer_request_handlers)
        thread.executor.submit.assert_called_once_with(thread._handle_peer_requests, peer_key)

    def test_transfer_failure_is_reported_and_remaining_pulls_are_skipped(self):
        thread = self._make_thread()
        thread._transfer_kv_cache_all_groups = MagicMock(
            side_effect=RuntimeError("transfer failed")
        )
        first_pull = self._recv_meta(all_task_done=False)
        last_pull = self._recv_meta(all_task_done=True)
        thread._mark_request_task_submitted(first_pull)
        thread._mark_request_task_submitted(last_pull)

        thread._handle_request(first_pull)
        self.assertTrue(thread._is_failed_recv_request("request-1"))
        thread._handle_request(last_pull)

        thread._transfer_kv_cache_all_groups.assert_called_once_with(first_pull)
        self.assertFalse(thread._is_failed_recv_request("request-1"))
        self.assertEqual(
            thread.get_and_clear_invalid_block_ids(),
            {10, 11},
        )
        self.assertEqual(thread.invalid_block_ids, set())
        thread.task_tracker.update_done_task_count.assert_called_once_with(
            "request-1"
        )
        self.assertEqual(thread.request_queue.task_done.call_count, 2)


class TestMooncakeHybridConnectorScheduler(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = object.__new__(MooncakeConnectorScheduler)
        scheduler.use_hybrid = True
        scheduler.use_compress = True
        scheduler.num_swa_blocks = [0, 2]
        # C4 exposes a 512-token logical block backed by one 128-token page.
        scheduler.group_block_size = [512, 128]
        scheduler._reqs_need_send = {}
        scheduler._reqs_need_recv = {}
        scheduler._private_circle_receiving_requests = {}
        scheduler._reqs_in_batch = set()
        scheduler.private_circle_group_ids = frozenset({1})
        scheduler.block_size = 128
        scheduler.engine_id = "engine"
        scheduler.side_channel_host = "127.0.0.1"
        scheduler.side_channel_port = 12345
        scheduler.tp_size = 1
        scheduler.multi_nodes_meta_mapping = {}
        return scheduler

    def test_compute_transfer_block_ids_trims_swa_groups(self):
        scheduler = self._make_scheduler()
        block_ids = (list(range(10)), [100, 101, 102, 103])

        transfer_block_ids = scheduler._compute_transfer_block_ids(block_ids, prompt_len=129)

        self.assertEqual(transfer_block_ids, ([0], [100, 101]))

    def test_request_finished_trims_logical_compressed_group_spans(self):
        scheduler = self._make_scheduler()
        scheduler.group_block_size = [512, 16384]
        scheduler.num_swa_blocks = [0, 0]
        request = MockRequest(
            "req-compressed",
            prompt_token_ids=list(range(513)),
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        )

        delay_free, params = scheduler.request_finished_all_groups(
            request,
            ([10, 11, 12], [20, 21]),
        )

        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        assert params is not None
        self.assertEqual(params["remote_block_ids"], ([10, 11], [20]))
        # This unused compatibility field remains in the legacy physical-block unit.
        self.assertEqual(params["num_prompt_blocks"], 5)

    def test_request_finished_trims_before_swa_clip(self):
        scheduler = self._make_scheduler()
        request = MockRequest(
            "req1",
            prompt_token_ids=list(range(129)),
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        )
        block_ids = (list(range(10)), [100, 101, 102, 103])

        delay_free, params = scheduler.request_finished_all_groups(request, block_ids)

        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        self.assertEqual(params["remote_block_ids"], ([0], [100, 101]))
        self.assertEqual(params["num_prompt_blocks"], 2)
        self.assertIn("req1", scheduler._reqs_need_send)

    def test_private_circle_receive_records_exact_import_and_match_group(self):
        scheduler = self._make_scheduler()
        scheduler._validate_private_layout = MagicMock()
        scheduler._private_page_ids = MagicMock(
            return_value=[200, 201, 202, 203]
        )
        private_layout = {
            "window_start": 128,
            "valid_length": 256,
            "window_size": 128,
        }
        request = MockRequest(
            "req1",
            prompt_token_ids=list(range(257)),
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_block_ids": ([30, 31], [40, 41, 42, 43]),
                "remote_engine_id": "remote-engine",
                "remote_host": "remote-host",
                "remote_port": 12345,
                "remote_request_id": "remote-req1",
                "private_circle_layout": private_layout,
            },
            status=RequestStatus.WAITING,
        )
        request.num_computed_tokens = 0
        request.private_circle_allocation = 3
        blocks = MagicMock()
        blocks.get_unhashed_block_ids_all_groups.return_value = (
            [10, 11],
            [],
        )

        scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens=256
        )

        self.assertEqual(
            request.private_circle_transfer_state,
            PRIVATE_CIRCLE_TRANSFER_RECEIVING,
        )
        self.assertEqual(request.private_circle_imported_window_start, 128)
        self.assertEqual(request.private_circle_imported_valid_length, 256)
        self.assertEqual(
            request.private_circle_transfer_shared_block_ids,
            frozenset({10, 11}),
        )
        _, local_block_ids, external_tokens = scheduler._reqs_need_recv[
            "req1"
        ]
        self.assertEqual(local_block_ids, ([10, 11], [200, 201, 202, 203]))
        self.assertEqual(external_tokens, 256)

    def test_private_circle_receive_rejects_incomplete_import(self):
        scheduler = self._make_scheduler()
        scheduler._validate_private_layout = MagicMock()
        scheduler._private_page_ids = MagicMock(return_value=[200, 201])
        request = MockRequest(
            "req1",
            prompt_token_ids=list(range(257)),
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_block_ids": ([30, 31], [40, 41]),
                "remote_engine_id": "remote-engine",
                "remote_host": "remote-host",
                "remote_port": 12345,
                "remote_request_id": "remote-req1",
                "private_circle_layout": {
                    "window_start": 128,
                    "valid_length": 255,
                    "window_size": 128,
                },
            },
            status=RequestStatus.WAITING,
        )
        request.num_computed_tokens = 0
        request.private_circle_allocation = 3
        blocks = MagicMock()
        blocks.get_unhashed_block_ids_all_groups.return_value = (
            [10, 11],
            [],
        )

        with self.assertRaisesRegex(
            RuntimeError, "does not cover the expected prefix"
        ):
            scheduler.update_state_after_alloc(
                request, blocks, num_external_tokens=256
            )

    def test_private_circle_receive_completion_publishes_ready_or_failed(self):
        scheduler = self._make_scheduler()
        ready_request = types.SimpleNamespace(
            request_id="ready",
            private_circle_transfer_shared_block_ids=frozenset({10, 11}),
            private_circle_transfer_state=PRIVATE_CIRCLE_TRANSFER_RECEIVING,
        )
        failed_request = types.SimpleNamespace(
            request_id="failed",
            private_circle_transfer_shared_block_ids=frozenset({20, 21}),
            private_circle_transfer_state=PRIVATE_CIRCLE_TRANSFER_RECEIVING,
        )
        scheduler._private_circle_receiving_requests = {
            "ready": ready_request,
            "failed": failed_request,
        }
        connector_output = types.SimpleNamespace(
            finished_recving={"ready", "failed"},
            invalid_block_ids={20},
        )

        scheduler.update_connector_output(connector_output)

        self.assertEqual(
            ready_request.private_circle_transfer_state,
            PRIVATE_CIRCLE_TRANSFER_READY,
        )
        self.assertEqual(
            failed_request.private_circle_transfer_state,
            PRIVATE_CIRCLE_TRANSFER_FAILED,
        )
        self.assertEqual(scheduler._private_circle_receiving_requests, {})

    def test_request_finished_uses_num_prompt_tokens(self):
        scheduler = self._make_scheduler()
        request = MockRequest(
            "req1",
            prompt_token_ids=None,
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
            num_prompt_tokens=129,
        )
        block_ids = (list(range(10)), [100, 101, 102, 103])

        delay_free, params = scheduler.request_finished_all_groups(request, block_ids)

        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        self.assertEqual(params["remote_block_ids"], ([0], [100, 101]))
        self.assertEqual(params["num_prompt_blocks"], 2)
