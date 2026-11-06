# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode

from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper


@pytest.mark.parametrize(
    "mode,skip,enpu,draft,expected_sync",
    [
        (CUDAGraphMode.FULL, True, False, False, False),
        (CUDAGraphMode.FULL, False, False, False, True),
        (CUDAGraphMode.FULL, False, True, False, False),
        (CUDAGraphMode.FULL, False, False, True, False),
        (CUDAGraphMode.PIECEWISE, False, False, False, False),
    ],
)
def test_replay_keeps_sparse_async_path_without_changing_default_fence(mode, skip, enpu, draft, expected_sync):
    wrapper = BreakableACLGraphWrapper.__new__(BreakableACLGraphWrapper)
    wrapper.skip_pre_replay_sync = skip
    wrapper.enable_enpu = enpu
    wrapper.use_eagle = draft
    entry = SimpleNamespace(output=object())
    stream = MagicMock()
    with (
        patch(
            "vllm_ascend.compilation.breakable_aclgraph.get_forward_context",
            return_value=SimpleNamespace(cudagraph_runtime_mode=mode),
        ),
        patch("vllm_ascend.compilation.breakable_aclgraph._EXTRA_CTX", SimpleNamespace(is_draft_model=draft)),
        patch("vllm_ascend.compilation.breakable_aclgraph.torch.npu.current_stream", return_value=stream),
        patch.object(BreakableCUDAGraphWrapper, "_replay") as replay,
    ):
        assert wrapper._replay(entry, (), {}) is entry.output
    assert stream.synchronize.call_count == int(expected_sync)
    replay.assert_called_once_with(entry, (), {})
