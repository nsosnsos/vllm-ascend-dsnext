from types import SimpleNamespace

import pytest

from vllm_ascend.speculative_mode import (
    SpeculativeMode,
    classify_speculative_mode,
)


@pytest.mark.parametrize(
    ("config", "expected"),
    (
        (None, SpeculativeMode.NONE),
        (
            SimpleNamespace(method="mtp", use_dspark=lambda: False),
            SpeculativeMode.MTP,
        ),
        (
            SimpleNamespace(method="dspark", use_dspark=lambda: True),
            SpeculativeMode.DSPARK,
        ),
        (SimpleNamespace(method="dspark"), SpeculativeMode.DSPARK),
        (
            SimpleNamespace(method="eagle3", use_dspark=lambda: False),
            SpeculativeMode.OTHER,
        ),
    ),
)
def test_classify_speculative_mode(config, expected):
    assert classify_speculative_mode(config) is expected


@pytest.mark.parametrize(
    "config",
    (
        SimpleNamespace(method="mtp", use_dspark=lambda: True),
        SimpleNamespace(method="dspark", use_dspark=lambda: False),
    ),
)
def test_classify_speculative_mode_rejects_contradictions(config):
    with pytest.raises(ValueError, match="Invalid speculative configuration"):
        classify_speculative_mode(config)
