# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical speculative-decoding mode classification."""

from enum import Enum
from typing import Any


class SpeculativeMode(str, Enum):
    NONE = "none"
    MTP = "mtp"
    DSPARK = "dspark"
    OTHER = "other"


ENGRAM_SPECULATIVE_MODES = frozenset(
    (SpeculativeMode.MTP, SpeculativeMode.DSPARK)
)


def classify_speculative_mode(speculative_config: Any | None) -> SpeculativeMode:
    """Return one authoritative mode and reject contradictory MTP/DSpark state."""
    if speculative_config is None:
        return SpeculativeMode.NONE

    method = getattr(speculative_config, "method", None)
    use_dspark = getattr(speculative_config, "use_dspark", None)
    reports_dspark = bool(use_dspark()) if callable(use_dspark) else False

    if method == "mtp":
        if reports_dspark:
            raise ValueError(
                "Invalid speculative configuration: method='mtp' cannot also "
                "report use_dspark()=True"
            )
        return SpeculativeMode.MTP
    if method == "dspark":
        if callable(use_dspark) and not reports_dspark:
            raise ValueError(
                "Invalid speculative configuration: method='dspark' must "
                "report use_dspark()=True"
            )
        return SpeculativeMode.DSPARK
    if reports_dspark:
        return SpeculativeMode.DSPARK
    return SpeculativeMode.OTHER
