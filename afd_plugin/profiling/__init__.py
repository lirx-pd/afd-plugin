# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in CAM profiling without importing a device runtime at package import."""

from __future__ import annotations

import os

PROFILE_ENABLE_ENV = "AFD_CAM_PROFILE_ENABLE"
BATCH_INFO_FIELDS = 5
SOURCE_TRACE_FIELDS = 2
WIRE_PROTOCOL_VERSION = 1


def batch_info_prefix_size(tp_size: int, expert_per_rank: int) -> int:
    """Count compact metadata fields before the per-source wire identity tail."""
    return BATCH_INFO_FIELDS + tp_size * (1 + expert_per_rank)


def runtime_profiling_enabled() -> bool:
    return os.environ.get(PROFILE_ENABLE_ENV) == "1"
