# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in CAM profiling without importing a device runtime at package import."""

from __future__ import annotations

import os

PROFILE_ENABLE_ENV = "AFD_CAM_PROFILE_ENABLE"
BATCH_INFO_FIELDS = 5
MAX_DURATION_S = 3600
MAX_LIGHTWEIGHT_DURATION_S = 7 * 24 * MAX_DURATION_S


def batch_info_prefix_size(tp_size: int, expert_per_rank: int) -> int:
    """Size of the unmodified CAM dispatch-recv metadata."""
    return BATCH_INFO_FIELDS + tp_size * (1 + expert_per_rank)


def runtime_profiling_enabled() -> bool:
    return os.environ.get(PROFILE_ENABLE_ENV) == "1"
