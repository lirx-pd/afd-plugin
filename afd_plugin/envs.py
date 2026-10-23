# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_ASYNC_CAM_LAYERED_GMM = "AFD_ASYNC_CAM_LAYERED_GMM"
AFD_ASYNC_CAM_MEGA_MOE = "AFD_ASYNC_CAM_MEGA_MOE"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def async_cam_layered_gmm_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_CAM_LAYERED_GMM, "").lower() in ENV_TRUE_VALUES


def async_cam_mega_moe_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_CAM_MEGA_MOE, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "AFD_ASYNC_CAM_LAYERED_GMM",
    "AFD_ASYNC_CAM_MEGA_MOE",
    "async_cam_mega_moe_enabled",
    "async_cam_layered_gmm_enabled",
    "force_balanced_topk_ids_enabled",
]
