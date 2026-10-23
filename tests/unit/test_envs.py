# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import pytest

from afd_plugin.envs import (
    AFD_ASYNC_CAM_LAYERED_GMM,
    AFD_ASYNC_CAM_MEGA_MOE,
    AFD_FORCE_BALANCED_TOPK_IDS,
    async_cam_layered_gmm_enabled,
    async_cam_mega_moe_enabled,
    force_balanced_topk_ids_enabled,
)


def test_force_balanced_topk_ids_env_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(AFD_FORCE_BALANCED_TOPK_IDS, raising=False)

    assert force_balanced_topk_ids_enabled() is False


def test_force_balanced_topk_ids_env_accepts_true_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(AFD_FORCE_BALANCED_TOPK_IDS, value)

        assert force_balanced_topk_ids_enabled() is True


@pytest.mark.parametrize(
    "name, enabled",
    [
        (AFD_ASYNC_CAM_LAYERED_GMM, async_cam_layered_gmm_enabled),
        (AFD_ASYNC_CAM_MEGA_MOE, async_cam_mega_moe_enabled),
    ],
)
def test_async_cam_compute_switch_defaults_and_boolean_values(
    monkeypatch, name, enabled
):
    monkeypatch.delenv(name, raising=False)
    assert enabled() is False
    for value in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv(name, value)
        assert enabled() is True
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(name, value)
        assert enabled() is False
