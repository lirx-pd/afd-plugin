# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Profiling uses the compact metadata returned by unmodified CAM operators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from afd_plugin.profiling import (
    BATCH_INFO_FIELDS,
    batch_info_prefix_size,
)


@pytest.mark.parametrize(
    "tp_size,experts,prefix", [(1, 1, 7), (2, 3, 13), (4, 16, 73), (8, 8, 77)]
)
def test_compact_prefix_contract(tp_size, experts, prefix):
    assert batch_info_prefix_size(tp_size, experts) == prefix
    assert BATCH_INFO_FIELDS == 5


def test_layout_import_needs_no_device_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; from afd_plugin.profiling import batch_info_prefix_size; "
            "assert batch_info_prefix_size(2, 3) == 13; "
            "assert not {'torch', 'torch_npu', 'vllm'} & sys.modules.keys()",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("ready", [True, False])
def test_tracer_host_slots_match_native_metadata(monkeypatch, ready):
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    from afd_plugin.profiling import tracing

    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *args, pin_memory=False, **kwargs: empty(*args, **kwargs)
    )
    monkeypatch.setattr(tracing, "EventTimestampReader", lambda: object())
    monkeypatch.setattr(
        torch_npu.npu,
        "Event",
        lambda **kwargs: SimpleNamespace(
            record=lambda *args: None, query=lambda: ready
        ),
    )
    monkeypatch.setattr(torch_npu.npu, "Stream", lambda **kwargs: object())

    def unexpected_sync(*args):
        pytest.fail("lazy PREPARE must not synchronize the whole device")

    monkeypatch.setattr(torch_npu.npu, "synchronize", unexpected_sync)
    monkeypatch.setattr(tracing, "RESOURCE_INIT_TIMEOUT_S", 0)

    if not ready:
        with pytest.raises(TimeoutError, match="event initialization deadline"):
            tracing.DeviceTracer(torch.device("cpu"), "ffn", ["a0", "a1"], 2, 3)
        return

    tracer = tracing.DeviceTracer(torch.device("cpu"), "ffn", ["a0", "a1"], 2, 3)

    assert all(slot.batch.shape == (13,) for slot in tracer.metadata_slots)
