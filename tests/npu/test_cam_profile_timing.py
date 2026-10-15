# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Allocated-device checks for the CANN 9.1 completed Event timestamp reader."""

from __future__ import annotations

import os
import threading
import time

import pytest

from afd_plugin.profiling.npu import EventTimestampReader

pytestmark = pytest.mark.npu
EVENT_TIMEOUT_S = 10


def runtime():
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires an allocated NPU and runtime opt-in")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return torch


def wait_complete(event):
    deadline = time.monotonic() + EVENT_TIMEOUT_S
    while not event.query():
        if time.monotonic() >= deadline:
            raise TimeoutError("NPU Event did not complete")
        time.sleep(0.001)


def test_timestamp_unit_and_reused_event_generation():
    torch = runtime()
    reader = EventTimestampReader()
    a = torch.randn(512, 512, dtype=torch.float16, device="npu")
    b = torch.randn_like(a)
    output = torch.empty_like(a)
    torch.mm(a, b, out=output)
    torch.npu.synchronize()
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    previous_end = 0
    for _ in range(8):
        begin.record()
        for _ in range(16):
            torch.mm(a, b, out=output)
        end.record()
        wait_complete(end)
        start_raw, end_raw = reader.read_raw(begin), reader.read_raw(end)
        assert start_raw is not None and end_raw is not None
        assert previous_end <= start_raw < end_raw
        # This reference synchronizes only after this isolated test is quiescent.
        elapsed_ms = begin.elapsed_time(end)
        assert (end_raw - start_raw) / elapsed_ms == pytest.approx(1000, rel=0.05)
        previous_end = end_raw


def test_completed_timestamp_does_not_wait_for_other_stream():
    torch = runtime()
    reader = EventTimestampReader()
    a = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    b = torch.randn_like(a)
    output = torch.empty_like(a)
    torch.mm(a, b, out=output)
    torch.npu.synchronize()
    ready = torch.npu.Event(enable_timing=True)
    ready.record()
    wait_complete(ready)
    busy = torch.npu.Stream()
    try:
        with torch.npu.stream(busy):
            for _ in range(512):
                torch.mm(a, b, out=output)
        assert not busy.query(), "independent stream finished before the probe"
        start = time.perf_counter()
        assert reader.read_raw(ready) is not None
        duration = time.perf_counter() - start
        assert not busy.query(), "timestamp read waited for unrelated device work"
        assert duration < 0.05
    finally:
        busy.synchronize()


def test_collector_thread_sets_device_and_reads_completed_event():
    torch = runtime()
    device = torch.npu.current_device()
    event = torch.npu.Event(enable_timing=True)
    event.record()
    wait_complete(event)
    timestamps = []
    errors = []

    def collect():
        try:
            torch.npu.set_device(device)
            timestamps.append(EventTimestampReader().read_raw(event))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=collect)
    thread.start()
    thread.join(EVENT_TIMEOUT_S)
    assert not thread.is_alive()
    assert not errors
    assert len(timestamps) == 1 and timestamps[0] is not None
