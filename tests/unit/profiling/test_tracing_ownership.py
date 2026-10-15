# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only ownership races using an uninitialized DeviceTracer instance."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from afd_plugin.profiling.model import host_span, span
from afd_plugin.profiling.session import Capture

tracing = pytest.importorskip("afd_plugin.profiling.tracing")

WAIT_TIMEOUT_S = 5


class FakeEvent:
    def __init__(self):
        self.recorded_streams = []

    def record(self, stream):
        self.recorded_streams.append(stream)


class RecordingWriter:
    def __init__(self):
        self.records = []

    def submit(self, record):
        self.records.append(record)
        return True


def _capture(generation=1):
    return Capture(
        "test-session",
        generation,
        RecordingWriter(),
        accepting=True,
        deadline_ns=time.monotonic_ns() + int(60e9),
    )


@pytest.fixture
def tracer(monkeypatch):
    tracer = tracing.DeviceTracer.__new__(tracing.DeviceTracer)
    tracer.device = torch.device("cpu")
    tracer.worker_id = "F0"
    tracer.clock_domain = "F0:device"
    tracer.lock = threading.Lock()
    tracer.slots = [tracing.EventSlot(FakeEvent(), FakeEvent())]
    tracer.metadata_slots = [
        tracing.MetadataSlot(
            FakeEvent(),
            torch.zeros((1, 2), dtype=torch.int64),
            torch.zeros(7, dtype=torch.int64),
        )
    ]
    tracer._host_records = {}
    tracer._retained = {}
    tracer._span_seq = 0
    tracer._dropped_spans = 0
    tracer._dropped_metadata = 0
    tracer._finalized_generation = 0
    tracer._anchor_pending = False
    stream = SimpleNamespace(npu_stream=7)
    monkeypatch.setattr(tracing.torch_npu.npu, "current_stream", lambda device: stream)
    return tracer


def test_ffn_control_is_read_only_after_copy_event_completes(tracer, monkeypatch):
    class ReadyEvent(FakeEvent):
        queries = 0

        def query(self):
            self.queries += 1
            return self.queries == 3

    ready = ReadyEvent()
    started_on = []
    copied = []
    source = torch.tensor([17, 5, 3], dtype=torch.int64)

    def read():
        assert ready.queries == 3
        return source.tolist()

    tracer._ffn_ready = ready
    tracer._ffn_control = SimpleNamespace(
        copy_=lambda tensor, *, non_blocking: copied.append((tensor, non_blocking)),
        tolist=read,
    )
    tracer.session = SimpleNamespace(
        start_pending_full=lambda: started_on.append(threading.get_ident())
    )
    yields: list[float] = []
    monkeypatch.setattr(tracing.time, "sleep", yields.append)

    assert tracer.read_ffn_control(source) == [17, 5, 3]
    assert copied[0][0] is source and copied[0][1] is True
    assert started_on == [threading.get_ident()] * 2
    assert yields == [0, 0]
    assert len(ready.recorded_streams) == 1


def test_completed_ffn_stream_does_not_pump_or_yield(tracer, monkeypatch):
    class CompletedEvent(FakeEvent):
        def query(self):
            return True

    ready = CompletedEvent()
    tracer._ffn_ready = ready

    def unexpected():
        pytest.fail("completed work must not enter the cooperative wait")

    tracer.session = SimpleNamespace(start_pending_full=unexpected)
    monkeypatch.setattr(tracing.time, "sleep", lambda _: unexpected())
    tracer.wait_for_stream()
    assert len(ready.recorded_streams) == 1


def test_span_admission_and_slot_registration_are_atomic_with_stop(tracer, monkeypatch):
    capture = _capture()
    execution = tracing.Execution(tracer, capture, "F0:call:1")
    checked = threading.Event()
    release_check = threading.Event()
    stop_attempted = threading.Event()
    stopped = threading.Event()
    observations = []
    errors = []
    original_admits = Capture.admits

    def gated_admits(self):
        admitted = original_admits(self)
        checked.set()
        if not release_check.wait(WAIT_TIMEOUT_S):
            raise TimeoutError("admission test barrier did not release")
        return admitted

    monkeypatch.setattr(Capture, "admits", gated_admits)

    def produce():
        try:
            tracer.span(execution, "FFN_COMPUTE")
        except BaseException as exc:
            errors.append(exc)

    def stop():
        try:
            stop_attempted.set()
            tracer.close_admission(capture)
            observations.append(tracer.pending_count())
            stopped.set()
        except BaseException as exc:
            errors.append(exc)

    producer = threading.Thread(target=produce, daemon=True)
    stopper = threading.Thread(target=stop, daemon=True)
    producer.start()
    try:
        assert checked.wait(WAIT_TIMEOUT_S)
        stopper.start()
        assert stop_attempted.wait(WAIT_TIMEOUT_S)
        # The gate is deliberately paused: the same lock must already be owned.
        assert tracer.lock.locked()
        assert not stopped.is_set()
    finally:
        release_check.set()
        producer.join(WAIT_TIMEOUT_S)
        if stopper.ident is not None:
            stopper.join(WAIT_TIMEOUT_S)
    assert not producer.is_alive() and not stopper.is_alive()
    assert not errors
    assert observations == [1]
    assert not capture.accepting
    assert tracer.slots[0].capture is capture
    assert len(tracer.slots[0].begin.recorded_streams) == 1


@pytest.mark.parametrize("terminal", ["closed", "finalized"])
def test_terminal_capture_cannot_reuse_or_modify_event_slots(tracer, terminal):
    capture, previous = _capture(), _capture(0)
    execution = tracing.Execution(tracer, capture, "F0:call:1")
    old_event = tracing.EventSlot(FakeEvent(), FakeEvent(), capture=previous)
    old_metadata = tracing.MetadataSlot(
        FakeEvent(),
        torch.zeros((1, 2), dtype=torch.int64),
        torch.zeros(7, dtype=torch.int64),
        capture=previous,
    )
    tracer.slots.insert(0, old_event)
    tracer.metadata_slots.insert(0, old_metadata)
    if terminal == "closed":
        tracer.close_admission(capture)
    else:
        tracer._finalized_generation = capture.generation
    with span(execution, "FFN_COMPUTE"), host_span(execution, "HOST_RECEIVE"):
        pass
    tracer.defer_metadata(execution, torch.ones((1, 2)), torch.ones(7))
    assert not tracer.retain_execution(execution)
    assert old_event.capture is previous and old_metadata.capture is previous
    assert tracer.slots[1].capture is None
    assert tracer.metadata_slots[1].capture is None
    assert not tracer._host_records and not tracer._retained
    assert tracer._dropped_metadata == 1
    for slot in tracer.slots:
        assert not slot.begin.recorded_streams and not slot.end.recorded_streams
    for slot in tracer.metadata_slots:
        assert not slot.ready.recorded_streams
        assert torch.count_nonzero(slot.trace) == 0
        assert torch.count_nonzero(slot.batch) == 0


def test_host_span_and_retained_execution_each_hold_pending_ownership(tracer):
    capture = _capture()
    execution = tracing.Execution(tracer, capture, "F0:call:1")
    assert tracer.pending_count() == 0
    host = host_span(execution, "HOST_RECEIVE")
    assert tracer.pending_count() == 1
    assert tracer.retain_execution(execution)
    assert tracer.pending_count() == 2
    host.__exit__(None, None, None)
    assert tracer.pending_count() == 1
    tracer.release_execution(execution)
    assert tracer.pending_count() == 0
    assert len(capture.writer.records) == 1
    assert capture.writer.records[0]["complete"]
