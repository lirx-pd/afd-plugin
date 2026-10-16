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
from afd_plugin.profiling.npu import FFNDeviceWaiter
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


@pytest.fixture
def device_waiter(monkeypatch):
    waiter = FFNDeviceWaiter.__new__(FFNDeviceWaiter)
    waiter.device = torch.device("cpu")
    waiter.begin = SimpleNamespace(query=lambda: True)
    waiter.end = SimpleNamespace(query=lambda: True)
    monkeypatch.setattr(
        tracing.torch_npu.npu, "current_stream", lambda device: "ffn-stream"
    )
    return waiter


def test_ffn_control_is_read_only_after_copy_event_completes(
    device_waiter, monkeypatch
):
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

    device_waiter._ready = ready
    device_waiter._control = SimpleNamespace(
        copy_=lambda tensor, *, non_blocking: copied.append((tensor, non_blocking)),
        tolist=read,
    )
    device_waiter.session = SimpleNamespace(
        advance_full=lambda **kwargs: started_on.append(threading.get_ident())
    )
    yields: list[float] = []
    monkeypatch.setattr(tracing.time, "sleep", yields.append)

    assert device_waiter.read_ffn_control(source) == [17, 5, 3]
    assert copied[0][0] is source and copied[0][1] is True
    assert started_on == [threading.get_ident()] * 3
    assert yields == [0, 0]
    assert len(ready.recorded_streams) == 1


def test_completed_ffn_stream_does_not_pump_or_yield(device_waiter, monkeypatch):
    class CompletedEvent(FakeEvent):
        def query(self):
            return True

    ready = CompletedEvent()
    device_waiter._ready = ready

    def unexpected():
        pytest.fail("completed work must not enter the cooperative wait")

    device_waiter.session = SimpleNamespace(advance_full=unexpected)
    monkeypatch.setattr(tracing.time, "sleep", lambda _: unexpected())
    device_waiter.wait_for_stream()
    assert len(ready.recorded_streams) == 1


def test_ffn_wait_checks_timing_event_after_control_copy(device_waiter, monkeypatch):
    ready = FakeEvent()
    ready.query = lambda: True
    device_waiter._ready = ready
    completed = iter((False, True))
    device_waiter.end = SimpleNamespace(query=lambda: next(completed))
    progressed = []
    device_waiter.session = SimpleNamespace(
        advance_full=lambda **kwargs: progressed.append(kwargs)
    )
    yields = []
    monkeypatch.setattr(tracing.time, "sleep", yields.append)

    device_waiter.wait_for_stream()

    assert progressed == [{"allow_stop": True}]
    assert yields == [0]


def test_full_dsv4_dbo_step_fits_before_device_events_complete(tracer):
    tracer.slots = [
        tracing.EventSlot(FakeEvent(), FakeEvent())
        for _ in range(tracing.SPAN_CAPACITY)
    ]
    capture = _capture()
    kinds = (
        "ATTENTION",
        "GATE",
        "DISPATCH_SEND",
        "SHARED_EXPERT",
        "COMBINE_RECV",
        "RESTORE_MERGE",
    )
    for stage in range(2):
        for layer in range(43):
            execution = tracing.Execution(tracer, capture, f"A0:{stage}:{layer}")
            for kind in kinds:
                with span(execution, kind):
                    pass
    assert tracer._dropped_spans == 0
    assert tracer.pending_count() == 2 * 43 * len(kinds)
    # Producer completion must not release events still owned by the device.
    assert all(slot.ended for slot in tracer.slots if slot.capture is capture)


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


def test_admitted_work_can_finish_metadata_and_spans_after_stop(tracer):
    capture = _capture()
    execution = tracing.Execution(tracer, capture, "F0:work:1")
    assert tracer.retain_execution(execution)
    tracer.close_admission(capture)
    with span(execution, "FFN_COMPUTE"), host_span(execution, "HOST_RECEIVE"):
        pass
    tracer.defer_metadata(
        execution,
        torch.ones((1, 2), dtype=torch.int64),
        torch.ones(7, dtype=torch.int64),
    )
    assert tracer.slots[0].ended
    assert tracer.metadata_slots[0].armed
    assert tracer._dropped_metadata == 0
    tracer.release_execution(execution)


def test_completed_receive_reuses_observer_without_capture_slot_ownership(tracer):
    capture = _capture()
    capture.start_ns = 200
    execution = tracing.Execution(tracer, capture, "F0:work:1")
    assert tracer.retain_execution(execution)
    tracer.reader = SimpleNamespace(read_raw=lambda event: event)
    tracer._range = None
    tracer.record_receive(execution, 10, 20, SimpleNamespace(npu_stream=7), 100, 300)
    record = capture.writer.records[0]
    assert record["kind"] == "DISPATCH_RECV" and record["complete"]
    assert record["device_begin_ns"] == 10000 and record["device_end_ns"] == 20000
    assert record["crosses_capture_start"] is True
    assert tracer.pending_count() == 1  # Only the admitted work remains owned.
    assert all(slot.capture is None for slot in tracer.slots)
    tracer.release_execution(execution)
    assert tracer.pending_count() == 0


def test_ffn_stop_waits_for_previous_compute_fence(device_waiter, monkeypatch):
    queries = iter([False, False, True])
    device_waiter._ready = SimpleNamespace(
        record=lambda stream: None, query=lambda: next(queries)
    )
    fences = iter([False, True])
    device_waiter.begin = SimpleNamespace(query=lambda: next(fences))
    observed = []
    device_waiter.session = SimpleNamespace(
        advance_full=lambda **kwargs: observed.append(kwargs)
    )
    monkeypatch.setattr(tracing.time, "sleep", lambda _: None)
    device_waiter.wait_for_stream()
    assert observed == [{"allow_stop": False}, {"allow_stop": True}]


@pytest.mark.parametrize(
    "failure", [RuntimeError("timestamp unavailable"), ValueError("bad event"), None]
)
def test_receive_timestamp_failure_closes_capture_and_releases_work(tracer, failure):
    capture = _capture()
    execution = tracing.Execution(tracer, capture, "F0:work:1")
    assert tracer.retain_execution(execution)

    def read(event):
        if failure is not None:
            raise failure
        return None

    tracer.reader = SimpleNamespace(read_raw=read)
    assert not tracer.record_receive(
        execution, 1, 2, SimpleNamespace(npu_stream=7), 100, 300
    )
    assert capture.error and not capture.accepting
    assert tracer._dropped_spans == 1
    assert tracer.pending_count() == 0 and not capture.writer.records
