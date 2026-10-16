# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fixed device resources for sparse CAM stage timing and deferred metadata."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_npu

from afd_plugin.profiling import batch_info_prefix_size
from afd_plugin.profiling.npu import EventTimestampReader

if TYPE_CHECKING:
    from afd_plugin.profiling.session import Capture, WorkerSession

# A queued DSV4 step can hold 2 microbatches * 43 layers * 6 device spans.
SPAN_CAPACITY = 1024
METADATA_CAPACITY = 16
RESOURCE_INIT_TIMEOUT_S = 2
ANCHOR_INTERVAL_NS = 1_000_000_000
NS_PER_DEVICE_TICK = 1000


@dataclass(slots=True)
class EventSlot:
    begin: torch_npu.npu.Event
    end: torch_npu.npu.Event
    capture: Capture | None = None
    record: dict | None = None
    stream: torch_npu.npu.Stream | None = None
    ended: bool = False
    exported: bool = False


@dataclass(slots=True)
class MetadataSlot:
    ready: torch_npu.npu.Event
    batch: torch.Tensor
    capture: Capture | None = None
    exec_ref: str = ""
    tensor: torch.Tensor | None = None
    exported: bool = False
    armed: bool = False


class Span:
    def __init__(self, tracer: DeviceTracer, slot: EventSlot) -> None:
        self.tracer = tracer
        self.slot = slot

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        slot = self.slot
        slot.end.record(slot.stream)
        assert slot.record is not None
        slot.record["host_end_ns"] = time.monotonic_ns()
        with self.tracer.lock:
            self.slot.ended = True


class HostSpan:
    def __init__(self, execution: Execution, record: dict) -> None:
        self.execution = execution
        self.record = record

    def __enter__(self) -> HostSpan:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        tracer = self.execution.tracer
        with tracer.lock:
            tracer._host_records.pop(self.record["span_id"])
            if not tracer.finalized(self.execution.capture):
                self.record.update(host_end_ns=time.monotonic_ns(), complete=True)
                self.execution.capture.writer.submit(self.record)


@dataclass(frozen=True, slots=True)
class Execution:
    tracer: DeviceTracer
    capture: Capture
    exec_ref: str


class DeviceTracer:
    """The collector owns pooled Events; the model thread owns receive timing."""

    def __init__(
        self,
        device: torch.device,
        worker_id: str,
        source_workers: list[str],
        tp_size: int,
        expert_per_rank: int,
    ) -> None:
        self.device = device
        self.worker_id = worker_id
        self.source_workers = source_workers
        self.tp_size = tp_size
        self.clock_domain = f"{worker_id}:device"
        self.lock = threading.Lock()
        self.reader = EventTimestampReader()
        self.session: WorkerSession | None = None
        self.slots = [
            EventSlot(
                torch_npu.npu.Event(enable_timing=True),
                torch_npu.npu.Event(enable_timing=True),
            )
            for _ in range(SPAN_CAPACITY)
        ]
        batch_size = batch_info_prefix_size(tp_size, expert_per_rank)
        self.metadata_slots = [
            MetadataSlot(
                torch_npu.npu.Event(),
                torch.empty(
                    (batch_size,),
                    dtype=torch.int64,
                    pin_memory=True,
                ),
            )
            for _ in range(METADATA_CAPACITY)
        ]
        self._free_slots = deque(self.slots)
        self._free_metadata = deque(self.metadata_slots)
        self.anchor = torch_npu.npu.Event(enable_timing=True)
        self.anchor_stream = torch_npu.npu.Stream(device=device)
        # PREPARE must not wait for an unrelated, possibly blocked CAM receive.
        for slot in self.slots:
            slot.begin.record(self.anchor_stream)
            slot.end.record(self.anchor_stream)
        for metadata in self.metadata_slots:
            metadata.ready.record(self.anchor_stream)
        self.anchor.record(self.anchor_stream)
        deadline = time.monotonic() + RESOURCE_INIT_TIMEOUT_S
        while not self.anchor.query():
            if time.monotonic() >= deadline:
                raise TimeoutError("LIGHTWEIGHT event initialization deadline exceeded")
            time.sleep(0.001)
        self._host_records: dict[str, tuple[Capture, dict]] = {}
        self._retained: dict[str, Execution] = {}
        self._span_seq = 0
        self._work_seq = 0
        self._dropped_spans = 0
        self._dropped_metadata = 0
        self._unresolved = 0
        self._anchors: list[tuple[int, int, int]] = []
        self._anchor_pending = False
        self._anchor_before = 0
        self._next_anchor = 0
        self._range: tuple[int, int] | None = None
        self._finalized_generation = 0

    def begin_capture(self) -> None:
        self._dropped_spans = 0
        self._dropped_metadata = 0
        self._unresolved = 0
        self._anchors = []
        self._range = None
        self._next_anchor = 0

    def execution(self, exec_ref: str) -> Execution | None:
        assert self.session is not None
        capture = self.session.capture
        if capture is None or not capture.admits():
            return None
        if capture.started_at_ns is None:
            capture.started_at_ns = time.time_ns()
        return Execution(self, capture, exec_ref)

    def new_work(self) -> Execution | None:
        assert self.session is not None
        if self.session.capture is None or not self.session.capture.admits():
            return None
        with self.lock:
            self._work_seq += 1
            reference = f"{self.worker_id}:work:{self._work_seq}"
        execution = self.execution(reference)
        if execution is not None and self.retain_execution(execution):
            return execution
        return None

    def record_receive(
        self,
        execution: Execution,
        begin_event: torch_npu.npu.Event,
        end_event: torch_npu.npu.Event,
        stream: torch_npu.npu.Stream,
        host_begin_ns: int,
        host_end_ns: int,
    ) -> bool:
        # The model thread owns this completed pair and reads it before reuse.
        capture = execution.capture
        try:
            begin = self.reader.read_raw(begin_event)
            end = self.reader.read_raw(end_event)
            if begin is None or end is None:
                raise RuntimeError("receive timing events are not complete")
        except (RuntimeError, ValueError) as exc:
            # Instrumentation failure must not abort the received model work.
            with self.lock:
                capture.error = f"{type(exc).__name__}: {exc}"
                capture.accepting = False
                self._dropped_spans += 1
                self._retained.pop(execution.exec_ref, None)
            return False
        begin, end = begin * NS_PER_DEVICE_TICK, end * NS_PER_DEVICE_TICK
        with self.lock:
            if self.finalized(capture):
                self._retained.pop(execution.exec_ref, None)
                return False
            self._span_seq += 1
            capture.writer.submit(
                {
                    "record_type": "SPAN",
                    "span_id": f"{self.worker_id}:{self._span_seq}",
                    "worker_id": self.worker_id,
                    "exec_ref": execution.exec_ref,
                    "kind": "DISPATCH_RECV",
                    "clock_domain": self.clock_domain,
                    "producer_id": str(threading.get_ident()),
                    "stream_id": stream.npu_stream,
                    "host_begin_ns": host_begin_ns,
                    "host_end_ns": host_end_ns,
                    "device_begin_ns": begin,
                    "device_end_ns": end,
                    "generation": capture.generation,
                    "complete": True,
                    "crosses_capture_start": host_begin_ns < capture.start_ns,
                }
            )
            self._range = (
                (min(self._range[0], begin), max(self._range[1], end))
                if self._range
                else (begin, end)
            )
        return True

    def span(self, execution: Execution, kind: str) -> Span | nullcontext:
        capture = execution.capture
        with self.lock:
            if self.finalized(capture) or (
                not capture.admits() and execution.exec_ref not in self._retained
            ):
                return nullcontext()
            if not self._free_slots:
                self._dropped_spans += 1
                return nullcontext()
            slot = self._free_slots.popleft()
            self._span_seq += 1
            stream = torch_npu.npu.current_stream(self.device)
            slot.capture = capture
            slot.stream = stream
            slot.ended = False
            slot.exported = False
            slot.record = {
                "record_type": "SPAN",
                "span_id": f"{self.worker_id}:{self._span_seq}",
                "worker_id": self.worker_id,
                "exec_ref": execution.exec_ref,
                "kind": kind,
                "clock_domain": self.clock_domain,
                "producer_id": str(threading.get_ident()),
                "stream_id": stream.npu_stream,
                "host_begin_ns": time.monotonic_ns(),
                "host_end_ns": None,
                "device_begin_ns": None,
                "device_end_ns": None,
                "generation": capture.generation,
                "complete": False,
            }
        slot.begin.record(stream)
        return Span(self, slot)

    def close_admission(self, capture: Capture) -> None:
        with self.lock:
            capture.accepting = False

    def retain_execution(self, execution: Execution) -> bool:
        with self.lock:
            if not execution.capture.admits() or self.finalized(execution.capture):
                return False
            if len(self._retained) >= SPAN_CAPACITY:
                self._dropped_spans += 1
                return False
            self._retained[execution.exec_ref] = execution
            return True

    def release_execution(self, execution: Execution) -> None:
        with self.lock:
            self._retained.pop(execution.exec_ref, None)

    def host_span(self, execution: Execution, kind: str) -> HostSpan | nullcontext:
        capture = execution.capture
        with self.lock:
            if self.finalized(capture) or (
                not capture.admits() and execution.exec_ref not in self._retained
            ):
                return nullcontext()
            if len(self._host_records) >= SPAN_CAPACITY:
                self._dropped_spans += 1
                return nullcontext()
            self._span_seq += 1
            record: dict = {
                "record_type": "SPAN",
                "span_id": f"{self.worker_id}:{self._span_seq}",
                "worker_id": self.worker_id,
                "exec_ref": execution.exec_ref,
                "kind": kind,
                "clock_domain": f"{self.worker_id}:host-monotonic",
                "producer_id": str(threading.get_ident()),
                "stream_id": "host",
                "host_begin_ns": time.monotonic_ns(),
                "host_end_ns": None,
                "device_begin_ns": None,
                "device_end_ns": None,
                "generation": capture.generation,
                "complete": False,
                "time_source": "HOST",
                "host_epoch_offset_ns": capture.host_epoch_offset_ns,
                "clock_uncertainty_ns": capture.clock_uncertainty_ns,
            }
            self._host_records[record["span_id"]] = (capture, record)
            return HostSpan(execution, record)

    def defer_metadata(self, execution: Execution, batch: torch.Tensor) -> None:
        with self.lock:
            if self.finalized(execution.capture) or (
                not execution.capture.admits()
                and execution.exec_ref not in self._retained
            ):
                self._dropped_metadata += 1
                return
            if not self._free_metadata:
                self._dropped_metadata += 1
                return
            slot = self._free_metadata.popleft()
            slot.capture = execution.capture
            slot.exec_ref = execution.exec_ref
            slot.tensor = batch
            slot.exported = False
            slot.armed = False
        # Preserve producer-stream ordering and tensor ownership through completion.
        slot.batch.copy_(batch, non_blocking=True)
        slot.ready.record(torch_npu.npu.current_stream(self.device))
        with self.lock:
            slot.armed = True

    def pending_count(self) -> int:
        with self.lock:
            return (
                len(self.slots)
                - len(self._free_slots)
                + len(self.metadata_slots)
                - len(self._free_metadata)
                + int(self._anchor_pending)
                + len(self._host_records)
                + len(self._retained)
            )

    def snapshot(self) -> dict:
        return {
            "pending_spans": self.pending_count(),
            "dropped_spans": self._dropped_spans,
            "dropped_metadata": self._dropped_metadata,
            "unresolved_spans": self._unresolved,
        }

    def _clock(self, capture: Capture) -> None:
        now = time.monotonic_ns()
        if self._anchor_pending:
            raw = self.reader.read_raw(self.anchor)
            if raw is None:
                return
            after = time.time_ns() - capture.clock_offset_ns
            if not self.finalized(capture):
                if len(self._anchors) >= MAX_DURATION_ANCHORS:
                    self._anchors.pop(0)
                self._anchors.append(
                    (raw * NS_PER_DEVICE_TICK, self._anchor_before, after)
                )
            self._anchor_pending = False
        if (
            capture.start_host_ns
            and not self.finalized(capture)
            and now >= self._next_anchor
            and (not capture.stop_requested or not self._anchors)
        ):
            self._anchor_before = time.time_ns() - capture.clock_offset_ns
            self.anchor.record(self.anchor_stream)
            self._anchor_pending = True
            self._next_anchor = now + ANCHOR_INTERVAL_NS

    def collect(self, capture: Capture) -> None:
        self._clock(capture)
        with self.lock:
            slots = [
                slot for slot in self.slots if slot.capture is not None and slot.ended
            ]
            metadata_slots = [
                slot
                for slot in self.metadata_slots
                if slot.capture is not None and slot.armed
            ]
        for slot in slots:
            begin, end = (
                self.reader.read_raw(slot.begin),
                self.reader.read_raw(slot.end),
            )
            if begin is None or end is None:
                continue
            begin, end = begin * NS_PER_DEVICE_TICK, end * NS_PER_DEVICE_TICK
            if not slot.exported:
                assert slot.record is not None and slot.capture is not None
                slot.record.update(
                    device_begin_ns=begin, device_end_ns=end, complete=True
                )
                slot.capture.writer.submit(slot.record)
                with self.lock:
                    self._range = (
                        (min(self._range[0], begin), max(self._range[1], end))
                        if self._range
                        else (begin, end)
                    )
            with self.lock:
                slot.capture = None
                slot.record = None
                self._free_slots.append(slot)
        for metadata_slot in metadata_slots:
            if not metadata_slot.ready.query():
                continue
            if not metadata_slot.exported:
                batch = metadata_slot.batch.tolist()
                assert metadata_slot.capture is not None
                base = batch[1]
                if not 0 <= base <= len(self.source_workers) - self.tp_size:
                    raise ValueError("CAM source rank outside profiling roster")
                metadata_slot.capture.writer.submit(
                    {
                        "record_type": "FFN_WORK",
                        "ffn_work_id": metadata_slot.exec_ref,
                        "layer_idx": batch[2],
                        "source_call_ids": [],
                        "source_workers": self.source_workers[
                            base : base + self.tp_size
                        ],
                        "source_relation": "temporal_source_candidate",
                        "batch_info": batch,
                    }
                )
            with self.lock:
                metadata_slot.capture = None
                metadata_slot.tensor = None
                self._free_metadata.append(metadata_slot)

    def finalized(self, capture: Capture) -> bool:
        return self._finalized_generation == capture.generation

    def finish_capture(self, capture: Capture) -> None:
        with self.lock:
            self._finalized_generation = capture.generation
            for owner, record in self._host_records.values():
                if owner is capture:
                    capture.writer.submit(dict(record))
                    self._unresolved += 1
            self._unresolved += sum(
                execution.capture is capture for execution in self._retained.values()
            )
            for slot in self.slots:
                if slot.capture is capture and not slot.exported:
                    assert slot.record is not None
                    capture.writer.submit(dict(slot.record))
                    slot.exported = True
                    self._unresolved += 1
            for metadata in self.metadata_slots:
                if metadata.capture is capture and not metadata.exported:
                    metadata.exported = True
                    self._unresolved += 1
        if self._anchors and self._range:
            raw, before, after = self._anchors[0]
            origin = (before + after) // 2
            uncertainty = (
                max(
                    abs((a + b) // 2 - (d - raw + origin)) + (b - a + 1) // 2
                    for d, a, b in self._anchors
                )
                + capture.clock_uncertainty_ns
            )
            capture.writer.submit(
                {
                    "record_type": "CLOCK_MAP",
                    "clock_domain": self.clock_domain,
                    "device_origin_ns": raw,
                    "host_origin_ns": origin,
                    "uncertainty_ns": uncertainty,
                    "valid_device_begin_ns": self._range[0],
                    "valid_device_end_ns": self._range[1],
                    "quality": "EMPIRICAL",
                    "slope_assumption": 1,
                    "anchors": self._anchors,
                }
            )
        if self.pending_count():
            capture.error = (
                capture.error or "device records unresolved at drain deadline"
            )
        capture.writer.submit(
            {
                "record_type": "SUMMARY",
                "worker_id": self.worker_id,
                "dropped_spans": self._dropped_spans,
                "dropped_metadata": self._dropped_metadata,
                "unresolved_spans": self._unresolved,
                "dropped_records": capture.writer.snapshot()["dropped_records"],
                "error": capture.error,
            }
        )
        self._anchors = []
        self._range = None
        self._next_anchor = 0


MAX_DURATION_ANCHORS = 3602
