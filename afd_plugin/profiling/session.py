# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Worker-owned profiling commands and admission deadlines."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.profiling.storage import JsonRecordWriter

if TYPE_CHECKING:
    from afd_plugin.profiling.full import RuntimeFullProfiler
    from afd_plugin.profiling.tracing import DeviceTracer

MAX_DURATION_S = 3600
LEASE_GRACE_S = 60
DRAIN_TIMEOUT_S = 2
FULL_OPERATION_TIMEOUT_S = 10


def _bounded_seconds(value: int | float | None, field: str, maximum: float) -> float:
    if value is None or type(value) not in (int, float) or not 0 < value <= maximum:
        raise ValueError(f"{field} must be in (0, {maximum}]")
    return float(value)


@dataclass(slots=True)
class Capture:
    session_id: str
    generation: int
    writer: JsonRecordWriter
    start_ns: int = 0
    deadline_ns: int = 0
    start_host_ns: int = 0
    clock_offset_ns: int = 0
    clock_uncertainty_ns: int = 0
    accepting: bool = False
    stop_requested: bool = False
    drain_deadline_ns: int = 0
    error: str | None = None
    started_at_ns: int | None = None
    flush_deadline_ns: int = 0
    host_epoch_offset_ns: int = 0
    mode: str = "LIGHTWEIGHT"
    full: RuntimeFullProfiler | None = None
    full_deadline_ns: int = 0

    def admits(self) -> bool:
        now = time.monotonic_ns()
        return self.accepting and self.start_ns <= now < self.deadline_ns


class WorkerSession:
    """RPCs only change admission; the collector owns device reads and draining."""

    def __init__(
        self,
        worker_id: str,
        role: str,
        output_dir: Path,
        tracer: DeviceTracer,
        identity: dict,
        *,
        full_factory: Callable[[Path], RuntimeFullProfiler] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.role = role
        self.output_dir = output_dir
        self.tracer = tracer
        self.identity = identity
        self.capture: Capture | None = None
        self.state = "IDLE"
        self.command_seq = 0
        self._generation = 0
        self._duration_s = 0.0
        self._lock = threading.Lock()
        self._last_command: tuple[str, dict] | None = None
        self._full_factory = full_factory
        self._full_poisoned = False

    @property
    def profiler_reusable(self) -> bool:
        with self._lock:
            full = self.capture.full if self.capture is not None else None
            return not self._full_poisoned and (full is None or not full.busy)

    def _command(self, name: str, command: dict) -> bool:
        # RPC payloads cross a process boundary; reject malformed identities.
        if not isinstance(command, dict):
            raise ValueError("command must be an object")
        session_id = command.get("session_id")
        if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id:
            raise ValueError("session_id must be a canonical UUID")
        sequence = command.get("command_seq")
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("command_seq must be a positive integer")
        if self.capture is not None and self.capture.session_id == session_id:
            if sequence == self.command_seq and self._last_command == (name, command):
                return False
            if sequence <= self.command_seq:
                raise ValueError("stale or conflicting command")
        elif name != "prepare":
            raise ValueError("unknown session")
        return True

    def prepare(self, command: dict, *, backend_busy: bool) -> dict:
        with self._lock:
            if not self._command("prepare", command):
                return self._status()
            if self.state not in {"IDLE", "COMPLETED", "FAILED"}:
                raise ValueError("a profiling session is already active")
            if self._full_poisoned or (
                self.capture is not None
                and self.capture.full is not None
                and self.capture.full.busy
            ):
                raise ValueError("runtime FULL backend is unavailable; restart worker")
            if self.tracer.pending_count():
                raise ValueError("previous generation still owns device resources")
            if (
                self.capture is not None
                and not self.capture.writer.snapshot()["closed"]
            ):
                raise ValueError("previous writer has not closed")
            if backend_busy:
                raise ValueError("an existing NPU profiler is active")
            mode = command.get("mode")
            if mode not in {"LIGHTWEIGHT", "FULL"}:
                raise ValueError("mode must be LIGHTWEIGHT or FULL")
            if mode == "FULL" and self._full_factory is None:
                raise ValueError("runtime FULL backend is not configured")
            duration = _bounded_seconds(
                command.get("duration_s"), "duration_s", MAX_DURATION_S
            )
            lease = _bounded_seconds(
                command.get("lease_s", duration + LEASE_GRACE_S),
                "lease_s",
                duration + LEASE_GRACE_S,
            )
            session_dir = self.output_dir / command["session_id"]
            session_dir.mkdir(parents=True, exist_ok=True)
            writer = JsonRecordWriter(session_dir / f"{self.worker_id}.jsonl")
            self._generation += 1
            self.capture = Capture(
                command["session_id"],
                self._generation,
                writer,
                deadline_ns=time.monotonic_ns() + int(lease * 1e9),
                mode=mode,
                full=self._full_factory(session_dir / self.worker_id / "full")
                if mode == "FULL" and self._full_factory is not None
                else None,
            )
            self._duration_s = duration
            self.tracer.begin_capture()
            self.state = "PREPARED"
            self.command_seq = command["command_seq"]
            self._last_command = ("prepare", dict(command))
            return self._status()

    def commit(self, command: dict) -> dict:
        with self._lock:
            if not self._command("commit", command):
                return self._status()
            if self.state != "PREPARED":
                raise ValueError("session is not prepared")
            capture = self.capture
            assert capture is not None
            start_host_ns = command.get("start_at_ns")
            if type(start_host_ns) is not int:
                raise ValueError("start_at_ns must be an integer")
            clocks = command.get("clock_by_worker")
            if not isinstance(clocks, dict) or not isinstance(
                clock := clocks.get(self.worker_id), dict
            ):
                raise ValueError("missing worker clock estimate")
            offset = clock.get("offset_ns")
            uncertainty = clock.get("uncertainty_ns")
            if (
                type(offset) is not int
                or type(uncertainty) is not int
                or uncertainty < 0
            ):
                raise ValueError("invalid worker clock estimate")
            worker_start = start_host_ns + offset
            now_wall, now_mono = time.time_ns(), time.monotonic_ns()
            start_delay_ns = worker_start - now_wall
            if not 0 < start_delay_ns <= int(LEASE_GRACE_S * 1e9):
                raise ValueError("start_at_ns must be within the next 60 seconds")
            lease = _bounded_seconds(
                command.get("lease_s", self._duration_s + LEASE_GRACE_S),
                "lease_s",
                self._duration_s + LEASE_GRACE_S,
            )
            if int(lease * 1e9) <= start_delay_ns:
                raise ValueError("commit lease_s must extend past start_at_ns")
            capture.start_host_ns = start_host_ns
            capture.clock_offset_ns = offset
            capture.clock_uncertainty_ns = uncertainty
            capture.host_epoch_offset_ns = now_wall - now_mono - offset
            capture.start_ns = now_mono + start_delay_ns
            capture.deadline_ns = min(
                now_mono + int(lease * 1e9),
                capture.start_ns + int(self._duration_s * 1e9),
            )
            capture.writer.submit(
                {
                    "record_type": "SESSION",
                    "schema_version": 1,
                    "session_id": capture.session_id,
                    "worker_id": self.worker_id,
                    "start_host_ns": start_host_ns,
                    "generation": capture.generation,
                    "identity": self.identity,
                    "requires_summary": True,
                    "mode": capture.mode,
                    "full_trace_dir": str(capture.full.trace_dir)
                    if capture.full is not None
                    else None,
                }
            )
            capture.accepting = capture.full is None
            self.state = "STARTING"
            self.command_seq = command["command_seq"]
            self._last_command = ("commit", dict(command))
            return self._status()

    def stop(self, command: dict) -> dict:
        with self._lock:
            if not self._command("stop", command):
                return self._status()
            capture = self.capture
            assert capture is not None
            if self.state not in {"COMPLETED", "FAILED"}:
                self._stop(capture)
            self.command_seq = command["command_seq"]
            self._last_command = ("stop", dict(command))
            return self._status()

    def _stop(self, capture: Capture) -> None:
        self.tracer.close_admission(capture)
        if not capture.stop_requested:
            capture.stop_requested = True
            capture.drain_deadline_ns = time.monotonic_ns() + int(DRAIN_TIMEOUT_S * 1e9)
            if capture.full is not None:
                capture.full.stop()
                capture.full_deadline_ns = time.monotonic_ns() + int(
                    FULL_OPERATION_TIMEOUT_S * 1e9
                )
        self.state = "STOPPING"

    def status(self, command: dict | None = None) -> dict:
        with self._lock:
            return self._status()

    def start_pending_full(self) -> None:
        """Run startup on the original model thread, outside the session lock."""
        with self._lock:
            capture = self.capture
            if (
                capture is None
                or self.state != "STARTING"
                or capture.full is None
                or capture.full.state != "STARTING"
                or time.monotonic_ns() >= capture.deadline_ns
            ):
                return
            full = capture.full
        full.step(startup_only=True)

    def _status(self) -> dict:
        capture = self.capture
        return {
            **self.identity,
            "worker_id": self.worker_id,
            "role": self.role,
            "state": self.state,
            "session_id": capture.session_id if capture else None,
            "command_seq": self.command_seq,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "capabilities": ["LIGHTWEIGHT", "FULL"]
            if self._full_factory is not None
            else ["LIGHTWEIGHT"],
            "error": capture.error if capture else None,
            "details": {
                "started_at_ns": capture.started_at_ns if capture else None,
                "writer": capture.writer.snapshot() if capture else None,
                "mode": capture.mode if capture else None,
                "full": capture.full.snapshot()
                if capture is not None and capture.full is not None
                else None,
                "profiler_poisoned": self._full_poisoned,
                **self.tracer.snapshot(),
            },
        }

    def tick(self) -> None:
        """Called by the device-bound collector, never by an inference thread."""
        with self._lock:
            capture = self.capture
            if capture is None or self.state in {"COMPLETED", "FAILED"}:
                return
            now = time.monotonic_ns()
            if self.state == "STARTING" and now >= capture.start_ns:
                if capture.full is not None:
                    if capture.full.state == "PREPARED":
                        capture.full.start()
                        capture.full_deadline_ns = now + int(
                            FULL_OPERATION_TIMEOUT_S * 1e9
                        )
                    if capture.full.started_at_ns is not None:
                        capture.started_at_ns = capture.full.started_at_ns
                        capture.accepting = True
                        self.state = "RUNNING"
                else:
                    if capture.started_at_ns is None:
                        capture.started_at_ns = time.time_ns()
                    self.state = "RUNNING"
            full = capture.full
            if full is not None:
                if full.error:
                    capture.error = capture.error or full.error
                    self._full_poisoned = True
                elif (
                    capture.full_deadline_ns
                    and now >= capture.full_deadline_ns
                    and not full.done.is_set()
                    and (full.started_at_ns is None or capture.stop_requested)
                ):
                    phase = "stop" if capture.stop_requested else "start"
                    capture.error = f"FULL profiler {phase} deadline exceeded"
                    self._full_poisoned = True
            output_error = capture.writer.snapshot()["error"]
            capture.error = capture.error or output_error
            if now >= capture.deadline_ns or capture.error:
                self._stop(capture)
            if not capture.error:
                self.tracer.collect(capture)
            if not capture.stop_requested:
                return
            if full is not None and not full.done.is_set() and not self._full_poisoned:
                return
            expired = now >= capture.drain_deadline_ns
            if self.tracer.pending_count() and not expired:
                return
            if not self.tracer.finalized(capture):
                self.tracer.finish_capture(capture)
                capture.flush_deadline_ns = now + int(DRAIN_TIMEOUT_S * 1e9)
            capture.writer.close(0)
            output = capture.writer.snapshot()
            if output["closed"]:
                capture.error = capture.error or output["error"]
                self.state = "FAILED" if capture.error else "COMPLETED"
            elif now >= capture.flush_deadline_ns:
                capture.error = capture.error or "writer flush deadline exceeded"
                self.state = "FAILED"
