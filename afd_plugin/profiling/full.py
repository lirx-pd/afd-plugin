# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native FULL lifecycle, advanced only by its model-thread owner."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from afd_plugin.compat.npu.profiler import (
    AFDNPUProfiler,
    AFDNPUProfilerRole,
    create_runtime_npu_profiler,
)


class RuntimeFullProfiler:
    def __init__(self, role: AFDNPUProfilerRole, trace_dir: Path) -> None:
        self.role = role
        self.trace_dir = trace_dir
        self.state = "PREPARED"
        self.started_at_ns: int | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._profiler: AFDNPUProfiler | None = None
        self._stepping = False

    @property
    def finished(self) -> bool:
        with self._lock:
            return self.state in {"COMPLETED", "FAILED"}

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy()

    def _busy(self) -> bool:
        return self._stepping or self.state in {"STARTING", "RUNNING", "STOPPING"}

    def start(self) -> None:
        with self._lock:
            if self.state == "PREPARED":
                self.state = "STARTING"

    def stop(self) -> None:
        with self._lock:
            if self.state in {"COMPLETED", "FAILED"}:
                return
            if self._profiler is None and not self._stepping:
                self.state = "COMPLETED"
            else:
                self.state = "STOPPING"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "trace_dir": str(self.trace_dir),
                "started_at_ns": self.started_at_ns,
                "error": self.error,
                "busy": self._busy(),
            }

    def step(self, *, startup_only: bool = False) -> None:
        with self._lock:
            if self.state in {"PREPARED", "COMPLETED", "FAILED"} or self._stepping:
                return
            self._stepping = True
        try:
            if self._profiler is None:
                profiler = create_runtime_npu_profiler(self.role, str(self.trace_dir))
                with self._lock:
                    self._profiler = profiler
                    self.started_at_ns = time.time_ns()
                    if self.state != "STOPPING":
                        self.state = "RUNNING"
            # A device-wait safe point must not enter synchronized stop.
            if startup_only:
                return
            with self._lock:
                profiler = self._profiler
                stopping = self.state == "STOPPING"
            assert profiler is not None
            if stopping:
                profiler.stop()
                with self._lock:
                    self.state = "COMPLETED"
            else:
                profiler.step()
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "FAILED"
        finally:
            with self._lock:
                self._stepping = False
