# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime requests for the existing runner-owned NPU profiler."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.compat.npu.profiler import (
    AFDNPUProfiler,
    AFDNPUProfilerRole,
    create_runtime_npu_profiler,
)

if TYPE_CHECKING:
    import torch

    from afd_plugin.profiling.session import WorkerSession

FFN_CONTROL_FIELDS = 3


class FFNDeviceWaiter:
    def __init__(self, device: torch.device, session: WorkerSession) -> None:
        import torch
        import torch_npu

        self.device = device
        self.session = session
        self._control = torch.empty(
            (FFN_CONTROL_FIELDS,), dtype=torch.int64, pin_memory=True
        )
        self._ready = torch_npu.npu.Event()
        # Warm the wait event before the FFN receive loop can block the device.
        self._ready.record()
        self._ready.synchronize()

    def wait_for_stream(self) -> None:
        import torch_npu

        self._ready.record(torch_npu.npu.current_stream(self.device))
        while not self._ready.query():
            self.session.start_pending_full()
            time.sleep(0)

    def read_ffn_control(self, control: torch.Tensor) -> list[int]:
        self._control.copy_(control, non_blocking=True)
        self.wait_for_stream()
        return self._control.tolist()


class RuntimeFullProfiler:
    def __init__(self, role: AFDNPUProfilerRole, trace_dir: Path) -> None:
        self.role = role
        self.trace_dir = trace_dir
        self.state = "PREPARED"
        self.started_at_ns: int | None = None
        self.error: str | None = None
        self.done = threading.Event()
        self._lock = threading.Lock()
        self._profiler: AFDNPUProfiler | None = None
        self._stepping = False
        self._stop_requested = False

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
            self._stop_requested = True
            if self._profiler is None and not self._stepping:
                self.state = "COMPLETED"
                self.done.set()
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
                    if not self._stop_requested:
                        self.state = "RUNNING"
            # A device-wait safe point must not enter synchronized stop.
            if startup_only:
                return
            with self._lock:
                profiler = self._profiler
                stopping = self._stop_requested
            assert profiler is not None
            if stopping:
                profiler.stop()
                with self._lock:
                    self.state = "COMPLETED"
                    self.done.set()
            else:
                profiler.step()
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "FAILED"
                self.done.set()
        finally:
            with self._lock:
                self._stepping = False
