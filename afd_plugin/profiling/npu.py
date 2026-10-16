# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Read completed, exclusively owned NPU Events without draining other streams."""

from __future__ import annotations

import ctypes
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch_npu.npu import Event

    from afd_plugin.profiling.session import WorkerSession


class EventTimestampReader:
    """The collector must set its NPU device and retain Event slot ownership."""

    def __init__(self) -> None:
        # These completed-event reads are short; releasing the GIL per read can
        # starve the collector for a full Python switch interval on every event.
        self._runtime = ctypes.PyDLL("libascendcl.so")
        self._get_timestamp = self._runtime.aclrtEventGetTimestamp
        self._get_timestamp.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._get_timestamp.restype = ctypes.c_int

    def read_raw(self, event: Event) -> int | None:
        # TorchNPU query also checks that this generation's record left its queue.
        if not event.query():
            return None
        handle = event.npu_event
        if handle == 0:
            raise ValueError("cannot timestamp an unrecorded NPU Event")
        timestamp = ctypes.c_uint64()
        status = self._get_timestamp(ctypes.c_void_p(handle), ctypes.byref(timestamp))
        if status != 0:
            raise RuntimeError(f"aclrtEventGetTimestamp failed with status {status}")
        return timestamp.value


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
        self.begin = torch_npu.npu.Event(enable_timing=True)
        self.end = torch_npu.npu.Event(enable_timing=True)
        self.begin.record()
        self.end.record()
        self.end.synchronize()

    def begin_receive(self) -> None:
        import torch_npu

        self.stream = torch_npu.npu.current_stream(self.device)
        self.host_begin_ns = time.monotonic_ns()
        # The receive's begin event also fences all preceding FFN computation.
        self.begin.record(self.stream)

    def end_receive(self) -> None:
        self.end.record(self.stream)
        self.host_end_ns = time.monotonic_ns()

    def wait_for_stream(self) -> None:
        import torch_npu

        self._ready.record(torch_npu.npu.current_stream(self.device))
        # TorchNPU may still report the timing Event pending after copy completion.
        while not self._ready.query() or not self.end.query():
            self.session.advance_full(allow_stop=self.begin.query())
            # Release the GIL without adding a timer delay to every receive.
            os.sched_yield()

    def read_ffn_control(self, control: torch.Tensor) -> list[int]:
        self._control.copy_(control, non_blocking=True)
        self.wait_for_stream()
        self.session.advance_full()
        return self._control.tolist()
