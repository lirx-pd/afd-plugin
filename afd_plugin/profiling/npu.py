# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Read completed, exclusively owned NPU Events without draining other streams."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_npu.npu import Event


class EventTimestampReader:
    """The collector must set its NPU device and retain Event slot ownership."""

    def __init__(self) -> None:
        self._runtime = ctypes.CDLL("libascendcl.so")
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
