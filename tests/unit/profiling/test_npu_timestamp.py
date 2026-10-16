# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from afd_plugin.profiling.npu import EventTimestampReader


class TimestampFunction:
    def __init__(self, status=0):
        self.status = status
        self.calls = []

    def __call__(self, handle, output):
        self.calls.append(handle.value)
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint64))[0] = 123456789
        return self.status


def make_reader(monkeypatch, status=0):
    function = TimestampFunction(status)
    library = SimpleNamespace(aclrtEventGetTimestamp=function)
    monkeypatch.setattr(ctypes, "PyDLL", lambda name: library)
    return EventTimestampReader(), function


def test_only_reads_completed_recorded_events(monkeypatch):
    reader, function = make_reader(monkeypatch)
    event = SimpleNamespace(query=lambda: False, npu_event=77)
    assert reader.read_raw(event) is None
    assert function.calls == []
    event.query = lambda: True
    assert reader.read_raw(event) == 123456789
    assert function.calls == [77]
    assert function.argtypes == [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    assert function.restype is ctypes.c_int


def test_rejects_unrecorded_event(monkeypatch):
    reader, function = make_reader(monkeypatch)
    event = SimpleNamespace(query=lambda: True, npu_event=0)
    with pytest.raises(ValueError, match="unrecorded"):
        reader.read_raw(event)
    assert function.calls == []


def test_runtime_error_is_not_replaced_with_synchronization(monkeypatch):
    reader, function = make_reader(monkeypatch, status=17)
    event = SimpleNamespace(query=lambda: True, npu_event=77)
    with pytest.raises(RuntimeError, match="status 17"):
        reader.read_raw(event)
    assert function.calls == [77]
