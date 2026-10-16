# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pipeline transitions reuse stream scopes and never allocate per-call records."""

from __future__ import annotations

import importlib
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from afd_plugin.profiling.session import Capture


@pytest.fixture
def tracer(monkeypatch):
    native = SimpleNamespace(_mark=Mock(), _mark_on_host=Mock())
    runtime = SimpleNamespace(
        _C=SimpleNamespace(
            _mstx=native, _npu_getCurrentStream=Mock(return_value=(11, 0, 20))
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", runtime)
    tracing = importlib.import_module("afd_plugin.profiling.tracing")
    monkeypatch.setattr(tracing, "torch_npu", runtime)
    tracer = tracing.DeviceTracer(SimpleNamespace(index=0), "f0")
    capture = Capture(
        "test",
        1,
        SimpleNamespace(submit=Mock(), snapshot=lambda: {"dropped_records": 0}),
        accepting=True,
        deadline_ns=time.monotonic_ns() + int(60e9),
    )
    tracer.session = SimpleNamespace(capture=capture)
    return tracer, capture, native


def test_adjacent_compute_reuses_one_track_scope_and_native_mark(tracer):
    tracer, capture, native = tracer
    first = tracer.execution()
    scope = tracer.span(first, "ATTENTION")
    for _ in range(10000):
        execution = tracer.execution()
        assert execution is first and tracer.span(execution, "GATE") is scope
        with scope:
            pass
    assert native._mark.call_count == 1
    assert native._mark.call_args.args == ("afd.cam|COMPUTE", 11, 0, 20, "afd.cam")
    assert len(tracer._streams) == 1 and len(first.scopes) == 3
    capture.writer.submit.assert_not_called()


def test_three_pipeline_states_without_stage_names_or_identity(tracer):
    tracer, capture, native = tracer
    execution = tracer.new_work()
    assert tracer.retain_execution(execution)
    for kind in (
        "ATTENTION",
        "GATE",
        "DISPATCH_SEND",
        "SHARED_EXPERT",
        "DBO_HOST_WAIT",
        "COMBINE_RECV",
        "RESTORE_MERGE",
    ):
        with tracer.span(execution, kind):
            pass
    tracer.release_execution(execution)
    assert [c.args[0] for c in native._mark.call_args_list] == [
        "afd.cam|COMPUTE",
        "afd.cam|COMMUNICATION",
        "afd.cam|COMPUTE",
        "afd.cam|WAIT",
        "afd.cam|COMPUTE",
        "afd.cam|WAIT",
    ]
    tracer.end_capture(capture)
    native._mark_on_host.assert_called_once_with("afd.cam|END", "afd.cam")
    tracer.finish_capture(capture)
    assert capture.writer.submit.call_args.args[0]["native_marks"] == 7
    assert tracer.pending_count() == 0


@pytest.mark.parametrize("kind", ["GMM_W13", "GMM_W2", "aten::mm", "OPERATOR"])
def test_operators_never_emit_native_marks(tracer, kind):
    tracer, _, native = tracer
    with tracer.span(tracer.new_work(), kind):
        pass
    native._mark.assert_not_called()


def test_streams_and_generations_keep_independent_states(tracer):
    tracer, capture, native = tracer
    first = tracer.execution()
    assert tracer.retain_execution(first)
    tracing = importlib.import_module("afd_plugin.profiling.tracing")
    tracing.torch_npu._C._npu_getCurrentStream.return_value = (12, 0, 20)
    second = tracer.execution()
    with tracer.span(first, "ATTENTION"), tracer.span(second, "FFN_COMPUTE"):
        pass
    assert native._mark.call_count == 2
    tracer.close_admission(capture)
    assert tracer.new_work() is None
    with tracer.span(first, "COMBINE_RECV"):
        pass
    tracer.release_execution(first)
    tracer.end_capture(capture)
    tracer.finish_capture(capture)
    tracer.begin_capture()
    with tracer.span(first, "ATTENTION"):
        pass
    assert native._mark.call_count == 3
    assert not tracer._streams


def test_unresolved_work_is_reported_at_drain_deadline(tracer):
    tracer, capture, _ = tracer
    tracer.retain_execution(tracer.new_work())
    tracer.finish_capture(capture)
    assert "unresolved" in capture.error
    assert tracer.snapshot()["unresolved_spans"] == 1


def test_native_failure_never_interrupts_model_execution(tracer):
    tracer, capture, native = tracer
    native._mark.side_effect = RuntimeError("failed")
    with tracer.span(tracer.new_work(), "FFN_COMPUTE"):
        completed = True
    assert completed and "native pipeline mark failed" in capture.error
    assert tracer.snapshot()["dropped_spans"] == 1


def test_exception_returns_stream_to_wait(tracer):
    tracer, _, native = tracer
    with pytest.raises(ValueError), tracer.span(tracer.new_work(), "ATTENTION"):
        raise ValueError("model failure")
    assert native._mark.call_args.args[0] == "afd.cam|WAIT"
