# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolated DeviceTracer probes with hard process deadlines."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from afd_plugin.profiling.model import host_span, span

PROBE_TIMEOUT_S = 120
KILL_GRACE_S = 5
WAIT_TIMEOUT_S = 10
START_DELAY_NS = 50_000_000
BUSY_ITERATIONS = 512
METADATA_ITERATIONS = 20
GENERATION_COUNT = 3
WORKER_ID = "ffn-test"
SOURCE_WORKERS = ("attn-0", "attn-1", "attn-2", "attn-3")

pytestmark = pytest.mark.npu


def _wait(condition, message: str, timeout: float = WAIT_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        time.sleep(0.001)


def _begin(session) -> Path:
    session_id = str(uuid.uuid4())
    session.prepare(
        {
            "session_id": session_id,
            "command_seq": 1,
            "mode": "LIGHTWEIGHT",
            "duration_s": 30,
        },
        backend_busy=False,
    )
    session.commit(
        {
            "session_id": session_id,
            "command_seq": 2,
            "start_at_ns": time.time_ns() + START_DELAY_NS,
            "clock_by_worker": {WORKER_ID: {"offset_ns": 0, "uncertainty_ns": 0}},
        }
    )
    _wait(session.capture.admits, "capture did not open admission")
    return session.output_dir / session_id / f"{WORKER_ID}.jsonl"


def _finish(session) -> dict:
    session.stop({"session_id": session.capture.session_id, "command_seq": 3})
    _wait(
        lambda: session.status()["state"] in {"COMPLETED", "FAILED"},
        "session did not finish",
    )
    return session.status()


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _source_call(session, execution) -> None:
    session.capture.writer.submit(
        {
            "record_type": "SOURCE_CALL",
            "source_call_id": execution.exec_ref,
            "group_id": "test-group",
            "stage_idx": 0,
            "layer_idx": 0,
            "fragment_ids": [],
        }
    )


def _probe_streams(torch, tracer, session, output: Path) -> dict:
    from afd_plugin.profiling.analyze import export_chrome_trace, reconstruct

    matrix = torch.ones((4096, 4096), dtype=torch.float16, device="npu")
    ready_target, busy_target = torch.empty_like(matrix), torch.empty_like(matrix)
    torch.mm(matrix, matrix, out=ready_target)
    torch.npu.synchronize()
    ready, busy = torch.npu.Stream(), torch.npu.Stream()
    # Bind admission while delaying the collector until both streams are queued.
    tracer.session = session
    path = _begin(session)
    ready_execution = tracer.execution(f"{WORKER_ID}:call:1")
    busy_execution = tracer.execution(f"{WORKER_ID}:call:2")
    _source_call(session, ready_execution)
    _source_call(session, busy_execution)
    with torch.npu.stream(ready), span(ready_execution, "FFN_COMPUTE"):
        torch.mm(matrix, matrix, out=ready_target)
    _wait(ready.query, "ready stream did not complete")
    with torch.npu.stream(busy), span(busy_execution, "FFN_COMPUTE"):
        for _ in range(BUSY_ITERATIONS):
            torch.mm(matrix, matrix, out=busy_target)
    assert not busy.query(), "busy stream completed before the collector probe"
    begin = time.monotonic()
    # Starting after both submissions makes the two-stream condition observable.
    session.start(tracer.device)
    _wait(
        lambda: any(
            record["record_type"] == "SPAN"
            and record["exec_ref"] == ready_execution.exec_ref
            for record in _records(path)
        ),
        "completed span was not exported",
    )
    elapsed = time.monotonic() - begin
    busy_pending = not busy.query()
    assert busy_pending, "collector waited for unrelated device work"
    assert elapsed < 0.1, elapsed
    _wait(busy.query, "busy stream did not complete")
    status = _finish(session)
    assert status["state"] == "COMPLETED", status
    records = _records(path)
    reconstructed = reconstruct(records)
    assert reconstructed["data_quality"] == "COMPLETE", reconstructed["errors"]
    spans = list(reconstructed["spans"].values())
    assert len(spans) == 2
    assert len({span["stream_id"] for span in spans}) == 2
    assert all(span["timing"]["duration_ns"] > 0 for span in spans)
    (output / "trace.json").write_text(json.dumps(export_chrome_trace(records)))
    return {
        "ready_collected_while_busy": busy_pending,
        "collector_seconds": elapsed,
        "status": status,
        "span_count": len(spans),
    }


def _probe_lazy_prepare(torch, tracer, session, output: Path) -> dict:
    from afd_plugin.profiling.analyze import reconstruct

    assert tracer is None and session.tracer is None
    assert session.status()["capabilities"] == ["LIGHTWEIGHT"]
    matrix = torch.ones((4096, 4096), dtype=torch.float16, device="npu")
    ready_target, busy_target = torch.empty_like(matrix), torch.empty_like(matrix)
    torch.mm(matrix, matrix, out=ready_target)
    torch.npu.synchronize()
    busy = torch.npu.Stream()
    with torch.npu.stream(busy):
        for _ in range(BUSY_ITERATIONS):
            torch.mm(matrix, matrix, out=busy_target)
    pending_before = not busy.query()
    assert pending_before, "unrelated work completed before lazy PREPARE"
    session_id = str(uuid.uuid4())
    begin = time.monotonic()
    session.prepare(
        {
            "session_id": session_id,
            "command_seq": 1,
            "mode": "LIGHTWEIGHT",
            "duration_s": 30,
        },
        backend_busy=False,
    )
    elapsed = time.monotonic() - begin
    pending_after = not busy.query()
    assert pending_after, "lazy PREPARE waited for unrelated device work"
    _wait(busy.query, "unrelated finite work did not complete")
    tracer = session.tracer
    assert tracer is not None and session.capture.full is None
    session.start(tracer.device)
    session.commit(
        {
            "session_id": session_id,
            "command_seq": 2,
            "start_at_ns": time.time_ns() + START_DELAY_NS,
            "clock_by_worker": {WORKER_ID: {"offset_ns": 0, "uncertainty_ns": 0}},
        }
    )
    _wait(session.capture.admits, "lazy capture did not open admission")
    execution = tracer.execution(f"{WORKER_ID}:call:1")
    assert execution is not None
    _source_call(session, execution)
    with span(execution, "FFN_COMPUTE"):
        torch.mm(matrix, matrix, out=ready_target)
    status = _finish(session)
    assert status["state"] == "COMPLETED", status
    path = session.output_dir / session_id / f"{WORKER_ID}.jsonl"
    reconstructed = reconstruct(_records(path))
    assert reconstructed["data_quality"] == "COMPLETE", reconstructed["errors"]
    assert len(reconstructed["spans"]) == 1
    recorded = next(iter(reconstructed["spans"].values()))
    assert recorded["timing"]["duration_ns"] > 0, recorded
    return {
        "pending_before_prepare": pending_before,
        "pending_after_prepare": pending_after,
        "prepare_seconds": elapsed,
        "status": status,
    }


def _probe_metadata(torch, tracer, session, output: Path) -> dict:
    from afd_plugin.profiling.analyze import reconstruct

    session.start(tracer.device)
    batch = torch.empty((11,), dtype=torch.int64, device="npu")
    value = torch.ones(16, device="npu")
    results = []
    for generation in range(1, GENERATION_COUNT + 1):
        path = _begin(session)
        expected = {}
        for index in range(METADATA_ITERATIONS):
            execution = tracer.new_work()
            assert execution is not None
            batch.copy_(torch.tensor([2, 2, index, 0, 0, 0, 0, 0, 0, 0, 0]))
            with span(execution, "FFN_COMPUTE"):
                value.add_(1)
            tracer.defer_metadata(execution, batch)
            tracer.release_execution(execution)
            expected[execution.exec_ref] = ["attn-2", "attn-3"]
            # The producer may reuse its wire buffers after the ordered D2H copy.
            batch.fill_(-1)
            _wait(
                lambda: tracer.pending_count() == 0, "metadata slot was not reclaimed"
            )
        status = _finish(session)
        assert status["state"] == "COMPLETED", status
        reconstructed = reconstruct(_records(path))
        assert len(reconstructed["spans"]) == METADATA_ITERATIONS
        assert {span["generation"] for span in reconstructed["spans"].values()} == {
            generation
        }
        assert {
            work_id: work["source_workers"]
            for work_id, work in reconstructed["ffn_works"].items()
        } == expected
        assert {
            work["layer_idx"] for work in reconstructed["ffn_works"].values()
        } == set(range(METADATA_ITERATIONS))
        assert all(
            len(work["batch_info"]) == 11
            for work in reconstructed["ffn_works"].values()
        )
        # This FFN-only probe intentionally has no Attention SOURCE_CALL records.
        assert reconstructed["data_quality"] == "PARTIAL"
        assert len(reconstructed["errors"]) == METADATA_ITERATIONS
        assert all(
            "missing aligned receive" in error for error in reconstructed["errors"]
        )
        results.append(
            {
                "generation": generation,
                "metadata_records": len(expected),
                "status": status,
            }
        )
    return {"generations": results}


def _probe_stop(torch, tracer, session, output: Path, *, host: bool = False) -> dict:
    session.start(tracer.device)
    path = _begin(session)
    execution = tracer.execution(f"{WORKER_ID}:call:1")
    assert execution is not None
    _source_call(session, execution)
    if host:
        assert tracer.retain_execution(execution)
    scope = (
        host_span(execution, "HOST_RECEIVE") if host else span(execution, "FFN_COMPUTE")
    )
    scope.__enter__()
    value = torch.ones(16, device="npu")
    value.add_(1)
    begin = time.monotonic()
    status = _finish(session)
    elapsed = time.monotonic() - begin
    assert status["state"] == "FAILED", status
    assert elapsed < 3, elapsed
    assert status["details"]["unresolved_spans"] == (2 if host else 1), status
    assert tracer.pending_count() > 0
    with pytest.raises(ValueError, match="previous generation still owns"):
        _begin(session)
    before = path.read_text()
    spans = [record for record in _records(path) if record["record_type"] == "SPAN"]
    assert len(spans) == 1 and not spans[0]["complete"]
    scope.__exit__(None, None, None)
    if host:
        assert tracer.pending_count() > 0
        with pytest.raises(ValueError, match="previous generation still owns"):
            _begin(session)
        tracer.release_execution(execution)
    _wait(
        lambda: tracer.pending_count() == 0, "late span did not release its resources"
    )
    assert path.read_text() == before, "late completion changed finalized output"
    assert session.capture.writer.snapshot()["dropped_records"] == 0
    next_path = _begin(session)
    next_execution = tracer.execution(f"{WORKER_ID}:call:2")
    assert next_execution is not None
    _source_call(session, next_execution)
    with span(next_execution, "FFN_COMPUTE"):
        value.add_(1)
    next_status = _finish(session)
    assert next_status["state"] == "COMPLETED", next_status
    next_spans = [
        record for record in _records(next_path) if record["record_type"] == "SPAN"
    ]
    assert len(next_spans) == 1 and next_spans[0]["generation"] == 2
    return {"stop_seconds": elapsed, "status": status, "next_status": next_status}


def _probe_host_stop(torch, tracer, session, output: Path) -> dict:
    return _probe_stop(torch, tracer, session, output, host=True)


def _probe_short(torch, tracer, session, output: Path) -> dict:
    from afd_plugin.profiling.analyze import export_chrome_trace, reconstruct

    tracer.session = session
    value = torch.ones(16, device="npu")
    value.add_(1)
    torch.npu.current_stream().synchronize()
    path = _begin(session)
    execution = tracer.execution(f"{WORKER_ID}:call:1")
    assert execution is not None
    _source_call(session, execution)
    with host_span(execution, "HOST_SUBMIT"), span(execution, "FFN_COMPUTE"):
        value.add_(1)
    # Stop before the first collector tick to exercise the missing-anchor case.
    status = session.stop({"session_id": session.capture.session_id, "command_seq": 3})
    assert status["details"]["started_at_ns"] is not None, status
    assert not tracer._anchors
    session.start(tracer.device)
    _wait(
        lambda: session.status()["state"] in {"COMPLETED", "FAILED"},
        "short capture did not finish",
    )
    status = session.status()
    assert status["state"] == "COMPLETED", status
    records = _records(path)
    reconstructed = reconstruct(records)
    assert reconstructed["data_quality"] == "COMPLETE", reconstructed["errors"]
    spans = list(reconstructed["spans"].values())
    assert len(spans) == 2
    host = next(span for span in spans if span.get("time_source") == "HOST")
    assert host["device_begin_ns"] is None and host["device_end_ns"] is None
    assert host["timing"]["duration_ns"] == host["host_end_ns"] - host["host_begin_ns"]
    assert host["timing"]["duration_ns"] > 0
    assert host["timing"]["aligned"]
    trace = export_chrome_trace(records)
    host_events = [
        event for event in trace["traceEvents"] if event.get("cat") == "CAM host stage"
    ]
    assert len(host_events) == 1
    assert host_events[0]["dur"] == host["timing"]["duration_ns"] / 1000
    assert host_events[0]["ts"] == host["timing"]["relative_begin_us"]
    (output / "trace.json").write_text(json.dumps(trace))
    return {"status": status, "host_duration_ns": host["timing"]["duration_ns"]}


def _probe(case: str, output: Path) -> None:
    import torch
    import torch_npu

    from afd_plugin.profiling.session import WorkerSession
    from afd_plugin.profiling.tracing import DeviceTracer

    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch_npu.npu.set_device(device)
    output.mkdir(parents=True, exist_ok=True)
    tracer = (
        None
        if case == "lazy_prepare"
        else DeviceTracer(device, WORKER_ID, list(SOURCE_WORKERS), 2, 2)
    )
    session = WorkerSession(
        WORKER_ID,
        "ffn",
        output,
        tracer,
        {},
        tracer_factory=lambda: DeviceTracer(
            device, WORKER_ID, list(SOURCE_WORKERS), 2, 2
        ),
    )
    probes: dict[str, Callable[..., dict]] = {
        "streams": _probe_streams,
        "metadata": _probe_metadata,
        "stop": _probe_stop,
        "short": _probe_short,
        "host_stop": _probe_host_stop,
        "lazy_prepare": _probe_lazy_prepare,
    }
    result = probes[case](torch, tracer, session, output)
    (output / "probe-result.json").write_text(json.dumps(result, indent=2))


@pytest.mark.parametrize(
    "case", ["streams", "metadata", "stop", "short", "host_stop", "lazy_prepare"]
)
def test_device_tracer_component(case, tmp_path):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires an allocated NPU and runtime opt-in")
    pytest.importorskip("torch_npu")
    output = tmp_path / case
    output.mkdir()
    with subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--probe", case, str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            log, _ = process.communicate(timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                log, _ = process.communicate(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                log, _ = process.communicate(timeout=KILL_GRACE_S)
            (output / "probe.log").write_text(log)
            pytest.fail(f"DeviceTracer {case} probe exceeded its deadline:\n{log}")
        (output / "probe.log").write_text(log)
        assert process.returncode == 0, log
    assert (output / "probe-result.json").exists()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--probe":
        raise SystemExit("usage: test_cam_profile_tracer.py --probe CASE OUTPUT")
    _probe(sys.argv[2], Path(sys.argv[3]))
