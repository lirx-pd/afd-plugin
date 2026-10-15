# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime activation of the existing public TorchNPU profiler."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

pytestmark = pytest.mark.npu
PROBE_TIMEOUT_S = 120
KILL_GRACE_S = 5
MATRIX_SIZE = 256
ITERATIONS = 4
BUSY_MATRIX_SIZE = 4096
BUSY_ITERATIONS = 1024
CPU_MARKER = "CAM_RUNTIME_FULL_PRODUCER"


def _probe(output: Path) -> None:
    import torch
    import torch_npu

    from afd_plugin.profiling.full import RuntimeFullProfiler

    for key, value in {
        "ENABLE": "0",
        "DIR": str(output / "unused-static-dir"),
        "WITH_STACK": "0",
    }.items():
        os.environ[f"AFD_NPU_ATTENTION_PROFILER_{key}"] = value
    results: list[dict] = []
    errors: list[str] = []

    def producer() -> None:
        try:
            device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
            torch_npu.npu.set_device(device)
            matrix = torch.ones(
                (MATRIX_SIZE, MATRIX_SIZE), device=device, dtype=torch.float16
            )
            target = torch.empty_like(matrix)
            torch.mm(matrix, matrix, out=target)
            torch_npu.npu.synchronize()
            for mode in ("idle", "active"):
                runtime = RuntimeFullProfiler("attention", output / mode)
                runtime.start()
                runtime.step()
                assert runtime.state == "RUNNING", runtime.snapshot()
                marker = f"{CPU_MARKER}_{mode}"
                with torch.profiler.record_function(marker):
                    if mode == "active":
                        for _ in range(ITERATIONS):
                            torch.mm(matrix, matrix, out=target)
                begin = time.monotonic()
                runtime.stop()
                runtime.step()
                assert runtime.done.is_set(), runtime.snapshot()
                assert runtime.state == "COMPLETED", runtime.snapshot()
                assert not runtime.busy, runtime.snapshot()
                traces = list(runtime.trace_dir.rglob("trace_view.json"))
                assert traces, runtime.snapshot()
                events = []
                for path in traces:
                    trace = json.loads(path.read_text())
                    events.extend(
                        trace if isinstance(trace, list) else trace["traceEvents"]
                    )
                cpu_markers = sum(event.get("name") == marker for event in events)
                producer_mm = sum(
                    event.get("name") == "aten::mm"
                    and event.get("tid") == threading.get_native_id()
                    for event in events
                )
                mm_kernels = sum(
                    "Task Type" in event.get("args", {})
                    and "aclnnMm_" in event.get("name", "")
                    for event in events
                )
                assert cpu_markers > 0, (mode, traces)
                if mode == "active":
                    assert producer_mm == ITERATIONS, producer_mm
                    assert mm_kernels == ITERATIONS, mm_kernels
                results.append(
                    {
                        "mode": mode,
                        "status": runtime.snapshot(),
                        "producer_tid": threading.get_native_id(),
                        "cpu_markers": cpu_markers,
                        "producer_mm": producer_mm,
                        "mm_kernels": mm_kernels,
                        "stop_seconds": time.monotonic() - begin,
                        "trace_paths": [str(path) for path in traces],
                    }
                )
        except BaseException:
            errors.append(traceback.format_exc())

    producer()
    assert not errors, "\n".join(errors)
    assert len({result["status"]["trace_dir"] for result in results}) == 2
    (output / "probe-result.json").write_text(json.dumps(results, indent=2))


def _probe_cooperative(output: Path) -> None:
    import uuid
    from functools import partial

    import torch
    import torch_npu

    from afd_plugin.profiling.full import FFNDeviceWaiter, RuntimeFullProfiler
    from afd_plugin.profiling.session import WorkerSession

    os.environ["AFD_NPU_FFN_PROFILER_ENABLE"] = "0"
    os.environ["AFD_NPU_FFN_PROFILER_WITH_STACK"] = "0"
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch_npu.npu.set_device(device)

    def unexpected_tracer():
        raise AssertionError("FULL must not allocate LIGHTWEIGHT resources")

    session = WorkerSession(
        "ffn-test",
        "ffn",
        output,
        None,
        {},
        full_factory=partial(RuntimeFullProfiler, "ffn"),
        tracer_factory=unexpected_tracer,
    )
    waiter = FFNDeviceWaiter(device, session)
    session.device_waiter = waiter
    session.start(device)
    matrix = torch.ones(
        (BUSY_MATRIX_SIZE, BUSY_MATRIX_SIZE), dtype=torch.float16, device=device
    )
    target = torch.empty_like(matrix)
    control = torch.tensor([123, 5, 17], dtype=torch.int64, device=device)
    torch.mm(matrix, matrix, out=target)
    pending = torch_npu.npu.Event()
    pending.record()
    torch_npu.npu.synchronize(device)
    session_id = str(uuid.uuid4())
    session.prepare(
        {
            "session_id": session_id,
            "command_seq": 1,
            "mode": "FULL",
            "duration_s": 30,
        },
        backend_busy=False,
    )
    session.commit(
        {
            "session_id": session_id,
            "command_seq": 2,
            "start_at_ns": time.time_ns() + 10_000_000,
            "clock_by_worker": {"ffn-test": {"offset_ns": 0, "uncertainty_ns": 0}},
        }
    )
    runtime = session.capture.full
    observations: list[dict] = []
    wait_calls = 0
    diagnostics = {
        "start_ns": session.capture.start_ns,
        "deadline_ns": session.capture.deadline_ns,
    }
    start_pending_full = session.start_pending_full
    wait_for_stream = waiter.wait_for_stream

    def observe_copy() -> None:
        diagnostics["pending_after_copy"] = not pending.query()
        wait_for_stream()

    def observe_start() -> None:
        nonlocal wait_calls
        wait_calls += 1
        pending_before = not pending.query()
        if wait_calls == 1:
            diagnostics["pending_after_event_record"] = pending_before
            diagnostics["first_full_state"] = runtime.state
            diagnostics["first_session_state"] = session.state
        start_pending_full()
        if runtime.started_at_ns is not None and not observations:
            session.tick()
            observations.append(
                {
                    "pending_before_start": pending_before,
                    "pending_after_start": not pending.query(),
                    "producer_tid": threading.get_native_id(),
                    "session_state": session.state,
                }
            )

    session.start_pending_full = observe_start
    waiter.wait_for_stream = observe_copy
    queue_begin = time.monotonic()
    for _ in range(BUSY_ITERATIONS):
        torch.mm(matrix, matrix, out=target)
    pending.record()
    diagnostics["pending_before_copy"] = not pending.query()
    assert diagnostics["pending_before_copy"], "work completed before receive-wait"
    read_begin = time.monotonic()
    assert waiter.read_ffn_control(control) == [123, 5, 17]
    diagnostics.update(
        wait_calls=wait_calls,
        queue_seconds=read_begin - queue_begin,
        read_seconds=time.monotonic() - read_begin,
    )
    assert observations, {"diagnostics": diagnostics, "status": session.status()}
    assert observations[0]["pending_before_start"], observations
    assert observations[0]["pending_after_start"], observations
    assert observations[0]["session_state"] == "RUNNING", observations
    assert observations[0]["producer_tid"] == threading.get_native_id()
    assert pending.query(), "control data became readable before work completed"
    marker = f"{CPU_MARKER}_cooperative"
    with torch.profiler.record_function(marker):
        torch.mm(matrix, matrix, out=target)
    session.stop({"session_id": session_id, "command_seq": 3})
    runtime.step()
    assert runtime.state == "COMPLETED", runtime.snapshot()
    deadline = time.monotonic() + 5
    while session.state not in {"COMPLETED", "FAILED"}:
        session.tick()
        assert time.monotonic() < deadline, session.status()
        time.sleep(0.001)
    assert session.state == "COMPLETED", session.status()
    assert session.tracer is None
    records = [
        json.loads(line)
        for path in output.rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert records
    assert all(record["record_type"] in {"SESSION", "SUMMARY"} for record in records)
    traces = list(runtime.trace_dir.rglob("trace_view.json"))
    assert traces, runtime.snapshot()
    events = []
    for path in traces:
        trace = json.loads(path.read_text())
        events.extend(trace if isinstance(trace, list) else trace["traceEvents"])
    assert any(event.get("name") == marker for event in events), traces
    assert any(
        event.get("name") == "aten::mm"
        and event.get("tid") == threading.get_native_id()
        for event in events
    ), traces
    assert any(
        "Task Type" in event.get("args", {}) and "aclnnMm_" in event.get("name", "")
        for event in events
    ), traces
    (output / "probe-result.json").write_text(
        json.dumps(
            {
                "startup": observations[0],
                "diagnostics": diagnostics,
                "status": session.status(),
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("probe", ["--probe", "--cooperative-probe"])
def test_runtime_full_preserves_public_profiler_outputs(tmp_path, probe):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires an allocated NPU and runtime opt-in")
    pytest.importorskip("torch_npu")
    with subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), probe, str(tmp_path)],
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
            (tmp_path / "probe.log").write_text(log)
            pytest.fail(f"Runtime FULL exceeded its deadline:\n{log}")
        (tmp_path / "probe.log").write_text(log)
        assert process.returncode == 0, log
    assert (tmp_path / "probe-result.json").exists()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in {"--probe", "--cooperative-probe"}:
        raise SystemExit(
            "usage: test_cam_profile_runtime_full.py --[cooperative-]probe OUTPUT"
        )
    (_probe_cooperative if sys.argv[1] == "--cooperative-probe" else _probe)(
        Path(sys.argv[2])
    )
