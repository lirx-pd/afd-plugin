# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded Full-profiler probes; these do not cover blocked CAM receive."""

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
THREAD_TIMEOUT_S = 60
STOP_TIMEOUT_S = 30
KILL_GRACE_S = 5
MATRIX_SIZE = 4096
ACTIVE_ITERATIONS = 64
IDLE_ITERATIONS = 4
OWNER_MARKER = "CAM_FULL_OWNER"
PRODUCER_MARKER = "CAM_FULL_PRODUCER"


def _probe(mode: str, output: Path) -> None:
    import torch
    import torch_npu

    from afd_plugin.compat.npu.profiler import create_afd_npu_profiler

    output.mkdir(parents=True, exist_ok=True)
    for key, value in {
        "ENABLE": "1",
        "WAIT": "0",
        "WARMUP": "0",
        "ACTIVE": "100000",
        "REPEAT": "1",
        "SKIP_FIRST": "0",
        "DIR": str(output),
        "WITH_STACK": "0",
    }.items():
        os.environ[f"AFD_NPU_ATTENTION_PROFILER_{key}"] = value
    device = int(os.environ.get("LOCAL_RANK", "0"))
    prepared = threading.Event()
    started = threading.Event()
    submitted = threading.Event()
    errors: list[str] = []
    result = {"mode": mode, "torch_npu": torch_npu.__version__}
    completion = []

    def producer():
        try:
            torch.npu.set_device(device)
            result["producer_tid"] = threading.get_native_id()
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                a = torch.ones(
                    (MATRIX_SIZE, MATRIX_SIZE), device="npu", dtype=torch.float16
                )
                b = torch.ones_like(a)
                target = torch.empty_like(a)
                torch.mm(a, b, out=target)
                stream.synchronize()
                prepared.set()
                if not started.wait(THREAD_TIMEOUT_S):
                    raise TimeoutError("owner did not start the profiler")
                iterations = ACTIVE_ITERATIONS if mode == "active" else IDLE_ITERATIONS
                with torch.profiler.record_function(PRODUCER_MARKER):
                    for _ in range(iterations):
                        torch.mm(a, b, out=target)
                end = torch.npu.Event()
                end.record(stream)
                completion.append(end)
                if mode == "idle":
                    stream.synchronize()
                submitted.set()
                # Keep buffers and the stream alive until the owner has stopped.
                if not stopped.wait(THREAD_TIMEOUT_S):
                    raise TimeoutError("owner did not stop the profiler")
        except BaseException:
            errors.append(traceback.format_exc())
            prepared.set()
            submitted.set()

    stopped = threading.Event()

    def owner():
        try:
            torch.npu.set_device(device)
            result["owner_tid"] = threading.get_native_id()
            if not prepared.wait(THREAD_TIMEOUT_S) or errors:
                raise RuntimeError("producer preparation failed")
            profiler = create_afd_npu_profiler("attention")
            assert profiler is not None
            with torch.profiler.record_function(OWNER_MARKER):
                owner_value = torch.ones(16, device="npu")
                owner_value.add_(1)
            started.set()
            if not submitted.wait(THREAD_TIMEOUT_S) or errors:
                raise RuntimeError("producer submission failed")
            result["pending_at_stop"] = not completion[0].query()
            begin = time.monotonic()
            profiler.stop()
            result["stop_seconds"] = time.monotonic() - begin
        except BaseException:
            errors.append(traceback.format_exc())
        finally:
            stopped.set()

    producer_thread = threading.Thread(
        target=producer, name="cam-producer", daemon=True
    )
    owner_thread = threading.Thread(target=owner, name="cam-profile-owner", daemon=True)
    producer_thread.start()
    owner_thread.start()
    deadline = time.monotonic() + THREAD_TIMEOUT_S
    for thread in (owner_thread, producer_thread):
        thread.join(max(0, deadline - time.monotonic()))
    if owner_thread.is_alive() or producer_thread.is_alive():
        raise TimeoutError("probe thread exceeded its deadline")
    result["errors"] = errors
    (output / "probe-result.json").write_text(json.dumps(result), encoding="utf-8")
    if errors:
        raise RuntimeError("\n".join(errors))


def _run_probe(mode: str, output: Path) -> dict:
    with subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--probe", mode, str(output)],
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
            pytest.fail(f"Full {mode} probe exceeded {PROBE_TIMEOUT_S}s:\n{log}")
        assert process.returncode == 0, log
    (output / "probe.log").write_text(log, encoding="utf-8")
    result = json.loads((output / "probe-result.json").read_text(encoding="utf-8"))
    trace_paths = list(output.rglob("trace_view.json"))
    assert trace_paths, f"Full profiler produced no readable trace:\n{log}"
    events = []
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        events.extend(trace if isinstance(trace, list) else trace["traceEvents"])
    result["owner_markers"] = sum(event.get("name") == OWNER_MARKER for event in events)
    result["producer_markers"] = sum(
        event.get("name") == PRODUCER_MARKER for event in events
    )
    result["producer_mm"] = sum(
        event.get("name") == "aten::mm" and event.get("tid") == result["producer_tid"]
        for event in events
    )
    result["kernel_events"] = sum(
        "Task Type" in event.get("args", {}) for event in events
    )
    result["producer_mm_kernels"] = sum(
        "Task Type" in event.get("args", {}) and "aclnnMm_" in event.get("name", "")
        for event in events
    )
    (output / "probe-result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


@pytest.mark.parametrize("mode", ["idle", "active"])
def test_full_owner_captures_producer_kernels_and_stops(mode, tmp_path):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires an allocated NPU and runtime opt-in")
    pytest.importorskip("torch_npu")
    output = tmp_path / mode
    result = _run_probe(mode, output)
    assert result["owner_tid"] != result["producer_tid"]
    assert result["stop_seconds"] < STOP_TIMEOUT_S, result
    assert result["pending_at_stop"] == (mode == "active"), result
    assert result["owner_markers"] > 0, result
    assert result["kernel_events"] > 0, result
    iterations = ACTIVE_ITERATIONS if mode == "active" else IDLE_ITERATIONS
    assert result["producer_mm_kernels"] == iterations, result


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--probe":
        raise SystemExit("usage: test_cam_profile_full.py --probe idle|active OUTPUT")
    _probe(sys.argv[2], Path(sys.argv[3]))
