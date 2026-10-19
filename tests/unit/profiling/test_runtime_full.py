# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
import sys
import threading
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.profiling import native as full_module
from afd_plugin.profiling import npu as npu_module
from afd_plugin.profiling import session as session_module
from afd_plugin.profiling.session import WorkerSession
from afd_plugin.profiling.worker import CAMProfileWorkerMixin
from tests.unit.profiling.test_session import (
    FakeClock,
    FakeTracer,
    commit_command,
    finish_writer,
    prepare_command,
    stop_command,
)


class FakeFullOwner:
    mode: str

    def __init__(self, trace_dir):
        self.trace_dir = trace_dir
        self.state = "PREPARED"
        self.started_at_ns = None
        self.error = None
        self.busy = False
        self.starts = 0
        self.steps = 0
        self.stops = 0
        self.block_stop = False
        self.stop_error = None

    @property
    def finished(self):
        return self.state in {"COMPLETED", "FAILED"}

    def start(self):
        self.starts += 1
        self.state = "STARTING"
        self.busy = True

    def activate(self, started_at_ns):
        self.started_at_ns = started_at_ns
        self.state = "RUNNING"

    def step(self, *, startup_only=False):
        self.steps += 1
        if self.mode == "LIGHTWEIGHT" and self.state == "STARTING":
            self.activate(1_700_000_000_000_000_000)

    def fail(self, error):
        self.error = error
        self.state = "FAILED"
        self.busy = False

    def stop(self):
        self.stops += 1
        if self.finished:
            return
        if self.stop_error:
            self.fail(self.stop_error)
        elif self.block_stop or (self.busy and self.started_at_ns is None):
            self.state = "STOPPING"
        else:
            self.state = "COMPLETED"
            self.busy = False

    def snapshot(self):
        return {
            "state": self.state,
            "trace_dir": str(self.trace_dir),
            "started_at_ns": self.started_at_ns,
            "error": self.error,
            "busy": self.busy,
        }


@pytest.fixture
def full_session(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(session_module, "time", clock)
    monkeypatch.setattr(session_module, "PROFILER_OPERATION_TIMEOUT_S", 0.05)
    owners = []

    def factory(path, mode):
        owner = FakeFullOwner(path)
        owner.mode = mode
        owners.append(owner)
        return owner

    session = WorkerSession(
        "attention-dp0",
        "attention",
        tmp_path,
        None,
        {"pid": 7},
        tracer_factory=FakeTracer,
        profiler_factory=factory,
    )
    yield session, clock, owners
    if session.capture is not None:
        assert session.capture.writer.close(1.0)


def start_full(session, clock):
    session.prepare(prepare_command(mode="FULL", duration_s=1), backend_busy=False)
    session.commit(commit_command(session, clock))
    clock.advance(0.1)
    session.tick()
    return session.capture.profiler


def activate_full(session, clock):
    owner = start_full(session, clock)
    owner.activate(clock.time_ns())
    session.tick()
    return owner


def make_worker(session):
    class NativeWorker:
        def profile(self, is_start=True, profile_prefix=None):
            return is_start, profile_prefix

    class Worker(CAMProfileWorkerMixin, NativeWorker):
        pass

    worker = Worker()
    worker.cam_profile_session = session
    worker.profiler = None
    worker.model_runner = SimpleNamespace(prof=None)
    return worker


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("initialized", [False, True])
def test_enabled_worker_creates_tracer_only_for_lightweight(
    tmp_path, monkeypatch, role, initialized
):
    class Connector(SimpleNamespace):
        pass

    connector = Connector(
        profile_worker_id=f"{role}-dp0",
        topology=SimpleNamespace(role=role),
        world_rank=0,
        attn_size=1,
        ffn_size=1,
        tp_size=1,
        comm_args=object() if initialized else None,
        is_initialized=initialized,
    )
    module = ModuleType("afd_plugin.connectors.npu.async_cam")
    monkeypatch.setattr(module, "CAMAsyncAFDConnector", Connector, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    monkeypatch.setenv("AFD_CAM_PROFILE_DIR", str(tmp_path))
    starts = []
    waiters = []
    tracer_calls = []
    waiter = SimpleNamespace()
    tracer = FakeTracer()
    monkeypatch.setattr(
        WorkerSession, "start", lambda session, device: starts.append((session, device))
    )

    def create_waiter(device, session):
        waiters.append((device, session))
        return waiter

    monkeypatch.setattr(npu_module, "FFNDeviceWaiter", create_waiter)
    worker = CAMProfileWorkerMixin()
    worker.device = "npu:0"
    worker.model_runner = SimpleNamespace(connector=connector, prof=None)

    def create_tracer(received_connector):
        tracer_calls.append(received_connector)
        return tracer

    monkeypatch.setattr(worker, "_create_cam_tracer", create_tracer)
    worker.initialize_cam_profiling()
    session = worker.cam_profile_session
    assert connector.profile_session is session
    assert starts == [(session, worker.device)]
    assert waiters == ([(worker.device, session)] if role == "ffn" else [])
    assert session.device_waiter is (waiter if role == "ffn" else None)
    assert session.tracer is None
    assert not tracer_calls

    try:
        assert worker.cam_profile_prepare(prepare_command(mode="FULL"))["error"] is None
        assert session.tracer is None
        assert not tracer_calls
        worker.cam_profile_stop(stop_command(session))
        finish_writer(session)
        command = prepare_command()
        assert worker.cam_profile_prepare(command)["error"] is None
        assert worker.cam_profile_prepare(dict(command))["error"] is None
        assert tracer_calls == [connector]
        assert session.tracer is tracer
        assert tracer.session is session
        assert tracer.beginnings == 1
    finally:
        if session.capture is not None:
            assert session.capture.writer.close(1.0)


@pytest.mark.parametrize("mode", [None])
def test_non_full_progress_does_not_wait_for_collector_lock(full_session, mode):
    session, _, _ = full_session
    if mode is not None:
        session.prepare(prepare_command(mode=mode), backend_busy=False)
    returned = threading.Event()

    def advance():
        session.advance_profiler()
        returned.set()

    owner = threading.Thread(target=advance)
    try:
        with session._lock:
            owner.start()
            assert returned.wait(1.0), "FULL check blocked behind unrelated collection"
    finally:
        owner.join(1.0)
    assert not owner.is_alive()


def test_runtime_owner_leaves_static_runner_profiler_separate_across_modes(
    full_session, monkeypatch
):
    session, clock, _ = full_session
    worker = make_worker(session)
    calls = []
    native = SimpleNamespace(
        step=lambda: calls.append("step"), stop=lambda: calls.append("stop")
    )

    def factory(role, trace_dir, **kwargs):
        calls.append("start")
        return native

    monkeypatch.setattr(full_module, "create_runtime_npu_profiler", factory)
    session._profiler_factory = partial(full_module.RuntimeProfiler, "attention")
    for mode in ("FULL", "LIGHTWEIGHT", "FULL"):
        status = worker.cam_profile_prepare(prepare_command(mode=mode, duration_s=1))
        assert status["state"] == "PREPARED" and status["error"] is None
        if mode == "FULL":
            assert worker.model_runner.prof is None
        worker.cam_profile_commit(commit_command(session, clock))
        clock.advance(0.1)
        session.tick()
        session.advance_profiler()
        session.tick()
        assert session.state == "RUNNING"
        worker.cam_profile_stop(stop_command(session))
        session.advance_profiler()
        finish_writer(session)
        assert session.state == "COMPLETED"
    assert calls == ["start", "stop"] * 3


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_idle_full_start_preserves_runner_ownership_and_stop_boundary(
    full_session, monkeypatch, role
):
    session, clock, _ = full_session
    session.role = role
    worker = make_worker(session)
    calls = []
    native = SimpleNamespace(
        step=lambda: calls.append(("step", threading.get_ident())),
        stop=lambda: calls.append(("stop", threading.get_ident())),
    )

    def factory(received_role, trace_dir, **kwargs):
        assert received_role == role
        calls.append(("start", threading.get_ident()))
        return native

    monkeypatch.setattr(full_module, "create_runtime_npu_profiler", factory)
    monkeypatch.setattr(full_module, "time", clock)
    session._profiler_factory = partial(full_module.RuntimeProfiler, role)

    assert worker.cam_profile_status({})["state"] == "IDLE"
    worker.cam_profile_prepare(prepare_command(mode="FULL", duration_s=1))
    session.advance_profiler(allow_stop=False)
    assert worker.cam_profile_status({})["state"] == "PREPARED"
    worker.cam_profile_commit(commit_command(session, clock))
    session.tick()
    session.advance_profiler(allow_stop=False)
    worker.cam_profile_status({})
    clock.advance(0.099)
    session.tick()
    session.advance_profiler(allow_stop=False)
    worker.cam_profile_status({})
    assert not calls

    clock.advance(0.001)
    session.tick()
    owner = session.capture.profiler
    assert owner.state == "STARTING"
    worker.cam_profile_status({})
    if role == "ffn":
        assert not calls
        session.advance_profiler(allow_stop=False)
    assert owner.state == "RUNNING"
    assert owner.started_at_ns == clock.time_ns()
    assert calls == [("start", threading.get_ident())]

    session.tick()
    assert session.state == "RUNNING"
    for _ in range(2):
        worker.cam_profile_status({})
        session.advance_profiler(allow_stop=False)
    assert len(calls) == 1

    worker.cam_profile_stop(stop_command(session))
    assert owner.state == "STOPPING"
    worker.cam_profile_status({})
    session.advance_profiler(allow_stop=False)
    if role == "ffn":
        assert len(calls) == 1
        assert not owner.finished
        session.advance_profiler()
    assert owner.finished
    finish_writer(session)
    assert session.state == "COMPLETED"
    assert calls[-1] == ("stop", threading.get_ident())


def test_idle_full_start_does_not_hold_session_lock_during_native_start(
    full_session, monkeypatch
):
    session, clock, _ = full_session
    entering_native = threading.Event()
    release_native = threading.Event()
    control_completed = threading.Event()
    startup_completed = threading.Event()
    resume_runner = threading.Event()
    result = []
    native_threads = []
    native = SimpleNamespace(
        step=lambda: None, stop=lambda: native_threads.append(threading.get_ident())
    )

    def factory(role, trace_dir, **kwargs):
        native_threads.append(threading.get_ident())
        entering_native.set()
        assert release_native.wait(2.0)
        return native

    monkeypatch.setattr(full_module, "create_runtime_npu_profiler", factory)
    session._profiler_factory = partial(full_module.RuntimeProfiler, "attention")
    start_full(session, clock)

    def run_model():
        session.advance_profiler(allow_stop=False)
        startup_completed.set()
        assert resume_runner.wait(2.0)
        session.advance_profiler()

    runner = threading.Thread(target=run_model)

    def control():
        result.append(session.status())
        result.append(session.stop(stop_command(session)))
        control_completed.set()

    controller = threading.Thread(target=control)
    runner.start()
    try:
        assert entering_native.wait(1.0)
        controller.start()
        assert control_completed.wait(1.0)
        assert result[0]["state"] == "STARTING"
        assert result[1]["state"] == "STOPPING"
        release_native.set()
        assert startup_completed.wait(1.0)
        assert session.capture.profiler.state == "STOPPING"
        assert not session.capture.profiler.finished
    finally:
        release_native.set()
        resume_runner.set()
        runner.join(2.0)
        if controller.ident is not None:
            controller.join(2.0)
    assert not runner.is_alive()
    assert not controller.is_alive()
    assert session.capture.profiler.finished
    assert native_threads == [runner.ident, runner.ident]
    finish_writer(session)
    assert session.state == "COMPLETED"


def test_full_waits_for_start_time_and_backend_activation(full_session):
    session, clock, owners = full_session
    prepared = session.prepare(
        prepare_command(mode="FULL", duration_s=1), backend_busy=False
    )
    owner = owners[0]
    assert prepared["capabilities"] == ["LIGHTWEIGHT", "FULL"]
    assert not session.capture.admits()
    session.commit(commit_command(session, clock))
    session.tick()
    clock.advance(0.099)
    session.tick()
    assert owner.starts == 0
    assert session.status()["details"]["started_at_ns"] is None

    clock.advance(0.001)
    session.tick()
    assert owner.starts == 1
    assert session.state == "STARTING"
    assert not session.capture.admits()
    assert session.capture.started_at_ns is None
    session.tick()
    assert owner.starts == 1

    owner.activate(clock.time_ns())
    session.tick()
    assert session.state == "RUNNING"
    assert not session.capture.admits()
    assert session.status()["details"]["started_at_ns"] == owner.started_at_ns
    session.stop(stop_command(session))
    assert not session.capture.admits()
    finish_writer(session)
    assert session.state == "COMPLETED"


@pytest.mark.parametrize("committed", [False, True])
def test_full_stop_before_start_does_not_start_native_backend(full_session, committed):
    session, clock, owners = full_session
    session.prepare(prepare_command(mode="FULL"), backend_busy=False)
    if committed:
        session.commit(commit_command(session, clock))
    session.stop(stop_command(session))
    clock.advance(0.2)
    finish_writer(session)
    assert session.state == "COMPLETED"
    assert owners[0].starts == 0
    assert owners[0].stops == 1
    assert session.capture.started_at_ns is None


def test_full_rpc_retries_do_not_repeat_native_operations(full_session):
    session, clock, owners = full_session
    prepare = prepare_command(mode="FULL", duration_s=1)
    session.prepare(prepare, backend_busy=False)
    session.prepare(dict(prepare), backend_busy=False)
    assert len(owners) == 1
    commit = commit_command(session, clock)
    session.commit(commit)
    session.commit(dict(commit))
    clock.advance(0.1)
    session.tick()
    owner = owners[0]
    owner.activate(clock.time_ns())
    session.tick()
    stop = stop_command(session)
    session.stop(stop)
    session.stop(dict(stop))
    finish_writer(session)
    session.stop(dict(stop))
    assert owner.starts == 1
    assert owner.stops == 1


def test_full_sessions_have_independent_owners_and_outputs(full_session):
    session, clock, owners = full_session
    paths = []
    for _ in range(2):
        owner = activate_full(session, clock)
        capture = session.capture
        expected = session.output_dir / capture.session_id / session.worker_id / "full"
        assert owner.trace_dir == expected
        status = session.status()
        assert status["details"]["mode"] == "FULL"
        assert status["details"]["native_profiler"]["trace_dir"] == str(expected)
        session.stop(stop_command(session))
        finish_writer(session)
        assert session.state == "COMPLETED"
        assert session.profiler_reusable
        output = session.output_dir / capture.session_id / f"{session.worker_id}.jsonl"
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert [record["record_type"] for record in records] == ["SESSION", "SUMMARY"]
        assert records[0]["record_type"] == "SESSION"
        assert records[0]["mode"] == "FULL"
        assert records[0]["full_trace_dir"] == str(expected)
        paths.append(expected)
    assert owners[0] is not owners[1]
    assert paths[0] != paths[1]


def test_full_never_allocates_or_admits_lightweight_tracing(full_session):
    session, clock, _ = full_session

    def unexpected_tracer():
        pytest.fail("FULL must not construct the lightweight tracer")

    session._tracer_factory = unexpected_tracer
    activate_full(session, clock)
    capture = session.capture
    assert session.tracer is None
    assert not capture.accepting
    assert not capture.admits()
    capture.accepting = True
    assert not capture.admits()
    capture.accepting = False
    session.stop(stop_command(session))
    finish_writer(session)
    assert session.state == "COMPLETED"
    assert session.tracer is None


def test_lightweight_uses_native_backend_without_collector_work(full_session):
    session, clock, owners = full_session
    session.prepare(prepare_command(), backend_busy=False)
    tracer = session.tracer
    assert tracer.beginnings == 1
    assert owners[0].mode == "LIGHTWEIGHT"
    assert session.capture.profiler is owners[0]
    session.commit(commit_command(session, clock))
    clock.advance(0.1)
    session.tick()
    assert not session.capture.admits()
    session.advance_profiler()
    session.tick()
    assert session.capture.admits()
    session.stop(stop_command(session))
    finish_writer(session)
    assert session.state == "COMPLETED"
    assert tracer.collections == 0
    assert tracer.finalized_generations == {session.capture.generation}


@pytest.mark.parametrize("phase", ["start", "stop"])
@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_full_backend_failure_preserves_rpc_and_permanently_blocks_reuse(
    full_session, phase, failure
):
    session, clock, _ = full_session
    owner = start_full(session, clock)
    worker = make_worker(session)
    if phase == "stop":
        owner.activate(clock.time_ns())
        session.tick()
        if failure == "error":
            owner.stop_error = "native stop failed"
        else:
            owner.block_stop = True
        worker.cam_profile_stop(stop_command(session))
    elif failure == "error":
        owner.fail("native start failed")

    if failure == "timeout":
        assert worker.cam_profile_status({})["state"] in {"STARTING", "STOPPING"}
        clock.advance(0.051)
    finish_writer(session)
    status = worker.cam_profile_status({})
    assert status["state"] == "FAILED"
    assert phase in status["error"]
    assert status["details"]["profiler_poisoned"] is True
    assert not session.capture.admits()
    assert not session.profiler_reusable
    if failure == "timeout":
        assert "deadline" in status["error"]
        owner.fail("native operation eventually returned")
    for mode in ("LIGHTWEIGHT", "FULL"):
        with pytest.raises(ValueError, match="restart worker"):
            session.prepare(prepare_command(mode=mode), backend_busy=False)
    with pytest.raises(ValueError, match="runtime CAM"):
        worker.profile(True)
    assert worker.profile(False) == (False, None)


def test_full_done_owner_still_busy_blocks_native_and_next_session(full_session):
    session, clock, _ = full_session
    owner = activate_full(session, clock)
    session.stop(stop_command(session))
    owner.busy = True
    finish_writer(session)
    assert session.state == "COMPLETED"
    assert not session.profiler_reusable
    with pytest.raises(ValueError, match="restart worker"):
        session.prepare(prepare_command(mode="FULL"), backend_busy=False)
    with pytest.raises(ValueError, match="runtime CAM"):
        make_worker(session).profile(True)
    owner.busy = False
    assert session.profiler_reusable
    assert (
        session.prepare(prepare_command(mode="FULL"), backend_busy=False)["state"]
        == "PREPARED"
    )
