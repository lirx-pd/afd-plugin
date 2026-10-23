# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph ownership contracts; fake graphs do not qualify NPU capture."""

import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
import torch  # noqa: E402

from afd_plugin.v1.worker.npu.async_ffn_model_runner import (  # noqa: E402
    AsyncNPUFFNModelRunner,
)


@pytest.fixture
def session_harness(monkeypatch):
    calls = []
    capturing = []
    connector = SimpleNamespace()

    class FakeGraph:
        fail_launch = False

        def __init__(self):
            self.operations = []

        def replay(self):
            calls.append("replay")
            if self.fail_launch:
                raise RuntimeError("partial launch")
            for operation in self.operations:
                operation()

        def reset(self):
            calls.append("reset")

    @contextmanager
    def capture(graph, *, pool, stream):
        calls.append("capture")
        capturing.append(graph)
        try:
            yield
        finally:
            capturing.pop()

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            NPUGraph=FakeGraph,
            graph=capture,
            graph_pool_handle=lambda: (1, 0),
            stream=lambda stream: nullcontext(),
            synchronize=lambda: calls.append("wait"),
        ),
        raising=False,
    )
    state = SimpleNamespace(
        token_nums_rankid_layeridx=torch.tensor([4, 0, 2, 0, 1]),
        group_list=torch.zeros(2),
        dynamic_scales=torch.ones(4),
    )
    payload = SimpleNamespace(
        hidden_states=torch.ones(4), context=SimpleNamespace(states=state)
    )

    def receive():
        calls.append("receive")
        return payload

    def execute(hidden, scales, counts, batch_info, *, retained_tensors):
        calls.append("compute")
        assert batch_info is state.token_nums_rankid_layeridx
        temporary = hidden.clone()
        retained_tensors.append(temporary)
        return temporary * 2

    def send(output, context):
        assert context is payload.context
        calls.append("send")

    connector.recv_attn_output = receive
    connector.send_ffn_output = send
    connector.capture_operation = lambda fn: capturing[-1].operations.append(fn)
    session = AsyncNPUFFNModelRunner.__new__(AsyncNPUFFNModelRunner)
    session.model = torch.nn.Linear(1, 1)
    session._layered_executor = execute
    session.connector = connector
    session._graph = None
    session._graph_capture_count = session._graph_replay_count = 0
    session._graph_pending = session._graph_closed = False
    session._graph_receiver_exited = False
    session._graph_failure = None
    session._initialize_graph(SimpleNamespace(synchronize=lambda: calls.append("wait")))
    return session, connector, calls


@pytest.fixture
def mega_session(session_harness):
    session, connector, calls = session_harness
    pending: list[tuple[int, int]] = []
    consumed: list[tuple[int, int]] = []
    completion = torch.zeros(1, dtype=torch.int64)

    def consume():
        consumed.append(pending.pop(0))

    class Executor:
        def __init__(self):
            self.completion = completion

        def __call__(self, *args):
            connector.capture_operation(consume)
            return self.completion

    session._mega_executor = Executor()
    connector.comm_args, connector.group_name = None, "fake-cam"
    return session, pending, consumed, completion, calls


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("count", [1, 5, 43])
def test_mega_graph_consumes_exactly_the_available_transactions(
    mega_session, monkeypatch, batch_size, count
):
    session, pending, consumed, completion, _ = mega_session
    monkeypatch.setattr(
        "afd_plugin.v1.worker.npu.async_ffn_model_runner.MEGA_GRAPH_BATCH_SIZE",
        batch_size,
    )
    session._capture_graph()
    for _ in range(2):
        # Transactions are chunks selected at replay time, including repeated layers.
        pending.extend((i % 3, i // 3) for i in range(count))
        expected = consumed + pending
        completion.fill_(-9)
        session._submit_graph(count)
        assert session._complete_graph()
        assert consumed == expected and pending == []
    session._graph_receiver_exited = True
    session._close_graph()
    session._close_graph()
    with pytest.raises(RuntimeError, match="closed"):
        session._submit_graph()


def test_mega_graph_status_failure_blocks_retry_and_release(mega_session):
    session, pending, _, completion, calls = mega_session
    session._capture_graph()
    pending.extend((i, 0) for i in range(5))
    session._submit_graph(5)
    completion.fill_(-2)
    with pytest.raises(RuntimeError, match="status -2"):
        session._complete_graph()
    session._graph_receiver_exited = True
    for operation in (
        session._submit_graph,
        session._complete_graph,
        session._close_graph,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    assert "reset" not in calls


def test_replay_retains_transaction_until_receiver_exit(session_harness):
    session, _, calls = session_harness
    session._capture_graph()
    temporary = weakref.ref(session._graph_tensors[0])
    for _ in range(3):
        session._submit_graph()
        assert session._complete_graph()
    assert calls.count("receive") == calls.count("compute") == calls.count("send") == 1
    assert calls.count("replay") == 3 and temporary() is not None
    with pytest.raises(RuntimeError, match="receiver exit"):
        session._close_graph()
    session._graph_receiver_exited = True
    session._close_graph()
    session._close_graph()
    assert calls.count("reset") == 1 and temporary() is None


def test_pending_device_work_prevents_replay_and_release(session_harness):
    session, _, calls = session_harness
    session._capture_graph()
    session._submit_graph()
    with pytest.raises(RuntimeError, match="not ready"):
        session._submit_graph()
    session._graph_receiver_exited = True
    with pytest.raises(RuntimeError, match="device completion"):
        session._close_graph()
    assert calls.count("replay") == 1 and "reset" not in calls


@pytest.mark.parametrize("count", [1, 43])
def test_batch_waits_once_without_host_metadata_or_store_poll(session_harness, count):
    session, _, calls = session_harness
    session._capture_graph()
    session._submit_graph(count)
    assert session._complete_graph()
    assert calls.count("replay") == session._graph_replay_count == count
    assert calls.count("wait") == 1
    assert calls[-1] == "wait"


@pytest.mark.parametrize("failure", ["submit", "completion"])
def test_failure_retains_resources_without_retry(session_harness, monkeypatch, failure):
    session, _, calls = session_harness
    session._capture_graph()
    if failure == "submit":
        session._graph.fail_launch = True
        operation = session._submit_graph
    else:
        session._submit_graph()

        def fail():
            raise RuntimeError("device synchronization failed")

        monkeypatch.setattr(torch.npu, "synchronize", fail)
        operation = session._complete_graph
    with pytest.raises(RuntimeError):
        operation()
    session._graph_receiver_exited = True
    for operation in (
        session._submit_graph,
        session._complete_graph,
        session._close_graph,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    assert session._graph_payload is not None and session._graph_output is not None
    assert "reset" not in calls


def test_partial_capture_never_retries_or_releases(session_harness):
    session, connector, calls = session_harness

    def fail_send(*args):
        raise RuntimeError("capture failed after receive")

    connector.send_ffn_output = fail_send
    with pytest.raises(RuntimeError, match="capture failed"):
        session._capture_graph()
    for operation in (
        session._capture_graph,
        session._submit_graph,
        session._close_graph,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    assert session._graph_payload is not None and session._graph_output is not None
    assert "replay" not in calls and "reset" not in calls
