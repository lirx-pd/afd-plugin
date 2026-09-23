# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm")

ubatching = importlib.import_module("afd_plugin.v1.worker.npu.ubatching")


@pytest.mark.parametrize("device_index", [0, 1])
@pytest.mark.parametrize("same_stream", [False, True])
def test_new_thread_binds_device_before_barrier_and_stream_query(
    monkeypatch, device_index, same_stream
):
    device = torch.device("npu", device_index)
    compute_stream = SimpleNamespace(device=device, stream_id=1)
    default_stream = (
        compute_stream if same_stream else SimpleNamespace(device=device, stream_id=0)
    )
    calls = []
    local = threading.local()

    def set_device(target):
        local.device = target
        calls.append(("device", target))

    def current_stream():
        calls.append(("query", local.device))
        assert local.device == device
        return default_stream

    def set_stream(stream):
        assert local.device == stream.device
        calls.append(("stream", stream))

    def ready():
        # Graph capture starts after this barrier, so device setup must precede it.
        assert calls == [("device", device)]
        calls.append(("ready", device))

    monkeypatch.setattr(torch.npu, "set_device", set_device)
    monkeypatch.setattr(torch.npu, "current_stream", current_stream)
    monkeypatch.setattr(torch.npu, "set_stream", set_stream)
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {})
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", [])
    monkeypatch.setattr(ubatching, "_DBO_CURRENT_STREAM", threading.local())
    monkeypatch.setattr(ubatching.forward_context, "_forward_context", None)
    contexts = ubatching.make_ubatch_contexts(
        2, compute_stream, [object(), object()], SimpleNamespace(wait=ready)
    )
    context = contexts[0]
    context.cpu_wait_event.set()

    def run():
        local.device = torch.device("npu", 0)
        with context:
            assert ubatching.dbo_current_stream() is compute_stream
            assert ubatching.forward_context._forward_context is context.forward_context
            assert ubatching.dbo_enabled()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run).result(timeout=5)

    expected = [("device", device), ("ready", device), ("query", device)]
    if not same_stream:
        expected.append(("stream", compute_stream))
    assert calls == expected
    assert context.cpu_signal_event.is_set()
    assert not context.cpu_wait_event.is_set()
    assert not ubatching.dbo_enabled()
    assert ubatching._CURRENT_CONTEXTS == [None, None]
