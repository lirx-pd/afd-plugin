# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector


class Execution:
    def __init__(self, tracer, exec_ref):
        self.tracer = tracer
        self.exec_ref = exec_ref


class Tracer:
    def __init__(self, *, active=True):
        self.active = active
        self.spans = []
        self.deferred = []
        self.released = []

    @contextmanager
    def span(self, execution, name):
        self.spans.append((execution.exec_ref, name))
        yield

    def execution(self, exec_ref):
        return Execution(self, exec_ref) if self.active else None

    def new_work(self):
        return self.execution("ffn-work")

    def retain_execution(self, execution):
        return True

    def release_execution(self, execution):
        self.released.append(execution)

    def record_receive(self, execution, *timing):
        self.spans.append((execution.exec_ref, "DISPATCH_RECV"))
        return True

    def defer_metadata(self, execution, source_trace, compact_metadata):
        self.deferred.append((execution, source_trace, compact_metadata))


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.delenv("AFD_CAM_OP_IO_LOG", raising=False)
    operators = SimpleNamespace(
        afd_async_dispatch_send=Mock(),
        afd_async_dispatch_recv=Mock(),
        afd_async_combine_send=Mock(),
        afd_async_combine_recv=Mock(return_value=torch.empty((2, 16))),
    )
    monkeypatch.setattr(torch.ops, "afd_ascend", operators)
    return operators


def connector(monkeypatch, *, role, enabled=True):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1" if enabled else "0")
    config = SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {"attn_ranks_per_dp": 2}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                hidden_size=16, num_experts_per_tok=2, n_routed_experts=8
            ),
        ),
    )
    afd_config = AFDConfig(
        connector="CAMAsyncAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )
    result = CAMAsyncAFDConnector(0, 0, config, afd_config, 0)
    result._initialized = True
    result.comm_args = torch.empty((1,))
    result._placeholder = torch.empty((1,))
    return result


def profile_session(tracer):
    return SimpleNamespace(
        tracer=tracer,
        advance_full=Mock(),
        device_waiter=SimpleNamespace(
            read_ffn_control=lambda control: control.tolist(),
            begin_receive=Mock(),
            end_receive=Mock(),
            begin=object(),
            end=object(),
            stream=object(),
            host_begin_ns=1,
            host_end_ns=2,
        ),
    )


def dispatch(connector, *, stage=0):
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=3, stage_idx=stage, seq_len=2
        )
    )
    connector.send_attn_output(
        torch.empty((2, 16)),
        context,
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        topk_weights=torch.ones((2, 2)),
    )
    return context


def recv_outputs(connector):
    prefix_size = 5 + connector.tp_size * (1 + connector.expert_per_rank)
    batch_info = torch.cat(
        (
            torch.arange(prefix_size, dtype=torch.int64),
            torch.tensor([1, (1 << 32) + 7, 1, (1 << 32) + 9]),
        )
    )
    return (
        torch.empty((16, 16)),
        torch.ones((16,)),
        batch_info,
        torch.ones((connector.expert_per_rank,), dtype=torch.int64),
    )


def test_dispatch_sequence_survives_idle_and_new_session_with_original_owner(
    monkeypatch, ops
):
    attn = connector(monkeypatch, role="attention")
    attn._profile_sequence = 1 << 32
    without_session = dispatch(attn)
    first_tracer = Tracer()
    attn.profile_session = SimpleNamespace(tracer=first_tracer)
    first_capture = dispatch(attn, stage=1)
    first_tracer.active = False
    idle = dispatch(attn)
    second_tracer = Tracer()
    attn.profile_session = SimpleNamespace(tracer=second_tracer)
    second_capture = dispatch(attn)

    contexts = [without_session, first_capture, idle, second_capture]
    expected = [(1 << 32) + offset for offset in range(1, 5)]
    assert [context.dispatch_seq for context in contexts] == expected
    assert [call.args[3] for call in ops.afd_async_dispatch_send.call_args_list] == (
        expected
    )
    assert without_session.profile is None
    assert idle.profile is None
    assert first_capture.profile.tracer is first_tracer
    assert second_capture.profile.tracer is second_tracer

    attn.recv_ffn_output(torch.empty((2, 16)), ubatch_idx=1, context=first_capture)

    assert first_capture.dispatch_seq == expected[1]
    assert first_tracer.spans == [
        (first_capture.profile.exec_ref, "DISPATCH_SEND"),
        (first_capture.profile.exec_ref, "COMBINE_RECV"),
    ]
    assert second_tracer.spans == [(second_capture.profile.exec_ref, "DISPATCH_SEND")]
    assert ops.afd_async_combine_recv.call_args.args[4] == 0


def test_recv_defers_tail_views_and_preserves_full_combine_metadata(monkeypatch, ops):
    ffn = connector(monkeypatch, role="ffn")
    tracer = Tracer()
    ffn.profile_session = profile_session(tracer)
    outputs = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = outputs
    batch_info = outputs[2]
    original = batch_info.clone()

    received = ffn.recv_attn_output(batch_size=2)
    states = received.context.states
    execution, source_trace, prefix = tracer.deferred.pop()

    assert execution is received.context.profile
    assert states.source_trace is source_trace
    assert source_trace.shape == (ffn.tp_size, 2)
    assert source_trace.tolist() == [[1, (1 << 32) + 7], [1, (1 << 32) + 9]]
    assert source_trace.untyped_storage().data_ptr() == batch_info.data_ptr()
    assert prefix.untyped_storage().data_ptr() == batch_info.data_ptr()
    assert source_trace.storage_offset() == len(prefix)
    assert len(prefix) == 5 + ffn.tp_size * (1 + ffn.expert_per_rank)
    assert prefix.storage_offset() == 0
    assert states.token_nums_rankid_layeridx is batch_info

    ffn.send_ffn_output(received.hidden_states, received.context)

    assert ops.afd_async_combine_send.call_args.args[2] is batch_info
    torch.testing.assert_close(batch_info, original)
    assert tracer.released == [execution]


@pytest.mark.parametrize("active_at_completion", [False, True])
def test_receive_admits_only_after_device_completion(
    monkeypatch, ops, active_at_completion
):
    ffn = connector(monkeypatch, role="ffn")
    tracer = Tracer(active=not active_at_completion)
    ffn.profile_session = profile_session(tracer)
    actual = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = actual

    def complete(control):
        assert not tracer.spans and not tracer.deferred and not tracer.released
        tracer.active = active_at_completion
        return control.tolist()

    ffn.profile_session.device_waiter.read_ffn_control = complete
    received = ffn.recv_attn_output(batch_size=2)

    assert received.context.states.token_nums_rankid_layeridx is actual[2]
    assert ops.afd_async_dispatch_recv.call_count == 1
    assert bool(received.context.profile) == active_at_completion
    assert len(tracer.deferred) == int(active_at_completion)
    assert tracer.spans == (
        [("ffn-work", "DISPATCH_RECV")] if active_at_completion else []
    )


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "idle"])
def test_no_capture_skips_metadata_views_and_keeps_full_metadata(
    monkeypatch, ops, enabled
):
    ffn = connector(monkeypatch, role="ffn", enabled=enabled)
    tracer = Tracer(active=False)
    if enabled:
        ffn.profile_session = profile_session(tracer)
    outputs = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = outputs
    batch_info = outputs[2]
    getitem = torch.Tensor.__getitem__

    def checked_getitem(tensor, item):
        assert tensor is not batch_info or item in (0, 2), (
            "inactive profiling must not slice source identities"
        )
        return getitem(tensor, item)

    monkeypatch.setattr(torch.Tensor, "__getitem__", checked_getitem)
    received = ffn.recv_attn_output(batch_size=2)
    ffn.send_ffn_output(received.hidden_states, received.context)

    assert received.context.profile is None
    assert received.context.states.source_trace is None
    assert not tracer.deferred
    assert ops.afd_async_dispatch_recv.call_args.args[2] == 0
    assert ops.afd_async_combine_send.call_args.args[2] is batch_info
    assert ops.afd_async_combine_send.call_args.args[3] == 0

    if not enabled:
        attn = connector(monkeypatch, role="attention", enabled=False)
        context = dispatch(attn)
        assert context.dispatch_seq == 0
        assert ops.afd_async_dispatch_send.call_args.args[3] == 0


def test_receive_timing_failure_does_not_abort_native_work(monkeypatch, ops):
    ffn = connector(monkeypatch, role="ffn")
    tracer = Tracer()
    tracer.record_receive = lambda *args: False
    ffn.profile_session = profile_session(tracer)
    actual = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = actual
    received = ffn.recv_attn_output(batch_size=2)
    assert received.context.profile is None
    assert received.hidden_states is actual[0]
    ffn.send_ffn_output(received.hidden_states, received.context)
    assert ops.afd_async_combine_send.call_args.args[2] is actual[2]
    assert not tracer.deferred
