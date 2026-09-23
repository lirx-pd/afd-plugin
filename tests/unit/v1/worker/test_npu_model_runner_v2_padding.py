# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU coverage of native V2 DP dispatch under CAMP padding."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")
pytest.importorskip("torch_npu")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils, dp_utils  # noqa: E402
from vllm.v1.worker.gpu import model_runner as native_v2  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as afd_v2  # noqa: E402


def _manager(mode=CUDAGraphMode.NONE):
    manager = object.__new__(cudagraph_utils.ModelCudaGraphManager)
    manager._graphs_captured = mode != CUDAGraphMode.NONE
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    manager._candidates = {}
    if mode != CUDAGraphMode.NONE:
        descriptor = cudagraph_utils.BatchExecutionDescriptor(
            cg_mode=mode,
            num_tokens=16,
            num_reqs=1,
        )
        manager._candidates = {(n, 0): [descriptor] for n in (6, 8)}
    return manager


def _mock_dp_collective(monkeypatch, counts, mode=CUDAGraphMode.NONE):
    calls = []
    cpu_group = object()
    monkeypatch.setattr(
        dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=cpu_group)
    )

    def all_reduce(tensor, *, group):
        assert group is cpu_group
        assert tensor.device.type == "cpu"
        calls.append(tensor.clone())
        tensor[0] = torch.tensor(counts, dtype=torch.int32)
        tensor[1].fill_(mode.value)
        tensor[2].zero_()

    monkeypatch.setattr(dp_utils.dist, "all_reduce", all_reduce)
    return calls


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("need_eager", [False, True])
def test_camp_v2_eager_pads_native_descriptor_and_dp_counts(
    monkeypatch, rank, need_eager
):
    calls = _mock_dp_collective(monkeypatch, [6, 8])
    original = native_v2.dispatch_cg_and_sync_dp
    with afd_v2._use_camp2p_eager_dp_padding():
        assert inspect.signature(
            native_v2.dispatch_cg_and_sync_dp, eval_str=True
        ) == inspect.signature(original, eval_str=True)
        descriptor, counts = native_v2.dispatch_cg_and_sync_dp(
            None if need_eager else _manager(),
            num_reqs=1,
            num_tokens=[6, 8][rank],
            uniform_token_count=None,
            dp_size=2,
            dp_rank=rank,
            need_eager=need_eager,
            num_active_loras=3,
        )
    assert len(calls) == 1
    assert calls[0][0, rank] == [6, 8][rank]
    assert descriptor.num_tokens == 8
    assert descriptor.num_reqs == 1
    assert descriptor.num_active_loras == 3
    assert descriptor.cg_mode == CUDAGraphMode.NONE
    assert counts.tolist() == [8, 8]
    assert native_v2.dispatch_cg_and_sync_dp is original


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE])
def test_camp_v2_preserves_native_graph_padding(monkeypatch, mode):
    calls = _mock_dp_collective(monkeypatch, [6, 8], mode)
    with afd_v2._use_camp2p_eager_dp_padding():
        descriptor, counts = native_v2.dispatch_cg_and_sync_dp(
            _manager(mode), 1, 6, None, 2, 0
        )
    assert len(calls) == 1
    assert descriptor.cg_mode == mode
    assert descriptor.num_tokens == 16
    assert counts.tolist() == [16, 16]


def test_camp_v2_preserves_dp1_without_collective(monkeypatch):
    calls = _mock_dp_collective(monkeypatch, [6])
    with afd_v2._use_camp2p_eager_dp_padding():
        descriptor, counts = native_v2.dispatch_cg_and_sync_dp(
            None, 1, 6, None, 1, 0, need_eager=True
        )
    assert calls == []
    assert descriptor.num_tokens == 6
    assert counts is None


def test_camp_v2_preserves_all_empty_batch(monkeypatch):
    calls = _mock_dp_collective(monkeypatch, [0, 0])
    with afd_v2._use_camp2p_eager_dp_padding():
        descriptor, counts = native_v2.dispatch_cg_and_sync_dp(
            None, 0, 0, None, 2, 0, need_eager=True
        )
    assert len(calls) == 1
    assert descriptor.num_tokens == descriptor.num_reqs == 0
    assert counts is None


def test_camp_v2_padding_scope_restores_dispatch_after_error():
    original = native_v2.dispatch_cg_and_sync_dp
    with (
        pytest.raises(RuntimeError, match="model failed"),
        afd_v2._use_camp2p_eager_dp_padding(),
    ):
        assert native_v2.dispatch_cg_and_sync_dp is not original
        raise RuntimeError("model failed")
    assert native_v2.dispatch_cg_and_sync_dp is original


@pytest.mark.parametrize(
    ("connector", "attention_size", "ffn_size", "dp_size", "should_pad"),
    [
        ("CAMP2pAFDConnector", 2, 1, 2, True),
        ("CAMP2pAFDConnector", 2, 2, 2, False),
        ("CAMP2pAFDConnector", 2, 1, 1, False),
        ("CAMAsyncAFDConnector", 2, 1, 2, False),
    ],
)
def test_camp_v2_execute_scopes_padding_to_dp_fan_in(
    monkeypatch, connector, attention_size, ffn_size, dp_size, should_pad
):
    _mock_dp_collective(monkeypatch, [6, 8])
    original = native_v2.dispatch_cg_and_sync_dp
    runner = object.__new__(afd_v2.AFDNPUAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(
        connector=connector,
        num_attention_ranks=attention_size,
        num_ffn_ranks=ffn_size,
    )
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )
    runner.dp_size = dp_size
    runner.prof = None
    runner.cudagraph_manager = None
    runner._afd_pending_metadata = None
    runner._afd_suppress_metadata_send = False
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_is_graph_replaying = False
    runner._afd_is_profile = False

    def execute_model(self, scheduler_output, intermediate_tensors, **kwargs):
        assert (native_v2.dispatch_cg_and_sync_dp is not original) == should_pad
        return native_v2.dispatch_cg_and_sync_dp(
            None,
            1,
            scheduler_output.total_num_scheduled_tokens,
            None,
            self.dp_size,
            0,
            need_eager=True,
        )

    monkeypatch.setattr(afd_v2.NPUModelRunnerV2, "execute_model", execute_model)
    descriptor, counts = runner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=6)
    )
    assert descriptor.num_tokens == (8 if should_pad else 6)
    if dp_size == 1:
        assert counts is None
    else:
        assert counts.tolist() == ([8, 8] if should_pad else [6, 8])
    assert native_v2.dispatch_cg_and_sync_dp is original
