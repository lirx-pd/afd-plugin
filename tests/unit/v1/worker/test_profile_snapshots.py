# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pipeline capture must never read or copy per-request runner input buffers."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm_ascend.worker.v2.model_runner import NPUModelRunner  # noqa: E402

from afd_plugin.v1.worker.npu.attention_model_runner import (  # noqa: E402
    AFDNPUAttentionModelRunner,
)
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (  # noqa: E402
    AFDNPUAttentionModelRunnerV2,
)


def test_mrv2_uses_native_input_preparation_without_snapshot_wrapper():
    assert AFDNPUAttentionModelRunnerV2.prepare_inputs is NPUModelRunner.prepare_inputs


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_mrv1_pipeline_never_reads_request_input_buffers(monkeypatch, enabled):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", enabled)
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    # No input_batch, request ids or profile snapshot fields are provided.
    assert runner._build_attention_metadata_with_ubatches(4, 2, 3) == ({}, None)


@pytest.mark.parametrize("failure", [False, True])
def test_attention_idles_pipeline_after_restoring_native_execution_state(
    monkeypatch, failure
):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    module = importlib.import_module(AFDNPUAttentionModelRunner.__module__)
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.prof = None
    runner.connector = object()
    runner._afd_is_graph_replaying = True
    pipeline = SimpleNamespace(tracer=SimpleNamespace(idle=Mock()), capture=object())
    monkeypatch.setattr(module, "begin_profile_step", lambda _: pipeline)

    def native_execute(self, *args):
        assert self._afd_live_execution
        if failure:
            raise ValueError("native execution failure")
        return "native result"

    monkeypatch.setattr(
        AFDNPUAttentionModelRunner.__bases__[0], "execute_model", native_execute
    )

    def idle(capture):
        assert capture is pipeline.capture
        assert not runner._afd_live_execution and runner._afd_is_graph_replaying

    pipeline.tracer.idle.side_effect = idle
    if failure:
        with pytest.raises(ValueError, match="native execution failure"):
            runner.execute_model(None)
    else:
        assert runner.execute_model(None) == "native result"
    pipeline.tracer.idle.assert_called_once_with(pipeline.capture)
