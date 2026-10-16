# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pipeline capture must never read or copy per-request runner input buffers."""

from types import SimpleNamespace

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
