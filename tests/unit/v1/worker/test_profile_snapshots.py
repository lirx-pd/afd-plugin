# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU identity snapshots around native runner input preparation."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm_ascend.worker.v2.model_runner import NPUModelRunner  # noqa: E402

from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.v1.worker.attention_metadata import (  # noqa: E402
    AFDMetadataProviderMixin,
)
from afd_plugin.v1.worker.npu.attention_model_runner import (  # noqa: E402
    AFDNPUAttentionModelRunner,
)
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (  # noqa: E402
    AFDNPUAttentionModelRunnerV2,
)


def _session(admits):
    return SimpleNamespace(capture=SimpleNamespace(admits=lambda: admits))


def test_mrv2_snapshot_preserves_exact_native_signature():
    assert inspect.signature(
        AFDNPUAttentionModelRunnerV2.prepare_inputs, eval_str=True
    ) == inspect.signature(NPUModelRunner.prepare_inputs, eval_str=True)


def test_mrv2_snapshots_returned_cpu_order_and_original_prompt(monkeypatch):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    runner = object.__new__(AFDNPUAttentionModelRunnerV2)
    runner.connector = SimpleNamespace(profile_session=_session(True))
    prompt_lengths = np.array([20, 999, 5])
    runner.req_states = SimpleNamespace(prompt_len=SimpleNamespace(np=prompt_lengths))
    batch = SimpleNamespace(
        req_ids=["resumed", "decode"],
        num_reqs=2,
        query_start_loc_np=np.array([0, 5, 6, 16]),
        num_computed_tokens_np=np.array([3, 21]),
        idx_mapping_np=np.array([2, 0]),
        is_prefilling_np=np.array([True, False]),
        num_tokens=6,
    )
    args = (object(), object(), object())

    def prepare(self, *received):
        assert self is runner and received == args
        return batch

    monkeypatch.setattr(NPUModelRunner, "prepare_inputs", prepare)
    assert runner.prepare_inputs(*args) is batch
    snapshot = runner._profile_snapshot
    batch.req_ids[:] = ["next", "next-2"]
    batch.query_start_loc_np.fill(0)
    batch.num_computed_tokens_np.fill(0)
    batch.is_prefilling_np.fill(False)
    prompt_lengths.fill(0)
    assert snapshot.req_ids == ("resumed", "decode")
    assert snapshot.query_offsets == (0, 5, 6)
    assert snapshot.computed_before == (3, 21)
    assert snapshot.prompt_lengths == (5, 20)
    assert snapshot.execution_modes == ("PREFILL", "DECODE")
    assert snapshot.real_extent == 6


@pytest.mark.parametrize(
    "enabled,session", [(False, None), (True, None), (True, _session(False))]
)
def test_mrv2_does_not_expand_inputs_without_admission(monkeypatch, enabled, session):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1" if enabled else "0")
    runner = object.__new__(AFDNPUAttentionModelRunnerV2)
    runner.connector = SimpleNamespace(profile_session=session)
    runner._profile_snapshot = object()
    batch = object()
    monkeypatch.setattr(NPUModelRunner, "prepare_inputs", lambda *args: batch)
    assert runner.prepare_inputs(None, None, None) is batch
    assert runner._profile_snapshot is None


def test_mrv1_snapshots_cpu_inputs_before_stage_metadata(monkeypatch):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.connector = SimpleNamespace(profile_session=_session(True))
    runner._afd_transaction_counter = 0
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.input_batch = SimpleNamespace(
        req_ids=["prefill", "decode", "padding"],
        num_computed_tokens_cpu_tensor=torch.tensor([4, 12, 0]),
        num_prompt_tokens_cpu_tensor=torch.tensor([10, 10, 0]),
    )
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 3, 4, 32]))
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 4)
    assert runner._build_attention_metadata_with_ubatches(4, 2, 3) == ({}, None)
    snapshot = runner._profile_snapshot
    assert runner._afd_pending_metadata.profile_snapshot is snapshot
    assert snapshot.req_ids == ("prefill", "decode")
    assert snapshot.query_offsets == (0, 3, 4)
    assert snapshot.computed_before == (4, 12)
    assert snapshot.prompt_lengths == (10, 10)
    assert snapshot.execution_modes == ("PREFILL", "DECODE")
    runner.input_batch.num_computed_tokens_cpu_tensor.fill_(100)
    runner.query_start_loc.cpu.fill_(0)
    assert snapshot.computed_before == (4, 12)
    assert snapshot.query_offsets == (0, 3, 4)


def test_mrv1_disabled_path_never_reads_input_buffers(monkeypatch):
    monkeypatch.delenv("AFD_CAM_PROFILE_ENABLE", raising=False)
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner._profile_snapshot = object()
    assert runner._build_attention_metadata_with_ubatches(4, 2, 3) == ({}, None)
    assert runner._profile_snapshot is None


def test_metadata_and_clones_share_only_immutable_snapshot():
    from afd_plugin.profiling.fragments import InputSnapshot

    snapshot = InputSnapshot.from_cpu(["r"], [0, 1], [5], [10], [True], 1)
    runner = AFDMetadataProviderMixin()
    runner.connector = object()
    runner._afd_transaction_counter = 0
    runner._profile_snapshot = snapshot
    metadata = runner.build_afd_metadata(None, 1)
    assert isinstance(metadata, AFDForwardContextMetadata)
    cloned = metadata.clone()
    assert cloned.profile_snapshot is metadata.profile_snapshot is snapshot
    assert cloned.tokens_lens is not metadata.tokens_lens
