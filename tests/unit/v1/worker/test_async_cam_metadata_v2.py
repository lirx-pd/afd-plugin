# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

from afd_plugin.model_executor.models.npu.async_cam_layout import AsyncMoeUbatchMetadata
from afd_plugin.v1.worker.npu import async_cam_metadata_v2 as module


def _batch(scheduled=(2, 6, 2), parent_padding=2):
    scheduled = np.asarray(scheduled, dtype=np.int32)
    computed = np.arange(1, len(scheduled) + 1, dtype=np.int32) * 10
    seq_lens = scheduled + computed
    query = np.concatenate(
        (np.zeros(1, dtype=np.int32), scheduled.cumsum(dtype=np.int32))
    )
    tokens = int(scheduled.sum())
    positions = torch.tensor(
        [
            int(prefix) + i
            for prefix, count in zip(computed, scheduled, strict=True)
            for i in range(count)
        ]
        + [0] * parent_padding,
        dtype=torch.int64,
    )
    return SimpleNamespace(
        num_tokens=tokens,
        num_tokens_after_padding=tokens + parent_padding,
        num_reqs=len(scheduled),
        num_scheduled_tokens=scheduled,
        query_start_loc=torch.from_numpy(query.copy()).to(torch.int32),
        query_start_loc_np=query,
        seq_lens=torch.from_numpy(seq_lens.copy()),
        seq_lens_np=seq_lens,
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens.copy()),
        num_computed_tokens_np=computed,
        positions=positions,
        is_prefilling_np=scheduled > 1,
        attn_state=AscendAttentionState.ChunkedPrefill,
    )


def _runner(*, split="token", sp=False, tp=4):
    return SimpleNamespace(
        connector=SimpleNamespace(extra_info=SimpleNamespace(async_moe_split=split)),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                use_sequence_parallel_moe=sp, tensor_parallel_size=tp
            )
        ),
        device=torch.device("cpu"),
        kernel_block_sizes=[128, 64],
        _afd_async_moe_ubatch_metadata=object(),
    )


class _Builder:
    requires_block_table_width = False

    def __init__(self, spec, layer_names, config, device):
        self.spec = spec
        self.calls = []

    def set_kernel_block_size(self, size):
        self.kernel_block_size = size

    def build(self, *, common_attn_metadata, **kwargs):
        self.calls.append(common_attn_metadata)
        return SimpleNamespace(common=common_attn_metadata, builder=self)


class _DSABuilder(_Builder, AscendDSAMetadataBuilder):
    def build(self, *, common_attn_metadata, **kwargs):
        cache = kwargs["common_ratio_to_sas_metadata"]
        layout = (
            tuple(common_attn_metadata.query_start_loc_cpu.tolist()),
            tuple(common_attn_metadata.seq_lens_cpu.tolist()),
            tuple(common_attn_metadata.positions.tolist()),
        )
        assert cache.setdefault("layout", layout) == layout
        assert kwargs["num_actual_reqs"] == common_attn_metadata.num_reqs
        self.common_ratio_to_sas_metadata = cache
        # The real DSA builder clears padded rows through this tensor.
        common_attn_metadata.block_table_tensor.zero_()
        return super().build(common_attn_metadata=common_attn_metadata, **kwargs)


def _group(layer_name, *, dsa=False, group_id=0):
    builder_cls = _DSABuilder if dsa else _Builder
    backend = SimpleNamespace(get_builder_cls=lambda: builder_cls)
    spec = SimpleNamespace(
        copy_with_new_block_size=lambda size: SimpleNamespace(block_size=size)
    )
    group = AttentionGroup(backend, [layer_name], spec, group_id)
    group.metadata_builders = [builder_cls(spec, group.layer_names, None, "cpu")]
    return group


@pytest.mark.parametrize("sp", [False, True])
def test_token_stages_preserve_request_prefixes_and_real_attention_span(sp):
    batch = _batch()
    runner = _runner(sp=sp)
    group = _group("layer")
    native_builder = group.metadata_builders[0]
    table = torch.arange(3, dtype=torch.int32).reshape(3, 1)
    slots = torch.arange(12, dtype=torch.int64).reshape(1, 12)

    result = module.build_async_cam_stage_metadata(
        runner, batch, (table,), slots, [[group]]
    )

    assert result.parent_input_tokens == 12
    assert [stage.actual_tokens for stage in result.stages] == [5, 5]
    assert [stage.input_tokens for stage in result.stages] == ([8, 8] if sp else [5, 5])
    first, second = (metadata["layer"].common for metadata in result.attn_metadata)
    assert first.query_start_loc_cpu.tolist() == [0, 2, 5]
    assert second.query_start_loc_cpu.tolist() == [0, 3, 5]
    assert first.seq_lens_cpu.tolist() == [12, 23]
    assert second.seq_lens_cpu.tolist() == [26, 32]
    assert (second.seq_lens_cpu - second.query_start_loc_cpu.diff()).tolist() == [
        23,
        30,
    ]
    assert first.positions.tolist() == [10, 11, 20, 21, 22]
    assert second.positions.tolist() == [23, 24, 25, 30, 31]
    assert first.slot_mapping.tolist() == list(range(5))
    assert second.slot_mapping.tolist() == list(range(5, 10))
    assert first.num_input_tokens == second.num_input_tokens == 5
    assert first.is_prefilling.tolist() == second.is_prefilling.tolist() == [True, True]
    assert group.metadata_builders[0] is native_builder
    assert not native_builder.calls
    assert len(group.metadata_builders) == 3
    assert group.metadata_builders[1] is not group.metadata_builders[2]
    assert all(
        builder.kernel_block_size == 128 for builder in group.metadata_builders[1:]
    )

    builders = tuple(group.metadata_builders)
    module.build_async_cam_stage_metadata(runner, batch, (table,), slots, [[group]])
    assert tuple(group.metadata_builders) == builders


def test_request_split_and_dsa_caches_are_shared_only_within_each_stage(monkeypatch):
    batch = _batch((2, 3, 2), parent_padding=1)
    runner = _runner(split="request")
    groups = [[_group("a", dsa=True)], [_group("b", dsa=True, group_id=1)]]
    tables = (
        torch.ones((3, 1), dtype=torch.int32),
        torch.full((3, 1), 2, dtype=torch.int32),
    )
    slots = torch.stack((torch.arange(8), torch.arange(8) + 100))
    materialized = []
    monkeypatch.setattr(
        module,
        "materialize_deepseek_attention_metadata",
        lambda metadata, positions, num_tokens: materialized.append(
            (metadata, positions.tolist(), num_tokens)
        ),
    )

    result = module.build_async_cam_stage_metadata(runner, batch, tables, slots, groups)

    assert [stage.actual_tokens for stage in result.stages] == [2, 5]
    assert [stage.request_slice for stage in result.stages] == [
        slice(0, 1),
        slice(1, 3),
    ]
    a0, a1 = groups[0][0].metadata_builders[1:]
    b0, b1 = groups[1][0].metadata_builders[1:]
    assert a0.common_ratio_to_sas_metadata is b0.common_ratio_to_sas_metadata
    assert a1.common_ratio_to_sas_metadata is b1.common_ratio_to_sas_metadata
    assert a0.common_ratio_to_sas_metadata is not a1.common_ratio_to_sas_metadata
    assert tables[0].tolist() == [[1], [1], [1]]
    assert tables[1].tolist() == [[2], [2], [2]]
    assert [count for _, _, count in materialized] == [2, 5, 2, 5]
    assert all(
        builder.kernel_block_size == 64
        for builder in groups[1][0].metadata_builders[1:]
    )
    assert result.attn_metadata[1]["b"].common.slot_mapping.tolist() == list(
        range(102, 107)
    )


@pytest.mark.parametrize(
    "scheduled,split,tp",
    [((), "token", 4), ((1,), "token", 4), ((4,), "request", 4), ((4,), "token", 1)],
)
def test_unsplittable_batches_keep_native_path(scheduled, split, tp):
    batch = _batch(scheduled, parent_padding=0)
    group = _group("layer")
    assert (
        module.build_async_cam_stage_metadata(
            _runner(split=split, tp=tp), batch, (), torch.empty((0, 0)), [[group]]
        )
        is None
    )
    assert len(group.metadata_builders) == 1


@pytest.mark.parametrize("instance_override", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_prepare_scope_preserves_native_metadata_and_restores_method(
    monkeypatch, instance_override, raises
):
    batch = _batch()
    full_metadata = {"layer": object()}
    events: list[str] = []

    class State:
        def prepare_attn(self, *args, **kwargs):
            events.append("native")
            return full_metadata

        def prepare_inputs(self, *args):
            return {"input_ids": batch.positions}

    state = State()
    if instance_override:
        state.__dict__.update(
            prepare_attn=state.prepare_attn, prepare_inputs=state.prepare_inputs
        )
    original_method = state.prepare_attn
    original_inputs_method = state.prepare_inputs
    runner = _runner()
    runner.model_state = state
    sidecar = object()

    def materialize(metadata, positions, count):
        assert metadata is full_metadata
        assert positions is batch.positions
        assert count == 10
        events.append("materialize")

    def build(*args):
        events.append("stages")
        if raises:
            raise RuntimeError("stage build failed")
        return sidecar

    monkeypatch.setattr(
        module, "materialize_deepseek_attention_metadata_by_layer", materialize
    )
    monkeypatch.setattr(module, "build_async_cam_stage_metadata", build)
    try:
        with module.use_async_cam_stage_metadata(runner):
            assert "prepare_attn" in state.__dict__
            assert (
                state.prepare_attn(
                    batch,
                    CUDAGraphMode.NONE,
                    (),
                    torch.empty(0),
                    [],
                    None,
                    for_capture=False,
                    ubatch_idx=0,
                )
                is full_metadata
            )
            assert runner._afd_async_moe_ubatch_metadata is sidecar
            assert state.prepare_inputs(batch, None) == {"input_ids": batch.positions}
            assert runner._afd_async_moe_ubatch_metadata is sidecar
    except RuntimeError as exc:
        assert raises and str(exc) == "stage build failed"
    else:
        assert not raises

    assert state.prepare_attn == original_method
    assert state.prepare_inputs == original_inputs_method
    assert ("prepare_attn" in state.__dict__) is instance_override
    assert ("prepare_inputs" in state.__dict__) is instance_override
    assert events == ["native", "materialize", "stages"]


def test_skip_attention_profile_plans_stages_with_none_metadata():
    batch = _batch()
    inputs = {"input_ids": batch.positions}

    class State:
        def prepare_attn(self, *args, **kwargs):
            pytest.fail("skip-Attention profile must not build KV metadata")

        def prepare_inputs(self, input_batch, req_states):
            assert input_batch is batch
            return inputs

    runner = _runner(sp=True)
    runner.model_state = State()
    runner._afd_async_moe_ubatch_metadata = None
    with module.use_async_cam_stage_metadata(runner):
        assert runner.model_state.prepare_inputs(batch, None) is inputs

    plan: AsyncMoeUbatchMetadata | None = runner._afd_async_moe_ubatch_metadata
    assert plan is not None
    assert plan.attn_metadata == [None, None]
    assert [stage.actual_tokens for stage in plan.stages] == [5, 5]
    assert [stage.input_tokens for stage in plan.stages] == [8, 8]
    assert plan.parent_input_tokens == 12
