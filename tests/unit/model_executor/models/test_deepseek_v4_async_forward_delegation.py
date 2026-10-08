# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

native = pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

import torch  # noqa: E402
import vllm.envs as envs  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm_ascend.ascend_forward_context import _EXTRA_CTX  # noqa: E402

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    async_cam_layout,
)
from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4_async_cam_forward as async_forward,
)
from afd_plugin.model_executor.models.npu.async_cam_layout import (  # noqa: E402
    AsyncMoeUbatchMetadata,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
)


def test_dsv4_async_metadata_lazily_delegates_to_async_cam_forward(monkeypatch):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    metadata = object()
    input_ids = object()
    positions = object()
    intermediate_tensors = object()
    sentinel = object()
    calls: list[tuple[object, ...]] = []

    def run_async(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        adapter,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda: metadata,
    )
    monkeypatch.setattr(
        async_forward,
        "run_async_moe_ubatch_forward",
        run_async,
    )

    result = model.forward(input_ids, positions, intermediate_tensors)

    assert result is sentinel
    assert calls == [
        (model, input_ids, positions, intermediate_tensors, metadata, None)
    ]


def test_dsv4_without_async_metadata_uses_native_forward(monkeypatch):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    sentinel = object()
    calls: list[tuple[object, ...]] = []

    def native_forward(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        adapter,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda: None,
    )
    monkeypatch.setattr(
        native.DeepseekV4Model,
        "forward",
        native_forward,
    )

    result = model.forward(None, object(), None, object())

    assert result is sentinel
    assert len(calls) == 1


def test_async_cam_forward_resolves_forward_context_entrypoint():
    assert callable(async_forward.get_forward_context)
    assert (
        "get_forward_context"
        in async_forward.run_async_moe_ubatch_forward.__code__.co_names
    )


@pytest.mark.parametrize("use_mrv2", [False, True])
@pytest.mark.parametrize("stage_idx", [0, 1])
def test_dsv4_stage_token_context_restores_parent_on_error(
    monkeypatch, use_mrv2, stage_idx
):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", use_mrv2)
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_forward, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        async_forward,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: group)
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[None, None],
        stages=(
            AsyncMoeStage(slice(0, 1), slice(0, 3), 4),
            AsyncMoeStage(slice(1, 2), slice(3, 5), 2),
        ),
        parent_input_tokens=5,
        use_sequence_parallel=True,
    )
    parent_kwargs = {
        "num_tokens": 5,
        "afd_metadata": SimpleNamespace(connector=object()),
    }
    parent = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs=parent_kwargs,
    )
    parent.num_tokens = 5

    def hc_pre(*_args):
        stage_context = get_forward_context()
        assert stage_context.additional_kwargs is not parent_kwargs
        assert _EXTRA_CTX.num_tokens == metadata.stages[stage_idx].actual_tokens
        raise RuntimeError("stage stopped")

    layer = SimpleNamespace(
        mlp=object.__new__(adapter.AFDDeepseekV4AttentionGateRemoteMoE),
        hc_pre=hc_pre,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
    )
    model = SimpleNamespace(
        embed_input_ids=lambda ids: ids.float().unsqueeze(-1),
        hc_mult=1,
        use_sequence_parallel_moe=True,
        layers=[layer],
        start_layer=0,
        end_layer=1,
    )
    monkeypatch.setattr(
        async_forward,
        "_run_two_stage_async_moe_schedule",
        lambda layers, compute, _send, _receive: compute(layers[0], stage_idx),
    )
    with override_forward_context(parent):
        with pytest.raises(RuntimeError, match="stage stopped"):
            async_forward.run_async_moe_ubatch_forward(
                model, torch.arange(5), torch.arange(5), None, metadata, None
            )
        assert get_forward_context() is parent
        assert _EXTRA_CTX.num_tokens == 5
        assert parent.additional_kwargs is parent_kwargs
        assert parent_kwargs["num_tokens"] == 5
