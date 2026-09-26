# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.forward_context import (  # noqa: E402
    get_forward_context as get_current_forward_context,
)

from afd_plugin.model_executor.models import (  # noqa: E402
    get_afd_metadata_from_forward_context,
)
from afd_plugin.model_executor.models.npu.async_cam_layout import (  # noqa: E402
    ASYNC_MOE_UBATCH_METADATA_KEY,
    AsyncMoeUbatchMetadata,
    get_async_moe_ubatch_metadata_from_forward_context,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
)


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
        afd_metadata={"stage": 1},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_get_afd_metadata_ignores_forward_context_attribute():
    forward_context = SimpleNamespace(
        additional_kwargs={},
        afd_metadata={"stage": 0},
    )

    assert get_afd_metadata_from_forward_context(forward_context) is None


def test_get_async_moe_ubatch_metadata_from_additional_kwargs():
    sidecar = {"ubatch_slices": ["stage0", "stage1"]}
    forward_context = SimpleNamespace(
        additional_kwargs={ASYNC_MOE_UBATCH_METADATA_KEY: sidecar},
    )

    assert (
        get_async_moe_ubatch_metadata_from_forward_context(forward_context) is sidecar
    )


@pytest.mark.parametrize(
    ("is_first_rank", "is_last_rank"),
    [(True, False), (False, True)],
)
def test_async_model_forward_preserves_pp_boundaries(
    monkeypatch,
    is_first_rank,
    is_last_rank,
):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    class FakeIntermediateTensors(dict):
        pass

    monkeypatch.setattr(async_forward, "IntermediateTensors", FakeIntermediateTensors)
    monkeypatch.setattr(
        async_forward,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_first_rank=is_first_rank,
            is_last_rank=is_last_rank,
        ),
    )
    forward_context = SimpleNamespace()
    afd_metadata = SimpleNamespace()
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        async_forward,
        "get_afd_metadata_from_forward_context",
        lambda context: afd_metadata if context is forward_context else None,
    )
    monkeypatch.setattr(
        async_forward,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda context: None,
    )

    schedule_calls: list[tuple[Any, ...]] = []

    def run_schedule(
        model,
        hidden_states,
        residual,
        positions,
        received_metadata,
        llama_4_scaling,
    ):
        schedule_calls.append(
            (
                model,
                hidden_states,
                residual,
                positions,
                received_metadata,
                llama_4_scaling,
            )
        )
        next_residual = (
            torch.zeros_like(hidden_states) if residual is None else residual + 2
        )
        return hidden_states + 1, next_residual

    monkeypatch.setattr(
        async_forward,
        "run_attention_gate_afd_forward",
        run_schedule,
    )
    norm_calls = []

    def run_norm(hidden_states, residual):
        norm_calls.append((hidden_states, residual))
        return hidden_states + residual, None

    model = SimpleNamespace(
        aux_hidden_state_layers=(),
        embed_input_ids=lambda input_ids: input_ids.to(torch.float32).unsqueeze(-1),
        _get_llama_4_scaling=lambda positions: None,
        norm=run_norm,
    )
    positions = torch.arange(2)
    if is_first_rank:
        input_ids = torch.tensor([3, 4])
        intermediate_tensors = None
        expected_hidden_states = model.embed_input_ids(input_ids)
        expected_residual = None
    else:
        input_ids = None
        expected_hidden_states = torch.full((2, 1), 5.0)
        expected_residual = torch.full((2, 1), 7.0)
        intermediate_tensors = FakeIntermediateTensors(
            {
                "hidden_states": expected_hidden_states,
                "residual": expected_residual,
            }
        )

    output = async_forward.run_model_forward(
        model,
        input_ids,
        positions,
        intermediate_tensors,
    )

    assert len(schedule_calls) == 1
    assert torch.equal(schedule_calls[0][1], expected_hidden_states)
    assert schedule_calls[0][2] is expected_residual
    assert schedule_calls[0][4] is afd_metadata
    scheduled_hidden_states = expected_hidden_states + 1
    scheduled_residual = (
        torch.zeros_like(expected_hidden_states)
        if expected_residual is None
        else expected_residual + 2
    )
    if is_last_rank:
        assert len(norm_calls) == 1
        assert torch.equal(output, scheduled_hidden_states + scheduled_residual)
    else:
        assert isinstance(output, FakeIntermediateTensors)
        assert torch.equal(output["hidden_states"], scheduled_hidden_states)
        assert torch.equal(output["residual"], scheduled_residual)


@pytest.mark.parametrize("in_profile_run", [False, True], ids=["regular", "profile"])
@pytest.mark.parametrize(
    "dense_prefix", [False, True], ids=["moe-only", "dense-prefix"]
)
def test_async_cam_profile_forward_runs_matched_connector_io(
    monkeypatch,
    in_profile_run,
    dense_prefix,
):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )
    from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

    forward_context = SimpleNamespace(
        additional_kwargs={},
        in_profile_run=in_profile_run,
        ubatch_idx=1,
        flash_comm_v1_enabled=True,
    )
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        npu_remote_moe,
        "get_afd_metadata_from_forward_context",
        lambda: forward_context.additional_kwargs["afd_metadata"],
    )

    events: list[tuple[Any, ...]] = []
    dispatch_layouts: list[object] = []
    restored_layouts: list[object] = []
    pending = []
    completed_layer_idx = None

    def prepare_dispatch_payload(
        hidden_states,
        topk_weights,
        topk_ids,
        router_logits,
        *,
        use_sequence_parallel,
    ):
        assert use_sequence_parallel is True
        layout = object()
        dispatch_layouts.append(layout)
        return SimpleNamespace(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            layout=layout,
        )

    def restore_dispatch_output(local_output, layout):
        events.append(("restore", completed_layer_idx, 1))
        restored_layouts.append(layout)
        return local_output

    monkeypatch.setattr(
        npu_remote_moe,
        "prepare_cam_dispatch_payload",
        prepare_dispatch_payload,
    )
    monkeypatch.setattr(
        npu_remote_moe,
        "restore_cam_dispatch_output",
        restore_dispatch_output,
    )

    def send_attn_output(hidden_states, context, **kwargs):
        metadata = context.metadata
        assert metadata.seq_lens == [hidden_states.shape[0]]
        assert kwargs["topk_weights"].shape == (hidden_states.shape[0], 1)
        assert kwargs["topk_ids"].dtype == torch.int32
        events.append(("send", metadata.layer_idx, metadata.stage_idx))
        pending.append((metadata.layer_idx, hidden_states))

    def recv_ffn_output(ref_tensor, ubatch_idx):
        nonlocal completed_layer_idx
        completed_layer_idx, dispatched = pending.pop(0)
        assert ref_tensor is dispatched
        events.append(("recv", completed_layer_idx, ubatch_idx))
        return ref_tensor + 10 * (completed_layer_idx - int(dense_prefix) + 1)

    def yield_attention(hidden_states, *, role):
        assert role == "attention"
        events.append(("yield", pending[-1][0], 1))
        return hidden_states + 1000

    monkeypatch.setattr(async_forward, "maybe_apply_dbo_yield", yield_attention)
    connector = SimpleNamespace(
        send_attn_output=send_attn_output,
        recv_ffn_output=recv_ffn_output,
    )
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=1)
    forward_context.additional_kwargs["afd_metadata"] = afd_metadata

    class _Runner:
        dispatch = npu_remote_moe.AFDAttentionGateMoERunner.dispatch
        layer_id = npu_remote_moe.AFDAttentionGateMoERunner.layer_id

        def __init__(self, layer_idx):
            self.layer_name = f"model.layers.{layer_idx}.mlp.experts"

        def combine(self, dispatch_ref, layout, *, stage_idx):
            assert self.layer_id == pending[0][0]
            return npu_remote_moe.AFDAttentionGateMoERunner.combine(
                self, dispatch_ref, layout, stage_idx=stage_idx
            )

    class _MoELayer:
        is_moe_layer = True

        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.mlp = SimpleNamespace(
                experts=_Runner(layer_idx), shared_experts=self.compute_shared
            )

        def compute_shared(self, hidden_states):
            events.append(("shared", self.layer_idx, 1))
            return 2 * hidden_states

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            events.append(("compute", self.layer_idx, 1))
            return (
                hidden_states + 1,
                residual + 2,
                torch.ones((hidden_states.shape[0], 1)),
                torch.zeros((hidden_states.shape[0], 1), dtype=torch.int32),
                torch.ones((hidden_states.shape[0], 1)),
            )

    class _DenseLayer:
        is_moe_layer = False

        def __call__(self, positions, hidden_states, residual, llama_4_scaling):
            events.append(("dense", 0))
            return hidden_states + 5, residual + 1

    moe_layers = [_MoELayer(int(dense_prefix) + offset) for offset in range(2)]
    layers: list[_MoELayer | _DenseLayer] = list(moe_layers)
    if dense_prefix:
        layers.insert(0, _DenseLayer())
    model = SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))
    expected_events: list[tuple[Any, ...]] = [("dense", 0)] if dense_prefix else []
    for layer in moe_layers:
        expected_events.extend(
            (event, layer.layer_idx, 1)
            for event in ("compute", "send", "shared", "yield", "recv", "restore")
        )

    for call_idx in range(2):
        events.clear()
        dispatch_layouts.clear()
        restored_layouts.clear()
        hidden_states = torch.full((2, 4), float(2 * call_idx))
        residual = torch.full_like(hidden_states, 7)
        output, output_residual = async_forward.run_attention_gate_afd_forward(
            model,
            hidden_states,
            residual,
            torch.arange(2),
            afd_metadata,
        )

        torch.testing.assert_close(
            output,
            9 * (hidden_states + 5 * int(dense_prefix)) + 62,
        )
        torch.testing.assert_close(output_residual, residual + 4 + int(dense_prefix))
        assert events == expected_events
        assert restored_layouts == dispatch_layouts
        assert not pending
        for layer in moe_layers:
            assert vars(layer.mlp.experts) == {
                "layer_name": f"model.layers.{layer.layer_idx}.mlp.experts"
            }


def test_deepseek_afd_wrapper_keeps_full_model_compile_enabled():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert "@native.support_torch_compile\nclass AFDDeepseekV2Model" in source
    assert "from __future__ import annotations" not in source
    assert "self.do_not_compile = True" not in source


def test_deepseek_afd_wrapper_treats_index_topk_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'self.is_v32 = hasattr(config, "index_topk")' in source
    assert "self.is_v32 = config.index_topk is not None" not in source
    assert "topk_tokens = config.index_topk" in source


def test_deepseek_afd_wrapper_treats_llama_4_scaling_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'getattr(self.config, "llama_4_scaling", None)' in source
    assert "self.config.llama_4_scaling" not in source


def test_deepseek_afd_attention_path_can_compute_gate_before_send():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    model_source = source.split("class AFDDeepseekV2Model", 1)[1].split(
        "class AFDDeepseekV2ForCausalLM",
        1,
    )[0]
    model_forward = model_source.split("    def forward(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]
    gate_runner = Path("afd_plugin/model_executor/npu/remote_moe.py").read_text()
    attention_gate_forward = executor_source.split(
        "def run_attention_gate_afd_forward(",
        1,
    )[1].split("def run_async_moe_ubatch_afd_forward(", 1)[0]

    assert 'if afd_role == "attention":' in source
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "def _forward_attention(" not in source
    assert "return super().forward(" in model_forward
    assert "deepseek_v2_async_cam_forward.run_model_forward(" in model_forward
    assert "compute_gate_topk(" in gate_runner
    assert "topk_weights=topk_weights" in gate_runner
    assert "topk_ids=topk_ids" in gate_runner
    assert "router_logits=router_logits" in gate_runner
    assert "layer.compute_attn_output(" in attention_gate_forward
    assert ".dispatch(" in attention_gate_forward
    assert ".combine(" in attention_gate_forward
    assert "send_attn_output(" not in executor_source
    assert "recv_ffn_output(" not in executor_source
    assert "prepare_cam_dispatch_payload(" not in executor_source
    assert "restore_cam_dispatch_output(" not in executor_source
    assert "topk_weights" in attention_gate_forward
    assert "topk_ids" in attention_gate_forward


def test_deepseek_afd_attention_gate_can_force_balanced_topk_ids():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path("afd_plugin/model_executor/npu/remote_moe.py").read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    compute_attn_output = source.split("    def compute_attn_output(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert "self.mlp.experts.compute_gate_topk(" in compute_attn_output
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "deepseek_v2_attention_gate" not in compute_attn_output
    helper_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    assert "def compute_attention_gate_topk(" not in helper_source
    assert "force_balanced_topk_ids_enabled" in gate_source
    assert "balanced_topk_ids = torch.arange(" in gate_source
    assert "topk_ids.copy_(" in gate_source
    assert "topk_weights, topk_ids = select_experts(" in (gate_source)
    assert "if force_balanced_topk_ids_enabled():" in gate_source
    assert (
        gate_source.index(
            "topk_weights, topk_ids = select_experts(",
        )
        < gate_source.index("if force_balanced_topk_ids_enabled():")
        < gate_source.index("return topk_weights.to(torch.float32)")
    )


def test_deepseek_afd_gate_on_attention_keeps_dense_layers_local():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()

    assert "self.is_moe_layer = is_moe_layer" in source
    assert "self.compute_gate_on_attention and not self.is_moe_layer" in source
    assert "if not layer.is_moe_layer:" in executor_source
    assert (
        "return _ATTENTION_ROLE if compute_gate_on_attention else _FFN_ROLE" in source
    )


def test_deepseek_compute_gate_on_attention_selects_backend_boundary():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'device_type not in ("cuda", "npu")' in source
    assert "self.mlp = AFDDeepseekV2RemoteExpertsMoE(" in source
    assert "GateOnlyRemoteMoE" not in source
    assert "AFDRemoteMoERunner.create(" in source
    assert 'prefix=f"{prefix}.mlp"' in source
    assert "self.mlp.experts.compute_gate_topk(" in source
    assert (
        "# NPU-only: gated MoE FFN compute consumes Attention-side topk payloads."
        in source
    )


@pytest.mark.parametrize(
    "dense_prefix", [False, True], ids=["moe-only", "dense-prefix"]
)
def test_async_moe_pipeline_preserves_stage_order(monkeypatch, dense_prefix):
    from afd_plugin.model_executor.models.npu import deepseek_v2_async_cam_forward
    from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

    events: list[tuple[Any, ...]] = []
    forward_context = SimpleNamespace(
        attn_metadata={"layer": "full"},
        additional_kwargs={},
        ubatch_idx=0,
        num_ubatches=1,
        num_tokens=4,
        pad_size=0,
        flash_comm_v1_enabled=True,
    )

    pending = {}
    dispatch_layouts = []
    restored_layouts = []

    def send_attn_output(hidden_states, context, **_kwargs):
        metadata = context.metadata
        assert metadata.seq_lens == [hidden_states.shape[0]]
        assert metadata.stage_idx not in pending
        events.append(("send", metadata.layer_idx, metadata.stage_idx))
        pending[metadata.stage_idx] = (metadata.layer_idx, hidden_states)

    def recv_ffn_output(ref_tensor, ubatch_idx):
        layer_idx, dispatched = pending.pop(ubatch_idx)
        assert ref_tensor is dispatched
        events.append(("recv", layer_idx, ubatch_idx))
        return ref_tensor + 10 * (layer_idx - int(dense_prefix) + 1)

    connector = SimpleNamespace(
        send_attn_output=send_attn_output,
        recv_ffn_output=recv_ffn_output,
    )
    parent_metadata = SimpleNamespace(
        stage_idx=0,
        connector=connector,
    )
    forward_context.additional_kwargs["afd_metadata"] = parent_metadata
    execution_plan = AsyncMoeUbatchMetadata(
        attn_metadata=[{"layer": "stage-0"}, {"layer": "stage-1"}],
        stages=[
            AsyncMoeStage(
                slice(0, 1),
                slice(0, 2),
                input_tokens=2,
            ),
            AsyncMoeStage(
                slice(1, 2),
                slice(2, 4),
                input_tokens=4,
            ),
        ],
        parent_input_tokens=4,
        use_sequence_parallel=True,
    )

    class _Runner:
        dispatch = npu_remote_moe.AFDAttentionGateMoERunner.dispatch
        layer_id = npu_remote_moe.AFDAttentionGateMoERunner.layer_id

        def __init__(self, layer_idx):
            self.layer_name = f"model.layers.{layer_idx}.mlp.experts"

        def combine(self, dispatch_ref, layout, *, stage_idx):
            assert self.layer_id == pending[stage_idx][0]
            return npu_remote_moe.AFDAttentionGateMoERunner.combine(
                self, dispatch_ref, layout, stage_idx=stage_idx
            )

    class _MoELayer:
        is_moe_layer = True

        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.mlp = SimpleNamespace(
                experts=_Runner(layer_idx), shared_experts=self.compute_shared
            )

        def compute_shared(self, hidden_states):
            stage_idx = next(
                stage_idx
                for stage_idx, (layer_idx, dispatched) in pending.items()
                if layer_idx == self.layer_idx and dispatched is hidden_states
            )
            events.append(("shared", self.layer_idx, stage_idx))
            return hidden_states + self.layer_idx - int(dense_prefix) + 1

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            stage_context = get_current_forward_context()
            stage_idx = stage_context.ubatch_idx
            assert positions == f"positions-{stage_idx}"
            assert llama_4_scaling == f"scaling-{stage_idx}"
            events.append(
                (
                    "compute",
                    self.layer_idx,
                    stage_idx,
                    stage_context.attn_metadata,
                    stage_context.num_tokens,
                    stage_context.pad_size,
                ),
            )
            topk = hidden_states[:, :1]
            return hidden_states, residual, topk, topk.to(torch.int32), None

    class _DenseLayer:
        is_moe_layer = False

        def __call__(self, positions, hidden_states, residual, llama_4_scaling):
            assert positions == "full-positions"
            assert llama_4_scaling == "full-scaling"
            events.append(("dense", 0))
            return hidden_states + 3, residual

    def build_stage_inputs(hidden_states, residual, positions, scaling, metadata):
        events.append(("split",))
        assert metadata is execution_plan
        return SimpleNamespace(
            hidden_states=[hidden_states[:1].clone(), hidden_states[2:].clone() + 1],
            residuals=[None, None],
            positions=["positions-0", "positions-1"],
            llama_4_scaling=["scaling-0", "scaling-1"],
        )

    def restore_stage_outputs(outputs, metadata):
        events.append(("restore-parent",))
        assert metadata is execution_plan
        assert not pending
        return tuple(outputs)

    def prepare_dispatch_payload(
        hidden_states, topk_weights, topk_ids, router_logits, **kwargs
    ):
        assert kwargs["use_sequence_parallel"] is True
        layout = object()
        dispatch_layouts.append(layout)
        return SimpleNamespace(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            layout=layout,
        )

    def restore_dispatch_output(output, layout):
        restored_layouts.append(layout)
        return output

    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_forward_context",
        lambda: forward_context,
    )
    monkeypatch.setattr(
        npu_remote_moe,
        "get_afd_metadata_from_forward_context",
        lambda: forward_context.additional_kwargs["afd_metadata"],
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "build_async_moe_stage_inputs",
        build_stage_inputs,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "restore_async_moe_stage_outputs",
        restore_stage_outputs,
    )
    monkeypatch.setattr(
        npu_remote_moe,
        "prepare_cam_dispatch_payload",
        prepare_dispatch_payload,
    )
    monkeypatch.setattr(
        npu_remote_moe,
        "restore_cam_dispatch_output",
        restore_dispatch_output,
    )

    def yield_attention(hidden_states, **_kwargs):
        events.append(("yield",))
        return hidden_states

    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "maybe_apply_dbo_yield",
        yield_attention,
    )
    moe_layers = [_MoELayer(int(dense_prefix) + offset) for offset in range(2)]
    layers: list[_MoELayer | _DenseLayer] = list(moe_layers)
    if dense_prefix:
        layers.insert(0, _DenseLayer())
    model = SimpleNamespace(start_layer=0, end_layer=len(layers), layers=layers)
    first_layer_idx = int(dense_prefix)
    last_layer_idx = first_layer_idx + 1
    expected_events: list[tuple[Any, ...]] = [("dense", 0)] if dense_prefix else []
    expected_events += [
        ("split",),
        ("compute", first_layer_idx, 0),
        ("send", first_layer_idx, 0),
        ("shared", first_layer_idx, 0),
        ("compute", first_layer_idx, 1),
        ("recv", first_layer_idx, 0),
        ("send", first_layer_idx, 1),
        ("shared", first_layer_idx, 1),
        ("compute", last_layer_idx, 0),
        ("recv", first_layer_idx, 1),
        ("send", last_layer_idx, 0),
        ("shared", last_layer_idx, 0),
        ("compute", last_layer_idx, 1),
        ("recv", last_layer_idx, 0),
        ("send", last_layer_idx, 1),
        ("shared", last_layer_idx, 1),
        ("recv", last_layer_idx, 1),
        ("restore-parent",),
    ]

    for call_idx in range(2):
        events.clear()
        dispatch_layouts.clear()
        restored_layouts.clear()
        output, residual = (
            deepseek_v2_async_cam_forward.run_async_moe_ubatch_afd_forward(
                model=model,
                hidden_states=torch.full((4, 8), float(2 * call_idx)),
                residual=None,
                positions="full-positions",
                afd_metadata=parent_metadata,
                async_moe_ubatch_metadata=execution_plan,
                llama_4_scaling="full-scaling",
            )
        )

        assert [event[:3] for event in events] == expected_events
        for event in (event for event in events if event[0] == "compute"):
            stage_idx = event[2]
            assert event[3] == {"layer": f"stage-{stage_idx}"}
            assert event[4] == 2
            assert event[5] == (0, 2)[stage_idx]
        expected = 4 * (2 * call_idx + 3 * int(dense_prefix)) + 44
        torch.testing.assert_close(output[0], torch.full((1, 8), float(expected)))
        torch.testing.assert_close(output[1], torch.full((2, 8), float(expected + 4)))
        assert restored_layouts == dispatch_layouts
        assert residual is None
        assert not pending
        assert forward_context.attn_metadata == {"layer": "full"}
        assert forward_context.additional_kwargs == {"afd_metadata": parent_metadata}
        assert forward_context.ubatch_idx == 0
        assert forward_context.num_ubatches == 1
        assert forward_context.num_tokens == 4
        assert forward_context.pad_size == 0
        for layer in moe_layers:
            assert vars(layer.mlp.experts) == {
                "layer_name": f"model.layers.{layer.layer_idx}.mlp.experts"
            }


def test_deepseek_afd_ffn_path_reuses_ascend_moe_mlp_after_attention_gate():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "compute_attention_gate_moe_ffn(" in compute_ffn_output
    assert "from afd_plugin.model_executor.models.npu import (" in compute_ffn_output
    assert "deepseek_v2_attention_gate," in compute_ffn_output
    assert "AFDF2ATransferPayload(" in compute_moe
    assert "MoEMlpComputeInput(" in compute_moe
    assert "unified_apply_mlp(" in compute_moe
    assert "routed_output, _ = unified_apply_mlp(" in compute_moe
    assert "quant_type == QuantType.W8A8" in compute_moe
    assert 'experts.get_eplb_parameter("w13_weight")' in compute_moe
    assert 'experts.get_eplb_parameter("w2_weight")' in compute_moe
    assert "experts.w13_weight" not in compute_moe
    assert "experts.w2_weight" not in compute_moe
    assert "w13_weight_scale_fp32" in compute_moe
    assert "w13_weight_scale_fp32_list" in compute_moe
    assert "w2_weight_scale_list" in compute_moe
    compute_moe_function = next(
        node
        for node in ast.parse(gate_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_attention_gate_moe_ffn"
    )
    quant_params_calls = [
        node
        for node in ast.walk(compute_moe_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MoEQuantParams"
    ]
    assert len(quant_params_calls) == 1
    # Check the contract without pinning formatting or optional quant fields.
    assert any(
        keyword.arg == "quant_type"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "quant_type"
        for keyword in quant_params_calls[0].keywords
    )
    assert "_gmmswigluquant_fusion_enabled()" in compute_moe
    assert "fusion=use_gmmswigluquant_fusion" in compute_moe
    assert "_compute_w8a8_shared_experts_from_int8(" in compute_moe
    assert "shared_input.dtype == torch.int8" in compute_moe
    assert 'getattr(layer.mlp, "swiglu_limit", None)' in compute_moe
    assert "fusion=False" not in compute_moe
    assert "output_dtype=torch.int32" in gate_source
    assert "npu_dequant_swiglu_quant(" in gate_source
    assert "activation_scale=pertoken_scale" in gate_source


@pytest.mark.parametrize(
    ("num_routed_tokens", "num_shared_tokens"),
    [(2, 2), (2, 0), (0, 2), (0, 0)],
)
@pytest.mark.parametrize(
    ("routed_scale_applied_in_topk", "expected_routed_value"),
    [(False, 2.0), (True, 1.0)],
)
def test_deepseek_afd_ffn_skips_empty_rank_local_moe_work(
    monkeypatch,
    num_routed_tokens,
    num_shared_tokens,
    routed_scale_applied_in_topk,
    expected_routed_value,
):
    from afd_plugin.model_executor.models.npu import deepseek_v2_attention_gate

    class FakeQuantType:
        NONE = "none"
        W8A8 = "w8a8"
        W4A8 = "w4a8"

    class KeywordArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    routed_calls = []

    def fake_unified_apply_mlp(*, mlp_compute_input):
        assert mlp_compute_input.quant.quant_type == FakeQuantType.W8A8
        assert mlp_compute_input.quant.is_per_channel_weight is False
        routed_calls.append(mlp_compute_input.hidden_states)
        return (
            torch.ones_like(
                mlp_compute_input.hidden_states,
                dtype=torch.bfloat16,
            ),
            None,
        )

    fake_moe_mlp: Any = ModuleType("vllm_ascend.ops.fused_moe.moe_mlp")
    fake_moe_mlp.unified_apply_mlp = fake_unified_apply_mlp
    fake_stage_contracts: Any = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_contracts",
    )
    fake_stage_contracts.MoEMlpComputeInput = KeywordArguments
    fake_stage_contracts.MoEWeights = KeywordArguments
    fake_stage_params: Any = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_params",
    )
    fake_stage_params.MoEQuantParams = KeywordArguments
    fake_quant_type: Any = ModuleType("vllm_ascend.quantization.quant_type")
    fake_quant_type.QuantType = FakeQuantType
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_mlp",
        fake_moe_mlp,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_stage_contracts",
        fake_stage_contracts,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_stage_params",
        fake_stage_params,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.quant_type",
        fake_quant_type,
    )

    shared_calls = []

    def fake_compute_shared(
        shared_experts,
        hidden_states,
        dynamic_scales,
        *,
        swiglu_limit,
        output_dtype,
    ):
        shared_calls.append(
            (shared_experts, hidden_states, dynamic_scales, swiglu_limit),
        )
        return torch.zeros_like(hidden_states, dtype=output_dtype)

    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "_compute_w8a8_shared_experts_from_int8",
        fake_compute_shared,
    )
    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "_gmmswigluquant_fusion_enabled",
        lambda: False,
    )

    shared_experts = object()
    experts = SimpleNamespace(
        quant_type=FakeQuantType.W8A8,
        dynamic_eplb=False,
        get_eplb_parameter=lambda name: name,
        activation="silu",
        _shared_experts=shared_experts,
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=experts,
            routed_scaling_factor=2.0,
        ),
    )
    hidden_states = torch.zeros((num_routed_tokens, 4), dtype=torch.int8)
    expand_x_shared = torch.zeros((num_shared_tokens, 4), dtype=torch.int8)

    output = deepseek_v2_attention_gate.compute_attention_gate_moe_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.zeros(2, dtype=torch.int64),
        dynamic_scales=torch.ones(num_routed_tokens),
        expand_x_shared=expand_x_shared,
        dynamic_scales_shared=torch.ones(num_shared_tokens),
        topk_scales=None,
        group_list_type=1,
        routed_scale_applied_in_topk=routed_scale_applied_in_topk,
    )

    assert len(routed_calls) == int(num_routed_tokens > 0)
    assert output.routed_output.shape == hidden_states.shape
    assert output.routed_output.dtype == torch.bfloat16
    if num_routed_tokens > 0:
        assert torch.equal(
            output.routed_output,
            torch.full_like(output.routed_output, expected_routed_value),
        )
    assert len(shared_calls) == int(num_shared_tokens > 0)
    if num_shared_tokens > 0:
        assert output.shared_output is not None
        assert output.shared_output.shape == expand_x_shared.shape
        assert shared_calls[0][3] is None
    else:
        assert output.shared_output is None


def test_deepseek_afd_ffn_compute_omits_stub_io_diagnostics():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "camp2p_stub_io_enabled()" not in source
    assert "_log_ffn_compute_step(" not in compute_ffn_output
    assert '"dense_mlp_begin"' not in compute_ffn_output
    assert '"dense_scaling_begin"' not in compute_ffn_output
    assert "_log_ffn_compute_step(" not in compute_moe
    assert '"routed_scaling_begin"' not in compute_moe
    assert '"shared_scaling_begin"' not in compute_moe
