# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Sparse model-boundary hooks with explicit native stage ownership."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from afd_plugin.profiling import runtime_profiling_enabled
from afd_plugin.profiling.fragments import build_fragments

if TYPE_CHECKING:
    from torch.nn import Module

    from afd_plugin.connectors import AFDForwardContextMetadata
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.model_executor.models.npu.async_cam_layout import CAMDispatchLayout
    from afd_plugin.profiling.tracing import Execution, HostSpan, Span


def span(execution: Execution | None, kind: str) -> Span | nullcontext:
    return (
        execution.tracer.span(execution, kind)
        if execution is not None
        else nullcontext()
    )


def host_span(execution: Execution | None, kind: str) -> HostSpan | nullcontext:
    return (
        execution.tracer.host_span(execution, kind)
        if execution is not None
        else nullcontext()
    )


def begin_profile_call(
    metadata: AFDForwardContextMetadata,
) -> tuple[int, Execution | None]:
    if not runtime_profiling_enabled():
        return 0, None
    attention = metadata.profile_attention
    if attention is not None:
        metadata.profile_attention = None
        return attention.sequence, attention.execution
    return cast("CAMAsyncAFDConnector", metadata.connector).begin_profile_call()


def record_source_call(
    execution: Execution | None,
    metadata: AFDForwardContextMetadata,
    layer_idx: int,
    layout: CAMDispatchLayout,
) -> None:
    if execution is None:
        return
    snapshot = metadata.profile_snapshot
    stage = metadata.stage_idx
    stage_begin = metadata.tokens_start_loc[
        stage if len(metadata.tokens_start_loc) > 1 else 0
    ]
    if layout.use_sequence_parallel:
        if metadata.profile_sp_origin is None:
            execution.capture.error = "sequence-parallel shard origin was not captured"
            execution.capture.accepting = False
            return
        local_begin = metadata.profile_sp_origin
    else:
        local_begin = layout.local_token_slice.start
    begin = stage_begin + local_begin
    end = begin + layout.local_tokens
    group = f"{execution.tracer.worker_id}:group:{metadata.transaction_id}"
    if metadata.profile_fragments is None:
        metadata.profile_fragments = {}
    key = (execution.capture.generation, stage, begin, end)
    if key not in metadata.profile_fragments:
        fragments = (
            build_fragments(snapshot, group, stage, begin, end) if snapshot else []
        )
        metadata.profile_fragments[key] = [
            fragment["fragment_id"] for fragment in fragments
        ]
        for fragment in fragments:
            execution.capture.writer.submit(fragment)
    execution.capture.writer.submit(
        {
            "record_type": "SOURCE_CALL",
            "source_call_id": execution.exec_ref,
            "group_id": group,
            "stage_idx": stage,
            "layer_idx": layer_idx,
            "fragment_ids": metadata.profile_fragments[key],
            "coverage": "COMPLETE"
            if snapshot is not None
            else "MISSING_INPUT_SNAPSHOT",
            "padded_tokens": layout.local_tokens
            - max(0, min(end, snapshot.real_extent) - begin)
            if snapshot
            else None,
            "expected_ffn_workers": execution.tracer.source_workers[
                cast("CAMAsyncAFDConnector", metadata.connector).attn_size :
            ],
        }
    )


@dataclass(slots=True)
class AttentionProfile:
    sequence: int
    execution: Execution | None
    scope: Span | nullcontext


def install_attention_hooks(
    module: Module,
    metadata_provider: Callable[[], AFDForwardContextMetadata | None],
) -> None:
    """Retain native V4 forward and attach timing only to its Attention module."""

    def begin(attention_module, args) -> None:
        metadata = metadata_provider()
        if metadata is None:
            return
        connector = cast("CAMAsyncAFDConnector", metadata.connector)
        sequence, execution = connector.begin_profile_call()
        scope = span(execution, "ATTENTION")
        metadata.profile_attention = AttentionProfile(sequence, execution, scope)
        scope.__enter__()

    def end(attention_module, args, output) -> None:
        metadata = metadata_provider()
        if metadata is not None and metadata.profile_attention is not None:
            metadata.profile_attention.scope.__exit__(None, None, None)

    module.register_forward_pre_hook(begin)
    module.register_forward_hook(end, always_call=True)
