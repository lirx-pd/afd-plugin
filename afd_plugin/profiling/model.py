# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Sparse model-boundary hooks with explicit native stage ownership."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

from afd_plugin.profiling import runtime_profiling_enabled

if TYPE_CHECKING:
    from torch.nn import Module

    from afd_plugin.connectors import AFDForwardContextMetadata
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.profiling.tracing import Execution, Span


def span(execution: Execution | None, kind: str) -> Span | nullcontext:
    return (
        execution.tracer.span(execution, kind)
        if execution is not None
        else nullcontext()
    )


def host_span(execution: Execution | None, kind: str) -> Span | nullcontext:
    return (
        execution.tracer.host_span(execution, kind)
        if execution is not None
        else nullcontext()
    )


def begin_profile_call(metadata: AFDForwardContextMetadata) -> Execution | None:
    if not runtime_profiling_enabled():
        return None
    attention = metadata.profile_attention
    if attention is not None:
        metadata.profile_attention = None
        return attention
    return cast("CAMAsyncAFDConnector", metadata.connector).begin_profile_call()


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
        execution = connector.begin_profile_call()
        metadata.profile_attention = execution
        if execution is not None:
            execution.tracer.transition(execution, "COMPUTE")

    module.register_forward_pre_hook(begin)
