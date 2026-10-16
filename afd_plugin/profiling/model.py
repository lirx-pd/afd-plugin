# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cache pipeline tracks at model entry and mark only phase boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch.nn import Module

    from afd_plugin.connectors import AFDConnectorBase, AFDForwardContextMetadata
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.profiling.tracing import Execution


def transition(execution: Execution | None, phase: str) -> None:
    if execution is not None:
        execution.tracer.transition(execution, phase)


def begin_profile_call(metadata: AFDForwardContextMetadata) -> Execution | None:
    execution = metadata.profile_pipeline
    if execution is not None and execution.tracer.retain_execution(execution):
        return execution
    return None


def begin_profile_step(connector: AFDConnectorBase) -> Execution | None:
    """Keep inter-layer compute contiguous; idle only at the model-step boundary."""
    session = cast("CAMAsyncAFDConnector", connector).profile_session
    if session is None or session.tracer is None:
        return None
    execution = session.tracer.execution()
    if execution is not None:
        execution.tracer.transition(execution, "COMPUTE")
    return execution


def install_pipeline_hook(
    module: Module,
    metadata_provider: Callable[[], AFDForwardContextMetadata | None],
) -> None:
    """One hook per model also runs on each native DBO child stream."""

    def begin(model_module, args) -> None:
        metadata = metadata_provider()
        if metadata is None:
            return
        metadata.profile_pipeline = begin_profile_step(metadata.connector)

    module.register_forward_pre_hook(begin)
