# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in worker RPC ownership; importing this module does not load TorchNPU."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from afd_plugin.profiling import runtime_profiling_enabled

if TYPE_CHECKING:
    import torch
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

    from afd_plugin.profiling.session import WorkerSession
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner


class CAMProfileWorkerMixin:
    device: torch.device
    model_runner: (
        AFDNPUAttentionModelRunner | AFDNPUAttentionModelRunnerV2 | AFDNPUFFNModelRunner
    )
    profiler: TorchNPUProfilerWrapper | None
    cam_profile_session: WorkerSession | None = None
    _cam_native_profiler_active = False

    def initialize_cam_profiling(self) -> None:
        if not runtime_profiling_enabled():
            return
        # Keep TorchNPU and the event pool out of workers with profiling disabled.
        from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
        from afd_plugin.profiling.session import WorkerSession
        from afd_plugin.profiling.tracing import DeviceTracer

        connector = self.model_runner.connector
        if not isinstance(connector, CAMAsyncAFDConnector):
            raise ValueError("runtime CAM profiling requires the async CAM connector")
        tracer = DeviceTracer(
            self.device,
            connector.profile_worker_id,
            connector.profile_peers,
            connector.tp_size,
            connector.expert_per_rank,
        )
        self.cam_profile_session = WorkerSession(
            connector.profile_worker_id,
            connector.topology.role,
            Path(os.environ.get("AFD_CAM_PROFILE_DIR", "cam-profiles")),
            tracer,
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "device": str(self.device),
                "cam_rank": connector.world_rank,
                "attention_ranks": connector.attn_size,
                "ffn_ranks": connector.ffn_size,
                "tp_size": connector.tp_size,
                "protocol_version": 1,
            },
        )
        connector.profile_session = self.cam_profile_session
        tracer.start(self.cam_profile_session)

    def _require_cam_profile_session(self) -> WorkerSession:
        if self.cam_profile_session is None:
            raise ValueError("runtime CAM profiling was not enabled at startup")
        return self.cam_profile_session

    def cam_profile_prepare(self, command: dict) -> dict:
        return self._cam_profile_command("prepare", command)

    def cam_profile_commit(self, command: dict) -> dict:
        return self._cam_profile_command("commit", command)

    def cam_profile_stop(self, command: dict) -> dict:
        return self._cam_profile_command("stop", command)

    def _cam_profile_command(
        self, action: Literal["prepare", "commit", "stop"], command: dict
    ) -> dict:
        session = self._require_cam_profile_session()
        try:
            if action == "prepare":
                return session.prepare(
                    command,
                    backend_busy=self._cam_native_profiler_active
                    or self.model_runner.prof is not None,
                )
            if action == "commit":
                return session.commit(command)
            return session.stop(command)
        except (ValueError, OSError) as exc:
            result = session.status()
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result

    def cam_profile_status(self, command: dict) -> dict:
        return self._require_cam_profile_session().status(command)

    # Ascend 8d4409d6 NPUWorker.profile also owns a native profiler. Its large
    # backend setup remains upstream; this guard adds only runtime exclusivity.
    # Signature matches upstream, including its omitted return annotation.
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # ### PATCH START: CAM session exclusivity
        session = self.cam_profile_session
        if (
            is_start
            and session is not None
            and session.state
            not in {
                "IDLE",
                "COMPLETED",
                "FAILED",
            }
        ):
            raise ValueError("a runtime CAM profiling session is active")
        # ### PATCH END: CAM session exclusivity
        if is_start:
            self._cam_native_profiler_active = True
        try:
            # This mixin is composed immediately before the pinned NPUWorker.
            result = super().profile(is_start, profile_prefix)  # type: ignore[misc]
        except Exception:
            if self.profiler is None:
                self._cam_native_profiler_active = False
            raise
        self._cam_native_profiler_active = is_start
        return result
