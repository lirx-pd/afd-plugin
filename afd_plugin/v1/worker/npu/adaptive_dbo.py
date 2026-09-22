# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Completed Attention forward/logits timing for synchronous Ascend DBO."""

from __future__ import annotations

import torch

from afd_plugin.v1.worker.adaptive_dbo import AdaptiveDBOPolicy


class AdaptiveDBORuntime:
    def __init__(self, max_step_ms: int, probe_interval: int) -> None:
        self.policy = AdaptiveDBOPolicy(max_step_ms, probe_interval)
        self.step = 0
        self.choice: tuple[int, tuple[int, ...], bool] | None = None
        self.start_event: torch.npu.Event | None = None
        self.end_event: torch.npu.Event | None = None
        self.pending = False
        self.elapsed_us = 0
        self.record_current = False

    def begin(self) -> None:
        if not self.record_current:
            return
        if self.start_event is None:
            self.start_event = torch.npu.Event(enable_timing=True)
            self.end_event = torch.npu.Event(enable_timing=True)
        self.start_event.record()

    def finish(self) -> None:
        if self.record_current:
            assert self.end_event is not None
            self.end_event.record()
            self.pending = True
            self.record_current = False

    def completed_sample(self) -> tuple[int, int]:
        if not self.pending:
            return 0, 0
        assert self.choice is not None
        assert self.start_event is not None and self.end_event is not None
        if not self.elapsed_us:
            # Keep one sample until every rank reports it; never wait on the device.
            if not self.end_event.query():
                return 0, 0
            self.elapsed_us = max(
                1, round(self.start_event.elapsed_time(self.end_event) * 1000)
            )
        return self.choice[0], self.elapsed_us

    def select(
        self,
        context: tuple[int, ...],
        eligible: bool,
        live: bool,
        sample_steps: list[int],
        elapsed_us: list[int],
        *,
        workload: tuple[int, ...] = (),
    ) -> bool:
        # All A ranks consume the same gathered observation and run the same
        # deterministic policy. This avoids a second control collective.
        if self.choice is not None:
            _, previous_context, previous_dbo = self.choice
            if all(step == self.choice[0] for step in sample_steps) and all(
                duration > 0 for duration in elapsed_us
            ):
                self.policy.observe(
                    previous_context, previous_dbo, max(elapsed_us) / 1000
                )
                self.choice = None
                self.pending = False
                self.elapsed_us = 0
        self.step += 1
        self.record_current = False
        if not live:
            return eligible
        use_dbo = self.policy.choose(
            context, eligible, can_sample=self.choice is None, workload=workload
        )
        if self.policy.needs_sample:
            self.choice = (self.step, context, use_dbo)
            self.record_current = True
        return use_dbo
