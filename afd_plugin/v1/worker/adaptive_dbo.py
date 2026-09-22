# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deterministic DBO selection from completed, globally matched step timings."""

from collections import OrderedDict
from dataclasses import dataclass, field
from math import isfinite
from statistics import median

_MAX_CONTEXTS = 32
_MIN_GAIN = 0.05
_EARLY_GAIN = 0.10
_MAX_WORKLOAD_DRIFT = 0.01
_PHASE_D2_BUDGET = 4
_PROBE_HEADROOM = 0.90
_PROBE_ACTIONS = (False, True, True, False, False, True, True, False)
_BUCKET_PRECISION_BITS = 5


def _workload_bucket(value: int) -> int:
    # Keep small shapes exact; larger buckets span at most 1/16 of their scale.
    width = 1 << max(0, (value - 1).bit_length() - _BUCKET_PRECISION_BITS)
    return ((value + width - 1) // width) * width


def make_adaptive_dbo_context(
    *,
    decode: bool,
    real_tokens: list[int],
    padded_tokens: int,
    context_bucket: int,
    execution_mode: int,
    requests: list[int],
    query_lengths: list[int],
    eligible: bool,
) -> tuple[int, ...]:
    """Share timing history for nearby shapes without changing execution shapes."""
    return (
        int(decode),
        int(eligible),
        *(_workload_bucket(value) for value in real_tokens),
        _workload_bucket(padded_tokens),
        context_bucket,
        execution_mode,
        *(_workload_bucket(value) for value in requests),
        *(_workload_bucket(value) for value in query_lengths),
    )


@dataclass
class _ContextState:
    use_dbo: bool = False
    remaining: int = 0
    samples: list[tuple[bool, float]] = field(default_factory=list)
    reason: str = "cold_start"
    workload: tuple[int, ...] | None = None


class AdaptiveDBOPolicy:
    """Compare D1/D2 within a context; forward latency is not a service SLO.

    All ranks must supply identical contexts, eligibility and observations.
    The caller owns step/epoch matching and reports each completed global step
    at most once, using its original context and final action. Warmup, dummy
    steps and unmatched or obsolete observations must not reach this policy.
    """

    def __init__(self, max_step_ms: int = 0, probe_interval: int = 128) -> None:
        if max_step_ms < 0 or probe_interval <= 0:
            raise ValueError(
                "max_step_ms must be non-negative and probe_interval must be positive"
            )
        self.max_step_ms = max_step_ms
        self.probe_interval = probe_interval
        self.reason = "cold_start"
        self.needs_sample = False
        self._contexts: OrderedDict[tuple[int, ...], _ContextState] = OrderedDict()
        self._phase_steps = [0, 0]
        self._phase_probes = [0, 0]
        self._phase_samples = [0, 0]

    def choose(
        self,
        context: tuple[int, ...],
        eligible: bool,
        *,
        can_sample: bool = True,
        workload: tuple[int, ...] = (),
    ) -> bool:
        self.needs_sample = False
        if not eligible:
            self.reason = "ineligible"
            return False
        phase = int(bool(context[0]))
        if self._phase_steps[phase] >= max(self.probe_interval, len(_PROBE_ACTIONS)):
            self._phase_steps[phase] = 0
            self._phase_probes[phase] = 0
            self._phase_samples[phase] = 0
        self._phase_steps[phase] += 1
        state = self._contexts.get(context)
        if state is None:
            state = _ContextState()
            self._contexts[context] = state
            if len(self._contexts) > _MAX_CONTEXTS:
                self._contexts.popitem(last=False)
        self._contexts.move_to_end(context)

        if state.remaining:
            # Residence counts executions, so settled steps need no timing events.
            state.remaining -= 1
            self.needs_sample = can_sample and bool(self.max_step_ms)
            self.reason = state.reason
            return state.use_dbo

        if not can_sample:
            self.reason = "sample_pending"
            return False
        if (
            state.workload is not None
            and workload
            and any(
                abs(current - initial) > _MAX_WORKLOAD_DRIFT * max(initial, 1)
                for current, initial in zip(workload, state.workload, strict=True)
            )
        ):
            # Buckets reuse decisions; a timing comparison also needs similar work.
            self._settle(state, False, "workload_changed")
            return False
        use_dbo = _PROBE_ACTIONS[len(state.samples)]
        if self._phase_samples[phase] >= len(_PROBE_ACTIONS) or (
            self._phase_probes[phase] >= _PHASE_D2_BUDGET
            and (use_dbo or not state.samples)
        ):
            self._settle(state, False, "probe_budget")
            return False
        if not state.samples:
            state.workload = workload
        if use_dbo:
            # Charge actual probes, including executions whose sample is delayed.
            self._phase_probes[phase] += 1
        self._phase_samples[phase] += 1
        self.needs_sample = True
        self.reason = "probe_d2" if use_dbo else "probe_d1"
        return use_dbo

    def observe(
        self, context: tuple[int, ...], use_dbo: bool, elapsed_ms: float
    ) -> None:
        if not isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        state = self._contexts.get(context)
        if state is None:
            return
        if self.max_step_ms and elapsed_ms > self.max_step_ms:
            self._settle(state, False, "latency_limit")
            return

        if state.remaining or state.workload is None:
            return

        expected_action = _PROBE_ACTIONS[len(state.samples)]
        if use_dbo != expected_action:
            # Delayed repeated actions cannot advance the alternating window.
            return
        if (
            self.max_step_ms
            and not use_dbo
            and elapsed_ms >= self.max_step_ms * _PROBE_HEADROOM
        ):
            self._settle(state, False, "latency_headroom")
            return

        state.samples.append((use_dbo, elapsed_ms))
        if len(state.samples) not in (len(_PROBE_ACTIONS) // 2, len(_PROBE_ACTIONS)):
            return
        d1 = [elapsed for dbo, elapsed in state.samples if not dbo]
        d2 = [elapsed for dbo, elapsed in state.samples if dbo]
        if max(d2) <= min(d1) * (1 - _EARLY_GAIN):
            self._settle(state, True, "dbo_gain")
        elif min(d2) > max(d1) * (1 - _MIN_GAIN):
            self._settle(state, False, "no_gain")
        elif len(state.samples) == len(_PROBE_ACTIONS):
            d1_ms, d2_ms = median(d1), median(d2)
            # ponytail: a range margin is conservative; no online statistical model.
            noise_ms = max(max(d1) - min(d1), max(d2) - min(d2)) / 2
            use_dbo = d2_ms + noise_ms <= d1_ms * (1 - _MIN_GAIN)
            self._settle(
                state, use_dbo, "dbo_gain" if use_dbo else "insufficient_evidence"
            )

    def _settle(self, state: _ContextState, use_dbo: bool, reason: str) -> None:
        state.use_dbo = use_dbo
        state.remaining = self.probe_interval
        state.samples.clear()
        state.workload = None
        state.reason = reason
        self.reason = reason
