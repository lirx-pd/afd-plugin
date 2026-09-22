# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from afd_plugin.v1.worker.npu import adaptive_dbo


@pytest.fixture
def events(monkeypatch):
    events = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.ready = False
            self.records = 0
            self.queries = 0
            self.elapsed_queries = 0
            self.elapsed_ms = 1.25
            events.append(self)

        def record(self):
            self.ready = False
            self.records += 1

        def query(self):
            self.queries += 1
            return self.ready

        def elapsed_time(self, end):
            self.elapsed_queries += 1
            return end.elapsed_ms

    monkeypatch.setattr(
        adaptive_dbo.torch, "npu", SimpleNamespace(Event=Event), raising=False
    )
    return events


def test_sample_is_retained_until_all_ranks_acknowledge_and_events_are_reused(events):
    runtime = adaptive_dbo.AdaptiveDBORuntime(20, 8)
    assert not runtime.select((64,), True, True, [0, 0], [0, 0])
    runtime.begin()
    runtime.finish()
    assert runtime.completed_sample() == (0, 0)
    assert runtime.pending
    events[-1].ready = True
    assert runtime.completed_sample() == (1, 1250)
    assert runtime.completed_sample() == (1, 1250)
    assert events[-1].queries == 2
    assert events[0].elapsed_queries == 1

    assert not runtime.select((64,), True, True, [1, 0], [1250, 0])
    assert runtime.policy.reason == "sample_pending"
    runtime.begin()
    runtime.finish()
    assert runtime.completed_sample() == (1, 1250)
    assert [event.records for event in events] == [1, 1]

    assert runtime.select((64,), True, True, [1, 1], [1250, 1300])
    assert runtime.completed_sample() == (0, 0)
    runtime.begin()
    runtime.finish()
    assert len(events) == 2
    assert [event.records for event in events] == [2, 2]
    assert runtime.completed_sample() == (0, 0)
    events[-1].elapsed_ms = 2.5
    events[-1].ready = True
    assert runtime.completed_sample() == (3, 2500)


@pytest.mark.parametrize(
    "sample_steps, durations",
    [([0, 1], [0, 1000]), ([1, 1], [1000, 0]), ([99, 99], [1000, 1000])],
)
def test_unmatched_observation_keeps_pending_probe_and_executes_d1(
    events, sample_steps, durations
):
    runtime = adaptive_dbo.AdaptiveDBORuntime(20, 8)
    context = (64,)
    assert not runtime.select(context, True, True, [0, 0], [0, 0])
    runtime.begin()
    runtime.finish()
    assert runtime.select(context, True, True, [1, 1], [10000, 11000])
    runtime.begin()
    runtime.finish()
    assert not runtime.select(context, True, True, sample_steps, durations)
    assert runtime.policy.reason == "sample_pending"
    assert runtime.choice == (2, context, True)
    runtime.begin()
    runtime.finish()
    assert [event.records for event in events] == [2, 2]
    assert runtime.select(context, True, True, [2, 2], [8000, 9000])
    assert runtime.choice == (4, context, True)


def test_slowest_rank_latency_forces_d1(events):
    runtime = adaptive_dbo.AdaptiveDBORuntime(20, 8)
    context = (64,)
    runtime.select(context, True, True, [0, 0], [0, 0])
    runtime.begin()
    runtime.finish()
    assert runtime.select(context, True, True, [1, 1], [10000, 11000])
    runtime.begin()
    runtime.finish()
    assert not runtime.select(context, True, True, [2, 2], [5000, 21000])
    assert runtime.policy.reason == "latency_limit"


@pytest.mark.parametrize(
    "eligible, live", [(True, False), (False, False), (False, True)]
)
def test_dummy_or_ineligible_steps_do_not_record_events(events, eligible, live):
    runtime = adaptive_dbo.AdaptiveDBORuntime(20, 8)
    assert runtime.select((64,), eligible, live, [0], [0]) is (eligible and not live)
    runtime.begin()
    runtime.finish()
    runtime.finish()
    assert events == []
    assert runtime.choice is None
    assert not runtime.record_current
    assert runtime.completed_sample() == (0, 0)


def test_repeated_finish_without_selection_does_not_overwrite_pending_sample(events):
    runtime = adaptive_dbo.AdaptiveDBORuntime(0, 128)
    assert not runtime.select((64,), True, True, [0], [0])
    runtime.begin()
    runtime.finish()
    events[-1].ready = True
    assert runtime.completed_sample() == (1, 1250)
    # A zero-token step can return before selecting another action.
    runtime.finish()
    assert [event.records for event in events] == [1, 1]
    assert runtime.completed_sample() == (1, 1250)


@pytest.mark.parametrize("d2_ms, expected_dbo", [(8.0, True), (10.0, False)])
def test_settled_steps_do_not_record_events_without_latency_limit(
    events, d2_ms, expected_dbo
):
    runtime = adaptive_dbo.AdaptiveDBORuntime(0, 128)
    context = (64,)
    sample_step, elapsed_us = 0, 0
    for expected_action in (False, True, True, False):
        assert (
            runtime.select(context, True, True, [sample_step], [elapsed_us])
            is expected_action
        )
        runtime.begin()
        runtime.finish()
        events[-1].elapsed_ms = d2_ms if expected_action else 10.0
        events[-1].ready = True
        sample_step, elapsed_us = runtime.completed_sample()

    for _ in range(3):
        assert (
            runtime.select(context, True, True, [sample_step], [elapsed_us])
            is expected_dbo
        )
        runtime.begin()
        runtime.finish()
        assert runtime.choice is None
        assert not runtime.record_current
        sample_step, elapsed_us = runtime.completed_sample()
        assert (sample_step, elapsed_us) == (0, 0)
    assert len(events) == 2
    assert [event.records for event in events] == [4, 4]
