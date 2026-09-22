# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import pytest

from afd_plugin.v1.worker.adaptive_dbo import AdaptiveDBOPolicy

_CONTEXT = (0, 1, 64, 64, 8, 0)


def _complete_window(
    policy: AdaptiveDBOPolicy,
    d1_ms: float,
    d2_ms: float,
    context: tuple[int, ...] = _CONTEXT,
    *,
    rounds: int = 1,
) -> None:
    for expected_action in (False, True, True, False) * rounds:
        action = policy.choose(context, eligible=True)
        assert action is expected_action
        assert policy.needs_sample
        policy.observe(context, action, d2_ms if action else d1_ms)


@pytest.mark.parametrize(
    ("d2_ms", "expected", "rounds"),
    [
        (10.0, True, 1),
        (18.0, True, 1),
        (19.0, True, 2),
        (19.1, False, 1),
        (25.0, False, 1),
    ],
)
def test_clear_outcomes_stop_early_and_small_gains_require_two_rounds(
    d2_ms, expected, rounds
):
    policy = AdaptiveDBOPolicy(max_step_ms=100)
    _complete_window(policy, d1_ms=20, d2_ms=d2_ms, rounds=rounds)
    assert policy.choose(_CONTEXT, eligible=True) is expected
    assert policy.reason == ("dbo_gain" if expected else "no_gain")


def test_small_gain_requires_four_completed_samples_per_action():
    policy = AdaptiveDBOPolicy(max_step_ms=100)
    for action in [False, True, True, False, False, True, True]:
        assert policy.choose(_CONTEXT, eligible=True) is action
        policy.observe(_CONTEXT, action, 19 if action else 20)
    assert policy.choose(_CONTEXT, eligible=True) is False
    assert policy.reason == "probe_d1"
    policy.observe(_CONTEXT, False, 20)
    assert policy.choose(_CONTEXT, eligible=True) is True
    assert policy.reason == "dbo_gain"


def test_noisy_timings_remain_insufficient_after_eight_samples():
    policy = AdaptiveDBOPolicy(max_step_ms=1000)
    d1_samples = iter([100, 10, 10, 10])
    d2_samples = iter([11, 11, 1, 11])
    for _ in range(8):
        action = policy.choose(_CONTEXT, eligible=True)
        policy.observe(_CONTEXT, action, next(d2_samples if action else d1_samples))
    assert policy.choose(_CONTEXT, eligible=True) is False
    assert policy.reason == "insufficient_evidence"


def test_residency_counts_choices_without_samples_and_replaces_previous_window():
    policy = AdaptiveDBOPolicy(probe_interval=2)
    _complete_window(policy, d1_ms=20, d2_ms=10)
    for _ in range(2):
        assert policy.choose(_CONTEXT, eligible=True) is True
        assert not policy.needs_sample
    _complete_window(policy, d1_ms=10, d2_ms=20)
    assert policy.choose(_CONTEXT, eligible=True) is False
    assert policy.reason == "no_gain"


@pytest.mark.parametrize("committed", [False, True])
def test_over_limit_d2_immediately_returns_to_d1_and_cools_down(committed):
    policy = AdaptiveDBOPolicy(max_step_ms=100, probe_interval=2)
    if committed:
        _complete_window(policy, d1_ms=20, d2_ms=10)
    else:
        assert policy.choose(_CONTEXT, eligible=True) is False
        policy.observe(_CONTEXT, False, 20)
    assert policy.choose(_CONTEXT, eligible=True) is True
    policy.observe(_CONTEXT, True, 101)
    for _ in range(2):
        assert policy.choose(_CONTEXT, eligible=True) is False
        assert policy.reason == "latency_limit"
        policy.observe(_CONTEXT, False, 20)
    assert policy.choose(_CONTEXT, eligible=True) is False
    policy.observe(_CONTEXT, False, 20)
    assert policy.choose(_CONTEXT, eligible=True) is True


@pytest.mark.parametrize("elapsed_ms", [90.0, 99.0, 100.0, 101.0])
def test_baseline_without_latency_headroom_does_not_probe_d2(elapsed_ms):
    policy = AdaptiveDBOPolicy(max_step_ms=100, probe_interval=2)
    for _ in range(12):
        assert policy.choose(_CONTEXT, eligible=True) is False
        policy.observe(_CONTEXT, False, elapsed_ms)


@pytest.mark.parametrize("committed", [False, True])
def test_ineligible_keeps_exploration_or_residency_for_later_eligible_work(committed):
    policy = AdaptiveDBOPolicy(max_step_ms=100, probe_interval=2)
    if committed:
        _complete_window(policy, d1_ms=20, d2_ms=10)
    else:
        assert policy.choose(_CONTEXT, eligible=True) is False
        policy.observe(_CONTEXT, False, 20)
    state = policy._contexts[_CONTEXT]
    previous_samples = state.samples.copy()
    previous_remaining = state.remaining
    previous_steps = policy._phase_steps.copy()
    assert policy.choose(_CONTEXT, eligible=False) is False
    assert policy.reason == "ineligible"
    assert not policy.needs_sample
    assert state.samples == previous_samples
    assert state.remaining == previous_remaining
    assert policy._phase_steps == previous_steps
    assert policy.choose(_CONTEXT, eligible=True) is True
    assert policy.reason == ("dbo_gain" if committed else "probe_d2")


def test_pending_sample_does_not_advance_exploration_or_spend_probe_budget():
    policy = AdaptiveDBOPolicy(max_step_ms=100)
    assert policy.choose(_CONTEXT, eligible=True) is False
    for _ in range(20):
        assert policy.choose(_CONTEXT, eligible=True, can_sample=False) is False
        assert policy.reason == "sample_pending"
        assert not policy.needs_sample
    policy.observe(_CONTEXT, False, 20)
    for action in [True, True, False]:
        assert policy.choose(_CONTEXT, eligible=True) is action
        assert policy.needs_sample
        policy.observe(_CONTEXT, action, 10 if action else 20)
    assert policy.choose(_CONTEXT, eligible=True) is True
    assert policy.reason == "dbo_gain"


@pytest.mark.parametrize(
    "elapsed_ms", [float("nan"), float("inf"), -float("inf"), -1, 0]
)
def test_invalid_observations_are_discarded(elapsed_ms):
    policy = AdaptiveDBOPolicy(max_step_ms=100)
    assert policy.choose(_CONTEXT, eligible=True) is False
    policy.observe(_CONTEXT, False, elapsed_ms)
    assert policy.choose(_CONTEXT, eligible=True) is False
    policy.observe(_CONTEXT, False, 20)
    policy.observe(_CONTEXT, True, elapsed_ms)
    assert policy.choose(_CONTEXT, eligible=True) is True
    policy.observe(_CONTEXT, True, 10)
    assert policy.choose(_CONTEXT, eligible=True) is True


def test_contexts_keep_independent_decisions_and_late_samples():
    policy = AdaptiveDBOPolicy(max_step_ms=100)
    other = (0, 1, 128, 128, 8, 0)
    _complete_window(policy, d1_ms=20, d2_ms=10)
    assert policy.choose(other, eligible=True) is False
    policy.observe(_CONTEXT, True, 10)
    assert policy.choose(other, eligible=True) is False
    assert policy.choose(_CONTEXT, eligible=True) is True


def test_lru_is_bounded_and_evicted_observations_do_not_recreate_context():
    policy = AdaptiveDBOPolicy(max_step_ms=100)
    _complete_window(policy, d1_ms=20, d2_ms=10, context=(0, 0))
    for index in range(1, 32):
        assert policy.choose((0, index), eligible=True) is False
    assert policy.choose((0, 0), eligible=True) is True
    assert policy.choose((0, 32), eligible=True) is False
    policy.observe((0, 1), False, 20)
    assert len(policy._contexts) == 32
    assert (0, 1) not in policy._contexts
    assert policy.choose((0, 1), eligible=True) is False
    assert policy.choose((0, 0), eligible=True) is True


def test_unexpected_action_does_not_advance_alternating_window():
    policy = AdaptiveDBOPolicy(max_step_ms=100)
    assert policy.choose(_CONTEXT, eligible=True) is False
    policy.observe(_CONTEXT, True, 10)
    _complete_window(policy, d1_ms=20, d2_ms=10)


def test_identical_global_observations_produce_identical_rank_decisions():
    policies = [AdaptiveDBOPolicy(max_step_ms=100, probe_interval=2) for _ in range(2)]
    for index in range(40):
        context = (index % 2,)
        eligible = index != 17
        actions = [policy.choose(context, eligible) for policy in policies]
        assert actions[0] == actions[1]
        assert policies[0].needs_sample == policies[1].needs_sample
        for policy, action in zip(policies, actions, strict=True):
            if policy.needs_sample:
                policy.observe(context, action, 10 if action else 20)
        assert policies[0].reason == policies[1].reason


@pytest.mark.parametrize(("max_step_ms", "probe_interval"), [(-1, 1), (100, 0)])
def test_requires_positive_limits(max_step_ms, probe_interval):
    with pytest.raises(ValueError, match="must be"):
        AdaptiveDBOPolicy(max_step_ms=max_step_ms, probe_interval=probe_interval)


@pytest.mark.parametrize("scale", [0.01, 1, 1000])
@pytest.mark.parametrize("d2_ms, expected", [(10, True), (25, False)])
def test_default_limit_selects_by_relative_gain(scale, d2_ms, expected):
    policy = AdaptiveDBOPolicy()
    _complete_window(policy, d1_ms=20 * scale, d2_ms=d2_ms * scale)
    assert policy.choose(_CONTEXT, eligible=True) is expected
    assert policy.reason == ("dbo_gain" if expected else "no_gain")


def test_probe_budget_is_shared_across_buckets_but_not_phases_or_settled_choices():
    policy = AdaptiveDBOPolicy()
    for bucket in range(2):
        _complete_window(policy, d1_ms=20, d2_ms=10, context=(0, bucket))
    assert policy.choose((0, 2), eligible=True) is False
    assert policy.reason == "probe_budget"
    assert not policy.needs_sample
    assert policy.choose((0, 0), eligible=True) is True
    assert policy.reason == "dbo_gain"
    assert not policy.needs_sample
    _complete_window(policy, d1_ms=20, d2_ms=10, context=(1, 0))
    assert policy.choose((1, 0), eligible=True) is True


def test_d2_attempts_spend_budget_even_without_completed_observations():
    policy = AdaptiveDBOPolicy()
    assert policy.choose(_CONTEXT, eligible=True) is False
    policy.observe(_CONTEXT, False, 20)
    for _ in range(4):
        assert policy.choose(_CONTEXT, eligible=True) is True
        assert policy.needs_sample
    assert policy.choose(_CONTEXT, eligible=True) is False
    assert policy.reason == "probe_budget"
    assert not policy.needs_sample


def test_all_probe_timings_are_bounded_and_only_eligible_steps_renew_budget():
    policy = AdaptiveDBOPolicy(probe_interval=10)
    for bucket in range(8):
        assert policy.choose((0, bucket), eligible=True) is False
        assert policy.needs_sample
    assert policy.choose((0, 8), eligible=True) is False
    assert policy.reason == "probe_budget"
    assert not policy.needs_sample
    for _ in range(20):
        assert policy.choose((0, 9), eligible=False) is False
        assert not policy.needs_sample
    assert policy.choose((0, 9), eligible=True) is False
    assert policy.reason == "probe_budget"
    assert not policy.needs_sample
    assert policy.choose((0, 10), eligible=True) is False
    assert policy.reason == "probe_d1"
    assert policy.needs_sample


@pytest.mark.parametrize("current", [989, 990, 1000, 1010, 1011])
def test_raw_workload_drift_above_one_percent_stops_comparison(current):
    policy = AdaptiveDBOPolicy()
    assert policy.choose(_CONTEXT, eligible=True, workload=(1000, 64)) is False
    policy.observe(_CONTEXT, False, 20)
    comparable = 990 <= current <= 1010
    assert policy.choose(_CONTEXT, eligible=True, workload=(current, 64)) is comparable
    assert policy.needs_sample is comparable
    assert policy.reason == ("probe_d2" if comparable else "workload_changed")


@pytest.mark.parametrize("max_step_ms", [0, 100])
def test_settled_timing_requires_a_latency_limit_and_available_events(max_step_ms):
    policy = AdaptiveDBOPolicy(max_step_ms=max_step_ms)
    _complete_window(policy, d1_ms=20, d2_ms=10)
    assert policy.choose(_CONTEXT, eligible=True) is True
    assert policy.needs_sample is bool(max_step_ms)
    assert policy.choose(_CONTEXT, eligible=True, can_sample=False) is True
    assert not policy.needs_sample
