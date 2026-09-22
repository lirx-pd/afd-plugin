# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def runner_module(monkeypatch):
    pytest.importorskip("vllm", reason="NPU runtime tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="NPU runtime tests require vLLM-Ascend")
    pytest.importorskip("torch_npu", reason="NPU runtime tests require torch-npu")
    from afd_plugin.v1.worker.npu import attention_model_runner

    group = SimpleNamespace(cpu_group=object())
    monkeypatch.setattr(attention_model_runner, "get_dp_group", lambda: group)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", Event)
    return attention_model_runner


def _new_runners(module, settings=((100, 128), (100, 128))):
    runners = []
    for rank, setting in enumerate(settings):
        runner = object.__new__(module.AFDNPUAttentionModelRunner)
        runner.dp_size = len(settings)
        runner.dp_rank = rank
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=len(settings),
                data_parallel_rank=rank,
                enable_dbo=True,
                use_ubatching=True,
                num_ubatches=2,
                ubatch_size=0,
                tensor_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                dbo_decode_token_threshold=32,
                dbo_prefill_token_threshold=128,
            )
        )
        runner.connector = SimpleNamespace(control_plane=object())
        runner._adaptive_dbo = (
            None if setting is None else module.AdaptiveDBORuntime(*setting)
        )
        runners.append(runner)
    return runners


def _inputs(module, num_tokens=64, **overrides):
    values = {
        "num_tokens_unpadded": num_tokens,
        "num_tokens_padded": num_tokens,
        "uniform_decode": True,
        "is_draft_model": False,
        "cudagraph_mode": module.CUDAGraphMode.NONE,
        "allow_dp_padding": True,
        "allow_microbatching": True,
        "live": True,
        "decode": True,
        "context_length": 1024,
        "num_reqs": 64,
        "max_query_len": 1,
    }
    values.update(overrides)
    return values


def _sync_all_ranks(
    monkeypatch, module, runners, inputs, samples=None, expected_error=None
):
    if samples is None:
        samples = [(0, 0)] * len(runners)
    columns = []
    for runner, values, sample in zip(runners, inputs, samples, strict=True):
        adaptive = runner._adaptive_dbo
        monkeypatch.setattr(adaptive, "completed_sample", lambda s=sample: s)
        columns.append(
            [
                values["num_tokens_unpadded"],
                values["num_tokens_padded"],
                values["cudagraph_mode"].value,
                values["uniform_decode"],
                values["allow_microbatching"],
                values["allow_dp_padding"] or values["is_draft_model"],
                values["live"],
                values["decode"],
                values["context_length"],
                *sample,
                adaptive.policy.max_step_ms,
                adaptive.policy.probe_interval,
                values["num_reqs"],
                values["max_query_len"],
            ]
        )
    gathered = torch.tensor(columns, dtype=torch.int64).T
    results = []
    collectives = []
    for rank, (runner, values) in enumerate(zip(runners, inputs, strict=True)):

        def all_reduce(packed, *, group, rank=rank):
            assert group is module.get_dp_group().cpu_group
            assert packed.device.type == "cpu"
            assert packed.dtype == torch.int64
            assert packed.shape == (15, len(runners))
            torch.testing.assert_close(packed[:, rank], gathered[:, rank])
            own_column_removed = packed.clone()
            own_column_removed[:, rank] = 0
            assert torch.count_nonzero(own_column_removed).item() == 0
            packed.copy_(gathered)
            collectives.append(rank)

        monkeypatch.setattr(module.dist, "all_reduce", all_reduce)
        if expected_error is not None:
            with pytest.raises(ValueError, match=expected_error):
                runner._sync_adaptive_dbo_metadata_across_dp(**values)
        else:
            results.append(runner._sync_adaptive_dbo_metadata_across_dp(**values))
            runner._adaptive_dbo.finish()
    assert collectives == list(range(len(runners)))
    return results


@pytest.mark.parametrize("path", ["single_rank", "no_control_plane", "skip"])
def test_static_metadata_fast_paths_do_not_collect(monkeypatch, runner_module, path):
    settings = (None,) if path == "single_rank" else (None, None)
    runner = _new_runners(runner_module, settings)[0]
    if path == "no_control_plane":
        runner.connector.control_plane = None
    if path == "skip":
        runner.vllm_config.parallel_config.enable_dbo = False
        runner.vllm_config.parallel_config.use_ubatching = False

    monkeypatch.setattr(
        runner_module, "should_skip_allreduce_across_dp_group", lambda *_args: True
    )

    def unexpected_collective(*_args, **_kwargs):
        pytest.fail("Static metadata fast paths must not run a collective")

    monkeypatch.setattr(runner_module.dist, "all_reduce", unexpected_collective)
    use_dbo, max_tokens, padded, mode = runner._sync_afd_metadata_across_dp(
        num_tokens_unpadded=48,
        num_tokens_padded=64,
        uniform_decode=True,
        cudagraph_mode=runner_module.CUDAGraphMode.FULL,
    )
    assert use_dbo is (path != "skip")
    assert max_tokens == 64
    assert mode == runner_module.CUDAGraphMode.FULL
    if path == "single_rank":
        assert padded is None
    else:
        assert padded.dtype == torch.int32
        assert padded.tolist() == [64, 64]


@pytest.mark.parametrize(
    "enable_dbo, allow_dp_padding, is_draft_model, expected_dbo, expected_padding",
    [
        (True, False, False, [True, False], [[80, 80], [80, 48]]),
        (False, False, False, [False, False], [[80, 48], [80, 48]]),
        (False, True, False, [False, False], [[80, 80], [80, 80]]),
        (False, False, True, [False, False], [[80, 80], [80, 80]]),
    ],
)
def test_static_metadata_preserves_collective_thresholds_and_padding(
    monkeypatch,
    runner_module,
    enable_dbo,
    allow_dp_padding,
    is_draft_model,
    expected_dbo,
    expected_padding,
):
    runners = _new_runners(runner_module, settings=(None, None))
    mode = runner_module.CUDAGraphMode.NONE
    gathered = torch.tensor([[80, 48], [80, 48], [mode.value] * 2], dtype=torch.int32)
    collectives = []
    monkeypatch.setattr(
        runner_module,
        "should_skip_allreduce_across_dp_group",
        lambda *_args: enable_dbo,
    )
    for rank, runner in enumerate(runners):
        runner.vllm_config.parallel_config.enable_dbo = enable_dbo
        runner.vllm_config.parallel_config.use_ubatching = enable_dbo

        def all_reduce(packed, *, group, rank=rank):
            assert group is runner_module.get_dp_group().cpu_group
            assert packed.device.type == "cpu"
            assert packed.dtype == torch.int32
            assert packed.shape == (3, 2)
            torch.testing.assert_close(packed[:, rank], gathered[:, rank])
            assert torch.count_nonzero(packed[:, 1 - rank]).item() == 0
            packed.copy_(gathered)
            collectives.append(rank)

        monkeypatch.setattr(runner_module.dist, "all_reduce", all_reduce)
        use_dbo, max_tokens, padded, synced_mode = runner._sync_afd_metadata_across_dp(
            num_tokens_unpadded=(80, 48)[rank],
            uniform_decode=(True, False)[rank],
            allow_dp_padding=allow_dp_padding,
            is_draft_model=is_draft_model,
        )
        assert use_dbo is expected_dbo[rank]
        assert max_tokens == 80
        assert padded.dtype == torch.int32
        assert padded.tolist() == expected_padding[rank]
        assert synced_mode == mode
    assert collectives == [0, 1]


@pytest.mark.parametrize(
    "real_tokens, ffn_size, live, use_dbo",
    [
        ((80, 48, 64, 32), 2, True, False),
        ((80, 48, 64, 32), 2, False, False),
        ((65, 65, 65, 65), 2, True, True),
        ((65, 65, 65, 65), 2, False, True),
        ((65,), 1, True, True),
    ],
)
def test_adaptive_stages_match_existing_camp_ffn_counts(
    monkeypatch, runner_module, real_tokens, ffn_size, live, use_dbo
):
    from afd_plugin.connectors.npu.camp2p import _num_tokens_for_ffn_rank
    from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key
    from afd_plugin.v1.worker.npu.ffn_model_runner import (
        _ffn_token_counts_across_ranks,
    )

    attention_size = len(real_tokens)
    runners = _new_runners(runner_module, settings=((100, 128),) * attention_size)
    inputs = [
        _inputs(runner_module, count, live=live, num_reqs=count)
        for count in real_tokens
    ]
    results = []
    for sample in [(0, 0), (1, 20000)] if live and use_dbo else [(0, 0)]:
        if attention_size == 1:
            runtime = runners[0]._adaptive_dbo
            monkeypatch.setattr(runtime, "completed_sample", lambda s=sample: s)
            results = [runners[0]._sync_adaptive_dbo_metadata_across_dp(**inputs[0])]
            runtime.finish()
        else:
            results = _sync_all_ranks(
                monkeypatch,
                runner_module,
                runners,
                inputs,
                samples=[sample] * attention_size,
            )

    connector = SimpleNamespace(attn_size=attention_size, ffn_size=ffn_size)
    for runner, result, real_count in zip(runners, results, real_tokens, strict=True):
        selected, padded_count, counts, _mode = result
        assert selected is use_dbo
        runner.vllm_config.parallel_config.is_moe_model = True
        if use_dbo:
            slices, _ = runner_module.maybe_create_ubatch_slices(
                True,
                torch.ones(real_count, dtype=torch.int32).numpy(),
                padded_count,
                real_count,
                runner.vllm_config,
            )
            metadata_list = dict(
                enumerate(
                    runner_module.build_ubatch_dp_metadata_list(
                        runner.vllm_config, slices
                    )
                )
            )
        else:
            assert counts.tolist() == [max(real_tokens)] * attention_size
            metadata_list = {
                0: runner_module.DPMetadata.make(
                    runner.vllm_config.parallel_config, padded_count, counts
                )
            }

        expected_key = []
        for stage_idx, metadata in metadata_list.items():
            stage_counts = metadata.num_tokens_across_dp_cpu.tolist()
            assert stage_counts == [stage_counts[0]] * attention_size
            expected = tuple(
                sum(stage_counts[rank::ffn_size]) for rank in range(ffn_size)
            )
            assert (
                tuple(
                    _ffn_token_counts_across_ranks(
                        connector, metadata_list, stage_idx, fallback=1
                    ).tolist()
                )
                == expected
            )
            assert (
                tuple(
                    _num_tokens_for_ffn_rank(
                        metadata_list,
                        stage_idx,
                        ffn_rank=rank,
                        attention_size=attention_size,
                        ffn_size=ffn_size,
                        fallback=1,
                    )
                    for rank in range(ffn_size)
                )
                == expected
            )
            expected_key.append((stage_idx, expected))
        assert make_ffn_graph_key(
            metadata_list, attention_size=attention_size, ffn_size=ffn_size
        ) == tuple(expected_key)


def test_mixed_prefill_decode_uses_global_threshold(monkeypatch, runner_module):
    runners = _new_runners(runner_module)
    inputs = [
        _inputs(runner_module),
        _inputs(runner_module, uniform_decode=False, decode=False),
    ]
    results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    assert [result[0] for result in results] == [False, False]
    assert all(runner._adaptive_dbo.choice is None for runner in runners)
    assert all(runner._adaptive_dbo.policy.reason == "ineligible" for runner in runners)


def test_eligible_mixed_prefill_decode_uses_prefill_context(monkeypatch, runner_module):
    runners = _new_runners(runner_module)
    inputs = [
        _inputs(runner_module, 128),
        _inputs(runner_module, 128, uniform_decode=False, decode=False),
    ]
    results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    assert [result[0] for result in results] == [False, False]
    assert all(runner._adaptive_dbo.choice[1][:2] == (0, 1) for runner in runners)
    assert all(runner._adaptive_dbo.policy.reason == "probe_d1" for runner in runners)


def test_eager_unequal_fan_in_above_threshold_uses_padded_d1(
    monkeypatch, runner_module
):
    runners = _new_runners(runner_module)
    assert runner_module.check_enable_ubatch(
        48, 80, uniform_decode=True, vllm_config=runners[0].vllm_config
    )
    inputs = [_inputs(runner_module, 80), _inputs(runner_module, 48)]
    results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    for use_dbo, max_tokens, padded, mode in results:
        assert use_dbo is False
        assert max_tokens == 80
        assert padded.dtype == torch.int32
        assert padded.tolist() == [80, 80]
        assert mode == runner_module.CUDAGraphMode.NONE


def test_idle_rank_forces_d1(monkeypatch, runner_module):
    runners = _new_runners(runner_module)
    inputs = [
        _inputs(runner_module),
        _inputs(runner_module, 1, num_tokens_padded=64, live=False, decode=False),
    ]
    results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    assert [result[0] for result in results] == [False, False]
    assert all(result[2].tolist() == [64, 64] for result in results)
    assert all(runner._adaptive_dbo.choice is None for runner in runners)


@pytest.mark.parametrize("veto_rank", [0, 1])
def test_any_rank_can_veto_microbatching(monkeypatch, runner_module, veto_rank):
    runners = _new_runners(runner_module)
    inputs = [_inputs(runner_module), _inputs(runner_module)]
    _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    baseline = _sync_all_ranks(
        monkeypatch, runner_module, runners, inputs, samples=[(1, 20000)] * 2
    )
    assert [result[0] for result in baseline] == [True, True]
    inputs[veto_rank]["allow_microbatching"] = False
    results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    assert [result[0] for result in results] == [False, False]
    assert all(runner._adaptive_dbo.policy.reason == "ineligible" for runner in runners)


@pytest.mark.parametrize("other_setting", [(101, 2), (100, 3)])
def test_adaptive_configuration_mismatch_fails_on_every_rank(
    monkeypatch, runner_module, other_setting
):
    runners = _new_runners(runner_module, settings=((100, 2), other_setting))
    inputs = [_inputs(runner_module), _inputs(runner_module)]
    _sync_all_ranks(
        monkeypatch,
        runner_module,
        runners,
        inputs,
        expected_error="adaptive DBO settings must match",
    )


def test_completed_global_samples_switch_all_ranks_d1_d2_d1(monkeypatch, runner_module):
    runners = _new_runners(runner_module)
    inputs = [_inputs(runner_module, context_length=256), _inputs(runner_module)]
    samples = [(0, 0), (0, 0)]
    actions = []
    for _ in range(5):
        results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs, samples)
        assert results[0][0] == results[1][0]
        action = results[0][0]
        actions.append(action)
        choices = [runner._adaptive_dbo.choice for runner in runners]
        assert choices[0] == choices[1]
        assert choices[0][1] == (1, 1, 64, 64, 64, 4, 0, 64, 64, 1, 1)
        durations = [8000, 10000] if action else [18000, 20000]
        samples = [
            (choice[0], duration)
            for choice, duration in zip(choices, durations, strict=True)
        ]
    assert actions == [False, True, True, False, True]
    assert all(runner._adaptive_dbo.policy.reason == "dbo_gain" for runner in runners)

    samples[1] = (samples[1][0], 101000)
    results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs, samples)
    assert [result[0] for result in results] == [False, False]
    assert all(
        runner._adaptive_dbo.policy.reason == "latency_limit" for runner in runners
    )


def test_missing_completed_rank_sample_does_not_train_policy(
    monkeypatch, runner_module
):
    runners = _new_runners(runner_module)
    inputs = [_inputs(runner_module), _inputs(runner_module)]
    _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    missing = _sync_all_ranks(
        monkeypatch, runner_module, runners, inputs, samples=[(1, 20000), (0, 0)]
    )
    assert [result[0] for result in missing] == [False, False]
    assert all(runner._adaptive_dbo.choice[0] == 1 for runner in runners)
    assert all(
        runner._adaptive_dbo.policy.reason == "sample_pending" for runner in runners
    )
    completed = _sync_all_ranks(
        monkeypatch, runner_module, runners, inputs, samples=[(1, 18000), (1, 20000)]
    )
    assert [result[0] for result in completed] == [True, True]
    assert all(runner._adaptive_dbo.policy.reason == "probe_d2" for runner in runners)


def test_adaptive_graph_mode_consensus_synchronizes_all_ranks(
    monkeypatch, runner_module
):
    runners = _new_runners(runner_module)
    inputs = [
        _inputs(runner_module, 80, cudagraph_mode=runner_module.CUDAGraphMode.FULL),
        _inputs(runner_module, 48, allow_dp_padding=False),
    ]
    results = _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    assert all(result[3] == runner_module.CUDAGraphMode.NONE for result in results)
    assert all(result[2].tolist() == [80, 80] for result in results)
    assert [result[0] for result in results] == [False, False]
    assert runners[0]._adaptive_dbo.choice == runners[1]._adaptive_dbo.choice


def test_prefill_query_shape_change_starts_a_new_context(monkeypatch, runner_module):
    runners = _new_runners(runner_module)
    inputs = [
        _inputs(
            runner_module,
            num_tokens=128,
            num_reqs=64,
            max_query_len=2,
            uniform_decode=False,
            decode=False,
        )
        for _ in runners
    ]
    _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    old_context = runners[0]._adaptive_dbo.choice[1]
    for values in inputs:
        values.update(num_reqs=32, max_query_len=4)
    results = _sync_all_ranks(
        monkeypatch, runner_module, runners, inputs, samples=[(1, 20000), (1, 20000)]
    )
    assert [result[0] for result in results] == [False, False]
    assert runners[0]._adaptive_dbo.choice[1] != old_context
    assert runners[0]._adaptive_dbo.choice == runners[1]._adaptive_dbo.choice


def test_timing_starts_after_dp_collective(monkeypatch, runner_module):
    runner = _new_runners(runner_module)[0]
    events = []

    def all_reduce(packed, *, group):
        assert group is runner_module.get_dp_group().cpu_group
        events.append("collective")
        packed[:, 1] = packed[:, 0]

    def begin():
        assert runner._adaptive_dbo.choice is not None
        events.append("begin")

    monkeypatch.setattr(runner_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(runner._adaptive_dbo, "begin", begin)
    runner._sync_adaptive_dbo_metadata_across_dp(**_inputs(runner_module))
    assert events == ["collective", "begin"]


def test_raw_context_drift_within_bucket_rejects_probe(monkeypatch, runner_module):
    runners = _new_runners(runner_module)
    inputs = [_inputs(runner_module), _inputs(runner_module)]
    _sync_all_ranks(monkeypatch, runner_module, runners, inputs)
    context = runners[0]._adaptive_dbo.choice[1]
    inputs[0]["context_length"] = 1040
    results = _sync_all_ranks(
        monkeypatch, runner_module, runners, inputs, samples=[(1, 20000), (1, 20000)]
    )
    assert [result[0] for result in results] == [False, False]
    assert all(
        runner._adaptive_dbo.policy.reason == "workload_changed" for runner in runners
    )
    assert all(runner._adaptive_dbo.choice is None for runner in runners)
    assert all(context in runner._adaptive_dbo.policy._contexts for runner in runners)


@pytest.mark.parametrize("failure", [None, "begin", "forward"])
def test_execute_scope_records_only_successful_forward(
    monkeypatch, runner_module, failure
):
    runner = _new_runners(runner_module)[0]
    runner.prof = None
    runner._afd_live_execution = False
    runner._afd_is_graph_replaying = False
    events = []

    def begin():
        events.append("begin")
        if failure == "begin":
            raise RuntimeError("begin")

    def forward(*_args):
        assert runner._afd_live_execution
        events.append("forward")
        # Upstream execution reaches begin through the DP metadata hook.
        runner._adaptive_dbo.begin()
        if failure == "forward":
            raise RuntimeError("forward")
        return "output"

    runner._adaptive_dbo = SimpleNamespace(
        begin=begin, finish=lambda: events.append("finish")
    )
    monkeypatch.setattr(runner_module, "step_afd_npu_profiler", lambda _prof: None)
    monkeypatch.setattr(runner_module.NPUModelRunner, "execute_model", forward)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            runner.execute_model(object())
        assert "finish" not in events
    else:
        assert runner.execute_model(object()) == "output"
        assert events == ["forward", "begin", "finish"]
    assert not runner._afd_live_execution


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_dummy_scope_preserves_static_state_and_restores_after_execution(
    monkeypatch, runner_module, adaptive, live, failure
):
    settings = ((100, 128),) if adaptive else (None,)
    runner = _new_runners(runner_module, settings)[0]
    runner._afd_live_execution = live
    observed = []

    def dummy(num_tokens, **_kwargs):
        assert num_tokens == 64
        observed.append(runner._afd_live_execution)
        if failure:
            raise RuntimeError("dummy failure")
        return "output"

    monkeypatch.setattr(runner, "_dummy_run_inference_mode", dummy)
    if failure:
        with pytest.raises(RuntimeError, match="dummy failure"):
            runner._dummy_run(64)
    else:
        assert runner._dummy_run(64) == "output"
    assert observed == [live and not adaptive]
    assert runner._afd_live_execution is live
