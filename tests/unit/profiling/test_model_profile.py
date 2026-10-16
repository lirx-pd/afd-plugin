# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace
from unittest.mock import Mock

from afd_plugin.profiling.model import begin_profile_call, install_pipeline_hook


def test_idle_pipeline_has_no_layer_lookup():
    assert begin_profile_call(SimpleNamespace(profile_pipeline=None)) is None


def test_pipeline_caches_track_but_checks_admission_each_layer():
    tracer = SimpleNamespace(retain_execution=Mock(return_value=True))
    execution = SimpleNamespace(tracer=tracer)
    metadata = SimpleNamespace(profile_pipeline=execution)
    assert begin_profile_call(metadata) is execution
    assert begin_profile_call(metadata) is execution
    assert metadata.profile_pipeline is execution
    tracer.retain_execution.return_value = False
    assert begin_profile_call(metadata) is None


def test_model_hook_refreshes_stream_for_each_dbo_child_and_generation():
    module = SimpleNamespace(register_forward_pre_hook=Mock())
    tracer = SimpleNamespace(execution=Mock(), transition=Mock())
    metadata = SimpleNamespace(
        connector=SimpleNamespace(profile_session=SimpleNamespace(tracer=tracer)),
        profile_pipeline=None,
    )
    install_pipeline_hook(module, lambda: metadata)
    hook = module.register_forward_pre_hook.call_args.args[0]
    for _ in range(3):
        execution = SimpleNamespace(tracer=tracer)
        tracer.execution.return_value = execution
        hook(module, ())
        assert metadata.profile_pipeline is execution
        tracer.transition.assert_called_with(execution, "COMPUTE")
    tracer.execution.return_value = None
    hook(module, ())
    assert metadata.profile_pipeline is None
