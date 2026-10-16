# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace
from unittest.mock import Mock

from afd_plugin.profiling.model import begin_profile_call, install_attention_hooks


def test_disabled_pipeline_never_reads_profiling_interfaces(monkeypatch):
    monkeypatch.delenv("AFD_CAM_PROFILE_ENABLE", raising=False)
    assert begin_profile_call(SimpleNamespace()) is None


def test_pipeline_reuses_hook_stream_without_call_identity(monkeypatch):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    execution = object()
    metadata = SimpleNamespace(profile_attention=execution)
    assert begin_profile_call(metadata) is execution
    assert metadata.profile_attention is None
    metadata.connector = SimpleNamespace(begin_profile_call=lambda: execution)
    assert begin_profile_call(metadata) is execution


def test_attention_hook_emits_only_compute_transition():
    module = SimpleNamespace(register_forward_pre_hook=Mock())
    execution = SimpleNamespace(tracer=SimpleNamespace(transition=Mock()))
    metadata = SimpleNamespace(
        connector=SimpleNamespace(begin_profile_call=lambda: execution),
        profile_attention=None,
    )
    install_attention_hooks(module, lambda: metadata)
    hook = module.register_forward_pre_hook.call_args.args[0]
    hook(module, ())
    assert metadata.profile_attention is execution
    execution.tracer.transition.assert_called_once_with(execution, "COMPUTE")
