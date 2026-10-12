# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import inspect

import pytest

native = pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402


def test_dsv4_model_uses_native_forward_without_afd_stage_plan(monkeypatch):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    sentinel = object()
    monkeypatch.setattr(native.DeepseekV4Model, "forward", lambda *_args: sentinel)

    assert adapter.AFDDeepseekV4Model.forward is native.DeepseekV4Model.forward
    assert model.forward(None, object(), None, object()) is sentinel


def test_dsv4_constructor_signatures_match_target():
    for adapter_class, native_class in (
        (adapter.AFDDeepseekV4DecoderLayer, native.DeepseekV4DecoderLayer),
        (adapter.AFDDeepseekV4Model, native.DeepseekV4Model),
    ):
        assert inspect.signature(adapter_class.__init__) == inspect.signature(
            native_class.__init__,
        )
