# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.profiling.fragments import InputSnapshot
from afd_plugin.profiling.model import begin_profile_call, record_source_call


class MemoryWriter:
    def __init__(self):
        self.records = []

    def submit(self, record):
        self.records.append(record)
        return True


def test_disabled_source_call_never_reads_profiling_interfaces(monkeypatch):
    monkeypatch.delenv("AFD_CAM_PROFILE_ENABLE", raising=False)
    assert begin_profile_call(SimpleNamespace()) == (0, None)


def test_idle_enabled_protocol_keeps_dispatch_sequence(monkeypatch):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    metadata = SimpleNamespace(
        profile_attention=None,
        connector=SimpleNamespace(begin_profile_call=lambda: (123, None)),
    )
    assert begin_profile_call(metadata) == (123, None)


def test_attention_hook_identity_is_consumed_from_its_owner(monkeypatch):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    execution = SimpleNamespace()
    metadata = SimpleNamespace(
        profile_attention=SimpleNamespace(sequence=17, execution=execution),
    )
    assert begin_profile_call(metadata) == (17, execution)
    assert metadata.profile_attention is None


def test_fragment_dictionary_is_emitted_again_for_a_new_capture_generation():
    metadata = SimpleNamespace(
        profile_snapshot=InputSnapshot.from_cpu(
            ["request"], [0, 4], [0], [4], [True], 4
        ),
        stage_idx=0,
        tokens_start_loc=[0],
        transaction_id="forward-1",
        profile_fragments=None,
        profile_sp_origin=None,
        connector=SimpleNamespace(attn_size=1),
    )
    layout = SimpleNamespace(
        use_sequence_parallel=False, local_token_slice=slice(0, 4), local_tokens=4
    )
    tracer = SimpleNamespace(worker_id="attention", source_workers=["attention", "ffn"])
    captures = []
    for generation in (1, 2):
        capture = SimpleNamespace(
            writer=MemoryWriter(),
            generation=generation,
            session_id=f"session-{generation}",
        )
        execution = SimpleNamespace(
            capture=capture, tracer=tracer, exec_ref=f"attention:call:{generation}"
        )
        record_source_call(execution, metadata, generation, layout)
        captures.append(capture)

    for capture in captures:
        fragments = {
            record["fragment_id"]
            for record in capture.writer.records
            if record["record_type"] == "FRAGMENT"
        }
        call = next(
            record
            for record in capture.writer.records
            if record["record_type"] == "SOURCE_CALL"
        )
        assert set(call["fragment_ids"]) <= fragments
