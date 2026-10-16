# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
        tokens_lens=[4],
        tokens_unpadded_lens=[4],
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


@pytest.mark.parametrize("total,split", [(2337, 1170), (3, 2), (8, 4)])
@pytest.mark.parametrize("cloned", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("explicit_unpadded", [False, True])
def test_stage_padding_does_not_claim_tokens_from_the_next_stage(
    total, split, cloned, sequence_parallel, explicit_unpadded
):
    snapshot = InputSnapshot.from_cpu(
        ["request"], [0, total], [0], [total], [True], total
    )
    intervals = []
    real_lengths = [split, total - split]
    physical_lengths = (
        [(length + 3) // 4 * 4 for length in real_lengths]
        if explicit_unpadded
        else real_lengths
    )
    for stage, (start, stop) in enumerate(((0, split), (split, total))):
        shard_tokens = (stop - start + 3) // 4
        for rank in range(4):
            origin = rank * shard_tokens
            metadata = SimpleNamespace(
                profile_snapshot=snapshot,
                stage_idx=stage,
                tokens_start_loc=[start] if cloned else [0, split],
                tokens_lens=[physical_lengths[stage]] if cloned else physical_lengths,
                tokens_unpadded_lens=(
                    ([stop - start] if cloned else real_lengths)
                    if explicit_unpadded
                    else []
                ),
                transaction_id="forward",
                profile_fragments=None,
                profile_sp_origin=origin if sequence_parallel else None,
                connector=SimpleNamespace(attn_size=4),
            )
            writer = MemoryWriter()
            execution = SimpleNamespace(
                capture=SimpleNamespace(writer=writer, generation=1),
                tracer=SimpleNamespace(
                    worker_id=f"attention-{rank}", source_workers=[]
                ),
                exec_ref=f"attention-{rank}:call:{stage}",
            )
            local_start = 0 if sequence_parallel else origin
            layout = SimpleNamespace(
                use_sequence_parallel=sequence_parallel,
                local_token_slice=slice(local_start, local_start + shard_tokens),
                local_tokens=shard_tokens,
            )
            record_source_call(execution, metadata, 0, layout)
            fragments = [
                row for row in writer.records if row["record_type"] == "FRAGMENT"
            ]
            intervals.extend(
                (row["token_start"], row["token_end"]) for row in fragments
            )
            assert all(
                start <= row["token_start"] < row["token_end"] <= stop
                for row in fragments
            )
            call = writer.records[-1]
            expected_real = max(0, min(shard_tokens, stop - start - origin))
            assert call["padded_tokens"] == shard_tokens - expected_real
    position = 0
    for begin, end in sorted(intervals):
        assert begin == position
        position = end
    assert position == total
