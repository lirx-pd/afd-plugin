# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from dataclasses import FrozenInstanceError

import pytest

from afd_plugin.profiling.fragments import InputSnapshot, build_fragments


def test_snapshot_copies_mutable_cpu_inputs():
    requests = ["prefill", "decode"]
    offsets, computed, prompts, prefilling = [0, 3, 4], [5, 20], [10, 10], [True, False]
    snapshot = InputSnapshot.from_cpu(
        requests, offsets, computed, prompts, prefilling, 4
    )
    requests[0] = "next-request"
    offsets[:] = [0, 1, 2]
    computed[:] = [100, 200]
    prompts[:] = [1000, 2000]
    prefilling[:] = [False, True]
    assert snapshot.req_ids == ("prefill", "decode")
    assert snapshot.query_offsets == (0, 3, 4)
    assert snapshot.computed_before == (5, 20)
    assert snapshot.prompt_lengths == (10, 10)
    assert snapshot.execution_modes == ("PREFILL", "DECODE")
    assert snapshot.real_extent == 4
    with pytest.raises(FrozenInstanceError):
        snapshot.real_extent = 8


def test_microbatch_and_sp_slice_cross_request_boundary():
    snapshot = InputSnapshot.from_cpu(
        ["chunked", "decode"], [0, 5, 7], [64, 20], [100, 10], [True, False], 7
    )
    fragments = build_fragments(snapshot, "worker:group", 1, 3, 6)
    assert [
        (
            row["request_id"],
            row["token_start"],
            row["token_end"],
            row["mode"],
            row["token_origin"],
        )
        for row in fragments
    ] == [
        ("chunked", 67, 69, "PREFILL", "PROMPT"),
        ("decode", 20, 21, "DECODE", "GENERATED"),
    ]
    assert all(
        row["group_id"] == "worker:group" and row["stage_idx"] == 1 for row in fragments
    )


def test_recomputed_generated_tokens_keep_native_prefill_mode():
    snapshot = InputSnapshot.from_cpu(["resumed"], [0, 6], [2], [5], [True], 6)
    fragments = build_fragments(snapshot, "attempt-1", 0, 0, 6)
    assert [
        (row["token_start"], row["token_end"], row["token_origin"]) for row in fragments
    ] == [(2, 5, "PROMPT"), (5, 8, "GENERATED")]
    assert {row["mode"] for row in fragments} == {"PREFILL"}
    assert len({row["fragment_id"] for row in fragments}) == 2
    retry = build_fragments(snapshot, "attempt-2", 0, 0, 6)
    assert {row["fragment_id"] for row in fragments}.isdisjoint(
        row["fragment_id"] for row in retry
    )


def test_padding_and_cached_prefix_are_excluded():
    snapshot = InputSnapshot.from_cpu(["cached"], [0, 8], [128], [200], [True], 5)
    fragments = build_fragments(snapshot, "g", 0, 2, 32)
    assert [(row["token_start"], row["token_end"]) for row in fragments] == [(130, 133)]
    assert build_fragments(snapshot, "g", 1, 5, 32) == []
    assert build_fragments(snapshot, "g", 1, 2, 2) == []


def test_empty_requests_and_zero_query_rows_produce_no_fragment():
    empty = InputSnapshot.from_cpu([], [0], [], [], [], 0)
    assert build_fragments(empty, "g", 0, 0, 16) == []
    snapshot = InputSnapshot.from_cpu(
        ["empty", "live"], [0, 0, 1], [0, 8], [2, 8], [True, False], 1
    )
    assert [row["request_id"] for row in build_fragments(snapshot, "g", 0, 0, 1)] == [
        "live"
    ]


@pytest.mark.parametrize("begin,end", [(-1, 1), (3, 2)])
def test_invalid_physical_interval_fails(begin, end):
    snapshot = InputSnapshot.from_cpu(["r"], [0, 1], [0], [1], [True], 1)
    with pytest.raises(ValueError, match="token interval"):
        build_fragments(snapshot, "g", 0, begin, end)
