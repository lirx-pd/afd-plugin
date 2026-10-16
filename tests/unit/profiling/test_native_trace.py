# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json

import pytest

from afd_plugin.profiling.analyze import _timing, read_jsonl, reconstruct
from afd_plugin.profiling.native_trace import read_native_trace


def header():
    return {
        "record_type": "SESSION",
        "schema_version": 1,
        "session_id": "test-session",
        "worker_id": "a0",
        "generation": 3,
        "start_host_ns": 1792136972000000000,
        "clock_offset_ns": 100,
        "clock_uncertainty_ns": 50,
        "native_trace_dir": "a0/lightweight",
    }


def phase(
    phase="COMPUTE", stage="ATTENTION", clock="D", sequence=1, device=True, **updates
):
    return {
        "name": f"afd.cam|3|a0:call:1|{phase}|{stage}|{clock}|{sequence}",
        "ph": "X",
        "pid": 1,
        "tid": 47,
        "ts": "1792136972859238.041",
        "dur": 15.82,
        "args": {"Physic Stream Id": 47} if device else {"domain": "afd.cam"},
        **updates,
    }


def write_trace(root, events):
    (root / "trace_view.json").write_text(json.dumps({"traceEvents": events}))


def test_layer_device_times_use_native_clock_and_skip_duplicate_host_launch_ranges(
    tmp_path,
):
    write_trace(
        tmp_path,
        [phase(), phase(device=False), phase("WAIT", "DBO_HOST_WAIT", "H", 2, False)],
    )
    device, host, mapping = read_native_trace(tmp_path, header())
    assert device["kind"] == "COMPUTE" and device["stage"] == "ATTENTION"
    assert device["device_begin_ns"] == 1792136972859238041
    assert device["device_end_ns"] == 1792136972859253861
    timing = _timing(device, [mapping], header()["start_host_ns"])
    assert timing["duration_ns"] == 15820 and timing["aligned"]
    assert host["kind"] == "WAIT" and host["device_begin_ns"] is None
    assert _timing(host, (), header()["start_host_ns"]) == timing


def test_layer_output_has_three_phase_totals_and_unions_overlapping_waits(tmp_path):
    events = [
        phase(dur=10),
        phase(
            "COMMUNICATION",
            "DISPATCH_SEND",
            sequence=2,
            ts="1792136972859258.041",
            dur=3,
        ),
        phase("WAIT", "COMBINE_RECV", sequence=3, ts="1792136972859278.041", dur=20),
        phase(
            "WAIT", "DBO_HOST_WAIT", "H", 4, False, ts="1792136972859280.041", dur=10
        ),
    ]
    write_trace(tmp_path, events)
    records = [
        header(),
        {
            "record_type": "SOURCE_CALL",
            "source_call_id": "a0:call:1",
            "group_id": "g",
            "stage_idx": 0,
            "layer_idx": 7,
            "fragment_ids": [],
        },
        *read_native_trace(tmp_path, header()),
    ]
    result = reconstruct(records)
    assert result["data_quality"] == "COMPLETE", result["errors"]
    assert result["layers"] == [
        {
            "worker_id": "a0",
            "execution_id": "a0:call:1",
            "role": "attention",
            "layer_idx": 7,
            "duration_ns": {"compute": 10000, "communication": 3000, "wait": 20000},
        }
    ]


def test_loader_resolves_native_files_after_moving_capture_and_rejects_escape(tmp_path):
    native = tmp_path / "a0" / "lightweight"
    native.mkdir(parents=True)
    write_trace(native, [phase()])
    path = tmp_path / "a0.jsonl"
    path.write_text(json.dumps(header()) + "\n")
    assert [r["record_type"] for r in read_jsonl([path])] == [
        "SESSION",
        "SPAN",
        "CLOCK_MAP",
    ]
    path.write_text(json.dumps({**header(), "native_trace_dir": "../escape"}) + "\n")
    with pytest.raises(ValueError, match="escapes session"):
        read_jsonl([path])


def test_missing_export_reports_offline_analysis_command(tmp_path):
    with pytest.raises(ValueError, match="analyse"):
        read_native_trace(tmp_path, header())


@pytest.mark.parametrize(
    "changes",
    [
        {"dur": -1},
        {"name": "afd.cam|4|a0:call:1|COMPUTE|ATTENTION|D|1"},
        {"name": "afd.cam|3|a0:call:1|OPERATOR|GMM_W13|D|1"},
    ],
)
def test_invalid_native_ranges_are_rejected(tmp_path, changes):
    write_trace(tmp_path, [phase(**changes)])
    with pytest.raises(ValueError):
        read_native_trace(tmp_path, header())


def test_missing_native_ranges_make_output_partial(tmp_path):
    write_trace(tmp_path, [phase()])
    records = [
        header(),
        *read_native_trace(tmp_path, header()),
        {
            "record_type": "SUMMARY",
            "worker_id": "a0",
            "native_spans": 2,
            "dropped_spans": 0,
            "dropped_metadata": 0,
            "unresolved_spans": 0,
            "dropped_records": 0,
        },
    ]
    result = reconstruct(records)
    assert result["data_quality"] == "PARTIAL"
    assert any("expected 2 native ranges, found 1" in e for e in result["errors"])
