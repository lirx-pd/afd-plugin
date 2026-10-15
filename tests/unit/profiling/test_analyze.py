from __future__ import annotations

import copy
import json
import random

import pytest

from afd_plugin.profiling.analyze import (
    export_chrome_trace,
    main,
    read_jsonl,
    reconstruct,
)

START_HOST_NS = 10_000_000_000
DEVICE_ORIGIN_NS = 1_000_000_000


def _span(span_id, worker, ref, kind, begin, end, **updates):
    return {
        "record_type": "SPAN",
        "span_id": span_id,
        "worker_id": worker,
        "exec_ref": ref,
        "kind": kind,
        "clock_domain": f"{worker}:device",
        "device_begin_ns": DEVICE_ORIGIN_NS + begin,
        "device_end_ns": DEVICE_ORIGIN_NS + end,
        "host_begin_ns": START_HOST_NS + begin,
        "host_end_ns": START_HOST_NS + end,
        "stream_id": "compute",
        "producer_id": f"{worker}:thread",
        "complete": True,
        "generation": 1,
        **updates,
    }


@pytest.fixture
def records():
    result = [
        {
            "record_type": "SESSION",
            "schema_version": 1,
            "session_id": "experiment",
            "start_host_ns": START_HOST_NS,
            "worker_id": "A0",
        }
    ]
    for worker in ("A0", "F0", "F1"):
        result.append(
            {
                "record_type": "CLOCK_MAP",
                "clock_domain": f"{worker}:device",
                "device_origin_ns": DEVICE_ORIGIN_NS,
                "host_origin_ns": START_HOST_NS,
                "uncertainty_ns": 2_000,
                "valid_device_begin_ns": DEVICE_ORIGIN_NS,
                "valid_device_end_ns": DEVICE_ORIGIN_NS + 200_000,
            }
        )
    for fragment, request, group, start, end in (
        ("f0", "request0", "group0", 0, 64),
        ("f1", "request0", "group1", 64, 128),
        ("shared", "request1", "group0", 0, 1),
    ):
        result.append(
            {
                "record_type": "FRAGMENT",
                "fragment_id": fragment,
                "request_id": request,
                "group_id": group,
                "stage_idx": 0,
                "token_start": start,
                "token_end": end,
                "mode": "PREFILL",
            }
        )
    for call, group, fragments in (
        ("call0", "group0", ["f0", "shared"]),
        ("call1", "group1", ["f1"]),
    ):
        result.append(
            {
                "record_type": "SOURCE_CALL",
                "source_call_id": call,
                "group_id": group,
                "stage_idx": 0,
                "layer_idx": 7,
                "fragment_ids": fragments,
            }
        )
    for work, call in (
        ("work0", "call0"),
        ("work1", "call0"),
        ("work2", "call0"),
        ("work3", "call1"),
    ):
        result.append(
            {
                "record_type": "FFN_WORK",
                "ffn_work_id": work,
                "source_call_ids": [call],
                "layer_idx": 7,
            }
        )
    result.extend(
        [
            _span("a0", "A0", "call0", "ATTENTION_GATE", 10_000, 30_000),
            _span("shared0", "A0", "call0", "SHARED_EXPERT", 35_000, 70_000),
            _span("a1", "A0", "call1", "ATTENTION_GATE", 90_000, 125_000),
            _span("ff0", "F0", "work0", "FFN_COMPUTE", 30_000, 90_000),
            _span("gmm0", "F0", "work0", "GMM_GATE_UP", 40_000, 50_000),
            _span("ff1", "F0", "work1", "FFN_COMPUTE", 95_000, 115_000),
            _span("ff2", "F1", "work2", "FFN_COMPUTE", 35_000, 105_000),
            _span("ff3", "F0", "work3", "FFN_COMPUTE", 130_000, 160_000),
        ]
    )
    return result


def test_host_span_uses_host_clock_without_device_mapping(records):
    records.append(
        _span(
            "host0",
            "A0",
            "call0",
            "HOST_SUBMIT",
            0,
            0,
            time_source="HOST",
            clock_domain="A0:host-monotonic",
            stream_id="host",
            host_begin_ns=2_000_000,
            host_end_ns=2_020_000,
            device_begin_ns=None,
            device_end_ns=None,
            host_epoch_offset_ns=START_HOST_NS - 2_000_000 + 10_000,
            clock_uncertainty_ns=300,
        )
    )
    result = reconstruct(records)
    assert result["data_quality"] == "COMPLETE"
    assert "A0:host-monotonic" not in result["clock_maps"]
    assert result["spans"]["host0"]["timing"] == {
        "duration_ns": 20_000,
        "relative_begin_us": 10,
        "relative_end_us": 30,
        "uncertainty_ns": 300,
        "aligned": True,
    }
    host_events = [
        event
        for event in export_chrome_trace(records)["traceEvents"]
        if event.get("cat") == "CAM host stage"
    ]
    assert len(host_events) == 1
    assert host_events[0]["ts"] == 10
    assert host_events[0]["dur"] == 20


def test_export_preserves_worker_identity_and_quality_fields(records):
    records[0]["requires_summary"] = True
    records[0]["identity"] = {"host": "worker-node", "device": "npu:0"}
    records.append(
        {
            "record_type": "SUMMARY",
            "worker_id": "A0",
            "dropped_spans": 0,
            "dropped_metadata": 1,
            "unresolved_spans": 0,
            "dropped_records": 0,
            "error": "metadata copy failed",
        }
    )
    result = reconstruct(records)
    exported = export_chrome_trace(records)
    assert exported["worker_sessions"] == result["worker_sessions"]
    assert exported["worker_summaries"] == result["worker_summaries"]
    assert exported["data_quality"] == result["data_quality"] == "PARTIAL"
    assert exported["errors"] == result["errors"]
    assert exported["worker_summaries"]["A0"]["error"] == "metadata copy failed"


def test_required_missing_summary_is_partial(records):
    records[0]["requires_summary"] = True
    result = reconstruct(records)
    assert result["data_quality"] == "PARTIAL"
    assert result["errors"] == ["worker A0: missing SUMMARY"]


def test_idle_summary_error_is_partial_with_zero_loss_counters(records):
    records = [{**records[0], "requires_summary": True}]
    records.append(
        {
            "record_type": "SUMMARY",
            "worker_id": "A0",
            "dropped_spans": 0,
            "dropped_metadata": 0,
            "unresolved_spans": 0,
            "dropped_records": 0,
            "error": "collector failed before the first device record",
        }
    )
    result = reconstruct(records)
    assert result["data_quality"] == "PARTIAL"
    assert len(result["errors"]) == 1
    assert "collector failed before the first device record" in result["errors"][0]


@pytest.mark.parametrize(
    "counter",
    ["dropped_spans", "dropped_metadata", "unresolved_spans", "dropped_records"],
)
def test_summary_loss_counters_are_partial(records, counter):
    records[0]["requires_summary"] = True
    summary = {
        "record_type": "SUMMARY",
        "worker_id": "A0",
        "dropped_spans": 0,
        "dropped_metadata": 0,
        "unresolved_spans": 0,
        "dropped_records": 0,
    }
    records.append(summary)
    assert reconstruct(records)["data_quality"] == "COMPLETE"
    summary[counter] = 1
    result = reconstruct(records)
    assert result["data_quality"] == "PARTIAL"
    assert result["errors"] == ["worker A0: dropped or unresolved records"]


def test_expected_ffn_worker_missing_is_partial(records):
    call = next(row for row in records if row.get("source_call_id") == "call0")
    call["expected_ffn_workers"] = ["F0", "F1"]
    assert reconstruct(records)["data_quality"] == "COMPLETE"
    records = [
        row
        for row in records
        if row.get("ffn_work_id") != "work2" and row.get("span_id") != "ff2"
    ]
    result = reconstruct(records)
    assert result["data_quality"] == "PARTIAL"
    assert result["errors"] == ["SOURCE_CALL call0: missing FFN workers ['F1']"]
    assert all("F1" not in pipeline["ffn_workers"] for pipeline in result["pipelines"])


def test_reconstruct_shuffled_chunks_and_shared_group(records):
    expected = reconstruct(records)
    random.Random(8).shuffle(records)
    assert reconstruct(records) == expected
    assert expected["data_quality"] == "COMPLETE"
    assert [
        (row["request_id"], row["group_id"], row["stage_idx"], row["layer_idx"])
        for row in expected["pipelines"]
    ] == [
        ("request0", "group0", 0, 7),
        ("request0", "group1", 0, 7),
        ("request1", "group0", 0, 7),
    ]
    first, second, shared = expected["pipelines"]
    assert first["ffn_workers"] == {
        "F0": [
            {"ffn_work_id": "work0", "span_ids": ["ff0", "gmm0"]},
            {"ffn_work_id": "work1", "span_ids": ["ff1"]},
        ],
        "F1": [{"ffn_work_id": "work2", "span_ids": ["ff2"]}],
    }
    assert shared["ffn_work_ids"] == first["ffn_work_ids"]
    assert shared["attention_span_ids"] == first["attention_span_ids"]
    assert second["ffn_work_ids"] == ["work3"]
    assert expected["fragments"]["f1"]["token_start"] == 64


def test_nested_and_shared_spans_are_not_counted_twice(records):
    result = reconstruct(records)
    assert result["ffn_compute_by_worker"] == [
        {"worker_id": "F0", "clock_domain": "F0:device", "duration_ns": 110_000},
        {"worker_id": "F1", "clock_domain": "F1:device", "duration_ns": 70_000},
    ]
    exported = export_chrome_trace(records)
    assert sum(event["ph"] == "X" for event in exported["traceEvents"]) == 8
    assert (
        sum(event["name"] == "SHARED_EXPERT" for event in exported["traceEvents"]) == 1
    )
    event = next(
        item for item in exported["traceEvents"] if item["args"].get("span_id") == "ff0"
    )
    assert event["ts"] == 30
    assert event["dur"] == 60
    assert type(event["pid"]) is int
    assert type(event["tid"]) is int
    assert event["args"]["timing"]["uncertainty_ns"] == 2_000
    assert exported["ffn_works"]["work0"]["span_ids"] == ["ff0", "gmm0"]


def test_compute_union_preserves_same_worker_overlap(records):
    records.append(_span("ff_overlap", "F0", "work0", "FFN_COMPUTE", 50_000, 100_000))
    result = reconstruct(records)
    assert result["ffn_compute_by_worker"][0]["duration_ns"] == 115_000


def test_missing_clock_keeps_raw_domain_and_duration(records):
    records = [record for record in records if record["record_type"] != "CLOCK_MAP"]
    result = reconstruct(records)
    assert result["clock_maps"] == {}
    span = result["spans"]["ff0"]
    assert result["data_quality"] == "PARTIAL"
    assert span["clock_domain"] == "F0:device"
    assert span["device_begin_ns"] == DEVICE_ORIGIN_NS + 30_000
    assert span["timing"]["duration_ns"] == 60_000
    assert span["timing"]["relative_begin_us"] is None
    exported = export_chrome_trace(records)
    assert all(event["ph"] == "M" for event in exported["traceEvents"])
    assert len(exported["unaligned_spans"]) == 8


def test_periodic_clock_maps_require_one_map_covering_whole_span(records):
    mapping = next(
        record for record in records if record.get("clock_domain") == "F0:device"
    )
    mapping["valid_device_end_ns"] = DEVICE_ORIGIN_NS + 100_000
    later = {
        **mapping,
        "valid_device_begin_ns": DEVICE_ORIGIN_NS + 120_000,
        "valid_device_end_ns": DEVICE_ORIGIN_NS + 200_000,
        "host_origin_ns": START_HOST_NS + 10_000,
    }
    records.append(later)
    result = reconstruct(records)
    assert result["spans"]["ff0"]["timing"]["relative_begin_us"] == 30
    assert result["spans"]["ff1"]["timing"]["aligned"] is False
    assert result["spans"]["ff3"]["timing"]["relative_begin_us"] == 140
    assert result["data_quality"] == "PARTIAL"


def test_missing_refs_and_incomplete_spans_are_partial(records):
    records = [record for record in records if record.get("fragment_id") != "f0"]
    work = next(record for record in records if record.get("ffn_work_id") == "work0")
    work["source_call_ids"].append("missing-call")
    records.extend(
        [
            _span("orphan", "A0", "missing-exec", "OTHER", 0, 1),
            _span(
                "unfinished",
                "A0",
                "call0",
                "COMBINE_RECV",
                0,
                1,
                complete=False,
                device_end_ns=None,
                host_end_ns=None,
            ),
        ]
    )
    result = reconstruct(records)
    assert result["data_quality"] == "PARTIAL"
    assert any("missing fragment f0" in error for error in result["errors"])
    assert any(
        "missing source_call missing-call" in error for error in result["errors"]
    )
    assert any("missing exec_ref missing-exec" in error for error in result["errors"])
    assert any("SPAN unfinished: incomplete" in error for error in result["errors"])
    assert "missing-call" not in result["source_calls"]
    assert result["spans"]["unfinished"]["timing"]["duration_ns"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_idx", 8),
        ("group_id", "wrong-group"),
        ("stage_idx", 1),
    ],
)
def test_conflicting_relations_are_rejected(records, field, value):
    call = next(record for record in records if record.get("source_call_id") == "call0")
    call[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        reconstruct(records)


@pytest.mark.parametrize("record_type", ["FRAGMENT", "SOURCE_CALL", "FFN_WORK", "SPAN"])
def test_duplicate_ids_are_rejected(records, record_type):
    records.append(
        copy.deepcopy(next(row for row in records if row["record_type"] == record_type))
    )
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct(records)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", True),
        ("device_begin_ns", "123"),
        ("host_begin_ns", float("nan")),
        ("complete", "true"),
    ],
)
def test_bad_span_values_are_rejected(records, field, value):
    span = next(row for row in records if row["record_type"] == "SPAN")
    span[field] = value
    with pytest.raises(ValueError):
        reconstruct(records)


def test_worker_headers_merge_only_when_session_fields_agree(records):
    records.append({**records[0], "worker_id": "F0"})
    assert len(reconstruct(records)["worker_sessions"]) == 2
    records[-1]["start_host_ns"] += 1
    with pytest.raises(ValueError, match="headers disagree"):
        reconstruct(records)
    records[-1] = dict(records[0])
    with pytest.raises(ValueError, match="duplicate SESSION"):
        reconstruct(records)


def test_read_jsonl_and_cli(tmp_path, records):
    source = tmp_path / "worker.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    assert read_jsonl([source]) == records
    output = tmp_path / "trace.json"
    main([str(tmp_path), "--output", str(output)])
    trace = json.loads(output.read_text(encoding="utf-8"))
    assert trace["data_quality"] == "COMPLETE"
    assert sum(event["ph"] == "X" for event in trace["traceEvents"]) == 8
    source.write_text('{"record_type":"SESSION_EXCERPT"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="worker.jsonl:1"):
        read_jsonl([source])


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_jsonl_nonfinite_numbers_keep_source_line(tmp_path, records, number):
    source = tmp_path / "worker.jsonl"
    source.write_text(
        json.dumps(records[0])
        + '\n{"record_type":"SUMMARY","extra":['
        + number
        + "]}\n"
    )
    with pytest.raises(ValueError, match="worker.jsonl:2:.*finite JSON"):
        read_jsonl([source])


def test_jsonl_schema_validation_is_owned_by_reconstruct(tmp_path, records):
    source = tmp_path / "worker.jsonl"
    records[0]["schema_version"] = 2
    source.write_text(json.dumps(records[0]) + "\n")
    parsed = read_jsonl([source])
    with pytest.raises(ValueError, match="schema_version"):
        reconstruct(parsed)
    source.write_text(json.dumps({**records[0], "schema_version": 1}) + "\n{invalid}\n")
    with pytest.raises(ValueError, match="worker.jsonl:2:"):
        read_jsonl([source])


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), b"bytes", {1}]
)
def test_reconstruct_rejects_invalid_nested_json_values(records, value):
    records[0]["identity"] = {"extra": [value]}
    with pytest.raises(ValueError, match="finite JSON"):
        reconstruct(records)


def test_reconstruct_rejects_cycles_but_accepts_shared_finite_values(records):
    shared = [1, 2.5, None, True]
    records[0]["identity"] = {"first": shared, "second": shared}
    assert reconstruct(records)["data_quality"] == "COMPLETE"
    shared.append(shared)
    with pytest.raises(ValueError, match="finite JSON"):
        reconstruct(records)


def test_unknown_schema_and_overlapping_maps_are_rejected(records):
    records[0]["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        reconstruct(records)
    records[0]["schema_version"] = 1
    records.append(dict(records[1]))
    with pytest.raises(ValueError, match="overlapping CLOCK_MAP"):
        reconstruct(records)
