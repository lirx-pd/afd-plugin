# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Profile-v1 wire checks driven by torchrun on allocated Ascend 910C devices.

AFD_PROFILE_TEST_EXTENSION optionally selects an isolated binding build;
ASCEND_CUSTOM_OPP_PATH and AFD_CUST_OPAPI_LIB_PATH select its matching OPP.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

from tests.npu.async_cam_common import HarnessConfig, NativeHarness


def make_config(
    topology: str, dtype: str, dynamic_quant: int, routing: str
) -> HarnessConfig:
    tp_size = 1 if topology == "single" else 2
    return HarnessConfig(
        attn_ranks=tp_size,
        moe_ranks=tp_size,
        tp_size=tp_size,
        lengths=(8,) * tp_size,
        dtype=dtype,
        dynamic_quant=dynamic_quant,
        routing=routing,
        capacity=8 * tp_size,
        max_seq_len=8 * tp_size,
        timeout_seconds=60,
    )


def _load_isolated_extension() -> None:
    extension_path = os.environ.get("AFD_PROFILE_TEST_EXTENSION")
    if extension_path is None:
        return
    spec = importlib.util.spec_from_file_location(
        "afd_plugin._C_ascend", extension_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load isolated binding {extension_path}")
    extension = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = extension
    spec.loader.exec_module(extension)


def run_probe(config: HarnessConfig) -> dict:
    os.environ["HCCL_BUFFSIZE"] = str(config.window_mb)
    os.environ["BATCH_SIZE_FACTOR"] = str(config.capacity / 262144)
    import torch
    import torch_npu  # noqa: F401

    device_name = torch.npu.get_device_name(int(os.environ["LOCAL_RANK"]))
    if not device_name.startswith(("Ascend910_93", "Ascend910C")):
        return {
            "status": "unsupported_soc",
            "reason": f"profile-v1 CAM requires ascend910_93/910C, found {device_name}",
        }
    _load_isolated_extension()
    from afd_plugin.compat.npu.ops import ensure_cam_async_profile_ops_available

    ensure_cam_async_profile_ops_available()
    saved_traces = []
    rounds = []
    with NativeHarness(config) as harness:
        dispatch_send = harness.operations["dispatch_send"]
        dispatch_recv = harness.operations["dispatch_recv"]
        sequence_base = 0

        def profile_send(*args):
            sequence = sequence_base + harness.rank if sequence_base else 0
            return dispatch_send(*args[:3], sequence, *args[4:])

        def profile_recv(*args):
            result = dispatch_recv(*args)
            assert len(result) == 4
            info = result[2]
            trace = info[-2 * config.tp_size :].view(config.tp_size, 2)
            assert trace.dtype == torch.int64
            assert tuple(trace.shape) == (config.tp_size, 2)
            assert (
                trace.untyped_storage().data_ptr() == info.untyped_storage().data_ptr()
            )
            tp_base = int(info[1].cpu())
            expected = [
                [1, sequence_base + source if sequence_base else 0]
                for source in range(tp_base, tp_base + config.tp_size)
            ]
            assert trace.cpu().tolist() == expected
            saved_traces.append((trace, expected))
            return result

        # Reuse one window across zero and nonzero sequences. Every metadata
        # output must retain its own tail after the shared slot is overwritten.
        schedule = (
            (1, None),
            (7, (1 << 32) + 7),
            (19, (1 << 32) + 65539),
            (7, (1 << 63) - 2),
            (1, None),
        )
        harness.operations["dispatch_send"] = profile_send
        harness.operations["dispatch_recv"] = profile_recv
        for layer_id, sequence in schedule:
            profile = sequence is not None
            sequence_base = sequence or 0
            report = harness.run_round(layer_id)
            assert not report["failures"], report["failures"]
            report["profile_v1"] = profile
            report["sequence_base"] = sequence
            rounds.append(report)
        for trace, expected in saved_traces:
            assert trace.cpu().tolist() == expected, (
                "source_trace changed after window reuse"
            )
        if harness.rank >= config.attn_ranks:
            assert len(saved_traces) == sum(report["chunks"] for report in rounds)
            assert len({trace.data_ptr() for trace, _ in saved_traces}) == len(
                saved_traces
            )
        gathered = harness.gather_reports({"rank": harness.rank, "rounds": rounds})
        if config.routing == "sparse" and config.moe_ranks == 2:
            empty_owner = gathered[-1]["rounds"]
            assert all(
                report["rows"] == 0 and report["chunks"] == 1 for report in empty_owner
            )
        if config.routing == "sparse" or config.moe_ranks == 1:
            assert all(
                report["chunks"] >= 2
                for report in gathered[config.attn_ranks]["rounds"]
            )
        return {
            "status": "passed",
            "environment": harness.environment(),
            "source_trace_count": len(saved_traces),
            "ranks": gathered,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("single", "tp2"), required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--dynamic-quant", type=int, choices=(0, 1), required=True)
    parser.add_argument("--routing", choices=("balanced", "sparse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = make_config(args.topology, args.dtype, args.dynamic_quant, args.routing)
    args.output.mkdir(parents=True, exist_ok=True)
    # torchrun workers have separate sessions; each worker must also enforce a
    # deadline if communicator setup hangs before the per-round watchdog starts.
    watchdog = threading.Timer(180, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        result = run_probe(config)
        path = args.output / f"rank-{os.environ['RANK']}.json"
        path.write_text(json.dumps(result, indent=2))
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    main()
