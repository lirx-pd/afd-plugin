# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in unmodified CAM communication and numerical regression probes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("topology,world_size", [("single", 2), ("tp2", 4)])
@pytest.mark.parametrize(
    "dtype,dynamic_quant,routing",
    [
        ("fp16", 0, "balanced"),
        ("fp16", 1, "sparse"),
        ("bf16", 0, "sparse"),
        ("bf16", 1, "balanced"),
    ],
)
def test_profile_wire_roundtrip(
    tmp_path, topology, world_size, dtype, dynamic_quant, routing
):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip(
            "requires AFD_RUN_ASCEND_OP_RUNTIME=1 and allocated Ascend 910C devices"
        )
    if os.environ.get("SOC_VERSION", "").lower() != "910c":
        pytest.skip("native CAM is built only for SOC_VERSION=910c (ascend910_93)")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if len([device for device in visible.split(",") if device.strip()]) < world_size:
        pytest.skip(f"requires at least {world_size} devices allocated with npu run")
    pytest.importorskip("torch_npu")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "tests.npu.cam_profile_wire",
        "--topology",
        topology,
        "--dtype",
        dtype,
        "--dynamic-quant",
        str(dynamic_quant),
        "--routing",
        routing,
        "--output",
        str(tmp_path),
    ]
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(visible.split(",")[:world_size])
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate(timeout=5)
        (tmp_path / "wire.log").write_text(stdout)
        pytest.fail(
            f"CAM wire subprocess exceeded 240 seconds; see {tmp_path / 'wire.log'}"
        )
    (tmp_path / "wire.log").write_text(stdout)
    assert process.returncode == 0, stdout
    reports = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    if any(report["status"] == "unsupported_soc" for report in reports):
        pytest.skip(
            "; ".join(
                report["reason"]
                for report in reports
                if report["status"] == "unsupported_soc"
            )
        )
    assert all(report["status"] == "passed" for report in reports)
