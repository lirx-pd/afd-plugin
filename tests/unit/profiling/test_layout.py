# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Profiling uses the compact metadata returned by unmodified CAM operators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from afd_plugin.profiling import (
    BATCH_INFO_FIELDS,
    batch_info_prefix_size,
)


@pytest.mark.parametrize(
    "tp_size,experts,prefix", [(1, 1, 7), (2, 3, 13), (4, 16, 73), (8, 8, 77)]
)
def test_compact_prefix_contract(tp_size, experts, prefix):
    assert batch_info_prefix_size(tp_size, experts) == prefix
    assert BATCH_INFO_FIELDS == 5


def test_layout_import_needs_no_device_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; from afd_plugin.profiling import batch_info_prefix_size; "
            "assert batch_info_prefix_size(2, 3) == 13; "
            "assert not {'torch', 'torch_npu', 'vllm'} & sys.modules.keys()",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
