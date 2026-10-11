# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real NPU streams/events and native thread handoff, isolated on each fault.

This checks the executor primitives, not checkpoint or CAMP2P correctness.
"""

import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

# Match tests/conftest.py and vllm serve: import ops before device_op to avoid
# Ascend's existing circular import in standalone test subprocesses.
import vllm_ascend.ops  # noqa: E402, F401
from vllm.config import VllmConfig  # noqa: E402
from vllm.forward_context import get_forward_context  # noqa: E402
from vllm.utils.torch_utils import current_stream  # noqa: E402
from vllm.v1.worker import ubatching  # noqa: E402
from vllm.v1.worker.gpu import model_runner  # noqa: E402, F401
from vllm.v1.worker.gpu.ubatch_utils import UBatchState  # noqa: E402
from vllm.v1.worker.ubatch_utils import UBatchSlice  # noqa: E402
from vllm.v1.worker.workspace import (  # noqa: E402
    current_workspace_manager,
    init_workspace_manager,
)
from vllm_ascend.ascend_forward_context import (  # noqa: E402
    get_mrv2_in_profile_run,
    override_mrv2_in_profile_run,
)
from vllm_ascend.utils import adapt_patch  # noqa: E402
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper  # noqa: E402

from afd_plugin.v1.worker.npu.ubatch_runner import NPUUBatchRunner  # noqa: E402


def _exercise(fault):
    # Construct only the native execution state; full initialization/metadata
    # and distributed communication are covered by the checkpoint run.
    adapt_patch(is_global_patch=True)
    torch.npu.set_device(0)
    runner = object.__new__(NPUUBatchRunner)
    runner.vllm_config = VllmConfig()
    runner.device = torch.device("npu:0")
    runner.num_ubatches = 2
    runner.comm_stream = torch.npu.Stream(device=runner.device)
    runner.capture_stream = torch.npu.Stream(device=runner.device)
    runner.ready_barrier = threading.Barrier(3, timeout=2)
    runner.sm_control = nullcontext()
    runner._pending_finish = None
    init_workspace_manager(runner.device, num_ubatches=2)
    contexts = [SimpleNamespace(stage=i) for i in range(2)]
    state = UBatchState(
        [UBatchSlice(slice(0, 1), slice(i * 4, (i + 1) * 4)) for i in range(2)],
        contexts,
    )
    parent = ubatching.forward_context._forward_context
    inputs = torch.arange(8, dtype=torch.float32, device=runner.device)
    trace = []
    injected: list[str] = []
    if fault == "device_init":
        # Native threads initialize before entering UBatchContext/barrier.
        set_device = torch.accelerator.set_device_index

        def fail_device(index):
            if not injected:
                injected.append(fault)
                raise RuntimeError("injected device_init")
            set_device(index)

        torch.accelerator.set_device_index = fail_device

    def fail():
        injected.append(fault)
        raise RuntimeError("injected " + fault)

    def model(input_ids, positions):
        stage = ubatching.dbo_current_ubatch_id()
        assert get_forward_context() is contexts[stage]
        assert torch.npu.current_device() == 0
        assert current_stream() == torch.npu.current_stream()
        assert get_mrv2_in_profile_run()
        trace.append((stage, "send"))
        workspace = current_workspace_manager().get_simultaneous(((4,), torch.float32))[
            0
        ]
        workspace.fill_(stage + 1)
        ubatching.dbo_switch_to_comm_sync()
        value = input_ids * 3 + positions
        if stage == 0 and fault == "after_send":
            fail()
        if stage == 0 and fault == "recv_hook":
            ubatching.dbo_register_recv_hook(fail)
        ubatching.dbo_yield_and_switch_from_comm_to_compute()
        ubatching.dbo_maybe_run_recv_hook()
        trace.append((stage, "consume"))
        torch.testing.assert_close(workspace.cpu(), torch.full((4,), stage + 1.0))
        if stage == 0 and fault == "owner_consume":
            fail()
        if stage == 0 and fault == "final_exit":
            # Native __exit__ runs this slot after removing registry identity.
            ubatching._CURRENT_CONTEXTS[stage].recv_hook = fail
        return value

    before = set(threading.enumerate())
    try:
        with override_mrv2_in_profile_run(True), torch_cuda_wrapper():
            result = runner.run(
                model, {"input_ids": inputs, "positions": inputs}, state
            )
        assert fault == "none", "injected failure was lost"
        torch.testing.assert_close(result.cpu(), torch.arange(8) * 4, check_dtype=False)
        assert trace == [(0, "send"), (1, "send"), (0, "consume"), (1, "consume")]
    except (RuntimeError, threading.BrokenBarrierError):
        if fault == "none":
            raise
        assert fault in injected, "failure occurred before the injection point"
    # Initialization failures may return from the parent barrier first. Check
    # that all native child threads and registry entries actually disappear.
    deadline = time.monotonic() + 3
    while set(threading.enumerate()) - before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (set(threading.enumerate()) - before)
    assert not ubatching.dbo_enabled()
    assert all(context is None for context in ubatching._CURRENT_CONTEXTS)
    assert ubatching.forward_context._forward_context is parent
    torch.npu.synchronize()
    print("NPU_DBO_PASS " + fault, flush=True)


@pytest.mark.parametrize(
    "fault",
    ["none", "device_init", "after_send", "recv_hook", "owner_consume", "final_exit"],
)
def test_native_npu_threads_and_bounded_failure(fault):
    if not torch.npu.is_available():
        pytest.skip("requires an allocated NPU")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), fault],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NPU_DBO_PASS " + fault in result.stdout


if __name__ == "__main__":
    _exercise(sys.argv[1])
