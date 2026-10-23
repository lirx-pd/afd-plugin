# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Binding contracts only; distributed math is tested by async_cam_runtime.py."""

import os

import pytest

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module")
def torch_runtime():
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires the opt-in compiled A3 extension")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

    ensure_cam_async_ops_available()
    return torch


def config():
    return [16, 256, 256, 2, 2, 1, 4, 1, 3, 1, 16, 2, 2, 2]


@pytest.mark.parametrize("use_out", [False, True])
def test_mega_moe_meta_shapes_and_weight_contract(torch_runtime, use_out):
    torch = torch_runtime
    anchor = torch.empty(1, dtype=torch.bfloat16, device="meta")
    comm = torch.empty(1, dtype=torch.float16, device="meta")
    layer_map = torch.empty(6, dtype=torch.int64, device="meta")
    scratch = torch.ops.afd_ascend.afd_mega_moe_ffn_workspace(anchor, config())
    w13 = [torch.empty(4, 256, 64, dtype=torch.int32, device="meta")] * 2
    w2 = [torch.empty(4, 256, 32, dtype=torch.int32, device="meta")] * 2
    s13 = [torch.empty(4, 2, 512, dtype=torch.int64, device="meta")] * 2
    s2 = [torch.empty(4, 2, 256, dtype=torch.int64, device="meta")] * 2
    b13 = [torch.empty(4, 512, dtype=torch.float32, device="meta")] * 2
    b2 = [torch.empty(4, 256, dtype=torch.float32, device="meta")] * 2
    args = (
        anchor,
        comm,
        w13,
        w2,
        s13,
        s2,
        b13,
        b2,
        layer_map,
        scratch,
        config(),
        "meta-contract",
        2.5,
    )
    op = torch.ops.afd_ascend.afd_mega_moe_ffn
    kwargs = {}
    if use_out:
        op = op.out
        kwargs["out"] = torch.empty(1, dtype=torch.int64, device="meta")
    output = op(*args, **kwargs)
    assert output.shape == (1,)
    assert output.dtype == torch.int64 and output.device.type == "meta"
    if use_out:
        assert output is kwargs["out"]
        with pytest.raises(RuntimeError, match="completion must have shape"):
            op(*args, out=torch.empty(2, dtype=torch.int64, device="meta"))
    with pytest.raises(RuntimeError, match="layer table length"):
        op(*args[:2], w13[:1], *args[3:], **kwargs)


@pytest.mark.parametrize(
    "index,value,message",
    [
        (10, 15, "complete expert"),
        (0, 32768, "uint16"),
        (1, 257, "multiples of 256"),
        (11, 3, "quantization groups"),
        (7, 3, "FFN rank"),
    ],
)
def test_mega_moe_rejects_invalid_static_bounds(torch_runtime, index, value, message):
    torch = torch_runtime
    bad = config()
    bad[index] = value
    anchor = torch.empty(1, dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match=message):
        torch.ops.afd_ascend.afd_mega_moe_ffn_workspace(anchor, bad)


def test_mega_moe_workspace_is_available_before_receiving(torch_runtime):
    torch = torch_runtime
    torch.npu.set_device(0)
    anchor = torch.empty(1, dtype=torch.bfloat16, device="npu")
    scratch = torch.ops.afd_ascend.afd_mega_moe_ffn_workspace(anchor, config())
    assert scratch.device == anchor.device
    assert scratch.dtype == torch.uint8 and scratch.ndim == 1 and scratch.numel() > 0
    larger = config()
    larger[10] = 262144
    other = torch.ops.afd_ascend.afd_mega_moe_ffn_workspace(anchor, larger)
    assert other.shape == scratch.shape
