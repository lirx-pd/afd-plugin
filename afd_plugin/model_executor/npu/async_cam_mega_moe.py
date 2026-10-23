# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded compact-v2 FFN transaction executed by one mixed A3 kernel."""

from __future__ import annotations

import torch

from afd_plugin.model_executor.npu.async_cam_w4a8 import (
    INT4_PER_INT32,
    AsyncCAMW4A8Executor,
)


class AsyncCAMMegaMoeExecutor:
    def __init__(
        self,
        weights: AsyncCAMW4A8Executor,
        *,
        max_seq_len: int,
        top_k: int,
        ffn_ranks: int,
        attn_ranks: int,
        rank: int,
        tp_size: int,
        capacity: int,
    ) -> None:
        experts, hidden, packed = weights.w13[0].shape
        intermediate = packed * INT4_PER_INT32 // 2
        if weights.output_dtype != torch.bfloat16:
            raise ValueError("MegaMoE requires BF16 output")
        self.config = [
            max_seq_len,
            hidden,
            intermediate,
            top_k,
            ffn_ranks,
            attn_ranks,
            experts,
            rank,
            ffn_ranks + attn_ranks,
            tp_size,
            capacity,
            1 if weights.w13_scale[0].dim() == 2 else weights.w13_scale[0].shape[1],
            weights.w2_scale[0].shape[1],
            len(weights.layer_ids),
        ]
        self.weights = weights
        self.anchor = torch.empty(1, dtype=torch.bfloat16, device=weights.w13[0].device)
        self.layer_map = weights.layer_id_to_slot
        if self.layer_map is None:
            self.layer_map = torch.arange(
                len(weights.layer_ids), dtype=torch.int64, device=self.anchor.device
            )
        # Resolve both schemas and validate static bounds before consuming CAM work.
        self.op = torch.ops.afd_ascend.afd_mega_moe_ffn.out
        self.completion = torch.zeros(1, dtype=torch.int64, device=self.anchor.device)
        self.scratch = torch.ops.afd_ascend.afd_mega_moe_ffn_workspace(
            self.anchor, self.config
        )

    def __call__(self, comm_args: torch.Tensor, group_name: str) -> torch.Tensor:
        weights = self.weights
        return self.op(
            self.anchor,
            comm_args,
            weights.w13,
            weights.w2,
            weights.w13_scale,
            weights.w2_scale,
            weights.w13_bias,
            weights.w2_bias,
            self.layer_map,
            self.scratch,
            self.config,
            group_name,
            weights.routed_scaling_factor,
            out=self.completion,
        )
