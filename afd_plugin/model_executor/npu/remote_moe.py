# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-owned Ascend routing for remote CAM experts."""

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.base_router import FusedMoERouter

from afd_plugin.envs import force_balanced_topk_ids_enabled
from afd_plugin.model_executor.remote_moe import (
    AFDRemoteMoERunner,
    remote_ffn_forward,
)


def validate_remote_moe_config() -> None:
    # The Ascend wrapper can enable local EPLB despite explicit factory kwargs.
    from vllm_ascend.ascend_config import get_ascend_config

    eplb_config = get_ascend_config().eplb_config
    if eplb_config.dynamic_eplb:
        raise RuntimeError(
            "Remote MoE does not support Attention-local dynamic_eplb",
        )
    if eplb_config.expert_map_path is not None:
        raise RuntimeError(
            "Remote MoE does not support Attention-local expert_map_path",
        )
    if eplb_config.num_redundant_experts != 0:
        raise RuntimeError(
            "Remote MoE does not support Attention-local Ascend redundant experts",
        )


class AFDAttentionGateMoERunner(AFDRemoteMoERunner):
    """Own routing while CAM's model loop schedules dispatch and combine."""

    @staticmethod
    def get_factory_kwargs(
        vllm_config: VllmConfig,
        gate: torch.nn.Module | None,
        num_shared_experts: int | None,
    ) -> dict[str, Any]:
        return {
            "gate": gate,
            "runner_args": {
                "mix_placement": bool(
                    vllm_config.additional_config.get("mix_placement", False)
                ),
                "num_shared_experts": num_shared_experts,
            },
        }

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
        *,
        mix_placement: bool,
        num_shared_experts: int | None,
    ) -> None:
        super().__init__(
            layer_name=layer_name,
            moe_config=moe_config,
            router=router,
            routed_experts=routed_experts,
            enable_dbo=enable_dbo,
            gate=gate,
            shared_experts=shared_experts,
            shared_expert_gate=shared_expert_gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.mix_placement = mix_placement
        self.num_shared_experts = num_shared_experts or 0

    def compute_gate_topk(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep Ascend imports lazy so GPU workers can load the same model shell.
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        assert self.gate is not None
        router_logits, _ = self.gate(hidden_states)
        experts = self.routed_experts
        num_experts = experts.global_num_experts
        if self.mix_placement:
            num_experts += self.num_shared_experts
        # CAM's FFN path applies the scale unless mixed placement folds it into
        # top-k weights. Never run MoERunner's final scaling/reduction again.
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=experts.top_k,
            use_grouped_topk=experts.use_grouped_topk,
            renormalize=experts.renormalize,
            topk_group=experts.topk_group,
            num_expert_group=experts.num_expert_group,
            custom_routing_function=experts.custom_routing_function,
            scoring_func=experts.scoring_func,
            routed_scaling_factor=(
                self.routed_scaling_factor if self.mix_placement else 1.0
            ),
            e_score_correction_bias=experts.e_score_correction_bias,
            mix_placement=self.mix_placement,
            num_logical_experts=router_logits.shape[1],
            num_shared_experts=self.num_shared_experts,
            num_experts=num_experts,
        )
        if force_balanced_topk_ids_enabled():
            balanced_topk_ids = torch.arange(
                topk_ids.numel(), device=topk_ids.device, dtype=torch.int64
            ).reshape(topk_ids.shape)
            topk_ids.copy_(
                balanced_topk_ids.remainder(router_logits.shape[1]).to(topk_ids.dtype)
            )
        return topk_weights.to(torch.float32), topk_ids, router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            raise NotImplementedError(
                "experts-boundary input_ids transport is not implemented",
            )
        topk_weights, topk_ids, router_logits = self.compute_gate_topk(hidden_states)
        return remote_ffn_forward(
            hidden_states,
            layer_idx=self.layer_id,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )
