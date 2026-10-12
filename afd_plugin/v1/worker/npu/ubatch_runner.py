# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Thin eager Ascend adapter over vLLM's MRV2 UBatchRunner (ced6857a)."""

import threading
from collections.abc import Callable
from contextlib import nullcontext
from copy import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import ubatching
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, UBatchState
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.workspace import use_workspace_lane
from vllm_ascend.ascend_forward_context import (
    get_mrv2_in_profile_run,
    override_mrv2_in_profile_run,
)
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from afd_plugin.compat.npu.runtime_config import uses_async_cam_dbo
from afd_plugin.compat.patches.npu.mrv2_ubatching import (
    use_async_cam_child_dp_metadata,
    use_npu_input_batch_slicing,
    use_npu_ubatch_contexts,
)
from afd_plugin.model_executor.models.npu.deepseek_attention_metadata import (
    materialize_deepseek_attention_metadata_by_layer,
)


class NPUUBatchRunner(UBatchRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        model_state: ModelState,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_num_reqs: int,
        *,
        kernel_block_sizes: list[int],
        timeout_s: float,
    ):
        super().__init__(
            vllm_config, device, model_state, attn_groups, kv_cache_config, max_num_reqs
        )
        self.ready_barrier = threading.Barrier(self.num_ubatches + 1, timeout=timeout_s)
        self.stage_attn_groups: list[list[list[AttentionGroup]]] = []
        with set_current_vllm_config(vllm_config):
            for _ in range(self.num_ubatches):
                groups = [
                    [copy(group) for group in cache_group]
                    for cache_group in attn_groups
                ]
                for group_idx, cache_group in enumerate(groups):
                    for group in cache_group:
                        group.create_metadata_builders(
                            vllm_config,
                            device,
                            kernel_block_size=kernel_block_sizes[group_idx],
                            num_metadata_builders=1,
                        )
                self.stage_attn_groups.append(groups)

    # Ascend 8d4409d6 asserts ubatch_idx == 0 and always selects builder zero.
    # Feed its existing builder function an independent group per native child;
    # no request/KV slicing or MRV1 state is reproduced here. Eager only.
    def _prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any] | None:
        # ### PATCH START: build Ascend metadata from native child coordinates.
        assert cudagraph_mode == CUDAGraphMode.NONE and not for_capture
        if input_batch.num_tokens == 0:
            # Ascend's existing no-metadata path zeros attention without touching
            # KV. The native context still runs padded tensors and A/F exchange.
            return None
        metadata = build_attn_metadata(
            attn_groups=self.stage_attn_groups[ubatch_idx],
            num_reqs=input_batch.num_reqs,
            num_actual_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            num_actual_tokens=input_batch.num_tokens,
            num_input_tokens=input_batch.num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=int(input_batch.num_scheduled_tokens.max(initial=0)),
            seq_lens=input_batch.seq_lens,
            seq_lens_np=input_batch.seq_lens_np,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            max_seq_len=self.model_state.max_model_len,
            # DSA builders clear padded rows in place. Keep writes in this child.
            block_tables=tuple(table.clone() for table in block_tables),
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            parallel_config=self.parallel_config,
            positions=input_batch.positions,
            attn_state=build_attn_state(
                self.vllm_config,
                input_batch.seq_lens_np,
                input_batch.num_reqs,
                input_batch.num_scheduled_tokens,
                input_batch.num_scheduled_tokens,
                kv_cache_config,
            ),
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
        )
        materialize_deepseek_attention_metadata_by_layer(
            metadata, input_batch.positions, input_batch.num_tokens
        )
        return metadata
        # ### PATCH END: build Ascend metadata from native child coordinates.

    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        previous_override = self.model_state.__dict__.get("prepare_attn")
        self.model_state.prepare_attn = self._prepare_attn
        try:
            with use_npu_input_batch_slicing():
                return super().prepare(
                    input_batch, block_tables, slot_mappings, cg_mode, for_capture
                )
        finally:
            if previous_override is None:
                del self.model_state.prepare_attn
            else:
                self.model_state.prepare_attn = previous_override

    def _make_forward_contexts(
        self,
        ubatch_slices: UBatchSlices,
        attn_metadata: list[dict[str, Any] | None],
        slot_mappings_by_layer: list[dict[str, torch.Tensor]],
        is_padding: list[torch.Tensor | None],
    ) -> list[ForwardContext]:
        with (
            use_async_cam_child_dp_metadata()
            if uses_async_cam_dbo(self.vllm_config)
            else nullcontext()
        ):
            contexts = super()._make_forward_contexts(
                ubatch_slices, attn_metadata, slot_mappings_by_layer, is_padding
            )
        for stage_idx, (context, stage) in enumerate(
            zip(contexts, ubatch_slices, strict=True)
        ):
            context.additional_kwargs = current_platform.set_additional_forward_context(
                context.attn_metadata,
                self.vllm_config,
                context.dp_metadata,
                num_tokens=stage.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ubatch_slices=ubatch_slices,
            )
            context.additional_kwargs.update(
                ubatch_idx=stage_idx, num_ubatches=len(ubatch_slices), dbo_enabled=True
            )
            mask = context.additional_kwargs.get("mc2_mask")
            if mask is not None:
                context.additional_kwargs["mc2_mask"] = mask.clone()
        return contexts

    def run(
        self, model: Callable, model_inputs: dict[str, Any], ubatch_state: UBatchState
    ) -> Any:
        is_profile = get_mrv2_in_profile_run()

        def forward(**inputs):
            with (
                set_current_vllm_config(self.vllm_config),
                override_mrv2_in_profile_run(is_profile),
                skip_dp_coordination()
                if uses_async_cam_dbo(self.vllm_config)
                else nullcontext(),
                use_workspace_lane(0),
            ):
                return model(**inputs)

        # Native prepare, threads, handoff and merge remain upstream. Restore the
        # parent context even if a child fails before native begin returns finish.
        with (
            override_forward_context(ubatching.forward_context._forward_context),
            use_npu_ubatch_contexts(),
        ):
            return super().run(forward, model_inputs, ubatch_state)
