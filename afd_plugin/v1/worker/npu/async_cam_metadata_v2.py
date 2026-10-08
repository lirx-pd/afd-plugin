# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MRV2 input adaptation for the existing model-owned Async CAM stages."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from types import MethodType
from typing import TYPE_CHECKING, cast

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.ubatch_utils import UBatchSlice
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo
from afd_plugin.model_executor.models.npu.async_cam_layout import AsyncMoeUbatchMetadata
from afd_plugin.model_executor.models.npu.deepseek_attention_metadata import (
    isolate_deepseek_attention_builder_inputs,
    materialize_deepseek_attention_metadata,
    materialize_deepseek_attention_metadata_by_layer,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import plan_async_moe_stages
from afd_plugin.v1.worker.npu.ubatch_utils import split_attn_metadata

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


def plan_async_cam_stage_metadata(
    runner: AFDNPUAttentionModelRunnerV2,
    input_batch: AscendInputBatch,
) -> AsyncMoeUbatchMetadata | None:
    """Plan real input coordinates, including dummy runs that skip Attention."""
    if input_batch.num_tokens == 0:
        return None
    extra_info = cast(AFDAsyncExtraInfo, runner.connector.extra_info)
    use_sp = runner.vllm_config.parallel_config.use_sequence_parallel_moe
    stages = plan_async_moe_stages(
        input_batch.num_scheduled_tokens[: input_batch.num_reqs],
        split=extra_info.async_moe_split,
        use_sequence_parallel=use_sp,
        tensor_parallel_size=runner.vllm_config.parallel_config.tensor_parallel_size,
    )
    if stages is None:
        return None
    return AsyncMoeUbatchMetadata(
        attn_metadata=[None for _ in stages],
        stages=stages,
        parent_input_tokens=input_batch.num_tokens_after_padding,
        use_sequence_parallel=use_sp,
    )


def build_async_cam_stage_metadata(
    runner: AFDNPUAttentionModelRunnerV2,
    input_batch: AscendInputBatch,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    attn_groups: list[list[AttentionGroup]],
) -> AsyncMoeUbatchMetadata | None:
    """Adapt native request coordinates without importing the MRV1 runner."""
    plan = plan_async_cam_stage_metadata(runner, input_batch)
    if plan is None:
        return None
    stages = plan.stages
    slices = [UBatchSlice(stage.request_slice, stage.token_slice) for stage in stages]
    stage_metadata: list[dict[str, AttentionMetadata]] = [{} for _ in stages]
    # DSA shares one request cache across KV groups, but never across stages.
    dsa_caches: list[dict] = [{} for _ in stages]
    num_reqs = input_batch.num_reqs
    query_start_loc_cpu = torch.from_numpy(
        input_batch.query_start_loc_np[: num_reqs + 1]
    )
    seq_lens_cpu = torch.from_numpy(input_batch.seq_lens_np[:num_reqs])
    computed_cpu = torch.from_numpy(input_batch.num_computed_tokens_np[:num_reqs])
    is_prefilling = torch.from_numpy(input_batch.is_prefilling_np[:num_reqs])
    max_query_len = int(input_batch.num_scheduled_tokens[:num_reqs].max())
    max_seq_len = int(seq_lens_cpu.max())
    for group_idx, groups in enumerate(attn_groups):
        common = AscendCommonAttentionMetadata(
            query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=input_batch.seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[:num_reqs],
            num_computed_tokens_cpu=computed_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=input_batch.num_tokens,
            num_input_tokens=input_batch.num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_tables[group_idx],
            slot_mapping=slot_mappings[group_idx],
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            is_prefilling=is_prefilling,
            causal=True,
        )
        for group in groups:
            if len(group.metadata_builders) < len(stages) + 1:
                # Preserve native builder zero and its kernel/storage block sizes.
                stage_group = copy(group)
                stage_group.create_metadata_builders(
                    runner.vllm_config,
                    runner.device,
                    kernel_block_size=runner.kernel_block_sizes[group_idx],
                    num_metadata_builders=len(stages),
                )
                group.metadata_builders.extend(stage_group.metadata_builders)
            for stage_idx, stage_common in enumerate(
                split_attn_metadata(slices, common)
            ):
                builder = group.get_metadata_builder(stage_idx + 1)
                isolate_deepseek_attention_builder_inputs(builder, stage_common)
                if isinstance(builder, AscendDSAMetadataBuilder):
                    metadata = builder.build(
                        common_prefix_len=0,
                        common_attn_metadata=stage_common,
                        num_actual_reqs=stage_common.num_reqs,
                        common_ratio_to_sas_metadata=dsa_caches[stage_idx],
                    )
                    dsa_caches[stage_idx] = builder.common_ratio_to_sas_metadata
                else:
                    metadata = builder.build(
                        common_prefix_len=0,
                        common_attn_metadata=stage_common,
                    )
                # Attention sees real global rows; TP padding belongs to HC/FFN.
                materialize_deepseek_attention_metadata(
                    metadata, stage_common.positions, stage_common.num_input_tokens
                )
                for layer_name in group.layer_names:
                    stage_metadata[stage_idx][layer_name] = metadata
    return replace(plan, attn_metadata=stage_metadata)


@contextmanager
def use_async_cam_stage_metadata(
    runner: AFDNPUAttentionModelRunnerV2,
) -> Iterator[None]:
    """Scope input adaptation to this native model-state instance."""
    state = runner.model_state
    original_prepare = state.prepare_attn
    original_inputs = state.prepare_inputs
    previous_overrides = {
        name: state.__dict__[name]
        for name in ("prepare_attn", "prepare_inputs")
        if name in state.__dict__
    }

    # Patch reason: native MRV2 prepares only full-batch builder-zero metadata.
    # Patch functionality: keep native preparation, then build model-owned stages.
    # Signature: matches AscendModelState.prepare_attn at vLLM-Ascend
    # 99d96c3da44adfd94e277173b1395eb46659f11f.
    # Delegation exception: native request/KV preparation remains in the saved
    # bound method; only the Async CAM stage sidecar is added here.
    # Removal plan: use a native per-batch metadata extension hook when available.
    def prepare_attn(
        self: ModelState,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict:
        # ### PATCH START: adapt model-owned Async CAM stages.
        metadata = original_prepare(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
        )
        materialize_deepseek_attention_metadata_by_layer(
            metadata, input_batch.positions, input_batch.num_tokens
        )
        runner._afd_async_moe_ubatch_metadata = build_async_cam_stage_metadata(
            runner, input_batch, block_tables, slot_mappings, attn_groups
        )
        return metadata
        # ### PATCH END: adapt model-owned Async CAM stages.

    # Patch reason: native profile skips prepare_attn but still calls this hook.
    # Patch functionality: install a stage plan with None attention metadata,
    # preserving the backend's existing skip-Attention profile path.
    # Signature: matches DefaultModelState.prepare_inputs at vLLM ced6857a.
    # Delegation exception: native model-specific inputs are preserved.
    # Removal plan: share the native per-batch metadata extension hook.
    def prepare_inputs(
        self: ModelState, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        # ### PATCH START: plan stages for skip-Attention profile forwards.
        inputs = original_inputs(input_batch, req_states)
        if runner._afd_async_moe_ubatch_metadata is None:
            runner._afd_async_moe_ubatch_metadata = plan_async_cam_stage_metadata(
                runner, cast(AscendInputBatch, input_batch)
            )
        return inputs
        # ### PATCH END: plan stages for skip-Attention profile forwards.

    try:
        state.prepare_attn = MethodType(prepare_attn, state)
        state.prepare_inputs = MethodType(prepare_inputs, state)
        yield
    finally:
        del state.prepare_attn
        del state.prepare_inputs
        state.__dict__.update(previous_overrides)
