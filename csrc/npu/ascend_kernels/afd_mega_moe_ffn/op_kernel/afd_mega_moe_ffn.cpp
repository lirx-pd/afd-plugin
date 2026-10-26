// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "afd_mega_moe_ffn_tiling.h"
#include "afd_async_dispatch_recv.h"
#undef TemplateMC2TypeClass
#undef TemplateMC2TypeFunc

// Reuse the checked-in MSD math without importing its standalone kernel entry.
#define GMM_SWIGLU_QUANT_V2_A8W4_MSD
namespace GroupedMatmulDequantSwigluQuant {
using GMMSwigluQuantV2BaseParams = Cam::MegaGmmParams;
using GMMSwigluQuantV2 = Cam::MegaQuantParams;
} // namespace GroupedMatmulDequantSwigluQuant
#include "grouped_matmul_swiglu_quant_v2_layered_a8w4_msd_pre.h"
#include "grouped_matmul_swiglu_quant_v2_layered_a8w4_msd_mid.h"
#include "grouped_matmul_swiglu_quant_v2_layered_a8w4_msd_post.h"

using Cam::AfdMegaMoeFFNTilingData;

namespace AfdMegaMoe {
using namespace AscendC;
using namespace GroupedMatmulDequantSwigluQuant;
using A = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using B = matmul::MatmulType<TPosition::GM, CubeFormat::NZ, int4b_t, false>;
using C = matmul::MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using MM = MMImplTypeCustom<A, B, C>;

template <HardEvent event> __aicore__ inline void Fence() {
    const auto id = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
}
__aicore__ inline int64_t ReadImmutableCount(GM_ADDR base, uint32_t index) {
    return *(reinterpret_cast<volatile __gm__ int64_t *>(base) + index);
}
__aicore__ inline int64_t ReadControl(GM_ADDR base, uint32_t index) {
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(base) + index);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
    // Polling must reload after cache invalidation; a non-volatile load may be hoisted.
    return *(reinterpret_cast<volatile __gm__ int64_t *>(base) + index);
}

__aicore__ inline void WriteCompletion(GM_ADDR output, int64_t value, TPipe &pipe) {
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            // The caller clears this status once per batch, never between its transactions.
            const auto previous = ReadControl(output, 0);
            if (previous < 0) value = previous;
            pipe.Reset();
            TBuf<TPosition::VECCALC> buf;
            pipe.InitBuffer(buf, 32);
            auto local = buf.Get<int64_t>();
            local.SetValue(0, value);
            GlobalTensor<int64_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(output));
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(gm, local, {1U, 8U, 0U, 0U, 0U});
            Fence<HardEvent::MTE3_S>();
        }
    }
}

__aicore__ inline GMAddrParams Addresses(GM_ADDR x, GM_ADDR scales, GM_ADDR counts, GM_ADDR weight,
                                         GM_ADDR wScale, GM_ADDR bias, GM_ADDR y, GM_ADDR yScale,
                                         GM_ADDR compute, const Cam::MegaGmmParams &p,
                                         int64_t layer) {
    GMAddrParams a{};
    a.xGM = x;
    a.xScaleGM = scales;
    a.groupListGM = counts;
    a.weightGM = weight;
    a.weightScaleGM = wScale;
    a.weightAuxiliaryMatrixGM = bias;
    a.yGM = y;
    a.yScaleGM = yScale;
    a.curLayer = layer;
    a.layerNum = p.layerNum;
    a.workSpaceGM = compute;
    a.workSpaceOffset1 = p.workSpaceOffset1 / 2;
    a.workSpaceOffset2 = p.workSpaceOffset1;
    a.workSpaceOffset3 = p.workSpaceOffset1 + p.workSpaceOffset2 / 2;
    return a;
}

__aicore__ inline void OutputTile(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                  const Cam::AfdMegaMoeFFNTilingData &t, GM_ADDR biasList,
                                  int64_t layer, uint32_t expert, uint32_t offset, uint32_t rows,
                                  GM_ADDR counts, TPipe &pipe, uint32_t id, uint32_t vectors) {
    if ASCEND_IS_AIV {
        const Cam::MegaRowRange range(rows, id, vectors);
        if (range.begin == range.end) return;
        pipe.Reset();
        const uint32_t h = t.recv.hiddenSize;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, h * 18 + 32);
        auto hi = buf.Get<half>();
        auto lo = buf.GetWithOffset<half>(h, h * 2);
        auto v = buf.GetWithOffset<float>(h, h * 4);
        auto tmp = buf.GetWithOffset<float>(h, h * 8);
        auto bias = buf.GetWithOffset<float>(h, h * 12);
        auto y = buf.GetWithOffset<bfloat16_t>(h, h * 16);
        constexpr uint32_t SCALE_BATCH_ROWS = 8;
        auto scale = buf.GetWithOffset<float>(SCALE_BATCH_ROWS, h * 18);
        GlobalTensor<half> raw;
        GlobalTensor<float> biases, scales;
        GlobalTensor<bfloat16_t> output;
        raw.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(scratch + w.compute2 + t.g2.workSpaceOffset1));
        biases.SetGlobalBuffer(GetLayerTensorAddr<float>(layer, t.g2.layerNum, biasList));
        scales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.zScales));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch + w.y));
        uint32_t groupEnd = ReadImmutableCount(counts, expert) - offset;
        uint32_t loadedExpert = UINT32_MAX;
        for (uint32_t r = range.begin; r < range.end; ++r) {
            while (r >= groupEnd) groupEnd += ReadImmutableCount(counts, ++expert);
            if (loadedExpert != expert) {
                DataCopy(bias, biases[expert * h], h);
                loadedExpert = expert;
            }
            DataCopy(hi, raw[2 * r * h], h);
            DataCopy(lo, raw[(2 * r + 1) * h], h);
            const uint32_t scaleIndex = (r - range.begin) % SCALE_BATCH_ROWS;
            if (scaleIndex == 0) {
                const uint32_t scaleRows = range.end - r < SCALE_BATCH_ROWS ? range.end - r : SCALE_BATCH_ROWS;
                DataCopyPad(scale, scales[r], {1U, uint32_t(scaleRows * sizeof(float)), 0U, 0U, 0U},
                            {false, 0U, 0U, 0U});
                Fence<HardEvent::MTE2_S>();
            }
            Fence<HardEvent::MTE2_V>();
            Cast(v, hi, RoundMode::CAST_NONE, h);
            Cast(tmp, lo, RoundMode::CAST_NONE, h);
            PipeBarrier<PIPE_V>();
            Muls(v, v, 16.0f, h);
            PipeBarrier<PIPE_V>();
            Add(v, v, tmp, h);
            PipeBarrier<PIPE_V>();
            Add(v, v, bias, h);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::S_V>();
            Muls(v, v, scale.GetValue(scaleIndex), h);
            PipeBarrier<PIPE_V>();
            Cast(y, v, RoundMode::CAST_RINT, h);
            if (t.routedScale != 1.0f) {
                // Match the existing BF16 output mul_: round before scaling.
                PipeBarrier<PIPE_V>();
                Cast(v, y, RoundMode::CAST_NONE, h);
                PipeBarrier<PIPE_V>();
                Muls(v, v, t.routedScale, h);
                PipeBarrier<PIPE_V>();
                Cast(y, v, RoundMode::CAST_RINT, h);
            }
            Fence<HardEvent::V_MTE3>();
            DataCopy(output[r * h], y, h);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::V_MTE2>();
        }
        pipe.Reset();
    }
}
} // namespace AfdMegaMoe

#include "afd_mega_moe_pipeline.h"

namespace AfdMegaMoe {
__aicore__ inline int64_t ValidateTransaction(GM_ADDR scratch, GM_ADDR layerMap,
                                             const Cam::MegaWorkspace &w,
                                             const AfdMegaMoeFFNTilingData &t, TPipe &pipe) {
    const auto layerId = ReadControl(scratch + w.batch, 2);
    const auto layer = layerId >= 0 && layerId < t.layerMapSize ? ReadControl(layerMap, layerId) : -1;
    if (layer < 0 || layer >= t.g1.layerNum) return -1;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, Cam::MegaAlign(t.recv.routeExpertNumPerMoe * sizeof(int64_t)));
    auto counts = buffer.Get<int64_t>();
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(scratch + w.counts));
    Fence<HardEvent::MTE3_MTE2>();
    DataCopyPad(counts, gm, {1U, uint32_t(t.recv.routeExpertNumPerMoe * sizeof(int64_t)), 0U, 0U, 0U},
                {false, 0U, 0U, 0U});
    Fence<HardEvent::MTE2_S>();
    uint64_t total = 0;
    for (uint32_t e = 0; e < t.recv.routeExpertNumPerMoe; ++e) {
        const auto count = counts.GetValue(e);
        if (count < 0 || uint64_t(count) > t.capacity - total) return -2;
        total += count;
    }
    StoreControl(scratch + w.validation, TOTAL_ROWS, total);
    return layer;
}
} // namespace AfdMegaMoe

extern "C" __global__ __aicore__ void
afd_mega_moe_ffn(GM_ADDR anchor, GM_ADDR commArgs, GM_ADDR layerMap, GM_ADDR scratch, GM_ADDR w13,
                 GM_ADDR w2, GM_ADDR s13, GM_ADDR s2, GM_ADDR b13, GM_ADDR b2, GM_ADDR completion,
                 GM_ADDR workspace, GM_ADDR tiling) {
    using namespace AfdMegaMoe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(AfdMegaMoeFFNTilingData);
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(AfdMegaMoeFFNTilingData, t, tiling);
        const Cam::MegaWorkspace w(t.capacity, t.recv.hiddenSize, t.intermediate,
                                   t.recv.routeExpertNumPerMoe, t.recv.tpSize);
        TPipe pipe;
        PipelineFlags flags(scratch + w.flags);
        if ASCEND_IS_AIV {
            flags.Clear(t.recv.aivNum, t.g1.coreNum);
            if (GetBlockIdx() == 0) {
                StoreControl(scratch + w.batch, 2, -1);
                Cam::AfdAsyncDispatchRecvTilingData dr{};
                dr.moeDistributeDispatchInfo = t.recv;
                MoeDistributeDispatchImpl::AfdAsyncDispatchRecv<bfloat16_t, int8_t, true> rx;
                rx.Init(anchor, scratch + w.tileX, scratch + w.tileScales, scratch + w.batch,
                        scratch + w.counts, workspace, &pipe, &dr, commArgs, 1);
                rx.Process(false);
                pipe.Reset();
                const auto result = ValidateTransaction(scratch, layerMap, w, t, pipe);
                StoreControl(scratch + w.validation, 0, result);
                pipe.Reset();
            }
        }
        SyncAll<false>();
        const auto layer = ReadControl(scratch + w.validation, 0);
        if (layer < 0) {
            WriteCompletion(completion, layer, pipe);
            return;
        }
        // The owner validated counts; refresh each consumer's immutable scalar view.
        GlobalTensor<int64_t> counts;
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(scratch + w.counts));
        for (uint32_t e = 0; e < t.recv.routeExpertNumPerMoe; e += 64 / sizeof(int64_t)) {
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counts[e]);
        }
        TileCursor cursor(scratch + w.counts, t.recv.routeExpertNumPerMoe);
        const bool splitVectors = t.recv.aivNum == Cam::MEGA_MAX_AIVS &&
                                  ReadControl(scratch + w.validation, TOTAL_ROWS) >
                                      Cam::MEGA_RING_SLOTS * Cam::MEGA_TILE_ROWS;
        if ASCEND_IS_AIC {
            ComputePipeline(scratch, w, t, flags, cursor, w13, w2, s13, s2, b13, b2, layer, pipe);
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() < Cam::MEGA_COMM_AIVS) {
                TransportPipeline(scratch, w, t, flags, cursor, pipe);
            } else if (splitVectors && GetBlockIdx() >= t.recv.aivNum - OUTPUT_AIVS) {
                OutputPipeline(scratch, w, t, flags, cursor, b2, layer, pipe);
            } else {
                VectorPipeline(scratch, w, t, flags, cursor, w13, w2, s13, s2, b13, b2, layer, pipe,
                               splitVectors ? OUTPUT_AIVS : 0);
            }
        }
        SyncAll<false>();
        WriteCompletion(completion, 0, pipe);
    }
}
