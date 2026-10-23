// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "afd_mega_moe_ffn_tiling.h"
#include "afd_async_dispatch_recv.h"
#undef TemplateMC2TypeClass
#undef TemplateMC2TypeFunc
#include "afd_async_combine_send.h"
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
__aicore__ inline int64_t ReadControl(GM_ADDR base, uint32_t index) {
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(base) + index);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
    return gm.GetValue(0);
}

__aicore__ inline void WriteCompletion(GM_ADDR output, int64_t value, TPipe &pipe) {
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
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

__aicore__ inline void PackTile(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                const Cam::AfdMegaMoeFFNTilingData &t, uint64_t row, uint32_t rows,
                                uint32_t expert, TPipe &pipe) {
    if ASCEND_IS_AIV {
        pipe.Reset();
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, Cam::MegaAlign(t.recv.hiddenSize) +
                                 Cam::MegaAlign(t.recv.routeExpertNumPerMoe * 8) + 32);
        auto x = buf.Get<int8_t>();
        auto counts = buf.GetWithOffset<int64_t>(t.recv.routeExpertNumPerMoe,
                                                 Cam::MegaAlign(t.recv.hiddenSize));
        auto scale = buf.GetWithOffset<float>(
            8, Cam::MegaAlign(t.recv.hiddenSize) + Cam::MegaAlign(t.recv.routeExpertNumPerMoe * 8));
        GlobalTensor<int8_t> src, dst;
        GlobalTensor<float> sSrc, sDst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.x));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.tileX));
        sSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scales));
        sDst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.tileScales));
        for (uint32_t r = GetBlockIdx(); r < rows; r += t.recv.aivNum) {
            DataCopy(x, src[(row + r) * t.recv.hiddenSize], t.recv.hiddenSize);
            DataCopyPad(scale, sSrc[row + r], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(dst[r * t.recv.hiddenSize], x, t.recv.hiddenSize);
            DataCopyPad(sDst[r], scale, {1U, 4U, 0U, 0U, 0U});
            Fence<HardEvent::MTE3_MTE2>();
        }
        if (GetBlockIdx() == 0) {
            for (uint32_t e = 0; e < t.recv.routeExpertNumPerMoe; ++e)
                counts.SetValue(e, e == expert ? rows : 0);
            GlobalTensor<int64_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(scratch + w.tileCounts));
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(gm, counts, {1U, uint32_t(t.recv.routeExpertNumPerMoe * 8), 0U, 0U, 0U});
            Fence<HardEvent::MTE3_S>();
        }
        pipe.Reset();
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

__aicore__ inline void MatmulStage(MM::MT &mm, GMAddrParams &a, const Cam::MegaGmmParams &p,
                                   WorkSpaceSplitConfig &split, TCubeTiling &tiling, TPipe &pipe) {
    GMMA8W4PreProcess pre;
    GMMA8W4MidProcess<MM> mid(mm);
    pre.Init(a, &p);
    mid.Init(a, &p);
    pre.Process(split, 0, &pipe);
    if ASCEND_IS_AIV {
        pipe.Reset();
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tiling);
    }
    mid.Process(split, 0);
    if ASCEND_IS_AIC {
        mm.End();
        // Init allocates new TPipe buffer records for every matrix stage.
        pipe.Reset();
    }
    SyncAll<false>();
}

__aicore__ inline void OutputTile(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                  const Cam::AfdMegaMoeFFNTilingData &t, GM_ADDR biasList,
                                  int64_t layer, uint32_t expert, uint64_t outRow, uint32_t rows,
                                  TPipe &pipe) {
    if ASCEND_IS_AIV {
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
        auto scale = buf.GetWithOffset<float>(8, h * 18);
        GlobalTensor<half> raw;
        GlobalTensor<float> biases, scales;
        GlobalTensor<bfloat16_t> output;
        raw.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(scratch + w.compute + t.g2.workSpaceOffset1));
        biases.SetGlobalBuffer(GetLayerTensorAddr<float>(layer, t.g2.layerNum, biasList));
        scales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.zScales));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch + w.y));
        DataCopy(bias, biases[expert * h], h);
        Fence<HardEvent::MTE2_V>();
        for (uint32_t r = GetBlockIdx(); r < rows; r += t.recv.aivNum) {
            DataCopy(hi, raw[2 * r * h], h);
            DataCopy(lo, raw[(2 * r + 1) * h], h);
            DataCopyPad(scale, scales[r], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            Fence<HardEvent::MTE2_V>();
            Fence<HardEvent::MTE2_S>();
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
            Muls(v, v, scale.GetValue(0), h);
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
            DataCopy(output[(outRow + r) * h], y, h);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::V_MTE2>();
        }
        pipe.Reset();
    }
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
        if ASCEND_IS_AIV {
            Cam::AfdAsyncDispatchRecvTilingData dr{};
            dr.moeDistributeDispatchInfo = t.recv;
            MoeDistributeDispatchImpl::AfdAsyncDispatchRecv<bfloat16_t, int8_t, true> rx;
            rx.Init(anchor, scratch + w.x, scratch + w.scales, scratch + w.batch,
                    scratch + w.counts, workspace, &pipe, &dr, commArgs, 1);
            rx.Process();
            pipe.Reset();
        }
        // All AIVs run DR, and every AIC/AIV reaches each mixed barrier once.
        SyncAll<false>();
        const auto layerId = ReadControl(scratch + w.batch, 2);
        const auto layer =
            layerId >= 0 && layerId < t.layerMapSize ? ReadControl(layerMap, layerId) : -1;
        if (layer < 0 || layer >= t.g1.layerNum) {
            WriteCompletion(completion, -1, pipe);
            return;
        }
        uint64_t total = 0;
        for (uint32_t e = 0; e < t.recv.routeExpertNumPerMoe; ++e) {
            const auto count = ReadControl(scratch + w.counts, e);
            if (count < 0 || uint64_t(count) > t.capacity - total) {
                WriteCompletion(completion, -2, pipe);
                return;
            }
            total += count;
        }
        TCubeTiling cube1 = *reinterpret_cast<TCubeTiling *>(t.cube1);
        TCubeTiling cube2 = *reinterpret_cast<TCubeTiling *>(t.cube2);
        MM::MT mm;
        uint64_t base = 0;
        for (uint32_t e = 0; e < t.recv.routeExpertNumPerMoe; ++e) {
            const uint32_t count = ReadControl(scratch + w.counts, e);
            for (uint32_t offset = 0; offset < count; offset += Cam::MEGA_TILE_ROWS) {
                const uint32_t rows =
                    count - offset < Cam::MEGA_TILE_ROWS ? count - offset : Cam::MEGA_TILE_ROWS;
                PackTile(scratch, w, t, base + offset, rows, e, pipe);
                SyncAll<false>();
                // MSD helpers read both tile counts and token scales through scalar cache.
                GlobalTensor<int64_t> controls;
                controls.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int64_t *>(scratch + w.tileCounts));
                DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE,
                                         DcciDst::CACHELINE_OUT>(controls);
                WorkSpaceSplitConfig split{
                    rows, 1,   0, int64_t(e), int64_t(e), int64_t(e), Cam::MEGA_TILE_ROWS,
                    rows, true};
                auto a1 = Addresses(scratch + w.tileX, scratch + w.tileScales,
                                    scratch + w.tileCounts, w13, s13, b13, scratch + w.z,
                                    scratch + w.zScales, scratch + w.compute, t.g1, layer);
                MatmulStage(mm, a1, t.g1, split, cube1, pipe);
                GMMA8W4PostProcess act;
                act.Init(a1, &t.g1, &t.quant);
                act.Process(split, 0, &pipe);
                if ASCEND_IS_AIV {
                    pipe.Reset();
                }
                SyncAll<false>();
                auto a2 =
                    Addresses(scratch + w.z, scratch + w.zScales, scratch + w.tileCounts, w2, s2,
                              b2, scratch + w.y, nullptr, scratch + w.compute, t.g2, layer);
                MatmulStage(mm, a2, t.g2, split, cube2, pipe);
                OutputTile(scratch, w, t, b2, layer, e, base + offset, rows, pipe);
                SyncAll<false>();
            }
            base += count;
        }
        if ASCEND_IS_AIV {
            Cam::AfdAsyncCombineSendTilingData cs{};
            cs.moeDistributeCombineInfo = t.send;
            MoeDistributeCombineSendImpl::AfdAsyncCombineSend<bfloat16_t> tx;
            tx.Init(scratch + w.y, workspace, &pipe, &cs, commArgs, scratch + w.batch, 1);
            tx.Process();
            pipe.Reset();
        }
        SyncAll<false>();
        WriteCompletion(completion, 0, pipe);
    }
}
