// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_PIPELINE_H
#define AFD_MEGA_MOE_PIPELINE_H

namespace AfdMegaMoe {
enum Stage : uint32_t { RECEIVED, PRE1, G1, PRE2, G2, OUTPUT, SENT };

__aicore__ inline void StoreControl(GM_ADDR base, uint32_t index, int64_t value) {
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(base) + index);
    gm.SetValue(0, value);
    // AIC has no UB DMA: explicitly flush its private scalar cache line.
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
}

class PipelineFlags {
public:
    __aicore__ explicit PipelineFlags(GM_ADDR base) : base_(base) {}
    __aicore__ void Clear(uint32_t vectors) {
        for (uint32_t i = GetBlockIdx(); i < Cam::MEGA_PIPELINE_STAGES * (Cam::MEGA_MAX_AIVS + 1);
             i += vectors)
            StoreControl(base_ + i * Cam::MEGA_ALIGNMENT, 0, 0);
    }
    __aicore__ void Wait(Stage stage, uint32_t sequence) const {
        WaitAt(Address(stage, 0), sequence);
    }
    __aicore__ void Arrive(Stage stage, uint32_t sequence, uint32_t id, uint32_t count) const {
        if ASCEND_IS_AIC {
            Fence<HardEvent::FIX_S>();
        }
        if ASCEND_IS_AIV {
            Fence<HardEvent::MTE3_S>();
        }
        StoreControl(Address(stage, id + 1), 0, sequence);
        if (id == 0) {
            for (uint32_t i = 1; i < count; ++i)
                WaitAt(Address(stage, i + 1), sequence);
            StoreControl(Address(stage, 0), 0, sequence);
        }
    }
private:
    __aicore__ GM_ADDR Address(Stage stage, uint32_t lane) const {
        return base_ + (stage * (Cam::MEGA_MAX_AIVS + 1) + lane) * Cam::MEGA_ALIGNMENT;
    }
    __aicore__ static void WaitAt(GM_ADDR address, uint32_t sequence) {
        while (ReadControl(address, 0) < sequence) {}
    }
    GM_ADDR base_;
};

struct Tile {
    uint32_t expert = 0, offset = 0, rows = 0, sequence = 0;
    __aicore__ WorkSpaceSplitConfig Split() const {
        return {rows, 1, 0, int64_t(expert), int64_t(expert), int64_t(expert),
                Cam::MEGA_TILE_ROWS, rows, true};
    }
};

class TileCursor {
public:
    __aicore__ TileCursor(GM_ADDR counts, uint32_t experts)
        : counts_(counts), experts_(experts), count_(ReadControl(counts, 0)) {}
    __aicore__ Tile Next() {
        while (expert_ < experts_) {
            if (offset_ < count_) {
                const uint32_t rows = count_ - offset_ < Cam::MEGA_TILE_ROWS ? count_ - offset_ : Cam::MEGA_TILE_ROWS;
                const Tile tile{expert_, offset_, rows, ++sequence_};
                offset_ += rows;
                return tile;
            }
            ++expert_;
            offset_ = 0;
            if (expert_ < experts_) count_ = ReadControl(counts_, expert_);
        }
        return {};
    }
private:
    GM_ADDR counts_;
    uint32_t experts_, count_, expert_ = 0, offset_ = 0, sequence_ = 0;
};

__aicore__ inline uint64_t Align32(uint64_t value) { return (value + 31) / 32 * 32; }

// Retains compact-v2's expert/TP order and releases its input only after every tile is drained.
class TileTransport {
public:
    __aicore__ TileTransport(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                             const AfdMegaMoeFFNTilingData &t, TPipe &pipe)
        : scratch_(scratch), w_(w), t_(t), pipe_(pipe) {
        context_ = reinterpret_cast<__gm__ HcclOpResParam *>(GetHcclContext<HCCL_GROUP_ID_0>());
        const auto &r = t_.recv;
        pipe_.Reset();
        const uint32_t metaBytes = (5 + r.tpSize + r.routeExpertNumPerMoe * r.tpSize) * 8;
        pipe_.InitBuffer(metadata_, Align32(metaBytes));
        pipe_.InitBuffer(buffer_, Align32(r.hiddenSize * 2 + 64 + r.routeExpertNumPerMoe * 8));
        auto meta = metadata_.Get<int64_t>();
        GlobalTensor<int64_t> batch;
        batch.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(scratch_ + w_.batch));
        DataCopyPad(meta, batch, {1U, metaBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        source_ = meta.GetValue(1);
        end_ = meta.GetValue(4);
        // Only this chunk's immutable counts become inclusive prefixes in private UB.
        for (uint32_t tp = 0; tp < r.tpSize; ++tp) {
            uint32_t prefix = 0;
            for (uint32_t e = 0; e < r.routeExpertNumPerMoe; ++e) {
                const auto index = 5 + r.tpSize + tp * r.routeExpertNumPerMoe + e;
                prefix += meta.GetValue(index);
                meta.SetValue(index, prefix);
            }
        }
        payload_ = Align32(5 * 8) + 2 * Align32(r.routeExpertNumPerMoe * 4);
        indices_ = payload_ + Align32((r.hiddenSize + 32) * (r.maxSeqLen / r.tpSize));
        sourceBytes_ = indices_ + Align32(r.maxSeqLen / r.tpSize * r.topk * 2);
        returnFlags_ = Align32(4ULL * (r.moeRankNum + r.routeExpertNumPerMoe * r.moeRankNum) * 48);
    }
    __aicore__ void Receive(const Tile &tile) {
        auto x = buffer_.Get<int8_t>();
        auto scales = buffer_.Get<float>();
        auto index = buffer_.GetWithOffset<uint16_t>(16, t_.recv.hiddenSize * 2 + 32);
        auto counts = buffer_.GetWithOffset<int64_t>(t_.recv.routeExpertNumPerMoe,
                                                     t_.recv.hiddenSize * 2 + 64);
        GlobalTensor<int8_t> dst;
        GlobalTensor<float> dstScale;
        const auto slot = scratch_ + w_.SlotOffset(tile.sequence);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(slot + w_.tileX));
        dstScale.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(slot + w_.tileScales));
        uint32_t tpBegin = 0;
        for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) {
            const uint32_t count = Count(tp, tile.expert);
            const uint32_t prefix = Prefix(tp, tile.expert);
            const auto source = reinterpret_cast<GM_ADDR>(context_->localWindowsIn) + (source_ + tp) * sourceBytes_;
            GlobalTensor<uint16_t> routes;
            GlobalTensor<int8_t> tokens;
            routes.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(source + indices_));
            tokens.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(source + payload_));
            for (uint32_t r = GetBlockIdx(); r < tile.rows; r += Cam::MEGA_COMM_AIVS) {
                const auto row = tile.offset + r;
                if (row < tpBegin || row >= tpBegin + count) continue;
                DataCopyPad(index, routes[prefix + row - tpBegin], {1U, 2U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                Fence<HardEvent::MTE2_S>();
                const uint32_t token = index.GetValue(0);
                Fence<HardEvent::S_MTE2>();
                DataCopy(x, tokens[token * (t_.recv.hiddenSize + 32)], t_.recv.hiddenSize + 32);
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(dst[r * t_.recv.hiddenSize], x, t_.recv.hiddenSize);
                DataCopyPad(dstScale[r], scales[t_.recv.hiddenSize / 4], {1U, 4U, 0U, 0U, 0U});
                Fence<HardEvent::MTE3_MTE2>();
            }
            tpBegin += count;
        }
        if (GetBlockIdx() == 0) {
            for (uint32_t e = 0; e < t_.recv.routeExpertNumPerMoe; ++e)
                counts.SetValue(e, e == tile.expert ? tile.rows : 0);
            GlobalTensor<int64_t> dstCounts;
            dstCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slot + w_.tileCounts));
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(dstCounts, counts, {1U, uint32_t(t_.recv.routeExpertNumPerMoe * 8), 0U, 0U, 0U});
        }
        Fence<HardEvent::MTE3_S>();
    }
    __aicore__ void Send(const Tile &tile) {
        auto local = buffer_.Get<bfloat16_t>();
        GlobalTensor<bfloat16_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch_ + w_.SlotOffset(tile.sequence) + w_.y));
        uint32_t tpBegin = 0;
        for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) {
            const uint32_t count = Count(tp, tile.expert);
            const uint32_t prefix = metadata_.Get<int64_t>().GetValue(5 + tp) + Prefix(tp, tile.expert);
            GlobalTensor<bfloat16_t> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(Peer(source_ + tp) + returnFlags_ +
                                 Align32(t_.recv.moeRankNum * 4)));
            for (uint32_t r = GetBlockIdx(); r < tile.rows; r += Cam::MEGA_COMM_AIVS) {
                const auto row = tile.offset + r;
                if (row < tpBegin || row >= tpBegin + count) continue;
                DataCopy(local, src[r * t_.recv.hiddenSize], t_.recv.hiddenSize);
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(dst[(prefix + row - tpBegin) * t_.recv.hiddenSize], local, t_.recv.hiddenSize);
                Fence<HardEvent::MTE3_MTE2>();
            }
            tpBegin += count;
        }
        Fence<HardEvent::MTE3_S>();
    }
    __aicore__ void Finish() {
        if (GetBlockIdx() != 0) return;
        auto flags = buffer_.Get<int32_t>();
        for (uint32_t e = 0; e < t_.recv.routeExpertNumPerMoe; ++e) flags.SetValue(e, e <= end_ ? 1 : 0);
        for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) {
            GlobalTensor<int32_t> dst;
            const auto local = reinterpret_cast<GM_ADDR>(context_->localWindowsIn) + (source_ + tp) * sourceBytes_;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(local + Align32(5 * 8) +
                                                                  Align32(t_.recv.routeExpertNumPerMoe * 4)));
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(dst, flags, {1U, uint32_t(t_.recv.routeExpertNumPerMoe * 4), 0U, 0U, 0U});
            Fence<HardEvent::MTE3_S>();
        }
        if (end_ == t_.recv.routeExpertNumPerMoe - 1) {
            auto zeros = buffer_.Get<int64_t>();
            for (uint32_t j = 0; j < 5; ++j) zeros.SetValue(j, 0);
            for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) {
                GlobalTensor<int64_t> dst;
                dst.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(context_->localWindowsIn +
                                       (source_ + tp) * sourceBytes_));
                Fence<HardEvent::S_MTE3>();
                DataCopyPad(dst, zeros, {1U, 40U, 0U, 0U, 0U});
                Fence<HardEvent::MTE3_S>();
            }
            for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) {
                flags.SetValue(0, metadata_.Get<int64_t>().GetValue(5 + tp) + 1);
                GlobalTensor<int32_t> dst;
                dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Peer(source_ + tp) + returnFlags_) +
                                   t_.recv.moeRankId - t_.recv.attnRankNum);
                Fence<HardEvent::S_MTE3>();
                DataCopyPad(dst, flags, {1U, 4U, 0U, 0U, 0U});
                Fence<HardEvent::MTE3_S>();
            }
        }
    }
private:
    __aicore__ uint32_t Count(uint32_t tp, uint32_t expert) {
        return Prefix(tp, expert + 1) - Prefix(tp, expert);
    }
    __aicore__ uint32_t Prefix(uint32_t tp, uint32_t expert) {
        return expert == 0 ? 0 : metadata_.Get<int64_t>().GetValue(
            5 + t_.recv.tpSize + tp * t_.recv.routeExpertNumPerMoe + expert - 1);
    }
    __aicore__ GM_ADDR Peer(uint32_t rank) const {
        return reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ HcclRankRelationResV2 *>(
                   context_->remoteRes[rank].nextDevicePtr)->windowsIn);
    }
    GM_ADDR scratch_;
    const Cam::MegaWorkspace &w_;
    const AfdMegaMoeFFNTilingData &t_;
    TPipe &pipe_;
    TBuf<TPosition::VECCALC> buffer_, metadata_;
    __gm__ HcclOpResParam *context_;
    uint32_t source_, end_;
    uint64_t payload_, indices_, sourceBytes_, returnFlags_;
};

__aicore__ inline void TransportPipeline(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                         const AfdMegaMoeFFNTilingData &t, const PipelineFlags &flags,
                                         TileCursor receive, TPipe &pipe) {
    TileTransport transport(scratch, w, t, pipe);
    auto send = receive;
    auto next = receive.Next();
    for (uint32_t i = 0; i < Cam::MEGA_RING_SLOTS && next.rows; ++i) {
        transport.Receive(next);
        flags.Arrive(RECEIVED, next.sequence, GetBlockIdx(), Cam::MEGA_COMM_AIVS);
        next = receive.Next();
    }
    for (auto tile = send.Next(); tile.rows; tile = send.Next()) {
        flags.Wait(OUTPUT, tile.sequence);
        transport.Send(tile);
        flags.Arrive(SENT, tile.sequence, GetBlockIdx(), Cam::MEGA_COMM_AIVS);
        // All senders have read the slot before any receiver overwrites it.
        flags.Wait(SENT, tile.sequence);
        if (next.rows) {
            transport.Receive(next);
            flags.Arrive(RECEIVED, next.sequence, GetBlockIdx(), Cam::MEGA_COMM_AIVS);
            next = receive.Next();
        }
    }
    transport.Finish();
    pipe.Reset();
}

__aicore__ inline void InvalidateScalarInputs(GM_ADDR address) {
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address));
    DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(gm);
}

__aicore__ inline void ComputeStage(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                    const AfdMegaMoeFFNTilingData &t, const PipelineFlags &flags,
                                    const Tile &tile, GM_ADDR weight, GM_ADDR scale, GM_ADDR bias,
                                    int64_t layer, bool first, MM::MT &mm, TPipe &pipe) {
    flags.Wait(first ? PRE1 : PRE2, tile.sequence);
    const auto slot = scratch + w.SlotOffset(tile.sequence);
    InvalidateScalarInputs(slot + w.tileCounts);
    const auto &p = first ? t.g1 : t.g2;
    auto a = Addresses(nullptr, nullptr, slot + w.tileCounts, weight, scale, bias, nullptr, nullptr,
                       slot + (first ? w.compute : w.compute2), p, layer);
    auto split = tile.Split();
    TCubeTiling cube = *reinterpret_cast<const TCubeTiling *>(first ? t.cube1 : t.cube2);
    mm.SetSubBlockIdx(0);
    mm.Init(&cube);
    GMMA8W4MidProcess<MM> mid(mm);
    mid.Init(a, &p);
    mid.Process(split, 0);
    mm.End();
    pipe.Reset();
    flags.Arrive(first ? G1 : G2, tile.sequence, GetBlockIdx(), t.g1.coreNum);
}

__aicore__ inline void ComputePipeline(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                       const AfdMegaMoeFFNTilingData &t, const PipelineFlags &flags,
                                       TileCursor cursor, GM_ADDR w13, GM_ADDR w2, GM_ADDR s13,
                                       GM_ADDR s2, GM_ADDR b13, GM_ADDR b2, int64_t layer, TPipe &pipe) {
    MM::MT mm;
    auto tile = cursor.Next();
    if (!tile.rows) return;
    ComputeStage(scratch, w, t, flags, tile, w13, s13, b13, layer, true, mm, pipe);
    while (tile.rows) {
        auto next = cursor.Next();
        if (next.rows) ComputeStage(scratch, w, t, flags, next, w13, s13, b13, layer, true, mm, pipe);
        ComputeStage(scratch, w, t, flags, tile, w2, s2, b2, layer, false, mm, pipe);
        tile = next;
    }
}

__aicore__ inline void PrepareTile(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                   const AfdMegaMoeFFNTilingData &t, const PipelineFlags &flags,
                                   const Tile &tile, GM_ADDR weight, GM_ADDR scale, GM_ADDR bias,
                                   int64_t layer, bool first, TPipe &pipe) {
    const auto slot = scratch + w.SlotOffset(tile.sequence);
    if (first) flags.Wait(RECEIVED, tile.sequence);
    InvalidateScalarInputs(slot + w.tileCounts);
    const auto &p = first ? t.g1 : t.g2;
    auto a = Addresses(slot + (first ? w.tileX : w.z), slot + (first ? w.tileScales : w.zScales),
                       slot + w.tileCounts, weight, scale, bias, nullptr, nullptr,
                       slot + (first ? w.compute : w.compute2), p, layer);
    auto split = tile.Split();
    GMMA8W4PreProcess pre;
    pre.Init(a, &p);
    pre.SetVectorGroup(GetBlockIdx() - Cam::MEGA_COMM_AIVS, t.recv.aivNum - Cam::MEGA_COMM_AIVS);
    pre.Process(split, 0, &pipe);
    pipe.Reset();
    flags.Arrive(first ? PRE1 : PRE2, tile.sequence, GetBlockIdx() - Cam::MEGA_COMM_AIVS,
                 t.recv.aivNum - Cam::MEGA_COMM_AIVS);
}

__aicore__ inline void VectorPipeline(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                      const AfdMegaMoeFFNTilingData &t, const PipelineFlags &flags,
                                      TileCursor cursor, GM_ADDR w13, GM_ADDR w2, GM_ADDR s13,
                                      GM_ADDR s2, GM_ADDR b13, GM_ADDR b2, int64_t layer, TPipe &pipe) {
    auto tile = cursor.Next();
    if (!tile.rows) return;
    const uint32_t id = GetBlockIdx() - Cam::MEGA_COMM_AIVS;
    const uint32_t vectors = t.recv.aivNum - Cam::MEGA_COMM_AIVS;
    PrepareTile(scratch, w, t, flags, tile, w13, s13, b13, layer, true, pipe);
    while (tile.rows) {
        auto next = cursor.Next();
        // Feed G1(next) before postprocessing this tile so Cube and Vector can overlap.
        if (next.rows) PrepareTile(scratch, w, t, flags, next, w13, s13, b13, layer, true, pipe);
        const auto slot = scratch + w.SlotOffset(tile.sequence);
        flags.Wait(G1, tile.sequence);
        InvalidateScalarInputs(slot + w.tileCounts);
        auto a = Addresses(slot + w.tileX, slot + w.tileScales, slot + w.tileCounts, w13, s13,
                           b13, slot + w.z, slot + w.zScales, slot + w.compute, t.g1, layer);
        auto split = tile.Split();
        GMMA8W4PostProcess act;
        act.Init(a, &t.g1, &t.quant);
        act.SetVectorGroup(id, vectors);
        act.Process(split, 0, &pipe);
        Fence<HardEvent::MTE3_S>();
        pipe.Reset();
        // PRE2 reads the same row partition as postprocess, with MTE3 drained before reuse.
        PrepareTile(scratch, w, t, flags, tile, w2, s2, b2, layer, false, pipe);
        flags.Wait(G2, tile.sequence);
        OutputTile(slot, w, t, b2, layer, tile.expert, 0, tile.rows, pipe);
        flags.Arrive(OUTPUT, tile.sequence, id, vectors);
        tile = next;
    }
}
} // namespace AfdMegaMoe
#endif
