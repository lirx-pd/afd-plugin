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
    __aicore__ void Clear(uint32_t vectors, uint32_t cubes) {
        for (uint32_t stage = 0; stage < Cam::MEGA_PIPELINE_STAGES; ++stage) {
            const uint32_t participants = stage == RECEIVED || stage == SENT
                                              ? Cam::MEGA_COMM_AIVS
                                              : (stage == G1 || stage == G2 ? cubes : vectors - Cam::MEGA_COMM_AIVS);
            for (uint32_t lane = GetBlockIdx(); lane <= participants; lane += vectors)
                StoreControl(Address(Stage(stage), lane), 0, 0);
        }
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
        // Leaders publish after their members; the root polls only leaders.
        constexpr uint32_t group = 4;
        if (id % group == 0) {
            const uint32_t end = id + group < count ? id + group : count;
            for (uint32_t i = id + 1; i < end; ++i)
                WaitAt(Address(stage, i + 1), sequence);
        }
        StoreControl(Address(stage, id + 1), 0, sequence);
        if (id == 0) {
            for (uint32_t i = group; i < count; i += group)
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
    uint32_t expert = 0, offset = 0, rows = 0, sequence = 0, lastExpert = 0;
    __aicore__ WorkSpaceSplitConfig Split() const {
        return {rows, 1, 0, int64_t(expert), int64_t(expert), int64_t(lastExpert),
                Cam::MEGA_TILE_ROWS, rows, true};
    }
};

// Counts were invalidated and validated at kernel entry.
class TileCursor {
public:
    __aicore__ TileCursor(GM_ADDR counts, uint32_t experts)
        : counts_(counts), experts_(experts), count_(ReadImmutableCount(counts, 0)) {}
    __aicore__ Tile Next() {
        SkipEmpty();
        if (expert_ == experts_) return {};
        Tile tile{expert_, offset_, 0, ++sequence_, expert_};
        while (tile.rows < Cam::MEGA_TILE_ROWS && expert_ < experts_) {
            const uint32_t remaining = Cam::MEGA_TILE_ROWS - tile.rows;
            const uint32_t rows = count_ - offset_ < remaining ? count_ - offset_ : remaining;
            tile.rows += rows;
            tile.lastExpert = expert_;
            offset_ += rows;
            SkipEmpty();
        }
        return tile;
    }
private:
    __aicore__ void SkipEmpty() {
        while (expert_ < experts_ && offset_ == count_) {
            ++expert_;
            offset_ = 0;
            if (expert_ < experts_) count_ = ReadImmutableCount(counts_, expert_);
        }
    }
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
        pipe_.InitBuffer(buffer_, Align32(r.hiddenSize * 2 + 32 + Cam::MEGA_ROUTE_INDEX_BYTES +
                                           r.routeExpertNumPerMoe * 8));
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
        auto index = buffer_.GetWithOffset<uint16_t>(Cam::MEGA_TILE_ROWS, t_.recv.hiddenSize * 2 + 32);
        auto counts = buffer_.GetWithOffset<int64_t>(t_.recv.routeExpertNumPerMoe,
                                                     t_.recv.hiddenSize * 2 + 32 + Cam::MEGA_ROUTE_INDEX_BYTES);
        GlobalTensor<int8_t> dst;
        GlobalTensor<float> dstScale;
        const auto slot = scratch_ + w_.SlotOffset(tile.sequence);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(slot + w_.tileX));
        dstScale.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(slot + w_.tileScales));
        if (GetBlockIdx() == 0) {
            for (uint32_t e = 0; e < t_.recv.routeExpertNumPerMoe; ++e) counts.SetValue(e, 0);
        }
        uint32_t tileRow = 0;
        for (uint32_t expert = tile.expert; expert <= tile.lastExpert; ++expert) {
            const uint32_t offset = expert == tile.expert ? tile.offset : 0;
            uint32_t total = 0;
            for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) total += Count(tp, expert);
            const uint32_t rows = total - offset < tile.rows - tileRow ? total - offset : tile.rows - tileRow;
            if (GetBlockIdx() == 0) counts.SetValue(expert, rows);
            uint32_t tpBegin = 0;
            for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) {
                const uint32_t count = Count(tp, expert);
                const uint32_t prefix = Prefix(tp, expert);
                const auto source = reinterpret_cast<GM_ADDR>(context_->localWindowsIn) + (source_ + tp) * sourceBytes_;
                GlobalTensor<uint16_t> routes;
                GlobalTensor<int8_t> tokens;
                routes.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(source + indices_));
                tokens.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(source + payload_));
                const uint32_t begin = offset > tpBegin ? offset : tpBegin;
                const uint32_t end = offset + rows < tpBegin + count
                                         ? offset + rows : tpBegin + count;
                const uint32_t first = begin < end ? Cam::MegaFirstRow(begin - offset, tileRow, GetBlockIdx()) : rows;
                if (begin < end && offset + first < end) {
                    // Fetch a contiguous route segment once, instead of synchronizing every row.
                    Fence<HardEvent::S_MTE2>();
                    DataCopyPad(index, routes[prefix + begin - tpBegin],
                                {1U, uint32_t((end - begin) * sizeof(uint16_t)), 0U, 0U, 0U},
                                {false, 0U, 0U, 0U});
                    Fence<HardEvent::MTE2_S>();
                }
                for (uint32_t r = first; begin < end && offset + r < end; r += Cam::MEGA_COMM_AIVS) {
                    const auto row = offset + r;
                    const uint32_t token = index.GetValue(row - begin);
                    Fence<HardEvent::S_MTE2>();
                    DataCopy(x, tokens[token * (t_.recv.hiddenSize + 32)], t_.recv.hiddenSize + 32);
                    Fence<HardEvent::MTE2_MTE3>();
                    DataCopy(dst[(tileRow + r) * t_.recv.hiddenSize], x, t_.recv.hiddenSize);
                    DataCopyPad(dstScale[tileRow + r], scales[t_.recv.hiddenSize / 4], {1U, 4U, 0U, 0U, 0U});
                    Fence<HardEvent::MTE3_MTE2>();
                }
                tpBegin += count;
            }
            tileRow += rows;
        }
        if (GetBlockIdx() == 0) {
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
        uint32_t tileRow = 0;
        for (uint32_t expert = tile.expert; expert <= tile.lastExpert; ++expert) {
            const uint32_t offset = expert == tile.expert ? tile.offset : 0;
            uint32_t total = 0;
            for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) total += Count(tp, expert);
            const uint32_t rows = total - offset < tile.rows - tileRow ? total - offset : tile.rows - tileRow;
            uint32_t tpBegin = 0;
            for (uint32_t tp = 0; tp < t_.recv.tpSize; ++tp) {
                const uint32_t count = Count(tp, expert);
                const uint32_t prefix = metadata_.Get<int64_t>().GetValue(5 + tp) + Prefix(tp, expert);
                GlobalTensor<bfloat16_t> dst;
                dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(Peer(source_ + tp) + returnFlags_ +
                                     Align32(t_.recv.moeRankNum * 4)));
                for (uint32_t r = (GetBlockIdx() + Cam::MEGA_COMM_AIVS - tileRow % Cam::MEGA_COMM_AIVS) % Cam::MEGA_COMM_AIVS;
                     r < rows; r += Cam::MEGA_COMM_AIVS) {
                    const auto row = offset + r;
                    if (row < tpBegin || row >= tpBegin + count) continue;
                    DataCopy(local, src[(tileRow + r) * t_.recv.hiddenSize], t_.recv.hiddenSize);
                    Fence<HardEvent::MTE2_MTE3>();
                    DataCopy(dst[(prefix + row - tpBegin) * t_.recv.hiddenSize], local, t_.recv.hiddenSize);
                    Fence<HardEvent::MTE3_MTE2>();
                }
                tpBegin += count;
            }
            tileRow += rows;
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
        if (next.rows) {
            // G2 releases the slot's inputs; its output occupies a separate region.
            flags.Wait(G2, tile.sequence);
            transport.Receive(next);
            flags.Arrive(RECEIVED, next.sequence, GetBlockIdx(), Cam::MEGA_COMM_AIVS);
            next = receive.Next();
        }
        flags.Wait(OUTPUT, tile.sequence);
        transport.Send(tile);
        flags.Arrive(SENT, tile.sequence, GetBlockIdx(), Cam::MEGA_COMM_AIVS);
        // Drain all senders before advancing or releasing the source window.
        flags.Wait(SENT, tile.sequence);
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
    GMMA8W4MidProcess<MM> mid(mm);
    mid.Init(a, &p);
    mid.Process(split, 0);
    flags.Arrive(first ? G1 : G2, tile.sequence, GetBlockIdx(), t.g1.coreNum);
}

__aicore__ inline void ComputePipeline(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                       const AfdMegaMoeFFNTilingData &t, const PipelineFlags &flags,
                                       TileCursor cursor, GM_ADDR w13, GM_ADDR w2, GM_ADDR s13,
                                       GM_ADDR s2, GM_ADDR b13, GM_ADDR b2, int64_t layer, TPipe &pipe) {
    MM::MT mm;
    auto tile = cursor.Next();
    if (!tile.rows) return;
    // One sequential context covers both GMMs; SetOrgShape/SetSingleShape select each job.
    const bool largerK = t.g1.K / t.g1.quantGroupNum >= t.g2.K / t.g2.quantGroupNum;
    TCubeTiling cube = *reinterpret_cast<const TCubeTiling *>(largerK ? t.cube1 : t.cube2);
    mm.SetSubBlockIdx(0);
    mm.Init(&cube);
    ComputeStage(scratch, w, t, flags, tile, w13, s13, b13, layer, true, mm, pipe);
    while (tile.rows) {
        auto next = cursor.Next();
        if (next.rows) ComputeStage(scratch, w, t, flags, next, w13, s13, b13, layer, true, mm, pipe);
        ComputeStage(scratch, w, t, flags, tile, w2, s2, b2, layer, false, mm, pipe);
        tile = next;
    }
    mm.End();
    pipe.Reset();
}

__aicore__ inline void PrepareTile(GM_ADDR scratch, const Cam::MegaWorkspace &w,
                                   const AfdMegaMoeFFNTilingData &t, const PipelineFlags &flags,
                                   const Tile &tile, GM_ADDR weight, GM_ADDR scale, GM_ADDR bias,
                                   int64_t layer, TPipe &pipe) {
    const auto slot = scratch + w.SlotOffset(tile.sequence);
    flags.Wait(RECEIVED, tile.sequence);
    const auto &p = t.g1;
    auto a = Addresses(slot + w.tileX, slot + w.tileScales,
                       slot + w.tileCounts, weight, scale, bias, nullptr, nullptr,
                       slot + w.compute, p, layer);
    auto split = tile.Split();
    GMMA8W4PreProcess pre;
    pre.Init(a, &p);
    pre.SetVectorGroup(GetBlockIdx() - Cam::MEGA_COMM_AIVS, t.recv.aivNum - Cam::MEGA_COMM_AIVS);
    pre.Process(split, 0, &pipe);
    pipe.Reset();
    flags.Arrive(PRE1, tile.sequence, GetBlockIdx() - Cam::MEGA_COMM_AIVS,
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
    PrepareTile(scratch, w, t, flags, tile, w13, s13, b13, layer, pipe);
    auto next = cursor.Next();
    if (next.rows) PrepareTile(scratch, w, t, flags, next, w13, s13, b13, layer, pipe);
    while (tile.rows) {
        const auto slot = scratch + w.SlotOffset(tile.sequence);
        flags.Wait(G1, tile.sequence);
        InvalidateScalarInputs(slot + w.tileCounts);
        auto a = Addresses(slot + w.tileX, slot + w.tileScales, slot + w.tileCounts, w13, s13,
                           b13, slot + w.compute2, slot + w.zScales, slot + w.compute, t.g1, layer);
        auto split = tile.Split();
        GMMA8W4PostProcess act;
        act.Init(a, &t.g1, &t.quant);
        act.SetVectorGroup(id, vectors);
        act.SetMsdOutput(slot + w.compute2);
        act.Process(split, 0, &pipe);
        Fence<HardEvent::MTE3_S>();
        pipe.Reset();
        // The rounded INT8 activation has already been packed into G2's MSD input.
        flags.Arrive(PRE2, tile.sequence, id, vectors);
        // Feed G1 two tiles ahead while Cube consumes this tile's prepared G2 input.
        auto ahead = cursor.Next();
        if (ahead.rows) PrepareTile(scratch, w, t, flags, ahead, w13, s13, b13, layer, pipe);
        flags.Wait(G2, tile.sequence);
        // Refilled inputs do not imply that the old output has been sent.
        if (tile.sequence > Cam::MEGA_RING_SLOTS)
            flags.Wait(SENT, tile.sequence - Cam::MEGA_RING_SLOTS);
        OutputTile(slot, w, t, b2, layer, tile.expert, tile.offset, tile.rows,
                   scratch + w.counts, pipe);
        flags.Arrive(OUTPUT, tile.sequence, id, vectors);
        tile = next;
        next = ahead;
    }
}
} // namespace AfdMegaMoe
#endif
