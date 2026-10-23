// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_WORKSPACE_H
#define AFD_MEGA_MOE_WORKSPACE_H
#include <cstdint>
#if defined(__CCE_AICORE__)
#define AFD_MEGA_HOST_DEVICE __aicore__
#else
#define AFD_MEGA_HOST_DEVICE
#endif
namespace Cam {
enum MegaConfigIndex : uint32_t {
    MEGA_MAX_SEQ_LEN = 0,
    MEGA_HIDDEN_SIZE = 1,
    MEGA_INTERMEDIATE_SIZE = 2,
    MEGA_TOP_K = 3,
    MEGA_FFN_RANKS = 4,
    MEGA_ATTN_RANKS = 5,
    MEGA_LOCAL_EXPERTS = 6,
    MEGA_RANK_ID = 7,
    MEGA_WORLD_SIZE = 8,
    MEGA_ATTN_TP = 9,
    MEGA_CAPACITY = 10,
    MEGA_W13_GROUPS = 11,
    MEGA_W2_GROUPS = 12,
    MEGA_LAYER_COUNT = 13,
    MEGA_CONFIG_SIZE = 14,
};
constexpr uint64_t MEGA_ALIGNMENT = 512;
constexpr uint32_t MEGA_TILE_ROWS = 128;
constexpr uint32_t MEGA_CUBE_TILING_BYTES = 1024;
AFD_MEGA_HOST_DEVICE constexpr uint64_t MegaAlign(uint64_t n) {
    return (n + MEGA_ALIGNMENT - 1) / MEGA_ALIGNMENT * MEGA_ALIGNMENT;
}

constexpr uint32_t MEGA_RING_SLOTS = 2;
constexpr uint32_t MEGA_COMM_AIVS = 8;
constexpr uint32_t MEGA_MAX_AIVS = 48;
constexpr uint32_t MEGA_PIPELINE_STAGES = 7;

// One cache line per writer; no payload-sized allocation depends on chunk capacity.
struct MegaWorkspace {
    uint64_t batch, counts, flags, tileX, tileScales, tileCounts, z, zScales;
    uint64_t compute, compute2, y, slotBytes, bytes;
    AFD_MEGA_HOST_DEVICE MegaWorkspace(uint64_t /*capacity*/, uint64_t h, uint64_t i, uint64_t e,
                                       uint64_t tp) {
        batch = 0;
        counts = MegaAlign((5 + tp + e * tp) * sizeof(int64_t));
        flags = counts + MegaAlign(e * sizeof(int64_t));
        tileX = flags + MEGA_PIPELINE_STAGES * (MEGA_MAX_AIVS + 1) * MEGA_ALIGNMENT;
        tileScales = tileX + MegaAlign(MEGA_TILE_ROWS * h);
        tileCounts = tileScales + MegaAlign(MEGA_TILE_ROWS * sizeof(float));
        z = tileCounts + MegaAlign(e * sizeof(int64_t));
        zScales = z + MegaAlign(MEGA_TILE_ROWS * i);
        compute = zScales + MegaAlign(MEGA_TILE_ROWS * sizeof(float));
        compute2 = compute + MegaAlign(2 * MEGA_TILE_ROWS * h + 8 * MEGA_TILE_ROWS * 2 * i);
        y = compute2 + MegaAlign(2 * MEGA_TILE_ROWS * i + 8 * MEGA_TILE_ROWS * h);
        slotBytes = y + MegaAlign(MEGA_TILE_ROWS * h * sizeof(uint16_t)) - tileX;
        bytes = tileX + MEGA_RING_SLOTS * slotBytes;
    }
    AFD_MEGA_HOST_DEVICE uint64_t SlotOffset(uint32_t sequence) const {
        return ((sequence - 1) % MEGA_RING_SLOTS) * slotBytes;
    }
};

} // namespace Cam
#undef AFD_MEGA_HOST_DEVICE
#endif
