// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// Host-only bounds check; does not qualify Ascend compilation or device execution.
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "utils/op_kernel/mega_moe_workspace.h"
int main() {
    for (uint64_t h : {256, 7168, 8192})
        for (uint64_t i : {256, 2048, 6144})
            for (uint64_t capacity : {16, 8192, 262144})
                for (uint64_t e : {1, 4, 256})
                    for (uint64_t tp : {1, 8}) {
                        const Cam::MegaWorkspace w(capacity, h, i, e, tp);
                        assert(w.batch == 0);
                        assert(w.counts >= (5 + tp + e * tp) * 8);
                        assert(w.flags >= w.counts + e * 8);
                        assert(w.tileX >= w.flags + Cam::MEGA_PIPELINE_STAGES *
                                                       (Cam::MEGA_MAX_AIVS + 1) * Cam::MEGA_ALIGNMENT);
                        assert(w.tileScales >= w.tileX + Cam::MEGA_TILE_ROWS * h);
                        assert(w.tileCounts >= w.tileScales + Cam::MEGA_TILE_ROWS * 4);
                        assert(w.z >= w.tileCounts + e * 8);
                        assert(w.zScales >= w.z + Cam::MEGA_TILE_ROWS * i);
                        assert(w.compute >= w.zScales + Cam::MEGA_TILE_ROWS * 4);
                        assert(w.compute2 >= w.compute + 2 * Cam::MEGA_TILE_ROWS * h +
                                                        8 * Cam::MEGA_TILE_ROWS * 2 * i);
                        assert(w.y >= w.compute2 + 2 * Cam::MEGA_TILE_ROWS * i +
                                                  8 * Cam::MEGA_TILE_ROWS * h);
                        assert(w.tileX + w.slotBytes >= w.y + Cam::MEGA_TILE_ROWS * h * 2);
                        assert(w.bytes == w.tileX + Cam::MEGA_RING_SLOTS * w.slotBytes);
                        for (uint64_t offset :
                             {w.batch, w.counts, w.flags, w.tileX, w.tileScales, w.tileCounts,
                              w.z, w.zScales, w.compute, w.compute2, w.y, w.slotBytes, w.bytes})
                            assert(offset % Cam::MEGA_ALIGNMENT == 0);
                        for (uint32_t sequence = 1; sequence < 100; ++sequence) {
                            assert(w.SlotOffset(sequence) + w.y + Cam::MEGA_TILE_ROWS * h * 2 <= w.bytes);
                            assert(w.SlotOffset(sequence) != w.SlotOffset(sequence + 1));
                            assert(w.SlotOffset(sequence) == w.SlotOffset(sequence + Cam::MEGA_RING_SLOTS));
                        }
                        const Cam::MegaWorkspace larger(capacity * 2, h, i, e, tp);
                        assert(larger.bytes == w.bytes);
                    }
}
