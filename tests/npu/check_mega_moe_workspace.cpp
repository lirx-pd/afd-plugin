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
                        assert(w.scales >= w.x + capacity * h);
                        assert(w.batch >= w.scales + capacity * 4);
                        assert(w.counts >= w.batch + (5 + tp + e * tp) * 8);
                        assert(w.y >= w.counts + e * 8);
                        assert(w.tileX >= w.y + capacity * h * 2);
                        assert(w.tileScales >= w.tileX + Cam::MEGA_TILE_ROWS * h);
                        assert(w.tileCounts >= w.tileScales + Cam::MEGA_TILE_ROWS * 4);
                        assert(w.z >= w.tileCounts + e * 8);
                        assert(w.zScales >= w.z + Cam::MEGA_TILE_ROWS * i);
                        assert(w.compute >= w.zScales + Cam::MEGA_TILE_ROWS * 4);
                        assert(w.bytes >= w.compute + 2 * Cam::MEGA_TILE_ROWS * h +
                                              8 * Cam::MEGA_TILE_ROWS * 2 * i);
                        assert(w.bytes >= w.compute + 2 * Cam::MEGA_TILE_ROWS * i +
                                              8 * Cam::MEGA_TILE_ROWS * h);
                        for (uint64_t offset :
                             {w.x, w.scales, w.batch, w.counts, w.y, w.tileX, w.tileScales,
                              w.tileCounts, w.z, w.zScales, w.compute, w.bytes})
                            assert(offset % Cam::MEGA_ALIGNMENT == 0);
                        const Cam::MegaWorkspace larger(capacity * 2, h, i, e, tp);
                        // Increasing capacity must not enlarge compute scratch.
                        assert(larger.bytes - larger.tileX == w.bytes - w.tileX);
                    }
}
