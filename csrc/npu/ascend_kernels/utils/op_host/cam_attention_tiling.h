// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ATTENTION_TILING_H
#define AFD_CAM_ATTENTION_TILING_H
#include <cstdint>

namespace Cam {
constexpr uint32_t CAM_SMALL_BATCH_TOKENS = 32;
constexpr uint32_t CAM_SMALL_ATTENTION_AIVS = 8;

// Combine reads dispatch statistics using the same AIV row count.
inline uint32_t AttentionAivs(uint32_t batch, uint32_t available) {
    return batch <= CAM_SMALL_BATCH_TOKENS && available > CAM_SMALL_ATTENTION_AIVS
               ? CAM_SMALL_ATTENTION_AIVS : available;
}
} // namespace Cam
#endif
