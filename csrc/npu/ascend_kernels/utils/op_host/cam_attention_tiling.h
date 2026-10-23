// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ATTENTION_TILING_H
#define AFD_CAM_ATTENTION_TILING_H
#include <cstdint>

namespace Cam {
constexpr uint32_t CAM_SMALL_BATCH_TOKENS = 32;
constexpr uint32_t CAM_SMALL_DISPATCH_AIVS = 8;
constexpr uint32_t CAM_SMALL_COMBINE_AIVS = 8;

inline uint32_t DispatchAivs(uint32_t batch, uint32_t available) {
    return batch <= CAM_SMALL_BATCH_TOKENS && available > CAM_SMALL_DISPATCH_AIVS
               ? CAM_SMALL_DISPATCH_AIVS : available;
}
inline uint32_t CombineAivs(uint32_t batch, uint32_t available) {
    return batch <= CAM_SMALL_BATCH_TOKENS && available > CAM_SMALL_COMBINE_AIVS
               ? CAM_SMALL_COMBINE_AIVS : available;
}
} // namespace Cam
#endif
