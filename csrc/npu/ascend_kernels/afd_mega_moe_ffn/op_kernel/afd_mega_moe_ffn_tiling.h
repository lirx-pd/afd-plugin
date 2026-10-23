// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_FFN_TILING_H
#define AFD_MEGA_MOE_FFN_TILING_H
#include <cstdint>
#include "afd_async_dispatch_recv_tiling.h"
#include "afd_async_combine_send_tiling.h"
#include "mega_moe_workspace.h"

namespace Cam {
// Same fields consumed by the existing A8W4 MSD helpers, without a second tiling ABI.
struct MegaGmmParams {
    int64_t groupNum, coreNum, K, N, M, baseM, baseN, mLimit;
    int64_t workSpaceOffset1, workSpaceOffset2, quantGroupNum, isSingleTensor;
    int64_t groupListType, smoothScaleDimNum, singleN;
    uint32_t layerNum;
};
struct MegaQuantParams {
    int64_t maxProcessRowNum, groupListLen, tokenLen;
};
struct AfdMegaMoeFFNTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    AfdAsyncDispatchRecvInfo recv;
    AfdAsyncCombineSendInfo send;
    MegaGmmParams g1, g2;
    MegaQuantParams quant;
    uint64_t capacity;
    uint32_t intermediate, layerMapSize;
    float routedScale;
    alignas(8) uint8_t cube1[MEGA_CUBE_TILING_BYTES];
    alignas(8) uint8_t cube2[MEGA_CUBE_TILING_BYTES];
};
} // namespace Cam
#endif
