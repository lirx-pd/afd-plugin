// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <cmath>
#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/hccl/hccl_tiling.h"
#include "ops_log.h"
#include "../op_kernel/afd_mega_moe_ffn_tiling.h"

namespace optiling {
namespace {
// config: maxSeqLen,H,I,topK,F,A,E,rank,world,TP,capacity,G1groups,G2groups,layers.
constexpr int64_t MEGA_GMM_TASK_N = 512;
constexpr uint64_t SYSTEM_WORKSPACE = 16 * 1024 * 1024;
uint64_t Align32(uint64_t n) { return (n + 31) / 32 * 32; }

bool CubeTiling(platform_ascendc::PlatformAscendC &platform, int64_t k, int64_t n, int64_t groups, int64_t taskN,
                uint8_t *dest) {
    matmul_tiling::MatmulApiTiling mm(platform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_INT4);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ,
                matmul_tiling::DataType::DT_INT4);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetShape(128, taskN, k / groups);
    mm.SetOrgShape(2 * Cam::MEGA_TILE_ROWS, n, k);
    mm.SetFixSplit(128, std::min<int64_t>(256, taskN), std::min<int64_t>(256, k / groups));
    mm.SetBufferSpace(-1, -1, -1);
    TCubeTiling data;
    if (mm.GetTiling(data) == -1 || data.GetDataSize() > Cam::MEGA_CUBE_TILING_BYTES)
        return false;
    data.SaveToBuffer(dest, Cam::MEGA_CUBE_TILING_BYTES);
    return true;
}

ge::graphStatus Tiling(gert::TilingContext *ctx) {
    const auto attrs = ctx->GetAttrs();
    const auto config = attrs->GetListInt(0);
    if (config == nullptr || config->GetSize() != Cam::MEGA_CONFIG_SIZE)
        return ge::GRAPH_FAILED;
    const auto c = config->GetData();
    for (size_t j = 0; j < Cam::MEGA_CONFIG_SIZE; ++j)
        if (c[j] <= 0 || c[j] > 262144)
            return ge::GRAPH_FAILED;
    if (c[Cam::MEGA_WORLD_SIZE] != c[Cam::MEGA_FFN_RANKS] + c[Cam::MEGA_ATTN_RANKS] ||
        c[Cam::MEGA_RANK_ID] < c[Cam::MEGA_ATTN_RANKS] ||
        c[Cam::MEGA_RANK_ID] >= c[Cam::MEGA_WORLD_SIZE] ||
        c[Cam::MEGA_ATTN_RANKS] % c[Cam::MEGA_ATTN_TP] ||
        c[Cam::MEGA_MAX_SEQ_LEN] % c[Cam::MEGA_ATTN_TP] ||
        c[Cam::MEGA_MAX_SEQ_LEN] / c[Cam::MEGA_ATTN_TP] * c[Cam::MEGA_TOP_K] > UINT16_MAX ||
        c[Cam::MEGA_MAX_SEQ_LEN] > 262144 || c[Cam::MEGA_HIDDEN_SIZE] % 256 ||
        c[Cam::MEGA_INTERMEDIATE_SIZE] % 256 || c[Cam::MEGA_HIDDEN_SIZE] > 8192 ||
        c[Cam::MEGA_INTERMEDIATE_SIZE] > 8192 ||
        c[Cam::MEGA_HIDDEN_SIZE] % c[Cam::MEGA_W13_GROUPS] ||
        c[Cam::MEGA_INTERMEDIATE_SIZE] % c[Cam::MEGA_W2_GROUPS] ||
        (c[Cam::MEGA_HIDDEN_SIZE] / c[Cam::MEGA_W13_GROUPS]) % 64 ||
        (c[Cam::MEGA_INTERMEDIATE_SIZE] / c[Cam::MEGA_W2_GROUPS]) % 64 ||
        c[Cam::MEGA_TOP_K] > c[Cam::MEGA_FFN_RANKS] * c[Cam::MEGA_LOCAL_EXPERTS] ||
        c[Cam::MEGA_WORLD_SIZE] > 384 || c[Cam::MEGA_LOCAL_EXPERTS] > 256 ||
        c[Cam::MEGA_CAPACITY] < c[Cam::MEGA_MAX_SEQ_LEN] || c[Cam::MEGA_CAPACITY] > 262144 ||
        !std::isfinite(*attrs->GetFloat(2)))
        return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const auto cubes = platform.GetCoreNumAic();
    const auto vectors = platform.GetCoreNumAiv();
    if (cubes == 0 || vectors != 2 * cubes || vectors > 48 || vectors <= Cam::MEGA_COMM_AIVS)
        return ge::GRAPH_FAILED;
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    // Existing MSD postprocess holds whole gate/up rows in UB.
    if (29ULL * c[Cam::MEGA_INTERMEDIATE_SIZE] + 1024 > ub ||
        7ULL * std::max(c[Cam::MEGA_HIDDEN_SIZE], c[Cam::MEGA_INTERMEDIATE_SIZE]) + 256 > ub)
        return ge::GRAPH_FAILED;
    const uint64_t meta =
        Align32((5 + c[Cam::MEGA_ATTN_TP] + c[Cam::MEGA_LOCAL_EXPERTS] * c[Cam::MEGA_ATTN_TP]) * 8);
    const uint64_t drFixed = Align32(5 * 8) + meta + Align32(c[Cam::MEGA_LOCAL_EXPERTS] * 8) +
                             2 * Align32(c[Cam::MEGA_LOCAL_EXPERTS] * 4) +
                             Align32(c[Cam::MEGA_HIDDEN_SIZE] + 32) +
                             Align32(c[Cam::MEGA_LOCAL_EXPERTS] * 8);
    const uint64_t csFixed = meta + Align32(c[Cam::MEGA_LOCAL_EXPERTS] * c[Cam::MEGA_ATTN_TP] * 4) +
                             Align32(c[Cam::MEGA_ATTN_TP] * 4);
    const uint64_t transport = meta + Align32(c[Cam::MEGA_HIDDEN_SIZE] * 2 + 32 + Cam::MEGA_ROUTE_INDEX_BYTES +
                                               c[Cam::MEGA_LOCAL_EXPERTS] * 8);
    const uint64_t routeIndices = Align32(
        ((c[Cam::MEGA_MAX_SEQ_LEN] / c[Cam::MEGA_ATTN_TP] * c[Cam::MEGA_TOP_K] +
          vectors - 1) / vectors) * 2);
    if (transport > ub || drFixed + routeIndices > ub ||
        csFixed + std::max<uint64_t>(c[Cam::MEGA_HIDDEN_SIZE] * 2, c[Cam::MEGA_FFN_RANKS] * 4) > ub)
        return ge::GRAPH_FAILED;
    uint64_t window = GetMaxWindowSize();
    const auto perSource =
        Align32(5 * 8) + 2 * Align32(c[Cam::MEGA_LOCAL_EXPERTS] * 4) +
        Align32((c[Cam::MEGA_HIDDEN_SIZE] + 32) *
                (c[Cam::MEGA_MAX_SEQ_LEN] / c[Cam::MEGA_ATTN_TP])) +
        Align32(c[Cam::MEGA_MAX_SEQ_LEN] / c[Cam::MEGA_ATTN_TP] * c[Cam::MEGA_TOP_K] * 2);
    const auto dispatchStats = Align32(
        4 * (c[Cam::MEGA_FFN_RANKS] + c[Cam::MEGA_FFN_RANKS] * c[Cam::MEGA_LOCAL_EXPERTS]) * 48);
    const auto returnBytes = dispatchStats + Align32(4 * c[Cam::MEGA_FFN_RANKS]) +
                             c[Cam::MEGA_MAX_SEQ_LEN] / c[Cam::MEGA_ATTN_TP] * c[Cam::MEGA_TOP_K] *
                                 c[Cam::MEGA_HIDDEN_SIZE] * 2;
    if (perSource * c[Cam::MEGA_ATTN_RANKS] + 32 > window || returnBytes > window)
        return ge::GRAPH_FAILED;
    auto data = ctx->GetTilingData<Cam::AfdMegaMoeFFNTilingData>();
    std::memset(data, 0, sizeof(*data));
    AscendC::Mc2CcTilingConfig comm(attrs->GetAttrPointer<char>(1), 8U,
                                    "AlltoAll=level0:fullmesh;level1:pairwise");
    comm.GetTiling(data->mc2InitTiling);
    comm.GetTiling(data->mc2CcTiling1);
    data->recv = {0,
                  uint32_t(c[Cam::MEGA_MAX_SEQ_LEN]),
                  uint32_t(c[Cam::MEGA_HIDDEN_SIZE]),
                  uint32_t(c[Cam::MEGA_TOP_K]),
                  uint32_t(c[Cam::MEGA_FFN_RANKS]),
                  uint32_t(c[Cam::MEGA_ATTN_RANKS]),
                  uint32_t(c[Cam::MEGA_LOCAL_EXPERTS]),
                  uint32_t(c[Cam::MEGA_RANK_ID]),
                  uint32_t(c[Cam::MEGA_WORLD_SIZE]),
                  uint32_t(c[Cam::MEGA_ATTN_TP]),
                  uint32_t(vectors),
                  ub,
                  window,
                  uint64_t(c[Cam::MEGA_CAPACITY]),
                  1};
    data->send = {0,
                  uint32_t(c[Cam::MEGA_MAX_SEQ_LEN]),
                  uint32_t(c[Cam::MEGA_HIDDEN_SIZE]),
                  uint32_t(c[Cam::MEGA_TOP_K]),
                  uint32_t(c[Cam::MEGA_FFN_RANKS]),
                  uint32_t(c[Cam::MEGA_ATTN_RANKS]),
                  uint32_t(c[Cam::MEGA_LOCAL_EXPERTS]),
                  uint32_t(c[Cam::MEGA_RANK_ID]),
                  uint32_t(c[Cam::MEGA_WORLD_SIZE]),
                  uint32_t(c[Cam::MEGA_ATTN_TP]),
                  uint32_t(vectors),
                  ub,
                  window};
    const auto taskN = std::min({MEGA_GMM_TASK_N, c[Cam::MEGA_HIDDEN_SIZE],
                                2 * c[Cam::MEGA_INTERMEDIATE_SIZE]});
    auto fill = [&](Cam::MegaGmmParams &p, int64_t k, int64_t n, int64_t groups) {
        p = {c[Cam::MEGA_LOCAL_EXPERTS],
             int64_t(cubes),
             k,
             n,
             Cam::MEGA_TILE_ROWS,
             128,
             taskN,
             Cam::MEGA_TILE_ROWS,
             2 * Cam::MEGA_TILE_ROWS * k,
             8 * Cam::MEGA_TILE_ROWS * n,
             groups,
             1,
             1,
             0,
             0,
             uint32_t(c[Cam::MEGA_LAYER_COUNT])};
    };
    fill(data->g1, c[Cam::MEGA_HIDDEN_SIZE], 2 * c[Cam::MEGA_INTERMEDIATE_SIZE],
         c[Cam::MEGA_W13_GROUPS]);
    fill(data->g2, c[Cam::MEGA_INTERMEDIATE_SIZE], c[Cam::MEGA_HIDDEN_SIZE],
         c[Cam::MEGA_W2_GROUPS]);
    data->quant = {1, c[Cam::MEGA_LOCAL_EXPERTS], 2 * c[Cam::MEGA_INTERMEDIATE_SIZE]};
    data->capacity = c[Cam::MEGA_CAPACITY];
    data->intermediate = c[Cam::MEGA_INTERMEDIATE_SIZE];
    data->layerMapSize = ctx->GetInputShape(2)->GetStorageShape().GetDim(0);
    data->routedScale = *attrs->GetFloat(2);
    const Cam::MegaWorkspace layout(c[Cam::MEGA_CAPACITY], c[Cam::MEGA_HIDDEN_SIZE],
                                    c[Cam::MEGA_INTERMEDIATE_SIZE], c[Cam::MEGA_LOCAL_EXPERTS],
                                    c[Cam::MEGA_ATTN_TP]);
    if (ctx->GetInputShape(3)->GetStorageShape().GetDim(0) != int64_t(layout.bytes) ||
        data->layerMapSize == 0 ||
        !CubeTiling(platform, c[Cam::MEGA_HIDDEN_SIZE], 2 * c[Cam::MEGA_INTERMEDIATE_SIZE],
                    c[Cam::MEGA_W13_GROUPS], taskN, data->cube1) ||
        !CubeTiling(platform, c[Cam::MEGA_INTERMEDIATE_SIZE], c[Cam::MEGA_HIDDEN_SIZE],
                    c[Cam::MEGA_W2_GROUPS], taskN, data->cube2))
        return ge::GRAPH_FAILED;
    // MIX 1:2 launches one block per Cube, with both sibling AIVs participating.
    ctx->SetBlockDim(cubes);
    ctx->SetTilingKey(1);
    ctx->SetScheduleMode(1); // SyncAll requires the complete mixed core group.
    ctx->GetWorkspaceSizes(1)[0] = SYSTEM_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}
} // namespace
} // namespace optiling

namespace ops {
class AfdMegaMoeFFN : public OpDef {
  public:
    explicit AfdMegaMoeFFN(const char *name) : OpDef(name) {
        Input("anchor").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("commArgs").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("layerMap").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        Input("scratch").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        for (const char *w : {"w13", "w2"})
            Input(w).ParamType(DYNAMIC).DataType({ge::DT_INT32}).Format({ge::FORMAT_FRACTAL_NZ});
        for (const char *s : {"s13", "s2"})
            Input(s).ParamType(DYNAMIC).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        for (const char *b : {"b13", "b2"})
            Input(b).ParamType(DYNAMIC).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        Output("completion").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        Attr("config").ListInt();
        Attr("hccl_group_name").String();
        Attr("routed_scale").Float();
        SetInferShape([](gert::InferShapeContext *ctx) {
            auto shape = ctx->GetOutputShape(0);
            shape->SetDimNum(1);
            shape->SetDim(0, 1);
            return ge::GRAPH_SUCCESS;
        });
        SetInferDataType([](gert::InferDataTypeContext *ctx) {
            ctx->SetOutputDataType(0, ge::DT_INT64);
            return ge::GRAPH_SUCCESS;
        });
        AICore().AddConfig("ascend910_93");
        AICore().SetTiling(optiling::Tiling);
        MC2().HcclGroup({"hccl_group_name"});
    }
};
OP_ADD(AfdMegaMoeFFN);
} // namespace ops
