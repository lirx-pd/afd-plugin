// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <cmath>
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_extension/op_api_common.h"
#include "afd_mega_moe_ffn/op_api/aclnn_afd_mega_moe_ffn.h"
#include "utils/op_kernel/mega_moe_workspace.h"

namespace afd_plugin::mega_moe {
namespace {
void check_config(at::IntArrayRef c) {
  TORCH_CHECK(c.size() == Cam::MEGA_CONFIG_SIZE, "MegaMoE config must have 14 entries");
  for (auto n : c)
    TORCH_CHECK(n > 0 && n <= 262144, "MegaMoE config entries must be in [1, 262144]");
  TORCH_CHECK(
      c[Cam::MEGA_WORLD_SIZE] == c[Cam::MEGA_FFN_RANKS] + c[Cam::MEGA_ATTN_RANKS] &&
          c[Cam::MEGA_RANK_ID] >= c[Cam::MEGA_ATTN_RANKS] &&
          c[Cam::MEGA_RANK_ID] < c[Cam::MEGA_WORLD_SIZE] &&
          c[Cam::MEGA_WORLD_SIZE] <= 384,
      "MegaMoE requires an FFN rank in the A+F CAM domain");
  TORCH_CHECK(
      c[Cam::MEGA_ATTN_RANKS] % c[Cam::MEGA_ATTN_TP] == 0 &&
          c[Cam::MEGA_MAX_SEQ_LEN] % c[Cam::MEGA_ATTN_TP] == 0 &&
          c[Cam::MEGA_MAX_SEQ_LEN] / c[Cam::MEGA_ATTN_TP] * c[Cam::MEGA_TOP_K] <=
              UINT16_MAX,
      "MegaMoE TP layout exceeds compact-v2 uint16 bounds");
  TORCH_CHECK(c[Cam::MEGA_HIDDEN_SIZE] % 256 == 0 &&
                  c[Cam::MEGA_INTERMEDIATE_SIZE] % 256 == 0 &&
                  c[Cam::MEGA_HIDDEN_SIZE] <= 8192 &&
                  c[Cam::MEGA_INTERMEDIATE_SIZE] <= 8192,
              "MegaMoE H/I must be multiples of 256 and no larger than 8192");
  TORCH_CHECK(c[Cam::MEGA_HIDDEN_SIZE] % c[Cam::MEGA_W13_GROUPS] == 0 &&
                  c[Cam::MEGA_INTERMEDIATE_SIZE] % c[Cam::MEGA_W2_GROUPS] == 0 &&
                  (c[Cam::MEGA_HIDDEN_SIZE] / c[Cam::MEGA_W13_GROUPS]) % 64 == 0 &&
                  (c[Cam::MEGA_INTERMEDIATE_SIZE] / c[Cam::MEGA_W2_GROUPS]) % 64 == 0,
              "MegaMoE W4A8 quantization groups must be K-aligned to 64");
  TORCH_CHECK(c[Cam::MEGA_CAPACITY] >= c[Cam::MEGA_MAX_SEQ_LEN],
              "MegaMoE capacity must fit a complete expert (max_seq_len)");
  TORCH_CHECK(
      c[Cam::MEGA_LOCAL_EXPERTS] <= 256 &&
          c[Cam::MEGA_TOP_K] <= c[Cam::MEGA_FFN_RANKS] * c[Cam::MEGA_LOCAL_EXPERTS],
      "MegaMoE expert count/top-k is invalid");
}
void check_tensor(const at::Tensor &x, const at::Tensor &anchor, at::ScalarType dtype) {
  TORCH_CHECK(x.device() == anchor.device() && x.scalar_type() == dtype &&
                  x.is_contiguous() && !x.requires_grad(),
              "MegaMoE tensor device/dtype/layout mismatch");
}
at::Tensor prepare(const at::Tensor &anchor, at::IntArrayRef config) {
  check_config(config);
  if (anchor.is_privateuseone()) {
    TORCH_CHECK(GetOpApiFuncAddr("aclnnAfdMegaMoeFFN") != nullptr &&
                    GetOpApiFuncAddr("aclnnAfdMegaMoeFFNGetWorkspaceSize") != nullptr,
                "MegaMoE ACLNN entry points are missing; rebuild the A3 run package");
  }
  const Cam::MegaWorkspace w(config[Cam::MEGA_CAPACITY], config[Cam::MEGA_HIDDEN_SIZE],
                             config[Cam::MEGA_INTERMEDIATE_SIZE],
                             config[Cam::MEGA_LOCAL_EXPERTS],
                             config[Cam::MEGA_ATTN_TP]);
  const c10::OptionalDeviceGuard guard(anchor.is_privateuseone() ? at::device_of(anchor)
                                                                 : std::nullopt);
  return at::empty({int64_t(w.bytes)}, anchor.options().dtype(at::kByte));
}
template <bool NPU>
at::Tensor run(const at::Tensor &anchor, const at::Tensor &comm, at::TensorList w13,
               at::TensorList w2, at::TensorList s13, at::TensorList s2,
               at::TensorList b13, at::TensorList b2, const at::Tensor &map,
               at::Tensor &scratch, at::IntArrayRef c, c10::string_view group,
               double routed_scale) {
  check_config(c);
  check_tensor(anchor, anchor, at::kBFloat16);
  check_tensor(comm, anchor, at::kHalf);
  check_tensor(map, anchor, at::kLong);
  check_tensor(scratch, anchor, at::kByte);
  TORCH_CHECK(anchor.dim() == 1 && anchor.numel() == 1 && map.dim() == 1 &&
                  map.numel() > 0 && map.numel() <= 262144,
              "MegaMoE anchor/map shape mismatch");
  TORCH_CHECK(std::isfinite(routed_scale) && !group.empty(),
              "MegaMoE scale/group is invalid");
  const Cam::MegaWorkspace w(c[Cam::MEGA_CAPACITY], c[Cam::MEGA_HIDDEN_SIZE],
                             c[Cam::MEGA_INTERMEDIATE_SIZE], c[Cam::MEGA_LOCAL_EXPERTS],
                             c[Cam::MEGA_ATTN_TP]);
  TORCH_CHECK(scratch.dim() == 1 && scratch.numel() == int64_t(w.bytes),
              "MegaMoE scratch size mismatch");
  for (auto list : {w13, w2, s13, s2, b13, b2})
    TORCH_CHECK(int64_t(list.size()) == c[Cam::MEGA_LAYER_COUNT],
                "MegaMoE layer table length mismatch");
  for (int64_t l = 0; l < c[Cam::MEGA_LAYER_COUNT]; ++l) {
    check_tensor(w13[l], anchor, at::kInt);
    check_tensor(w2[l], anchor, at::kInt);
    check_tensor(s13[l], anchor, at::kLong);
    check_tensor(s2[l], anchor, at::kLong);
    check_tensor(b13[l], anchor, at::kFloat);
    check_tensor(b2[l], anchor, at::kFloat);
    TORCH_CHECK(
        w13[l].sizes() ==
                at::IntArrayRef({c[Cam::MEGA_LOCAL_EXPERTS], c[Cam::MEGA_HIDDEN_SIZE],
                                 c[Cam::MEGA_INTERMEDIATE_SIZE] / 4}) &&
            w2[l].sizes() == at::IntArrayRef({c[Cam::MEGA_LOCAL_EXPERTS],
                                              c[Cam::MEGA_INTERMEDIATE_SIZE],
                                              c[Cam::MEGA_HIDDEN_SIZE] / 8}),
        "MegaMoE packed weight shape mismatch");
    TORCH_CHECK(s13[l].numel() == c[Cam::MEGA_LOCAL_EXPERTS] * c[Cam::MEGA_W13_GROUPS] *
                                      2 * c[Cam::MEGA_INTERMEDIATE_SIZE] &&
                    s2[l].numel() == c[Cam::MEGA_LOCAL_EXPERTS] *
                                         c[Cam::MEGA_W2_GROUPS] *
                                         c[Cam::MEGA_HIDDEN_SIZE],
                "MegaMoE scale shape mismatch");
    TORCH_CHECK(
        b13[l].sizes() == at::IntArrayRef({c[Cam::MEGA_LOCAL_EXPERTS],
                                           2 * c[Cam::MEGA_INTERMEDIATE_SIZE]}) &&
            b2[l].sizes() ==
                at::IntArrayRef({c[Cam::MEGA_LOCAL_EXPERTS], c[Cam::MEGA_HIDDEN_SIZE]}),
        "MegaMoE compensation shape mismatch");
  }
  if constexpr (NPU) {
    for (auto list : {w13, w2})
      for (const auto &weight : list)
        TORCH_CHECK(vllm_ascend::NPUBridge::GetNpuStorageImplDesc(weight).npu_format_ ==
                        ACL_FORMAT_FRACTAL_NZ,
                    "MegaMoE requires the existing NZ-packed W4A8 weights");
  }
  const c10::OptionalDeviceGuard guard(anchor.is_privateuseone() ? at::device_of(anchor)
                                                                 : std::nullopt);
  auto output = at::empty({1}, anchor.options().dtype(at::kLong));
  if constexpr (NPU) {
    // IntArrayRef selects the ACL int-array converter; std::vector does not.
    std::string group_name(group.data(), group.size());
    char *group_ptr = const_cast<char *>(group_name.c_str());
    EXEC_NPU_CMD(aclnnAfdMegaMoeFFN, anchor, comm, map, scratch, w13, w2, s13, s2, b13,
                 b2, c, group_ptr, routed_scale, output);
  }
  return output;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(afd_ascend, m) {
  m.def("afd_mega_moe_ffn_workspace(Tensor anchor, int[] config) -> Tensor");
  m.def(
      "afd_mega_moe_ffn(Tensor anchor, Tensor comm_args, Tensor[] w13, Tensor[] w2, "
      "Tensor[] s13, Tensor[] s2, Tensor[] b13, Tensor[] b2, Tensor layer_map, "
      "Tensor(a!) scratch, int[] config, str group_name, float routed_scale=1.0) -> "
      "Tensor");
}
TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, m) {
  m.impl("afd_mega_moe_ffn_workspace", &prepare);
  m.impl("afd_mega_moe_ffn", &run<true>);
}
TORCH_LIBRARY_IMPL(afd_ascend, Meta, m) {
  m.impl("afd_mega_moe_ffn_workspace", &prepare);
  m.impl("afd_mega_moe_ffn", &run<false>);
}
}  // namespace afd_plugin::mega_moe
