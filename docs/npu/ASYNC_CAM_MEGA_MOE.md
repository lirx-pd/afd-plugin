# Async CAM MegaMoE phase 1 (A3)

Status: experimental A3 implementation; qualification scope is recorded below.
The branch is `feat/async-cam-megamoe-phase1`, based on afd-plugin
`df8d6a0c403c3ca1b4ed9cf1687e7204908bebec`.

## Scope and entry points

`afd_mega_moe_ffn` is a single bounded `MIX_AIC_1_2` kernel:

```text
all AIV: compact-v2 DR
all cores: barrier, validate layer/counts
for each nonempty expert and each <=128-row tile:
    AIV: copy activation/scales and publish tile counts
    AIV: existing MSD INT8 -> high/low INT4 preprocessing
    AIC: GMM1 with packed NZ INT4 weights
    AIV: existing dequant/compensation/SwiGLU/full-row quantization
    AIV: existing MSD preprocessing of the quantized activation
    AIC: GMM2
    AIV: dequant/compensation -> BF16 -> routed scaling -> BF16
all AIV: compact-v2 CS
all cores: drain, write completion status
```

Attention still runs its existing dispatch, shared experts and weighted combine.
The wire layout and `cam_async_routed_only_compact_v2` identity are unchanged.
The batch header's total row count is not used as the chunk's valid length;
compute uses the sum of the selected expert counts.

This implements the phase-ordered P1 candidate. Communication/compute overlap,
per-tile return, multi-slot credits, stream_v3 and source preemption are P2/P4 work.
There is no claimed speedup relative to the existing layered FULL baseline.

The standalone DR/CS classes are reused only with **every AIV participating**,
with their original vector-only barriers inside each phase. Every AIC and AIV
then executes the same mixed barriers. P2 cannot assign a subset of AIVs to
these classes without replacing their barriers and core indexing. The mixed
operator uses batch scheduling and one launch block per Cube (two sibling AIVs),
covering all available A3 cores. This follows the
[Ascend SyncAll contract](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0204.html).

## Ownership and numerical contract

- `AsyncCAMMegaMoeExecutor` owns stable layer mapping, weights, anchor and scratch.
  The connector owns the HCCL domain and windows. Graph replay reuses the scratch
  on one stream. Shutdown retains the existing receiver-exit and device-completion
  requirements before releasing either owner.
- Whole-chunk expanded input/output remain in scratch for compatibility. Only
  GMM/activation intermediates are bounded to 128 rows. There is no duplicate
  full-model weight conversion or new weight copy.
- Input and output route order, expert-chunk continuation and last-chunk completion
  are inherited from DR/CS. Empty experts skip compute; an empty FFN still runs CS.
- W4A8 per-channel and per-group use the existing MSD preprocessing and GMM1
  epilogue. GMM2 reuses the MSD matmul and implements its output epilogue. This
  selection may differ in rounding from the layered autotiling path; compare
  numerical outputs before accepting it. `swiglu_limit` remains ignored as in
  the current layered executor. Routed scaling rounds to BF16 before multiplying.
- All mutable cross-phase control reads invalidate the relevant cache line.
  Each packed tile also invalidates scalar cache before the MSD helpers read
  its counts and token scales.
  Each Cube matmul stage resets its TPipe after `Matmul.End()` so buffer records
  do not accumulate across experts and tiles.
  Scratch and window data transfers use MTE; barriers do not substitute for
  cache maintenance.
- The output tensor is an INT64 status: `0` success, `-1` invalid layer mapping,
  `-2` invalid chunk counts. Invalid control data does **not** publish successful
  remote completion. The service must be treated as failed and cleaned up at
  process level; device-status propagation to the control plane is not yet
  qualified. The finite harness checks the status after completion.

## Supported bounds

The implementation currently requires A3, BF16 CAM anchor/output,
`dynamicQuant=1`, homogeneous layered W4A8, FFN TP=1, and no FFN two-stream.
H/I are multiples of 256 and at most 8192; UB budgeting may reject smaller
limits for a specific H/I pair. Quantization group K is a multiple of 64.
There are at most 256 local experts and at most 384 A+F ranks.

`max_seq_len` must divide Attention TP exactly. The conservative source-route
bound is `(max_seq_len / TP) * top_k <= 65535`. Receive capacity must be at least
`max_seq_len`, so a complete expert fits even under worst-case routing. Explicit
window and UB checks also run in host tiling. The legacy receive-capacity
`BATCH_SIZE_FACTOR` still applies. These are P1 bounds, not newly claimed
capabilities of the older operators.

## Build and enable

Build on the repository's prepared CANN/torch-npu A3 environment:

```bash
SOC_VERSION=910c pip install -e . -v --no-build-isolation
```

The A3 registry includes the new ACLNN operator and binding. A5 builds exclude
it. For an existing Async CAM deployment, set on the FFN workers:

```bash
export VLLM_PLUGINS=ascend,afd
export AFD_ASYNC_CAM_LAYERED_GMM=1
export AFD_ASYNC_CAM_MEGA_MOE=1
```

The default is disabled. The enable switch fails if layered W4A8 is unavailable
or two-stream is requested; it does not fall back after consuming a message.
FULL graph signatures include the backend identity. Configuration and transport
requirements in [ASYNC_FFN_RUNTIME.md](ASYNC_FFN_RUNTIME.md) also apply.

## Qualification commands

The host-only workspace check requires a C++17 compiler:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined,address \
  -I csrc/npu/ascend_kernels tests/npu/check_mega_moe_workspace.cpp \
  -o /tmp/check_mega_moe_workspace
ASAN_OPTIONS=detect_leaks=0 /tmp/check_mega_moe_workspace
pytest -q -ra tests/unit/test_envs.py tests/unit/package/test_ascend_build_files.py
```

The workspace check covers 162 H/I/capacity/expert/TP combinations, including
64-bit allocation sizes, disjoint regions, alignment and capacity-independent
compute scratch. Leak detection is disabled because this sandbox is traced;
address and undefined-behavior checks remain enabled. It does not compile an
Ascend kernel.

On three isolated A3 devices, run a finite synthetic-weight reference first,
then the new eager and FULL variants:

```bash
export VLLM_PLUGINS=ascend,afd
export SOC_VERSION=910c
export ASCEND_RT_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="$PWD/afd_plugin/_cann_ops_custom/vendors/afd-plugin${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
# Keep the same flags and saved reference for all three runs.
timeout --kill-after=20s 180s torchrun --standalone --nproc-per-node=3 \
  tests/npu/async_cam_runtime.py \
  --boundary --tile-boundary --save-outputs /tmp/cam-layered
timeout --kill-after=20s 180s torchrun --standalone --nproc-per-node=3 \
  tests/npu/async_cam_runtime.py \
  --boundary --tile-boundary --mega-moe --reference-outputs /tmp/cam-layered
timeout --kill-after=20s 180s torchrun --standalone --nproc-per-node=3 \
  tests/npu/async_cam_runtime.py \
  --boundary --tile-boundary --mega-moe --full-graph \
  --reference-outputs /tmp/cam-layered
```

Repeat all three commands with `--per-group`; use a distinct reference path.
Also repeat with `--random-weights --routed-scale=2.5` to cover dense signed
weights distinct across experts and FFN ranks, compensation and BF16 scaling order.
Add `--intermediate-size=512` to the three matching commands to exercise H != I.
Repeat with `--attn-ranks=2` and four devices/processes to exercise source
interleaving. This is two independent Attention sources, not Attention TP=2;
the real-checkpoint follow-up below separately covers Attention TP=4 in eager
mode.

The compiled binding also has Meta contract checks and an NPU workspace/export
availability check. These do not validate communication or arithmetic:

```bash
AFD_RUN_ASCEND_OP_RUNTIME=1 pytest -q tests/npu/test_async_cam_mega_moe_meta.py
```

The fixture tests non-contiguous layer IDs, repeated layers, empty experts/FFN,
129-row experts split into 128+1 compute tiles, old expert-chunk continuation,
fixed-address replay,
poisoned unused scratch and cleanup. New-path results are compared with the
saved layered output using `rtol=0.04, atol=2e-5`; unchanged layered FULL retains
its exact-match check. These are synthetic-fixture thresholds, not approval
for a real checkpoint's accuracy budget.

Before enabling real service traffic, complete invalid-layer/count fault
handling, long-running lifecycle checks, and a profiler showing one FFN mixed
kernel. Extend checkpoint accuracy coverage beyond the specific eager case
below. Then compare against layered eager and layered FULL with
identical topology, capacity, routing and model settings. Measure rather than
infer the performance benefit.

## Current evidence

Validated on 2026-10-23 with A3 / ascend910_93 and CANN 9.1.0. The isolated
source build compiled the mixed kernel, generated ACLNN API and PyTorch binding.
No framework package was replaced. MegaMoE is disabled by default. All
qualification references explicitly enabled layered W4A8; fused runs selected
the additional MegaMoE switch.

The following synthetic matrix predates the TPipe reset fix described below.
Its recorded source/binary hashes apply to that earlier build; it does not
establish FULL-mode coverage of the fixed build.

| Synthetic workload | H / I | Attention / FFN ranks | Layered eager reference | Mega eager | Mega FULL |
| --- | --- | --- | --- | --- | --- |
| Per-channel diagonal | 256 / 256 | 1 / 2 | PASS | PASS | PASS |
| Per-group diagonal | 256 / 256 | 1 / 2 | PASS | PASS | PASS |
| Per-channel dense signed, routed scale 2.5 | 256 / 256 | 1 / 2 | PASS | PASS | PASS |
| Per-group dense signed, routed scale 2.5 | 256 / 512 | 1 / 2 | PASS | PASS | PASS |
| Per-group dense signed, source switching, routed scale 2.5 | 256 / 512 | 2 / 2 | PASS | PASS | PASS |

All five cases used capacity 256, four local experts, top-k 2 and layer IDs
2/5. Every run exercised 7/16/129-token requests, sparse/empty FFN work and
expert continuation. FULL captured once, replayed with stable addresses and
poisoned unused scratch, disabled Python transactions during replay, and
completed cleanup. Two Attention sources were scheduled in sequence; this is
neither concurrent fan-in qualification nor Attention TP=2.

The unchanged layered FULL path also passed an **exact** comparison with its
eager reference on the four-rank dense per-group H=256/I=512 case. The compiled
binding's seven Meta/workspace checks passed. On WSL, the selected unit modules
had 49 passes; modules requiring unavailable torch/vLLM and the opt-in NPU
checks skipped. Python lint/compilation and the 162-case ASan/UBSan workspace
check passed. The remote AArch64 host also passed the plain workspace check;
its ASan link was unavailable because `libasan` was absent.

The four saved-output workloads (per-group diagonal, dense per-channel, dense
per-group and two-source dense per-group) had maximum absolute error 0 in both
fused modes. The declared acceptance thresholds remain `rtol=0.04, atol=2e-5`;
this observation does not define a real model accuracy budget.

[Validation evidence](ASYNC_CAM_MEGA_MOE_VALIDATION.json) records source and
binary SHA-256 values, package versions, finite-run summaries and numerical error
summaries. The source audit covers the runtime, build registration and native
fixtures. Diagnostic probes were removed before these qualification runs.

### Real-checkpoint eager follow-up

The fixed build completed 300 GSM8K samples with DeepSeek-V4-Flash DSpark W4A8:
**285/300 (95.00%)** for both strict-match and flexible-extract. The run used
one A3 host with 16 dies, Attention DP2/TP4 and FFN DP8/TP1/EP8. All eight FFN
ranks reported `mega_moe` for all 43 layers. Both roles ran eager, with DBO and
FFN two-stream disabled. Evaluation used test rows 0–299, 8-shot, greedy
generation, 512 output tokens and concurrency 12. No fresh baseline was run,
as requested; this score does not establish parity with the layered backend.

Real-model warmup exposed accumulated TPipe buffer records across matmul
stages: `AfdMegaMoeFFN` failed with 507015 at AIC PC offset `0x4058`. An eager
H=4096/I=2048, 32-local-expert finite fixture reproduced the same fault. Resetting
the AIC TPipe after each `Matmul.End()` fixed that reproduction; all seven
fixture workloads then passed, followed by the real-checkpoint test. Native
artifacts were rebuilt and hashes verified before evaluation.

The post-fix 64K launch passed model profiling but failed the Attention KV
capacity check (34.5 GiB required, 21.57 GiB available). The completed run used
`max-model-len=32768`, retaining MBT=32768 and all evaluation settings without
sample truncation. Inference completed with exit code 0 and no inference errors.
The E2E harness terminated its services afterward; FFN shutdown exceptions and
resource-tracker warnings remain in the logs. No tagged test processes remained
and test ports were free. The iTASK allocation `afd-mrv2-async` remains Running
at the user's request.

[GSM8K evidence](ASYNC_CAM_MEGA_MOE_GSM8K_300.json) records the fixed source and
binary hashes, settings, scores, retained artifact hashes and remaining limits.
This follow-up covers that checkpoint and Attention TP=4 in eager mode. The
fixed build's FULL mode, dedicated concurrent-source stress, invalid-control
recovery, long-running lifecycle, kernel profiling and performance benefit
remain unqualified.
