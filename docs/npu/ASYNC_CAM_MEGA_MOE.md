# Async CAM MegaMoE (A3)

Status: experimental A3 implementation. A runnable Phase 2 version is available on
`feat/async-cam-megamoe-phase2`, based on Phase 1 commit `22334e2`.
The current functional upgrade fixes small-batch DS/CR at 8 AIVs, batches four
MegaMoE transactions per FULL graph, and caches immutable chunk metadata.
It remains disabled by default. The
[A3 tile pipeline redesign](ASYNC_CAM_MEGA_MOE_TILE_PIPELINE.md) uses three slots,
separate input/output release conditions, two-tile PRE1 lookahead and
16 communication / 32 compute AIVs. The current
[expert-packed tiles](ASYNC_CAM_MEGA_MOE_EXPERT_PACKING.md) combine adjacent
experts in 512-row tiles to reduce repeated initialization and synchronization.
The historical Phase 2 measurements below describe the pre-upgrade implementation.

## Scope and entry points

`afd_mega_moe_ffn` is a single bounded `MIX_AIC_1_2` kernel:

```text
AIV 0: select compact-v2 source/expert chunk and publish metadata
all cores: validate layer/counts, initialize per-invocation sequence flags
16 communication AIVs: receive three tiles, then refill after G2 / send completed output
remaining AIVs: INT8->INT4 prep, SwiGLU/full-row quantization, BF16 output
all AICs: G1(0), G1(1), G2(0), G1(2), G2(1), ..., G2(last)
communication AIV 0: release consumed input and publish final compact-v2 completion
all cores: drain, write completion status
```

Three 512-row slots hold independent G1 and G2 intermediates. A tile can span
adjacent experts and multiple Attention TP sources. Its expert range and partial
first-expert offset retain the original route order. The producer publishes
its sequence only after every participating core has drained its writes.
Input staging is refilled after G2; BF16 output is reused only after all senders
have consumed it. These regions have independent release conditions.
G1/G2 share the AIC pool; no second mixed kernel competes for AIV resources.
On a full A3 die the allocation is 24 AIC, 16 communication AIV and 32 compute AIV.
This is the fixed split used for the current functional qualification.

Attention still runs its existing dispatch, shared experts and weighted combine.
The wire layout and `cam_async_routed_only_compact_v2` identity are unchanged.
Data returns per tile, but Attention observes completion only after the final
expert chunk; no partial Attention consumption is introduced.
The selected source stays fixed for one invocation. Stream-v3 credits,
independent-source scheduling and source preemption remain outside Phase 2.

The standalone DR runs metadata-only on AIV 0. Its payload path and vector-only
barriers remain unchanged for existing callers. The ring uses explicit group
completion flags instead of placing subset participants in `SyncAll`.
Only entry and exit use mixed global barriers. The mixed kernel uses batch
scheduling with one block per AIC, including its two sibling AIVs.

## Ownership and numerical contract

- `AsyncCAMMegaMoeExecutor` owns stable layer mapping, weights, anchor, scratch
  and completion status.
  The connector owns the HCCL domain and windows. Graph replay reuses the scratch
  on one stream. Shutdown retains the existing receiver-exit and device-completion
  requirements before releasing either owner.
- Scratch contains metadata, per-core completion flags and three 512-row slots.
  Its size is independent of receive capacity. G1 and G2 have separate scratch
  in each slot. There is no duplicate full-model weight conversion or new copy.
- Input and output preserve compact-v2 expert/TP route order, expert-chunk
  continuation and final completion. Empty experts skip compute; an empty FFN
  still releases input and publishes completion.
- W4A8 per-channel and per-group use the existing MSD preprocessing and GMM1
  epilogue. GMM2 reuses the MSD matmul and implements its output epilogue. This
  selection may differ in rounding from the layered autotiling path; compare
  numerical outputs before accepting it. `swiglu_limit` remains ignored as in
  the current layered executor. Routed scaling rounds to BF16 before multiplying.
- All mutable cross-phase control reads invalidate the relevant cache line.
  Each tile stage invalidates scalar cache before the MSD helpers read its
  counts and token scales. Each flag writer has a separate aligned cache line;
  scalar control writes explicitly flush it, and readers invalidate before polling.
  AIV publication waits for MTE3; AIC publication waits for FIX. Replay clears
  flags at entry after the previous invocation has drained on the same stream.
  Each Cube matmul stage resets its TPipe after `Matmul.End()` so buffer records
  do not accumulate across experts and tiles.
  Scratch and window data transfers use MTE; barriers do not substitute for
  cache maintenance.
- The output tensor is an INT64 status: `0` success, `-1` invalid layer mapping,
  `-2` invalid chunk counts. Invalid control data does **not** publish successful
  remote completion. The service must be treated as failed and cleaned up at
  process level. FULL checks the first error after its bounded batch completes,
  then blocks replay and resource release. Cross-role cancellation and recovery
  are not provided; an invocation waiting for a failed peer can still block.

## Fixed functional upgrade

Attention DS and CR use at most 8 AIVs for 1–32 local tokens; larger batches
retain the platform's full AIV count. The CR tiling records the DS statistic
row count separately from its own execution count. It uses dispatch prefixes
only when both operators partition tokens identically. The reserved 48-row
window region and compact-v2 wire layout remain unchanged. This rule applies
to existing Async CAM callers as well as MegaMoE.

FULL captures a single-transaction graph and a graph containing four serial
transactions. Submission of N chunks uses N // 4 batched replays and N % 4
single replays, without rounding N upward. A chunk is selected by compact-v2
at runtime; graph position does not identify a fixed model layer. Both graphs
share the stream, pool and scratch. The graph signature includes the batch size.

The new `afd_mega_moe_ffn.out` overload reuses an initialized INT64 completion
tensor. A negative status is sticky across subsequent transactions. FULL zeros
it once before submission and reads it after the existing batch fence; it does
not clear it inside either captured graph. The original overload allocates a
zero-initialized status for each independent call. Callers of `.out` must
initialize their status before starting a new independent batch.

Each communication core loads the current chunk's immutable metadata into UB
once and computes per-TP expert prefixes there. Receive/send reuse these values;
the tile cursor reads each expert's aggregate count once. Metadata and payload
buffers remain live until the communication pipeline finishes. Host tiling
checks their combined UB footprint. Dynamic stage flags, slot visibility,
global scalar-cache invalidation and MTE3/FIX fences retain their existing rules.

Device-side multi-transaction polling (P3) is deferred. Each native invocation
still consumes one compact-v2 chunk. Qualification of the fixed version focuses
on function and accuracy. The subsequent profiling report qualifies a scoped
control-path optimization; the optimal core count remains unestablished.

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
`BATCH_SIZE_FACTOR` still applies. These retain the P1 bounds, not newly claimed
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
129-row experts, old expert-chunk continuation and fixed-address replay,
poisoned unused scratch and cleanup. New-path results are compared with the
saved layered output using `rtol=0.04, atol=2e-5`; unchanged layered FULL retains
its exact-match check. These are synthetic-fixture thresholds, not approval
for a real checkpoint's accuracy budget.

Before enabling real service traffic, complete invalid-layer/count fault
handling, long-running lifecycle checks, and a profiler showing one FFN mixed
kernel. Extend checkpoint accuracy coverage beyond the tested checkpoint and
topology below. Then compare against layered eager and layered FULL with
identical topology, capacity, routing and model settings. Measure rather than
infer the performance benefit.

## CANN 9.1 const fix and fresh-kernel qualification

`TileTransport::Count()` and `Prefix()` are non-const because CANN 9.1
`TBuf::Get()` has no const overload. Receive/send callers are already non-const;
this fix changes no routing or numerical operations.

On 2026-10-23, all existing MegaMoE kernel `.o` and `.json` artifacts were removed
before rebuilding. The build checked compiler logs as well as exit codes,
required a newly written kernel with a different hash, and verified the installed
kernel against it. The new SHA-256 is
`8b4b0ee93d70513e3a32ca4dae655bda24981bf25ad20a5f8be5aa061f3f1146`.
Kernel compilation, packaging and installation passed on CANN 9.1.0 / A3.

The new kernel passed 25 interface/graph tests, FP16/BF16 communication with
quantization off/on, the TP2 32/33-token boundary, and eager/FULL small-capacity
and H4096/I2048 finite comparisons against saved layered outputs.

The same 300-document GSM8K evaluation completed with **283/300 (94.33%)** for
strict-match and flexible-extract. Attention remains DP2/TP4 eager, FFN is
DP8/TP1/EP8 FULL, with 8-shot greedy evaluation, 512 output tokens, concurrency
12 and MBT/max-model-len 32768. All eight FFN ranks captured two graphs and
recorded at least 176,128 completed logical transactions each. Source and native
binary hashes matched before and after evaluation; inference errors were zero.

All document, prompt and target hashes match the historical Phase 2 run, as do
model metadata, dataset and package versions. Against its 286/300 score, two
answers improved and five worsened; 165 generated texts differ. No fresh baseline
was run. The observed score is 1 percentage point lower; accuracy parity and the
cause of the difference are not established. The invalidated 287/300 upgrade
run below is not a reference for the new kernel.

The model test and complete qualification both exited 0. Test processes and
ports were released; both iTASK allocations remain Running. Forced service
shutdown still logs existing exceptions and resource-tracker warnings.
Changed-file pre-commit passes. Performance, graceful shutdown and long-running
fault recovery remain unqualified.

[Fresh-kernel validation](ASYNC_CAM_MEGA_MOE_CONST_FIX_VALIDATION.json) records
build identity, actual scores, sample comparison and the retained archive hash.

## Superseded upgrade qualification

**Invalidated:** the build log contains a CANN 9.1 const-correctness error in
`TileTransport::Prefix()`. Packaging continued with a pre-existing kernel.
The recorded 287/300 run and finite tests below do not qualify the upgraded
kernel, even though source and installed-binary hashes were recorded.
The fresh-kernel qualification above replaces this invalidated conclusion.

### Original observations

The combined DS8/CR8, K=4 graph and metadata-cache build passed the following
checks on A3 with CANN 9.1.0:

- 25 compiled Meta/workspace and graph-runner tests, including exact transaction
  counts, remainder batches, sticky errors and resource-retention guards.
- Existing four-operator CAM tests against the CPU oracle for FP16/BF16 with
  dynamic quantization off/on, sparse routing and an empty FFN. A separate
  TP2 case uses 32 and 33 local tokens to cross the small-batch tiling boundary.
- Eager and FULL against saved layered outputs with dense signed per-group
  weights, H256/I512, routed scale 2.5, two sources and capacity 256. Both FULL
  ranks capture two graphs and complete 16 and 14 logical transactions.
- Eager and FULL at H4096/I2048, 32 experts and capacity 262144, including empty
  FFN work, repeated layers and 129-token workloads spanning multiple tiles.
  Both FULL ranks capture two graphs and complete seven logical transactions.
- 596 local CPU tests and 33 subtests; 184 unavailable or opt-in checks skip.
  The 162-case host workspace check passes ASan/UBSan with leak detection off.

The independent graph candidate also completed native K=1/2/4 submissions of
1, 5, 43 and 43 transactions, then released its resources. An invalid layer map
produced status -1; a subsequent successful native call retained -1. This
checks first-error retention, not recovery from a failed peer or a stuck kernel.
All functional fixtures completed their cleanup.

The combined build completed **287/300 GSM8K samples (95.67%)** for both
strict-match and flexible-extract on `afd-cross-task-f-20261021`. It uses the
same DeepSeek-V4-Flash DSpark W4A8 checkpoint, Attention DP2/TP4 eager and
FFN DP8/TP1/EP8 FULL configuration as the historical Phase 2 run. The 8-shot,
greedy, 512-output-token, concurrency-12 evaluation settings and 32768 token
limits are unchanged. The historical Phase 2 score is 286/300; no fresh baseline
was run, and this comparison does not establish accuracy parity.

All 300 document, prompt and target hashes match that historical run. Model
metadata, dataset and package versions also match. Four answers improved and
three worsened; 166 generated texts differ. All eight FFN ranks selected
MegaMoE for 43 layers, captured two graphs and recorded at least 176,128
successfully completed logical transactions each. Counts include warmup,
smoke and evaluation work; they are not Python replay-call counts.

Source and native binary hashes match before and after evaluation. There were
no inference errors or missing/duplicate documents. The first archive verifier
encountered TCP TIME_WAIT on port 6455 after process cleanup; the unchanged
verifier passed once the port was released, without rerunning inference.
The harness still logs exceptions and resource-tracker warnings during forced
shutdown. Test processes are gone, ports are free, and both existing iTASK
allocations remain Running. Graceful shutdown remains unqualified.

Changed-file pre-commit passes. Repository-wide pre-commit still has unrelated
pre-existing failures; local pytest excludes unconfigured model E2E tests.
[Upgrade evidence](ASYNC_CAM_MEGA_MOE_UPGRADE_VALIDATION.json) records the fixed
source/binary identities, test scopes, sample comparison and artifact hashes.

## Historical Phase 2 qualification before this upgrade

The runnable version uses two 128-row slots with separate G1/G2 scratch,
8 communication AIVs and 40 compute AIVs. At H=4096, I=2048, 32 local experts
and Attention TP=4, scratch is 23,773,696 bytes (22.67 MiB), independent of
receive capacity. The entry points, flags and compact-v2 wire format are
unchanged from Phase 1.

The frozen kernel passed eager and FULL finite tests at H=4096/I=2048 with
32 local experts and receive capacity 262144. These compare all seven
7/16/129-token workloads against saved layered outputs and cover empty FFN
work, repeated layers and slot reuse. FULL records one capture and seven
replays per FFN rank, stable addresses and poisoned scratch between requests.
A separate Attention TP4/FFN2 fixture passed 12 repetitions with distinct TP
source values and an analytic output check. These are operator tests, not a
real-model accuracy conclusion.

The small-capacity fixture also passed eager and FULL against saved layered
outputs with dense signed per-group weights, H=256/I=512, routed scale 2.5,
capacity 256 and two independent Attention sources. It covers the 129-row
case spanning two expert chunks, source switching, empty FFN work and poisoned
scratch. The two FULL FFN ranks captured once and completed 16 and 14 replays.

The compiled binding passed all seven Meta/workspace tests. Local pytest passed
596 tests and 33 subtests, with 183 tests skipped; the CPU environment does not
replace native NPU qualification. Pre-commit and the 162-case ASan/UBSan
workspace check passed.

The polling loop must reload its flag after cache invalidation: a non-volatile
load caused a reproducible wait despite the other participants publishing
their sequence. The frozen version uses a volatile GM load and explicit
per-writer cache flushes. Broader cache and DMA optimizations were not retained.

The following diagnostic measurements use the native kernel duration from
the CANN profiler, with dispatch completed before FFN launch. Each value is
the maximum across FFN ranks of the last five kernel durations averaged after
12 repetitions and a cold warmup. Both versions use the same synthetic
workload and topology. These are not end-to-end model performance results.

| Workload | Attention TP | Phase 1 (us) | Phase 2 (us) |
| --- | --- | --- | --- |
| 7 tokens | 1 | 676.704 | 1097.832 |
| 1024 tokens, one expert | 1 | 1368.604 | 2387.948 |
| 1024 tokens, balanced experts | 1 | 1593.324 | 2998.564 |
| 129 tokens, one expert | 4 | 926.336 | 1562.212 |

Phase 2 has not met a performance acceptance criterion. Further tuning and
an end-to-end performance comparison are deferred. No new real-model baseline
was run.

The frozen build completed **286/300 GSM8K samples (95.33%)**, for both
strict-match and flexible-extract, on `afd-cross-task-f-20261021`. DeepSeek-V4-Flash
DSpark W4A8 ran with Attention DP2/TP4 in eager mode and FFN DP8/TP1/EP8 in
FULL graph mode. The run retained 8-shot greedy evaluation, 512 output tokens,
concurrency 12, MBT/max-model-len 32768, DBO off and FFN two-stream off.
All eight FFN ranks selected MegaMoE for 43 layers, captured once and recorded
at least 176,128 successfully completed replays each. Exit code was 0; there
were no inference errors, missing documents or duplicate documents.

All 300 document, prompt and target hashes match the historical Phase 1 runs;
model metadata, dataset and package versions also match the Phase 1 FULL run.
Relative to its 288/300 score, two documents improved and four worsened, with
159 generated texts differing. The historical eager score was 285/300.
This establishes completion on this checkpoint, not numerical or accuracy parity.

The source and native-artifact hashes were verified before and after evaluation.
The harness cleaned up its service processes and freed its ports; its forced
shutdown still logs FFN exceptions and resource-tracker warnings. The iTASK
allocation remains Running. Graceful shutdown and long-running lifecycle are
not qualified by this run.

[Phase 2 evidence](ASYNC_CAM_MEGA_MOE_PHASE2_VALIDATION.json) records the frozen
source/binary hashes, finite checks, real-model results and diagnostic timings.

## Historical Phase 1 evidence

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
This follow-up covers that checkpoint and Attention TP=4 in eager mode.

### Real-checkpoint FFN FULL follow-up

The Phase 1 Python runner and the same repaired native kernel subsequently
passed 300 GSM8K samples on A3 task `afd-cross-task-f-20261021`: strict-match
and flexible-extract both **288/300 (96.00%)**. Attention remained eager with
DP2/TP4; FFN used FULL graph with DP8/TP1/EP8. All other evaluation settings
above were retained. Test-only wrappers recorded one capture and at least
176,128 successfully completed replays on each of the eight FFN ranks, with
MegaMoE active. Counts include startup, smoke and evaluation work.

All 300 document, prompt and target hashes matched the saved eager run. Seven
documents changed correctness (five improved, two worsened); 165 generated
texts differed. The historical eager score was 285/300. No fresh baseline was
run, and this cross-host comparison does not establish numerical parity or
an accuracy gain caused by graph mode.

[FFN FULL evidence](ASYNC_CAM_MEGA_MOE_GRAPH_GSM8K_300.json) records the source
and native hashes, scores and capture/replay observations. Test exit code was
0; no inference errors or remaining tagged test processes were found. The
iTASK allocation remains Running. Attention graph mode, dedicated
concurrent-source stress, invalid-control recovery, long-running lifecycle,
kernel profiling and performance benefit remain unqualified.
