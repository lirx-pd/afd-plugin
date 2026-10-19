# CAM Async Connector User Guide

`CAMAsyncAFDConnector` is the Ascend CAM-backed asynchronous connector for AFD
Attention/FFN disaggregation. It lets Attention workers compute MoE routing and
exchange routed-expert activations with independent FFN expert ranks
through CAM async dispatch/combine operators.

Use the pinned dependency pair in the
[root installation instructions](../../README.md#ascend-npu-installation).
Current runtime constraints and validation evidence are summarized in the
[runtime matrix](../design/module/execution_platforms.md#tested-runtime-matrix).

> [!WARNING]
> CAM async is experimental. The native DBO path requires target-device model
> validation; configuration acceptance and executor tests alone do not qualify
> model accuracy or performance.

## When to use this connector

Use this connector for eager asynchronous Ascend inference with these requirements:

- plugin-owned Ascend operators are built and installed on every node;
- Attention performs MoE gating before dispatch to FFN ranks;
- Attention execution is eager and AFD async-DP is enabled; **`async_dp=true` is required**;
- regular prefill and autoregressive decode steps use the same CAM path;
- optional ubatching uses exactly two native MRV2 DBO microbatches,
  including Attention DP1.

CAM async supports eager execution. Native DBO requires Attention-side gating,
PP=PCP=DCP=1, and two ubatches. A step below the native DBO threshold runs
unsplit locally. Both token thresholds must be at least two.
Attention DP replicas make that decision independently; FFN is connector-driven.

FFN also supports the experimental complete-transaction FULL mode described in
[Async FFN runtime](ASYNC_FFN_RUNTIME.md#selection-and-startup).
FULL requires MRV2 on both roles and accepts Attention-side DBO under the
existing constraints. Attention graph execution remains unsupported.

## ModelRunnerV2 deployment

Set `VLLM_USE_V2_MODEL_RUNNER=1` and `VLLM_PLUGINS=ascend,afd` on both roles.
Attention uses native Ascend ModelRunnerV2; FFN uses the independent
`AsyncNPUFFNModelRunner`. Neither role selects or falls back to MRV1. Use the pinned vLLM/vLLM-Ascend pair in the installation instructions.

The V2 Async CAM contract is:

- registered DeepSeek-V2/V3 MLA or DeepSeek-V4 models;
- `--enforce-eager` on Attention and `--enable-expert-parallel` on both roles, with
  `async_dp=true` and `compute_gate_on_attention=true`;
- on FFN, select eager with `--enforce-eager`; for FULL, omit `--enforce-eager`
  and pass `--compilation-config '{"cudagraph_mode":"FULL"}'`, following the narrow
  W4A8/runtime contract in the linked Async FFN guide;
- PP=PCP=DCP=1, static expert placement, and each role's configured rank count
  equal to its DP x TP size;
- `attn_ranks_per_dp` equal to Attention TP, and Attention
  `--max-num-batched-tokens` divisible by that TP size;
- `dynamicQuant=1` for DeepSeek-V4 Flash W4A8.

Attention-local FlashComm1/SP is supported. Set
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` only on Attention and `0` on FFN.
DeepSeek-V2/V3 Attention with effective model SP automatically selects Ascend
shared-expert DP: shared weights are replicated per rank, which computes its
local tokens. Without model SP, this automatic override is not applied.
Attention TP/SP and routed-expert parallelism are unchanged. MRV2 rejects
explicit `enable_shared_expert_dp=true` and finegrained TP configuration.
Use `--block-size=128` for V2-Lite MLA Attention; for V4,
use the model-specific settings in the
[DSV4 E2E profiles](../../tests/e2e/models/deepseek_v4_flash/config.py).

At every Attention DP size, `--enable-dbo` uses native child preparation,
per-ubatch model forwards, and output merge. Native
splitting can divide one request across microbatches. For DeepSeek-V2/V3, pass
`--enable-chunked-prefill` so a request crossing the microbatch boundary can
attend to its prefix KV. Native draft-model/MTP paths are also
unsupported.

## CAM async data flow

CAM async removes synchronization between data-parallel replicas and avoids the
rank-wide synchronization imposed by dispatch/combine all-to-all communication
at large expert-parallel sizes. When requests are unevenly distributed across
DP replicas, each replica can continue its CAM dispatch/combine work without
waiting for every other replica to reach the same communication point. This
eliminates unnecessary idle time caused by DP request imbalance.

One MoE layer follows this sequence:

1. Attention computes top-k expert IDs and weights.
2. `async_dispatch_send` sends hidden states and routing IDs into the CAM group.
3. Each FFN rank calls `async_dispatch_recv` and receives its routed expert
   tokens, compact metadata, per-expert counts, and optional dynamic-quant scales.
4. The FFN worker executes its local routed experts; Attention computes shared experts.
5. `async_combine_send` returns those outputs with the dispatch metadata.
6. Attention calls `async_combine_recv`; CAM routes, weights, and combines the
   expert results for the original tokens, then adds its local shared output.

CAM dispatch payloads carry the token-count and routing metadata. Consequently,
this connector does not use the separate Gloo DP-metadata control plane used by
the synchronous connectors.

## Topology and rank derivation

The connector creates one HCCL world with all Attention ranks first and all FFN
ranks second:

```text
world rank:  0    1   ...  A-1   A    A+1  ...  A+F-1
member:      A0   A1  ...  A_    F0   F1   ...  F_
```

For `A = num_attention_ranks` and `F = num_ffn_ranks`:

- Attention role rank `i` has world rank `i`;
- FFN role rank `j` has world rank `A + j`;
- world size is `A + F`;
- each role rank must be unique and within its role's configured rank count.

Attention ranks are `DP x TP`. `attn_ranks_per_dp` must equal the Attention TP
width passed to CAM. Before connector initialization, the connector factory
resolves the effective role rank as:

```text
role_rank =
    global_dp_rank * tp_size + tp_rank
```

The role rank is runtime state, not public configuration. vLLM's global DP
rank already includes `data_parallel_start_rank`, and the connector factory
uses one shared resolver before constructing any connector.

For example, a DP3TP2 Attention + DP2TP1/EP2 FFN topology uses:

```text
num_attention_ranks = 3 * 2 = 6
num_ffn_ranks = 2
attn_ranks_per_dp = 2

No role-rank field is configured. Attention runtime role ranks are `0..5`, and
FFN runtime role ranks are `0..1`.

CAM world ranks: A0..A5 = 0..5, F0..F1 = 6..7
```

FFN ranks follow expert parallel placement. The runtime derives experts per rank
from the model routed-expert count and `num_ffn_ranks`; use a model/topology in
which routed experts divide evenly across FFN ranks. All roles must use the same
model, routed-expert layout, quantization, HCCL address, rank counts, CAM
settings. Enable native DBO only on Attention; FFN must not receive
`--enable-dbo` or native ubatching flags. Legacy connector flags may remain in
shared JSON because their normalization is Attention-only. In an async deployment, every Attention (A)
and FFN (F) process must also set `--max-num-batched-tokens` to the same value.

## AFD configuration

Pass AFD configuration through vLLM's `--additional-config` under the `afd`
key. The presence of the `afd` object enables AFD; omit it to disable AFD.
There is no separate `--afd-config` option.

```jsonc
{
  "afd": {
    "role": "attention",
    "connector": "CAMAsyncAFDConnector",
    "async_dp": true,
    "host": "10.0.0.1",
    "port": 6239,
    "num_attention_ranks": 6,
    "num_ffn_ranks": 2,
    "compute_gate_on_attention": true,
    "connector_extra_config": {
      "dynamicQuant": 1,
      "attn_ranks_per_dp": 2,
      "hccl_buffer_size": 4096
    }
  }
}
```

### Common fields

| Field | Type | Default | Meaning and constraint |
| --- | --- | --- | --- |
| `role` | `"attention" \| "ffn"` | `"attention"` | Role owned by this process. |
| `connector` | `str` | `"P2pNcclAFDConnector"` | Must be `CAMAsyncAFDConnector`. |
| `async` / `async_dp` | `bool` | `false` | Must be `true`. `async` is the accepted compatibility alias for canonical `async_dp`. |
| `host` | `str` | `"127.0.0.1"` | **Must be the IP address of the node that owns Attention rank 0.** Every rank must use the same reachable value. |
| `port` | `int` | `1239` | HCCL rendezvous port in `1..65535`; it must be free and reachable. |
| `afd_process_group_timeout_s` | `int` | `120` | Positive AFD group rendezvous timeout in seconds; also bounds the NPU native DBO startup barrier, not device receive polling. |
| `num_attention_ranks` | `int` | `1` | Total Attention ranks across all DP and TP groups. |
| `num_ffn_ranks` | `int` | `1` | Total FFN expert ranks. |
| `compute_gate_on_attention` | `bool` | `false` | Must be `true`; CAM async runs MoE routing on Attention before dispatching to FFN ranks. |
| `connector_extra_config` | `dict` | `{}` | Connector-specific settings. Unknown top-level AFD fields are rejected. |

Compatibility aliases `afd_role`, `afd_connector`, `afd_host`, `afd_port`,
and `async` are accepted. New configurations should use canonical names,
including `async_dp`; enable DBO through the Attention CLI flags below.

### CAM async `connector_extra_config`

| Field | Type | Default | Meaning and constraint |
| --- | --- | --- | --- |
| `dynamicQuant` | `int` | `0` | Enables CAM dispatch/combine dynamic-quant metadata. Only `0` and `1` are accepted. With `1`, FFN receives quantized routed activations plus scale tensors and must return output compatible with combine-send. |
| `attn_ranks_per_dp` | `int` | `1` | Positive Attention TP rank count per DP replica. It is the CAM Attention grouping width and is independent of the FFN process's local TP size. |
| `async_moe_ubatching` | `bool` | `false` | Compatibility alias for native DBO on Attention, including DP1. Prefer `--enable-dbo`. |
| `async_moe_num_ubatches` | `int` | `2` | Legacy compatibility validation; only `2` is accepted. Native DBO owns the microbatch count. |
| `async_moe_split` | `str` | `"request"` | Deprecated compatibility field: accepts `"request"` or `"token"` but has no effect on splitting and logs a warning when supplied. Native DBO owns request/token boundaries. |
| `hccl_buffer_size` | `int` | unset | Positive CAM HCCL buffer size in MB. The override applies only to the `afd_async_cam` communication domain. When unset, HCCL uses `HCCL_BUFFSIZE`, then its built-in default. Configure the same value on every Attention and FFN rank in the domain. |

For a DP+SP deployment such as DP3TP2 Attention + DP2TP1/EP2 FFN, set
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` only on Attention and explicitly set it to
`0` on FFN. Attention then holds contiguous TP-local sequence shards. For plain
DP+TP Attention, leave FlashComm1 disabled on both roles. In either case, FFN
consumes expert-routed CAM work items and may independently use TP1.

## Native DBO and Async CAM ubatching

### vLLM native DBO

For eager DeepSeek-V2/V3 MLA or DeepSeek-V4 Attention at any DP size, enable
native DBO explicitly or retain `async_moe_ubatching=true` as its alias:

```text
--enable-dbo
--dbo-decode-token-threshold <N>
--dbo-prefill-token-threshold <N>
```

`--ubatch-size 2` is also normalized to DBO on this Async Attention path.
Do not enable native DBO on FFN. FFN consumes independent CAM work items and
never joins Attention's DBO thread schedule. Native thresholds select eligible
prefill/decode batches and must be at least the microbatch count, including when
using the legacy alias. Native slicing determines token and request ranges,
including parent padding and requests crossing a microbatch boundary. The model
owns SP padding and each owner's layout restoration/TP collectives.

CAM has one physical window per Attention rank. Before another ubatch sends,
a native peer receive hook completes the preceding raw receive with its owner's
explicit transfer context. The resumed owner restores layout and adds shared
expert output. The final native context exit drains any remaining receive hook.
Each call sends and receives once; transfer errors require restarting the A/F
group rather than retrying inside the executor.

### Attention DP1 compatibility

DP1 uses the same native slicing, child attention metadata, execution and merge
as DP > 1. The Ascend adapter extends native factory and threshold selection to
DP1 because upstream otherwise returns early there. Async child contexts start
with `dp_metadata=None`; FFN DP coordination belongs to the connector.
DP1 model correctness still needs the hardware checks in the
[NPU validation guide](TESTING.md#native-dbo-contracts-and-device-execution).

For shape-only runtime diagnostics, set
`AFD_ASYNC_MOE_LAYOUT_LOG=1` on Attention. The log reports the parent token
extent, CAM-local slice, padding, and whether the FFN result is TP all-gathered.
It does not inspect tensor values or force a device synchronization.

## Requirements

Use Ascend 910C, the pinned stack from the
[root installation instructions](../../README.md#ascend-npu-installation),
and plugin-owned source-built operators on every node.

Build the AFD operators from the repository root inside the container:

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 pip install -e . -v --no-build-isolation
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

The loader configures the packaged `afd-plugin` vendor directory and loads
`afd_plugin._C_ascend`. No external CAM wheel or vendor library is required.
Async CAM sends routed experts only. Attention owns shared MLP weights and
computes shared outputs in the model's token layout. DeepSeek-V2/V3 with
effective model SP selects replicated shared weights. DeepSeek-V4 preserves
the native shared-expert policy through `AFDDeepseekV4SharedExperts`, including
all-gather/reduce-scatter for TP-sharded weights with SP and all-reduce without
SP. Replicated weights use local SP tokens or split/gather full tokens. Shared
weights and intermediates are included in Attention memory profiling.

Set `connector_extra_config.hccl_buffer_size` when the CAM domain needs a
larger buffer. Also set `HCCL_BUFFSIZE` to that same effective size on every
rank: Async CAM host tiling reads the environment, whereas HCCL allocation
can use the domain override. For the JSON example above, export
`HCCL_BUFFSIZE=4096`. Changing only `hccl_buffer_size` can misalign the window
capacity and tail-state address. `HCCL_BUFFSIZE` also affects unrelated groups
without explicit overrides; see the [operator window contract](CAM_ASYNC_ROUTED_OPS.md#communication-window-configuration).

For the experimental FlashComm1/SP cases, set
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` only on Attention. Set it to `0` for plain TP
Attention and on every FFN process.

At initialization, the runtime verifies all four
`torch.ops.afd_ascend.afd_async_*` operators and logs the plugin version and
actual vendor library path. Missing source operators fail startup.

## Current limitations

- Attention is eager. FFN FULL targets homogeneous W4A8 layers on A3/910C;
  see [supported configuration](ASYNC_FFN_RUNTIME.md#selection-and-startup)
  and [validation commands](ASYNC_FFN_RUNTIME.md#validation).
  Broader models and topologies require separate validation.
- Regular prefill and autoregressive decode steps are supported; context
  parallel execution remains outside the supported Async CAM topology.
- Native DBO requires eager Attention, two ubatches, and no speculative
  decoding. FFN keeps its independent connector-driven execution.
- Attention DP1 reuses native DBO through scoped Ascend compatibility adapters;
  the legacy request/token split selector no longer controls boundaries.
- ModelRunnerV2 rejects PP, PCP, and DCP; V1 PCP and async MoE context
  parallelism remain unsupported.
- Routed experts should divide evenly across FFN ranks.

For startup, HCCL, CAM operator, and shutdown failures, see the
[NPU troubleshooting guide](TROUBLESHOOTING.md).

## Runtime profiling

Set `AFD_CAM_PROFILE_ENABLE=1`, `AFD_CAM_PROFILE_TOKEN` to the same nonempty
secret, and `AFD_CAM_PROFILE_DIR` to a writable output directory on every
Attention and FFN server before startup. Install the plugin's endpoint entry
point on every server and include `afd_profile` in `VLLM_PLUGINS`, preserving
the existing `ascend,afd` entries. Keep static AFD profiling and upstream
profiling stopped while a runtime session is active.

Runtime profiling supports eager FFN execution and MRV2 FFN FULL graph replay.
Graph capture remains a startup operation; profiling starts and stops the native
TorchNPU backend around replay, as in the upstream worker profiler.

Start one coordinator, listing each server API URL once, including FFN:

```bash
export AFD_CAM_PROFILE_TARGETS='["http://attention:8000", "http://ffn:8001"]'
export AFD_CAM_PROFILE_TOKEN='<shared-secret>'
export AFD_CAM_PROFILE_DIR='./cam-profiles'
python -m afd_plugin.profiling.control --host 127.0.0.1 --port 8765
```

Both modes use the TorchNPU profiling backend used by vLLM-Ascend.

For FFN FULL graphs, `LIGHTWEIGHT` collects NPU-only `Level0` kernel timestamps
without CPU operator callbacks, shapes or stacks. It reconstructs `WAIT` from
dispatch-receive, `COMPUTE` between receive and combine-send, and
`COMMUNICATION` during combine-send. Captured graphs do not rerun Python/MSTX
phase hooks. This graph mode produces more raw data than eager lightweight
mark-only capture; its pipeline covers the observed device kernel boundaries.
Missing, overlapping or unpaired DR/CS kernels make the export `PARTIAL`.

FFN graph profiling stops after the current replay batch completes. An idle DR
cannot be cancelled; if a batch remains blocked, the existing profiler operation
deadline reports failure. Continue the workload through stop rather than leaving
an incomplete graph batch pending. Profiling does not change graph resource
ownership or make graph shutdown cancellable.

`LIGHTWEIGHT` records a continuous **COMPUTE**, **COMMUNICATION**, **WAIT** pipeline
on Attention and FFN streams, with optional Attention dispatch annotations.
Eager execution uses NPU-only `Level_none` and the `afd.cam` MSTX domain. A native device mark
is submitted only when a stream changes phase; adjacent compute regions across
layers share one state. Stream tracks and the three phase messages are reused;
model/DBO entry caches the track instead of hooking every Attention layer.
CPU operator callbacks, kernel/task profiling, shapes, stacks/modules,
memory, FLOPs and hardware counters are off. There are no timing Event pools,
collector scans, request snapshots or relationship records. LIGHTWEIGHT also
disables TorchNPU's Python Event recording wrappers while retaining completion
fences. Control and JSONL writer threads sleep until notified or a deadline expires.

Send phases are COMMUNICATION; receives and DBO yields are WAIT. CAM fuses
transfer and polling, so WAIT includes receiving data. These are stream pipeline
states, including gaps between submissions, rather than pure kernel busy time.
Streams remain separate so DBO overlap is preserved. Native timestamps and the
coordinator's clock offset/uncertainty align Attention and FFN tracks; no layer or
request pairing is inferred. The initial interval before a stream's first marker
is unobserved. A native host END marker clips queued work at the capture boundary
and closes a stream waiting indefinitely for its peer.

Attention COMMUNICATION markers additionally carry `microbatch_id`, `layer_idx`,
`num_requests` and `num_tokens`; offline export keeps these in the interval's
Chrome trace `args`. Counts come from native CPU scheduler/batch metadata, exclude
padding, and describe the whole Attention microbatch before TP sharding.
`microbatch_id` is the stage slot within a step (zero for an unsplit batch),
not a globally unique batch ID. A request spanning two slices counts in both.
The layer is the remote MoE dispatch layer; merged COMPUTE intervals do not
represent complete individual layer timings. Unknown batches remain unannotated.
These fields extend existing marks without new device marks, Events, D2H copies,
barriers or per-layer JSONL records. FFN carries no added identity metadata.

Clock alignment groups workers by Linux boot ID, rather than hostname. Workers
on the coordinator's host use zero offset; workers sharing another host receive
one common estimate. After preparation, three status probes bound each worker's
clock offset between `worker_time - response_received` and
`worker_time - request_sent`. The intersection of these bounds determines the
host's offset and uncertainty, without assuming symmetric network/RPC latency.
Inconsistent bounds fail the session before commit. Hosts without a boot ID
retain separate worker estimates. Cross-host uncertainty describes calibration
time and assumes stable clock offsets during capture; it does not measure native
device-to-host timestamp mapping error or guarantee zero drift in long captures.

`FULL` retains CPU/NPU activities, Level2, shape recording and configured
stack/modules. Both modes use unchanged upstream CAM operators; `csrc` and the
wire protocol are unchanged. Layered FFN keeps layer selection on device and
skips unused host control copies in LIGHTWEIGHT and idle mode. Legacy FFN and
FULL retain their three control values.
The stream/MSTX bindings target TorchNPU 2.10 and need revalidation on upgrades.

For continuous long-running collection, LIGHTWEIGHT accepts up to seven days;
FULL retains its one-hour limit. For example, use
`{"mode":"LIGHTWEIGHT","duration":86400.0}` for one day. Collection does not
sample or discard intervals. Storage still grows with phase transitions;
removing identities and duplicate records reduces that growth, rather than
placing a fixed bound on the entire capture.

```bash
curl -sS -X POST http://127.0.0.1:8765/cam/profile/start \
  -H "Authorization: Bearer ${AFD_CAM_PROFILE_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"FULL","duration":10.0}'
```

Use the returned `session_id` to query `GET /cam/profile/{session_id}` or stop
early with `POST /cam/profile/{session_id}/stop`, using the same authorization
header. The duration is a shared wall-clock window; native initialization can
shorten each worker's active coverage. Inspect `details.started_at_ns` for the
actual start. A window ending before all backends activate fails explicitly.
Both runtime modes record continuously between explicit start and stop, without the
static `WAIT/WARMUP/ACTIVE/REPEAT/SKIP_FIRST` schedule. The existing static
profiler retains that schedule. Runtime does not modify environment variables;
the role's `AFD_NPU_*_PROFILER_WITH_STACK` setting applies to FULL only.

Each worker writes only common `SESSION`/`SUMMARY` records to
`<session_id>/<worker_id>.jsonl`, independent of the number of requests or layers.
FULL output remains in `<session_id>/<worker_id>/full/`. For both modes,
`details.native_profiler.trace_dir` identifies the native output location.

During LIGHTWEIGHT capture, CANN writes its native raw files. After native stop,
a background thread packs the dedicated worker directory into
`<session_id>/<worker_id>/lightweight.tar.gz` using fast gzip compression. It
verifies the gzip data and makes the archive durable before removing the raw
directory. Each worker then retains one archive and one small JSONL file.
Compression does not block native stop or parse traces during inference.
Wait for the final `.tar.gz` file before copying or analyzing a finished capture;
session completion can precede archiving. Live worker status exposes
`details.native_profiler.archive_state` and `archive_error`; the coordinator
manifest retains its stop-time snapshot. On failure, raw data or a completed
archive remains available for recovery. Raw data and the temporary archive
coexist while compression runs.

Export the aligned pipeline offline with the matching TorchNPU/CANN environment:

```bash
python -m afd_plugin.profiling.analyze ./cam-profiles/<session_id> \
  --native --output ./pipeline.json.gz
```

The analyzer safely extracts worker archives into temporary directories, invokes
the native parser offline and imports only phase-transition timestamps. The
compact gzip Chrome trace contains separate Attention/FFN stream tracks, without
per-event identity arguments or duplicate relationship tables. An output path
without `.gz` writes compact JSON instead. Previously exported native data and
old Event/layer JSONL captures remain readable.

Native operations stay on the model thread. The control thread requests lifecycle
changes and enforces deadlines; it never owns the native profiler. Idle Attention
advances pending startup and stop during status RPCs. FFN advances them while
querying its preallocated control-copy completion Event. Runtime sessions own
their native profiler independently of the runner's static `prof` object.

FFN records an untimed completion Event immediately before dispatch receive. That Event also
fences all preceding compute and combine-send work. Once it completes, runtime
profiling may stop while the next receive is still waiting. The runtime FFN instance
uses the pinned TorchNPU `stop_trace` implementation without its full-device
synchronize; all native stop, GC/Event cleanup, finalize and trace-handler steps
remain. Attention and static profiling retain their existing synchronization.
No receive cancellation or extra CAM control messages are required.

FFN retains two untimed completion fences for the mandatory control copy and
safe native lifecycle. The pre-receive stop fence is queried only when native
stop is pending. Stop closes admission, drains admitted work within the deadline
and reports unresolved work or missing native markers as PARTIAL. Runtime
profiling disabled at startup allocates no profiling resources.

## W4A8 layered GMM (experimental, disabled by default)

Set `AFD_ASYNC_CAM_LAYERED_GMM=1` on every FFN rank to use the two layered GMM
operators for compatible W4A8 routed experts. Leave it unset on Attention
ranks; enabling it on another role or connector fails at startup. The startup
log must show `actual=layered` to confirm that the new path is active.

```bash
# Use the same checkpoint, topology, capacity, and requests for both runs.
AFD_ASYNC_CAM_LAYERED_GMM=0 bash /path/to/W4A8-FFN-launch-script.sh
AFD_ASYNC_CAM_LAYERED_GMM=1 bash /path/to/W4A8-FFN-launch-script.sh
```

The intended configuration is Ascend 910C / `ascend910_93`, async CAM FFN,
Attention-side gate, `dynamicQuant=1`, eager execution (or the separately
qualified FFN FULL mode), and static expert
placement. `VLLM_USE_V2_MODEL_RUNNER=1` is required on both roles; FFN uses
its independent connector-driven runner. This path applies only to Ascend DeepSeek V4;
enabling the switch for another model fails at startup. All remote MoE layers must share geometry,
SiLU activation, quantization layout, and scaling semantics. The extractor
can represent per-channel and per-group parameters, but the pinned Ascend
loader rejects positive `group_size`; the current target is limited to
per-channel W4A8. DSV4 Flash has layered off/on 300-question comparisons and
real INT8-activation/INT4-weight operator traces in the
[runtime validation record](../design/module/execution_platforms.md#tested-runtime-matrix).
Per-group operator coverage does not extend the checkpoint loader to
per-group weights.
Shared experts remain on Attention. DSV4 applies routed scaling in top-k, so
FFN does not apply it again.

The graph change preserves the existing layered arithmetic: `swiglu_limit` is
recorded but is not applied by layered W13. Ordinary eager may apply it, so
FULL must be compared with corresponding layered eager, not treated as a fix
for their pre-existing difference. FULL reuses the existing native DR and CS operators. See
[resource lifetime](ASYNC_FFN_RUNTIME.md#resource-lifetime) for completion and
process-cleanup requirements.
Missing `w13_scale_bias` or `w2_scale_bias`, dynamic EPLB, non-SiLU
activation, or incompatible parameters fail before receiving work. Non-W4A8,
mixed-quantization, and heterogeneous layers stay on the legacy path with a
startup reason. Do not fill absent compensation parameters with zeros merely
to enable this path.

The new path uses the full dispatch-recv capacity and device expert counts. A
slice of the original metadata, or a device mapping, selects the layer; the
original metadata goes to combine-send. Empty ranks, chunks, and ubatches use
the existing communication protocol. With `AFD_CAM_OP_IO_LOG` enabled, the new
path logs only Tensor shape, dtype, and device without reading metadata values.
The Async eager worker synchronizes after each complete FFN layer loop; FULL
submits a bounded batch on one stream and waits once for that batch. These
submission loops do not define request boundaries. Measure peak
memory for the capacity-sized intermediates, and set `BATCH_SIZE_FACTOR` high
enough for every whole-expert chunk.

CPU contract tests cover layer mapping, repeated layer IDs, capacity, zero
counts, parameter rejection, startup selection, and the absence of explicit
metadata D2H in the new Python path. They do not establish NPU numerical
accuracy, empty-work completion, hidden synchronization, or a performance gain.
Run the compiled Meta and NPU numerical tests described in the
[testing guide](TESTING.md#layered-w4a8-validation) and the
[FULL runtime checks](ASYNC_FFN_RUNTIME.md#validation). Deployments require
target-device validation of completion, checkpoint accuracy and performance.

To revert, set `AFD_ASYNC_CAM_LAYERED_GMM=0` and restart the FFN processes.
When switching from FULL to legacy eager, also pass `--enforce-eager` explicitly
before restarting FFN.
For performance runs, disable `AFD_CAM_OP_IO_LOG` and
`AFD_FORCE_BALANCED_TOPK_IDS`, retain the actual-path startup log, and measure
the existing group synchronization separately.
