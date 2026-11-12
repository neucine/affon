# Optimization, Profiling, and Evidence

AFFON keeps optimization decisions, runtime measurement, and policy changes
separate. A `Program` describes semantics, compilation produces an immutable
`Executable`, and a `Session` optionally records what happened while that
executable ran. Measurements never rewrite a live executable or silently tune a
future compilation.

## Public controls

Session options control runtime behavior shared by all executables in that
session:

```ts
import { Session } from "affon:core"

const session = new Session({
  device: "cuda",
  determinism: "strict",
  telemetry: {
    computeTiming: "invocation",
    hardwareMetrics: true,
    bridgeTiming: true,
  },
})
```

Runtime measurements are published through `std:telemetry`, alongside user
spans and metrics. Backend timing and hardware metrics are independent.
Unsupported metrics are reported as unavailable, never as measured zeroes.

`computeTiming` selects backend timing scope:

- `"none"` records no backend timing samples. Invocation and kernel-count
  evidence can still be requested independently through `hardwareMetrics`.
- `"invocation"` records one deferred timing sample around all kernels launched
  by an invocation. This is the default choice for model profiling.
- `"kernel"` preserves the legacy event pair around every observed backend
  kernel launch, including successful cuBLAS operations.

`telemetry: { backendTiming: true }` remains compatible and selects
`computeTiming: "kernel"`. New code should set `computeTiming` explicitly.
Kernel counts are emitted once per invocation from backend launch accounting,
so choosing invocation timing or hardware-only observation does not infer the
count from retained timing samples. CUDA direct launches and successful
cuBLAS submissions share that accounting path.

`telemetry: { bridgeTiming: true }` adds opt-in spans around the Affon host
bridge: Program run, input binding, Compute submission, output wrapping, and
accelerator readback. Readback is split into device mapping/synchronization,
native conversion or repacking, and JavaScript copy where those phases exist.
The spans carry backend, dtype, element count, byte estimate, and Compute
storage ID. Use them to account for host time that occurs outside the Compute
execution span.

Nested `to_array()` conversion can dominate that readback span on large dense
tensors even when the device copy is negligible. Dense innermost rows are now
emitted with one `JS_NewArrayFrom` call per row instead of one QuickJS property
store per element. On a `[128,512]` f32 CUDA tensor, `to_bytes()` stayed near
0.05 ms while nested conversion fell from about 1.05 ms to 0.38 ms. Measure
`to_bytes()` and `to_array()` separately before attributing a parity gap to the
device copy. The retained split profile and reproduction script are in the
[full readback batch](../../test/benchmark/results/parity-cuda-readback-full2-2026-11-10/README.md).

On CUDA, `telemetry: { profilerMarkers: true }` adds NVTX ranges around each
backend invocation. Each range names its run, executable, physical step, and
invocation IDs for correlation with an Nsight Systems trace. The same option
marks `item()`, `to_array()`, and `to_bytes()` readback with tensor and storage
IDs. Invocation, execution, and readback ranges use independent start/end
handles, so nested cuBLAS ranges cannot corrupt the Affon correlation stack.
This is an opt-in diagnostic: it adds host overhead and does not time GPU
completion. See the [observability gap audit](./observability-gaps.md) for the
SQLite correlator, coverage checks, and remaining timeline limits.

The SQLite correlator uses the exported CUDA runtime, kernel, copy, memset, and
synchronization tables to partition device idle intervals. Its timeline output
reports synchronization, other CUDA API time, and host delay separately, with
the largest gaps linked to their preceding and following invocation IDs.
`capture.json` includes this summary for the trace selected by the model profile.

Compute identifiers are 64-bit values. JavaScript numbers cannot represent every
such value exactly, so telemetry and runtime evidence also publish
`compute.run_id_text`, `compute.executable_id_text`,
`compute.invocation_id_text`, submission and invocation range `*_id_text`
fields, `program_id_text`, and `executable_id_text` where applicable. Use those
decimal strings as the authoritative identities. Numeric mirrors are convenient
for display and ordering only; never use them to decide whether a large ID
correlates with an Nsight or runtime-evidence row.

Compilation options select one immutable executable variant:

```ts
const reference = session.compile(model, {
  optimizationLevel: "none",
  numericalPolicy: "exact_only",
  explanationLevel: "detailed",
})

const candidate = session.compile(model, {
  optimizationLevel: "safe",
  numericalPolicy: "backend_equivalent",
  optimizationGoal: "throughput",
  explanationLevel: "detailed",
  residualPolicy: "retain",
})
```

For dynamic shapes that require repeated compilation, `searchLimits` bounds
physical candidate exploration per compilation. The chosen limits and any
truncation are included in `executable.explain().executable.selection`.

```ts
session.compile(model, {
  searchLimits: {
    structuralStates: 16,
    structuralExpansions: 512,
    completeEvaluations: 64,
    physicalWork: 128,
    readyWork: 64,
  },
})
```

The controls have distinct meanings:

- `optimizationLevel`: `none` is the legalization/reference path; `safe`
  enables optional rules allowed by the numerical policy.
- `numericalPolicy`: `exact_only`, `backend_equivalent`, or `approximate` sets
  the strongest numerical class a rule may use. Choosing `approximate` grants
  permission but does not imply that an approximate rule exists.
- `optimizationGoal`: `balanced`, `latency`, `throughput`, or `memory` guides
  deterministic cost selection and residual planning.
- `explanationLevel`: `summary` retains accepted compiler decisions; `detailed`
  also retains rejected candidates and alternatives. It does not enable runtime
  timing.
- `residualPolicy`: `automatic`, `retain`, or `recompute` controls derivative
  residual handling. `retain` is the public default because it remains valid
  when fusion removes intermediate instruction boundaries.

Normalized options form part of the executable cache key. The same Program and
options return the cached variant; different options produce separate variants.

`denseInputs: true` opts into the compute core's existing dense-input guards for
every argument, parameter, and state tensor. Inputs must have dense strides and
zero byte offset; incompatible views fail with `GuardMismatch` instead of
silently copying or selecting a different implementation. The default remains
layout-generic. This is useful for frozen inference weights packed into the
backend's preferred matrix order once at model installation/loading time:

```ts
const inference = session.compile(model, { denseInputs: true })
```

The guarded executable has a distinct cache key and exposes its chosen physical
implementations through `explain()`. A dense-input promise does not imply that
every intermediate view is dense. Keep trained parameter layouts in the model's
declared order; inference packing is an explicit adapter step for fixed weights.

Inserted layout materializations are prepared as native `contiguous` copies on
the selected device. They depend on their producer and precede their consumer;
they do not read pending accelerator results through an unsynchronized host map.

## Explain the compiler decision

`Executable.explain()` returns an immutable, JSON-compatible explanation:

```ts
const explanation = candidate.explain()
```

The report includes Program identities, executable steps, source mappings,
chosen implementations, temporary resources, residual decisions, optimization
rules, numerical classes, target capability fingerprints, predicted costs, and
resulting step identities. It answers “what was selected and why?” Compiler
costs are predictions, not elapsed-time measurements.

## Capture runtime evidence

Warm up first, then wrap a representative interval with `std:telemetry`:

```ts
for (let index = 0; index < 3; index++) {
  const warmup = candidate.run(inputs, state)
  warmup.dispose()
}

import telemetry from "std:telemetry"

const output = telemetry.trace("model.candidate", () =>
  candidate.run(inputs, state)
)
const metrics = telemetry.metrics()
```

For a selected run whose evidence must be joined to source instructions and
physical implementation identity, opt into a bounded raw evidence buffer:

```ts
const session = new Session({
  device: "cuda",
  telemetry: {
    computeTiming: "invocation",
    bridgeTiming: true,
    captureRecords: 65536,
  },
})
const executable = session.compile(model)

session.resetEvidence()
const output = executable.run(inputs, state)
const evidence = session.runtimeEvidence(executable)
console.log(evidence.dropped_event_count)
```

`captureRecords` is a per-Session record limit, with a maximum of 4,194,304.
`resetEvidence()` should immediately precede the selected run so that cost is
paid only for the diagnosis. `runtimeEvidence()` returns Compute's aggregate
form: observed invocation and timing evidence joined to each physical step,
source instruction IDs, selected implementation, storage identity, shared
timing groups, phase-local live and active memory peaks, synchronization count,
and compiler-cost validations. Schema version 3 also reports
`kernel_count_evidence`. `exact` permits the aggregate kernel count to be treated
as the device-kernel total. `custom_dispatches_exact_library_fanout_unknown`
means direct launches and library calls are exact, while a library call may fan
out to multiple kernels. Check `dropped_event_count`; a nonzero value means the
requested buffer was too small and the aggregate is incomplete.

For an `optimize()` Program, `runtimeEvidence()` returns
`kind: "composite_optimization_evidence"` with two selected-run components:

- `gradient` owns the primal forward and backward execution;
- `optimizer` is the actual native update executable.

The gradient component's device time is shared by forward and backward nodes
when one backend invocation covers both. The composite states this attribution
instead of copying shared time into independent totals. It also marks the
source forward executable as unexecuted because the gradient executable produced
the primal result. The schema and count contract are checked by the training
capture runner described below.

The telemetry console uses a separate process-wide ring for HTTP trace export.
It is enabled only when `RUNTIME_TELEMETRY_CONSOLE=1`. Its initial capacity comes
from `RUNTIME_TELEMETRY_TRACE_RECORDS`; the default is 4,096 records and the
maximum is 1,048,576. The previous fixed capacity of 512 could be exhausted
before a warmed model step was collected, so a complete capture requires a
capacity sized for the number of records produced between snapshots. The
Apertus profile runner sets this through `--telemetry-records`, defaulting to
16,384. If more than the ring capacity is written before a snapshot, the
collector reports `missed: true`; increasing capacity does not guarantee
complete evidence unless that flag is false.

Compute execution spans are children of the active user trace. They contain
events for executable steps, backend invocations, CUDA kernels or Metal command
buffers, synchronization reasons, timing scopes, hardware-metric availability,
and metric samples. Stable `compute.*` counters and gauges cover executions,
invocations, backend time, synchronization, and storage.

## Capture a representative training step

Use the training observation runner when a training question requires more than
an aggregate event stream:

```sh
python3 test/benchmark/training/profile-training.py \
  --affon /path/to/affon \
  --device cuda \
  --output /tmp/affon-training-profile
```

The worker runs one warmup and one selected fixed-shape decoder
forward/backward plus AdamW update. The Python orchestrator collects telemetry,
invokes Nsight by default, exports SQLite, correlates kernel and API activity,
and validates the composite identities, submission counts, kernel-count
semantics, and selected outer trace. See
[`test/benchmark/training/README.md`](../../test/benchmark/training/README.md)
for backend, Nsight, and buffer controls.

The validated CUDA r6 capture distinguishes the two evidence classes:
490 custom launches, two cuBLAS calls, and an aggregate count of 492 in the
gradient component, versus 493 Nsight kernels because the cuBLAS calls accounted
for the library fan-out difference. The optimizer component has 320 custom
launches and 320 kernels with no library calls, so its count is `exact`. The
runner preserves the difference and requires the explicit unknown-fan-out
marker; it does not force the aggregate to equal the profiler total.

The selected outer training trace covered 33.065 ms of 41.569 ms wall time.
The remaining 8.504 ms is explicit, unassigned time rather than measured zero.
Use the capture to locate host submission, synchronization, and device gaps,
then run a separate uninstrumented workload for throughput.

## Capture deterministic inference

Use the inference Program profile to validate a different operation mix without
depending on a downloaded pretrained checkpoint:

```sh
python3 test/benchmark/hf-inference/profile-inference-program.py \
  --affon /path/to/affon \
  --device cuda \
  --output /tmp/affon-inference-profile
```

The worker runs one warmup and one selected tiny GPT-2 Program with deterministic
f32 weights. It exercises embedding, transformer blocks, layer normalization,
GELU, masked attention, softmax, matmul, logit readback, and token selection.
The runner verifies aggregate schema 3, kernel-count semantics, invocation and
step identity, custom/library counts, phase coverage, external Nsight activity
when available, and dropped telemetry.

The CUDA r1 capture matched 67 invocations and 338 custom kernels across all
three sources. All 713 device activities joined, with 5.368 ms of device gap
showing 357.853 us CUDA API time and 5.010 ms host delay. The selected trace
covered 7.496 ms of 7.985 ms wall time. On Metal the same Program passed with
64 invocations and 64 exact kernels, no drops, and 2.226 ms of the 5.081 ms
wall interval reported as unattributed. Metal currently validates internal
telemetry and runtime evidence rather than an external device trace.

Always compare like with like:

- identical Program revision, inputs, shapes, dtypes, and state;
- identical backend, target fingerprint, numerical policy, and build;
- explicit warmup and sample counts;
- exact output equality or a declared operation/dtype tolerance;
- median and tail latency, not one sample;
- invocation, kernel, synchronization, transfer, and peak-memory counts as
  well as wall-clock time.

For repeatable corpora and regression policy, use compute's
`bench-observability`, `bench-optimization`, and `compare-observability`
commands. Interactive timings are diagnostic, not sufficient policy evidence.

## Feed evidence back safely

Evidence may justify a later source change; it never changes the running
Session. Apply the narrowest supported change:

1. **Cost tweak:** change deterministic ordering only when repeated compatible
   measurements contradict the current descriptor.
2. **Shape guard:** restrict or enable an existing rule where correctness is
   proven but performance is shape-dependent.
3. **New implementation:** register an explicit category, numerical class,
   capability guard, stable identity, and cost descriptor.
4. **New rule:** add a versioned rule only when the semantic pattern, legality,
   source mapping, numerical contract, and phase ownership are new.
5. **Autotuning candidate:** declare an `equivalence_group` only for multiple
   implementations of the same guarded region with identical permitted outputs
   and effects. Fused versus unfused and native versus host fallback are not
   equivalent algorithm alternatives.

Every feedback change requires:

- `none` versus `safe` correctness evidence;
- legality, provenance, phase-ordering, and stable tie-break tests;
- CPU and every affected accelerator target;
- exact shape/dtype/target benchmark evidence;
- median, p95, backend-time, invocation, kernel, synchronization, transfer, and
  memory regression gates;
- an explanation assertion proving the intended selection;
- a pipeline or rule-version increment when identical policy inputs can select
  a different executable;
- updated durable evidence and architecture documentation.

Reject evidence when parity fails, traces overflow, target fingerprints differ,
the apparent gain comes only from added synchronization, or a microbenchmark
gain regresses the representative model corpus.

## Extending optimization

Preserve the compiler phase order:

1. semantic validation and target-independent transformation;
2. candidate discovery from Program facts and capabilities;
3. legality and numerical-policy filtering;
4. deterministic region selection;
5. layout and materialization planning;
6. final implementation selection;
7. dependency scheduling and resource/lifetime planning;
8. backend preparation;
9. immutable executable freeze;
10. runtime execution and optional observation.

Discovery must not read provisional implementation state, prepare handles, time
kernels, or mutate the Program. Runtime observation must not re-plan, select a
kernel, or modify policy. Program, Executable, Session, and backend ownership
remain separate and join only through stable identities and provenance.

When two genuinely equivalent implementations exist, bounded autotuning may be
added as a separate cache-isolated profile. It must have shape/dtype guards,
output-parity validation, warmup/sample/time budgets, stable tie-breaking, a
target-fingerprinted cache, and explanation evidence. Strict determinism must
continue to disable measurement-driven selection.
