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
import { Session } from "affon:compute"

const session = new Session({
  device: "cuda",
  determinism: "strict",
  telemetry: {
    backendTiming: true,
    hardwareMetrics: true,
  },
})
```

Runtime measurements are published through `std:telemetry`, alongside user
spans and metrics. Backend timing and hardware metrics are independent.
Unsupported metrics are reported as unavailable, never as measured zeroes.

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

Compute execution spans are children of the active user trace. They contain
events for executable steps, backend invocations, CUDA kernels or Metal command
buffers, synchronization reasons, timing scopes, hardware-metric availability,
and metric samples. Stable `compute.*` counters and gauges cover executions,
invocations, backend time, synchronization, and storage.

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
