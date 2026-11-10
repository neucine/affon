# Compute observability gap audit

Status: selected-run train/infer gates passed on 2026-11-09 for Apertus cached
decode on CUDA, a deterministic GPT-2 inference Program on CUDA and Metal, and
a representative fixed-shape CUDA training step. The baseline observation gate
for beginning performance optimization and parity-gap work is closed for this
workload matrix. Per-allocation and per-release intervals, explicit dependency
readiness, and additional library fan-out coverage remain conditional fidelity
work rather than blockers. This audit covers the Affon Program API and the
linked Compute runtime.

## Existing evidence path

| Layer | Available evidence | Limit |
| --- | --- | --- |
| `Program.inspect()` | Stable semantic nodes, scopes, roles, and provenance | No execution facts |
| `Executable.explain()` | Physical nodes, source mappings, selected offers, costs, and schedule | Predictions and decisions, not observed latency |
| Compute `EvidenceRecord` | Run, invocation, submission, backend timing, independent kernel-count and submission-kind evidence, hardware-metric, synchronization, and storage events | No complete host or device timeline |
| Compute `observe.aggregate()` | Step and source attribution, kernel counts, exact library/custom submission counts where the backend can observe them, timing samples, shared timings, selected physical costs, and cost validation | Availability of submission-kind evidence is explicit per backend; it does not prove kernel fan-out by itself |
| `std:telemetry` | Application spans and a projection of Compute evidence | Metrics discard individual invocation relationships |
| External profiler | Device timeline, driver activity, synchronization, copy and memset bytes, joined automatically to Affon invocation and readback IDs | Remaining gaps are intervals outside marked ranges; dependency state is inferred from API and synchronization coverage |

## Questions a sufficient capture must answer

For one selected, warmed invocation, using a fixed executable and input
contract:

1. Account for elapsed wall time by application phase, Affon bridge, Compute
   preparation and scheduling, backend submission, synchronization, readback,
   and cleanup. Report time outside these buckets as unknown. Host and device
   intervals can overlap; never sum them as though they were disjoint.
2. Link each backend invocation and its timing to its physical node and source
   instructions. Mark fused or batched timing as shared rather than dividing it
   across nodes without evidence.
3. Explain device active intervals, gaps between them, and the dependency or
   host delay preceding each gap. Aggregate kernel time alone is insufficient.
4. Distinguish actual storage allocation, pool reuse, transfer, materialization,
   and planned temporary bytes. Attribute measured bytes to a run and physical
   node where the runtime can establish ownership.
5. State timing source, coverage, dropped records, observation overhead, and
   unavailable hardware counters. A missing sample is never a zero duration.
6. Correlate an external device trace with the same run, submission, and
   invocation IDs when backend events cannot answer a question internally.

## Current gaps and next evidence changes

| Gap | Current state | Required change |
| --- | --- | --- |
| Host submission cost | Backend timing and total run spans exist, but no per invocation host call duration | Opt-in `invocation_host_timing` evidence keyed by invocation ID; first slice implemented in Compute and projected into Affon telemetry |
| Other host work | Compute records setup, execution, output staging, synchronization, and output publication when backend timing is enabled. Opt-in bridge timing adds Affon run, input binding, Compute call, output wrapping, and readback spans. The causal inference runtime adds spans for decode input construction and cleanup, logit selection, prior-cache release, and output cleanup. Readback separately measures device mapping/synchronization, native conversion or repacking, and JavaScript copying where applicable. The warmed Apertus, GPT-2, and training selections verify this hierarchy on the exercised paths. Per-allocation and per-tensor release intervals remain unavailable | Add narrower allocation/release intervals only where the aggregate shows a dominant unexplained ownership boundary |
| Device idle and dependency | Opt-in CUDA NVTX submission ranges carry run, executable, physical step, and invocation IDs. Readback ranges carry tensor, storage, and operation IDs. `tools/correlate-nsys-compute.py` joins marked ranges through CUDA API correlation IDs to kernels, copies, and memsets. For every run and readback gap it separates synchronization, cuBLAS API intent, other CUDA API coverage, and host delay, then reports unmatched activities. | The four-way partition is closed, but `host_delay_ns` remains inferred dependency or unrecorded host state. Add explicit dependency readiness only where operation-transition attribution is insufficient to choose the next experiment |
| Memory attribution | `storage_use` events identify tensor, temporary, and persistent allocations by session storage ID. Run and physical step are recorded when active; persistent allocation happens during compilation. The aggregate maps live persistent IDs to physical nodes or resource slots and separates setup bytes from step bytes. Phase events include boundary live/active bytes and phase-local peaks. Focused tests verify persistent allocation/release IDs and reuse across two Program runs. CUDA memcpy activities inside marked readback ranges contribute observed byte counts, and the warmed captures validate copy-byte and phase-peak accounting on the exercised paths | Add per-allocation/release intervals only for a dominant unexplained ownership boundary |
| Unified capture | `telemetry.captureRecords` creates a bounded per-Session evidence buffer. `Session.resetEvidence()` selects the run window and `Session.runtimeEvidence(executable)` exports Compute's aggregate with physical steps, source IDs, selected costs and implementations, backend counts, timings, phase memory, storage, synchronization, and cost validation. For an optimized Program it exports a gradient/optimizer composite. The Apertus, deterministic GPT-2 inference, and training runners select one warmed trace, record `Executable.explain()`, verify run, executable, invocation, and step identities across available sources, and place classified gaps and phase coverage in `capture.json`. Dropped records remain explicit | Extend acceptance checks to new backends and operation mixes when their evidence semantics differ |

The first model-level acceptance target passed on the RTX 3090. The selected
Apertus warmed decode matched 3,344 invocations across telemetry, runtime
evidence, and Nsight with no invocation mismatches. It observed 9,040 kernels,
48,334 device activities, 651 synchronization activities, zero unmatched device
activities, zero correlation conflicts, zero telemetry misses, and zero runtime
evidence drops. Readback and copy bytes are attributable. A second, smaller CPU
Program should validate event ordering and exact attribution. Baseline and
instrumented runs must use the same executable, inputs, output checks, and
target; measure observation overhead separately.

CUDA timing events are pooled and reused after their measured interval has
resolved. Timing is selectable as `none`, `invocation`, or `kernel`.
Hardware-only observation creates no event pair, invocation timing creates one
pair around the whole invocation, and kernel timing preserves the legacy pair
per observed backend operation. Kernel counts are recorded once per invocation
independently of timing retention.

The first Apertus count audit exposed a real gap: runtime evidence counted
6,799 direct Affon launches while Nsight saw 9,040 device kernels. The missing
2,241 kernels were launched inside cuBLAS and bypassed the driver launch
wrapper. CUDA now advances the same launch accounting after successful
`cublasSgemm` and `cublasGemmEx` submissions. It also records independent
`library_calls` and `custom_dispatches` for each invocation. A rerun reported
exactly 9,040 kernels with 6,799 direct launches and 2,241 kernels reached
through cuBLAS. All 257 selected-run matmul invocations took the cuBLAS path.
This boundary is an observed library operation, not a CUPTI kernel callback; a
future cuBLAS algorithm that fans one API call out to multiple kernels would
need explicit fan-out evidence rather than silently assuming one kernel.

Runtime evidence now preserves the selected compiler cost instead of replacing
it with a synthetic launch count. `cuda_matmul_strided` remains an
implementation family: its compile-time candidate can predict one custom
submission while the CUDA backend reports the actual library/custom split.
Those facts are separate so the selected cost is reproducible and the runtime
result is not rewritten to match a stale prediction. This is intentional:
cuBLAS availability is resolved dynamically, and exact invocation strides are
not available during candidate enumeration. The selected cost therefore stays
a planning fact, while submission kinds are runtime facts. CUDA supplies exact
submission-kind counts; Metal derives them from its selected physical cost;
unavailable counts remain absent rather than being serialized as measured
zeroes.

Runtime evidence schema version 3 publishes `kernel_count_evidence` alongside
the counts:

- `exact` means the selected run has no library-backed operation, so the
  observed direct-launch count is the device-kernel total.
- `custom_dispatches_exact_library_fanout_unknown` means direct launches and
  library API calls are exact, but a library API call may launch more than one
  kernel. The aggregate's `kernel_count` must not be presented as an exact
  device-kernel total in this case.

The r6 CUDA training capture exposed the distinction directly. Its gradient
component reported 490 custom launches, two cuBLAS API calls, and three cuBLAS
kernels in Nsight. The runtime aggregate's count of 492 therefore differed from
the 493 device kernels observed externally. The accounting difference is
retained as library fan-out evidence rather than forcing the aggregate to equal
the profiler total. The optimizer component reported 320 custom launches and
320 kernels with no library calls, so its aggregate is `exact`.

The selected Apertus decode recorded no `cuEventCreate` or
`cuEventDestroy_v2` calls, while the first large prefill still creates the
pool's high-water mark. Compared with kernel-scoped timing, invocation timing
reduced `cuEventRecord` from 10,130 calls to 3,189 and its selected-run CPU
coverage from 9.268 ms to 2.004 ms. The authoritative r3 capture measured
38.847 ms device-active time inside a 55.045 ms device span, leaving a
16.199 ms device gap. Its gap partition was 0.860 ms synchronization,
3.289 ms cuBLAS API, 6.243 ms other CUDA API, and 5.807 ms unrecorded host
delay. These are single-run profiling measurements; use the call-count change
as the deterministic result and repeated uninstrumented runs to establish
latency.

## CUDA timeline capture

Use `new Session({ device: "cuda", telemetry: { profilerMarkers: true } })`
for the selected profiling run. The runtime loads NVTX dynamically and emits
`compute.profiler_marker_availability` inside the execution span. An unavailable
NVTX library leaves execution valid but prevents correlation. Capture with
`--trace=cublas-verbose,cuda,nvtx,osrt`. Plain `cublas` tracing does not export
the API ranges needed for selected-run intent; `cublas-verbose` exports calls
such as `cublasSgemm_v2` and `cublasGemmEx` as `NVTX_EVENTS`. Those ranges
enclose the submitted CUDA launches but have no separate SQLite table or child
correlation ID. Export SQLite and run:

```sh
python3 tools/correlate-nsys-compute.py \
  --sqlite /path/to/profile.sqlite \
  --output /path/to/compute-timeline.json
```

The correlator requires `NVTX_EVENTS`, `StringIds`, kernel activity, and CUDA
runtime or driver API tables. It matches CUDA API calls inside each marked host
range on the same thread, then uses CUPTI correlation IDs for kernels, copies,
memsets, and synchronization. cuBLAS is recognized from host NVTX intervals and
is counted separately from the CUDA launch it submits. Timeline schema version
3 partitions each device gap into:

- `synchronization_ns`: correlated stream or device synchronization;
- `cublas_api_ns`: cuBLAS API intent outside synchronization overlap;
- `cuda_api_ns`: other correlated CUDA runtime or driver API execution;
- `host_delay_ns`: no matching CUDA API coverage.

The partition covers every gap exactly once. `host_delay_ns` identifies host
work or unrecorded dependency delay; it does not prove GPU starvation.
`cublas_api_ns` shows execution of the library entry point, while CUDA API time
shows the launch and event work it performs. These host intervals can overlap
device work. Profiling with thousands of NVTX ranges can be intrusive, so use
it for diagnosis and measure ordinary latency in a separate uninstrumented run.

The final Apertus r3 capture partitioned a 16.199 ms device gap as 0.860 ms
synchronization, 3.289 ms cuBLAS API, 6.243 ms other CUDA API, and 5.807 ms
host delay. Operation transitions account for all 5.807 ms of host delay.
This attribution is contextual: a transition's preceding operation identifies
where the interval follows in submission order, not the dependency that caused
the delay.

Invocation, execution, and readback markers use independent
`nvtxRangeStartA`/`nvtxRangeEnd` handles. The initial implementation used a
thread-local push/pop stack. Nested cuBLAS verbose ranges changed that stack, so
later Affon pop calls could close the wrong range and assign all invocations in
a run nearly the same end time. That produced false synchronization and cuBLAS
correlations. Independent handles keep each Affon interval bounded to its own
start and end; execution ranges join setup copies to the run without merging
invocation timelines, and readback ranges cannot enclose later executable runs.

Host delay by preceding operation led with `mul` 0.927 ms, `cat` 0.663 ms,
`add` 0.643 ms, `cast` 0.575 ms, `matmul` 0.513 ms, and `mean_axis` 0.504 ms.
Submission cost was concentrated differently:

| Operation | Calls | cuBLAS calls | Direct launches | Submission time |
| --- | ---: | ---: | ---: | ---: |
| `matmul` | 257 | 2,241 | 0 | 11.244 ms |
| `cat` | 64 | 0 | 2,560 | 4.931 ms |
| `mean_axis` | 129 | 0 | 1,345 | 3.612 ms |
| `mul` | 707 | 0 | 707 | 1.275 ms |
| `add` | 481 | 0 | 481 | 0.817 ms |
| `cast` | 386 | 0 | 386 | 0.685 ms |

These are observations for one warmed decode, not stable latency rankings.

## CUDA training capture

The training runner in
[`test/benchmark/training/`](../../test/benchmark/training/README.md) executes
one warmup and one selected fixed-shape optimizer step. The selected
`Executable.run()` contains two Compute executions: the gradient executable
produces the primal forward output and backward gradients, then the native
optimizer executable updates parameters. `Session.runtimeEvidence()` exports
both as a composite instead of pretending one executable owns the complete
step.

The authoritative r6 artifact is
`/home/chaoyang/Private/affon-observation-capture-20261109/training-cuda-r6-20261109`.
It contains the worker report, telemetry records, capture status, Nsight report,
SQLite export, and correlated timeline. The selected capture had 5,584
telemetry records, no missed console records, exactly one selected training
trace, and existing runs 2 and 3 capping the process-wide evidence window so the
selected step could be exported without unrelated records.

All 1,976 device activities joined to marked executions. There were no
unmatched device, synchronization, or cuBLAS activities and no conflicting
correlation IDs. The tool classified 343 synchronization activities outside
Affon-marked ranges separately; 19 cuBLAS ranges included 15 lifecycle calls
that were not selected-run compute.

| Component | Invocations | Custom kernels | cuBLAS calls | cuBLAS kernels | Device active | Device span | Device gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient plus primal forward | 176 | 490 | 2 | 3 | 1.976 ms | 4.883 ms | 2.907 ms |
| Optimizer update | 320 | 320 | 0 | 0 | 1.478 ms | 9.332 ms | 7.854 ms |

The gradient gap comprised 309 ns synchronization, 50.737 us cuBLAS API,
771.159 us other CUDA API, and 2.085 ms host delay. The optimizer gap comprised
644 ns synchronization, 1.589 ms CUDA API, and 6.265 ms host delay. These
single-run values support attribution, not a throughput claim. Host delay
remains an inferred dependency or unrecorded host state; operation-transition
attribution narrows its location.

The selected outer training trace covered 33.065 ms of a 41.569 ms wall
interval through the two executable runs. A further 8.504 ms was unattributed,
for a coverage ratio of 0.7954. This is visible remaining time, not an implicit
zero. The direct evidence covers the gradient/optimizer components, but it does
not yet explain every interval owned by the outer training workflow.

## Deterministic inference Program capture

`profile-inference-program.py` runs one warmed fixed-shape GPT-2 inference
Program with deterministic synthetic f32 weights. The Program exercises token
and position embeddings, two transformer blocks, layer normalization, GELU,
masked attention, softmax, matmul, projection, selected logit readback, and
native token selection. It is a backend and accounting probe, not a pretrained
model quality or throughput result.

The authoritative CUDA artifact is
`/home/chaoyang/Private/affon-observation-capture-20261109/inference-program-cuda-r1-20261109`.
Its selected run matched 67 invocations and 338 custom kernels across
telemetry, runtime evidence, and Nsight. Runtime evidence is schema 3 with
`kernel_count_evidence: exact`; there were no library calls. All 713 device
activities and 87 synchronization activities joined without conflicts. The
15 cuBLAS ranges were lifecycle calls outside selected compute and were
classified separately.

The CUDA selected run measured 781.954 us device-active time inside a 6.150 ms
device span. Its 5.368 ms device gap comprised 357.853 us correlated CUDA API
time and 5.010 ms inferred host delay, with no synchronization or cuBLAS API
inside the selected execution. The selected outer trace covered 7.496 ms of a
7.985 ms wall interval, for a ratio of 0.9387.

The same Program passed the internal selected-run checks on Metal: 770
telemetry records, no misses, one execution, 64 invocations, 64 custom kernels,
schema 3 exact counts, no dropped events, and matching compiled/runtime
executable IDs. The selected trace covered 2.855 ms of 5.081 ms wall time and
reported the remaining 2.226 ms as unattributed. Metal has no joined external
device timeline in this facility, so its result validates internal telemetry,
identity, counts, and phase coverage rather than Nsight activity correlation.
The retained artifact is
`/Users/chao.yang/Private/affon-observation-capture-20261109/inference-program-metal-20261109`.

An attempted Apertus cached-prefill capture did not produce an observability
artifact. The 320-capacity candidate was stopped after 18 minutes with memory
near the 30 GB host limit, and a capacity-8 retry exceeded a 10-minute diagnostic
budget. That behavior belongs to prefill planning and resource use, not to a
missing telemetry join. The successful deterministic GPT-2 Program supplies
the different inference mix for the observation gate without presenting the
incomplete Apertus prefill attempt as a pass.

## Sufficiency decision

The Apertus decode observation gaps that blocked performance work are closed:
run identity, invocation identity, physical step identity, exact kernel count,
cuBLAS representation, device activity, gap partition, host submission time,
memory ownership, and operation transition attribution now join in one report.
Use this report before changing kernels, planning, or parity-test performance.

The representative training acceptance check also joins execution identity,
component ownership, invocation and physical-step identity, custom/library
accounting, device activity, synchronization coverage, phase memory, and outer
phase coverage. It proves the facility works for a second training-shaped
workload on CUDA.

The deterministic GPT-2 inference Program also passed on CUDA and Metal. On
CUDA it joins telemetry, runtime evidence, and Nsight; on Metal it validates
telemetry, runtime evidence, exact counts, identity, and phase coverage. Together
with training, these close the baseline train/infer observation gate for the
exercised matrix. Performance optimization and parity-gap work can now start
without waiting for every specialty path to be instrumented.

The remaining work is conditional rather than blocking:

1. **New planner inputs:** validate actual submission kinds whenever a new
   backend, quantized path, or library operation is used for planning. Preserve
   `kernel_count_evidence` so unknown library fan-out never becomes an exact
   device-kernel claim.
2. **Dependency diagnosis:** add explicit dependency readiness only when a
   comparison cannot distinguish host scheduling from GPU dependency delay
   using operation-transition attribution and the existing correlated timeline.
3. **Memory diagnosis:** add per-allocation and per-release intervals only where
   aggregate phase and ownership evidence identifies an unexplained boundary.
4. **Architecture constraint:** keep backend metrics, exact counts, profiler
   markers, and host/device timelines as distinct evidence sources. Shared
   capture and export tooling must preserve each source's coverage and error
   semantics.

The baseline gate is sufficient for the next optimization cycle. It does not
claim that every backend path, hardware counter, or specialized library has
already been exercised.

The 2026-10-04 Compute observability audit established a working event and
aggregation foundation. Its completion claim covered the benchmark corpus and
disabled-profiling overhead, not the broader model-level sufficiency criteria
above.
