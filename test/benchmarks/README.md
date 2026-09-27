# Compute performance coverage

Model-independent, curated operation and sequence comparisons against PyTorch,
with correctness as a prerequisite. This suite lives at the compute API boundary;
no model files, HF adapter behavior or application requests are involved.

Current status and restart instructions: [optimization checkpoint](../../docs/core/compute-optimization-checkpoint.md) (2026-09-27).

## Run

Use a ReleaseFast Affon build and a Python environment containing `torch` and
`safetensors`. Run from any directory; the harness resolves its repository root.
No dependencies or fixtures are downloaded by the runner.

```sh
/path/to/python test/benchmarks/run.py \
  --affon /path/to/release/bin/affon \
  --compute /path/to/compute \
  --output /tmp/compute-bench-new
```

Metal/PyTorch MPS is the default; `--device cpu` selects both CPU implementations.
The binary must include the shared kernel context and internal graph execution
bridge for scoped sequence cases. Output directories must be fresh. The live
Compute `OpTag` enum supplies the operator inventory; new tags automatically appear
as uncovered until cases are added. Changes to the inventory also invalidate an
older regression baseline, so coverage growth cannot silently pass. This inventory does not enumerate fused
internal kernels or all public convenience functions.

Outputs include:

- `README.md`: per-case latency and Affon/PyTorch ratio (above one means slower).
- `summary.json`: coverage inventory, raw sample averages, median/p90, group
  medians, timing resolution, environment and binary/suite hashes.
- Four worker reports/logs and shared input/reference fixtures for reproduction.

The first saved snapshot is under `reports/metal-initial`. It is a local
measurement, not a universal performance guarantee or a parity claim.

## Cases and coverage

`cases.py` is the explicit case registry. The initial set covers f32 unary and
binary operations, suffix broadcasts, transposed inputs, axis/all reductions,
softmax, normalization and matmul. Matmul cases include rank-two/rank-three
variants, a batched transpose, and shapes on either side of current layout-aware
MPS selection thresholds. The historical baseline also retained a separate contiguous entry point; it is
now normalized through the common dispatcher.

Dependent relu/add sequences have separate eager and scoped cases. Lengths of
4, 20 and 35 produce 8, 40 and 70 encodings: below and across the 32-operation
chunk limit. Operator coverage counts exclude sequences, so exercising an op
inside a chain does not falsely claim an isolated benchmark.

Every run reports uncovered OpTags. Measured means these explicit cases passed,
not every shape, dtype or implementation passed. Metal f32 matmul selection is observed through native dispatch events in a separate
diagnostic worker. Compiled matmul epilogues are checked with fusion-hit counters.
Other operations remain **not instrumented**. Forced implementation comparisons, other
dtypes, additional strided layouts, branch patterns, memory pressure, other
backends and fused-kernel inventory are outstanding coverage dimensions. Add
cases as dispatch is consolidated; don't infer MPS/custom selection from shape.

## Measurement contract

CPU PyTorch generates deterministic shared safetensors inputs and complete output
references. Both workers check every output element with `atol=rtol=1e-4` before
accepting timings. Data transfer, input views, correctness readback and file I/O
are outside timing. Outputs are allocated by the operation in both frameworks.
Inputs include positive-domain log/sqrt and nonzero divisors. GELU uses the tanh
approximation on both sides. All-element reductions adapt PyTorch scalars to
Affon's one-element output shape. Arbitrary-axis normalization uses PyTorch
`layer_norm` with axis-moving views, not a hand-written arithmetic decomposition.

Both use inference/no-grad execution. PyTorch MPS CPU fallback is disabled. An
isolated eager operation is synchronized before the next invocation. Eager
sequences synchronize each op; scoped sequences synchronize once on return.
Affon may also drain at internal chunk limits. This deliberately measures both
completion contracts; eager-chain timing is not PyTorch's usual queued execution.
CPU has no GPU scope and the two sequence modes have equivalent semantics.

Fresh processes run sequentially in Affon/PyTorch/PyTorch/Affon order, with five
warmups and three samples of 100 invocations per case in each process by default.
PyTorch CPU threads are fixed to one. No concurrent jobs are started by the runner;
the host must be kept quiet by the operator. Thermal/background noise remains
possible. Group medians help expose run-order drift.

Affon uses its millisecond clock, amortized over repetitions. Any sample shorter
than 10 ms makes that case's ratio unresolved; increase `--repeats` to resolve it.
Use `--view-repeats` to independently increase repeats for cheap host views; it
defaults to `--repeats`, and each result records its actual repeat count.
At that minimum, clock quantization can still be material. Raw block totals are
preserved. p90 is across repeat-block averages, **not individual-call tail
latency**. Native command timing and legacy implementation-forcing environment
variables are removed from worker environments.

## Regression tracking

Competitive ratios show distance from PyTorch. Regression checks compare Affon
against a previous Affon result on the same host configuration and suite:

```sh
/path/to/python test/benchmarks/run.py \
  --affon /path/to/new/bin/affon --compute /path/to/compute \
  --output /tmp/compute-bench-next \
  --baseline /tmp/compute-bench-new/summary.json
```

The default regression tolerance is 15% on median; override with
`--regression-tolerance`. This is a configurable alert threshold, not a universal
parity target. A detected regression exits 2 after saving reports. Incompatible
suites/environments, missing cases, incorrect results or unresolved timers reject
the baseline comparison. Reproduce a flagged regression before attributing it to
code; this local harness cannot control host scheduling.

Run reporting checks with:

```sh
python3 -m unittest discover -s test/benchmarks -p 'test_*.py'
```

There is no CI scheduler or dedicated benchmark host configured by this change.
The executable runner and saved baseline make later scheduled regression runs
possible without depending on model regressions to discover a gap.


## Implementation visibility

The runner starts a separate correctness-checked Affon process with
`AFFON_METAL_DISPATCH_TRACE=1` before timing. It saves `dispatch.json` and attaches
per-case event deltas to the summary/table. The four timing workers have tracing
explicitly disabled; the Affon timing worker rejects leaked dispatch events.
An older native binary without events fails the Metal matmul diagnostic instead
of silently inferring the implementation from a shape.

Events come from the native selection sites, not a duplicate selector:

| Event suffix | Meaning |
|---|---|
| `mps_layout_eligible` | Layout-aware shape/stride/descriptor checks accepted MPS |
| `custom_layout_shape_policy` | Size policy (or empty batch) bypassed MPS |
| `custom_layout_unsupported_strides` | Layout cannot be represented by this MPS adapter |
| `custom_layout_descriptor_bounds` | Alignment or conservative descriptor bounds bypassed MPS |
| `custom_strided_matvec_f32` / `custom_strided_matmul_f32` | Actual custom pipeline choice |
| `mps_contiguous_f32` | Entered the legacy contiguous MPS implementation |
| `custom_offset_f32` / `custom_offset_many_f32` / `custom_tiled_f32` | Entered a direct custom implementation |
| `contiguous_mps_error_fallback` | Legacy MPS attempt failed and dispatch fell back to custom |

Counts are decision/entry events, not GPU command counts. One call can produce a
reason plus an implementation event, or an MPS attempt plus a custom fallback.
Events precede completion; correctness success is reported independently. No
selection rules or error behavior change in this instrumentation step. Integer matmul paths are outside this instrumentation scope. Fused epilogues use
the separate fusion-hit counters described below.

Current cases exercise size-policy custom selection and eligible layout-aware MPS.
Other instrumented reasons/entries are not claimed as exercised just because
instrumentation exists. In particular, both rank-two and rank-three cases in the
saved diagnostic reach the layout-aware path; the historical diagnostic did not exercise the older contiguous entry. The
consolidation described below now removes that separate policy.

The updated harness changes its suite fingerprint. Keep `metal-initial` as
historical evidence; use the new `metal-dispatch` report as the baseline for this
version rather than bypassing baseline compatibility checks.

### Consolidated ordinary f32 matmul

Ordinary contiguous/offset/batched/strided entries now share the layout-aware
selector described in [Metal matmul dispatch](../../docs/core/metal-matmul-dispatch.md).
Legacy contiguous-MPS, direct-offset/tiled and MPS-error-fallback events in the
historical table above no longer occur in the consolidated implementation. The
layout eligibility/rejection and custom strided pipeline events remain current.
The native direct-entry regression covers the previously separate entry points;
the model-independent timing cases and harness remain unchanged.


### Matmul batching cases

The suite now also includes four 20-step matmul/ReLU chains: custom-sized (2×16)
and MPS-sized (64×128), each in eager and scoped modes. Every step computes matmul
then ReLU. Weights mix an identity and an averaging matrix, keeping repeated
results bounded while preserving a real dependency. Both workers use identical
fixtures. PyTorch eager mode synchronizes each operation; scoped mode completes
the whole chain. Affon's scoped mode may split at its internal chunk limit.

The diagnostic report records scope statistics separately from timings. With
matmul batching, each of these chains encodes 40 operations in two submissions.
The previous implementation batches only the 20 ReLU operations, with matmul
acting as a synchronous boundary. The suite fingerprint changes with these
additional cases; the batching report includes fresh before/after runs of the
same expanded harness rather than comparing incompatible historical baselines.


### Compiled epilogue cases

The epilogue checkpoint contained 71 cases: 57 isolated cases covering 23 OpTags,
six relu/add chains, four matmul/ReLU chains and four compiled epilogue chains. The epilogue
cases run 40 dependent matmul+bias or matmul+bias+GELU steps, in eager and scoped
modes. Bias has the full output shape. Compilation happens outside timing, and
the diagnostic requires exactly 40 matching fusion hits per chain. Scoped native
execution records 40 encodings in two submissions. These counters verify fusion,
not MPS/custom dispatch policy.

PyTorch evaluates the equivalent compound expression eagerly (tanh GELU), without
`torch.compile`. Eager mode synchronizes after each compound step; scoped mode
synchronizes after the whole chain. Ratios therefore compare these specific
execution contracts, not compiled-framework parity.

Use `--filter '^fused-matmul-'` to run only this family. Filtered runs retain the
full OpTag inventory, so sequence-only reports correctly show zero isolated-op
coverage. Fresh before/after runs must use the same filter and harness version.
The [epilogue report](reports/metal-epilogue-batching/NOTES.md) records focused
performance results and full-suite correctness checks.


### Reduction, indexing and layout matrix

The coverage-expansion checkpoint defines 132 cases, including 14 sequences. On the validated
Metal/PyTorch MPS configuration, 129 pass and three declared unsupported
combinations remain visible. CPU runs 131 cases with one unsupported combination.
Isolated OpTag coverage grows from 23/66 to 45/66. This is API-boundary coverage,
not proof that every internal kernel or dispatch route was exercised.

New cases include min/max, population variance/std (`correction=0` in PyTorch),
and argmin/argmax across all elements or either axis, on contiguous and transposed
inputs. Gather and index_select exercise both axes with deterministic repeated
and reordered i64 indices. Layout cases include contiguous conversion, reshape,
permutation, stepped slices, squeeze/unsqueeze, concatenation and stacking.
Affon checks output dtype for every case and integer index outputs exactly;
PyTorch also checks output dtype and values. Primary numeric inputs remain f32: i64 index coverage does not imply
general i64 arithmetic coverage.

`summary.json` schema 2 retains per-case and per-operation variants: logical input
shapes, fixture shapes, input layouts and dtypes, output shape/dtype, axis and
other parameters, eager/scoped mode, per-framework completion contracts and observed native
implementation events. Sequence variants do not inflate isolated OpTag coverage.
Uninstrumented implementations remain explicitly unobserved. GPU route coverage
cannot be inferred from shape or from a successful output.

The public `transpose` API uses permutation internally, so these view cases count
as `permute`; they do not claim the separate internal `transpose` OpTag.
Both workers prepare transposed input views before timing.
Supported layout views measure host API/view construction, without a PyTorch GPU
fence per invocation. Device operations still complete per call. Affon materializes stepped slices
while PyTorch returns a view; the report records this completion difference.
Contiguous
conversion includes each framework's actual behavior, including PyTorch's no-op
on an already contiguous input; it is not a forced copy bandwidth benchmark.

Declared unsupported cases are excluded before timing and retained in the report:

- Transposed reshape: Affon's eager API requires contiguous input; PyTorch would
  copy. The harness does not silently add a contiguous conversion.
- Transposed all-element argmin/argmax on MPS: PyTorch 2.9 rejects these views.
  Their CPU comparisons pass. Revisit this declared limitation on PyTorch upgrades.

Reports rank resolved, correct cases by excess median milliseconds over PyTorch,
with isolated operations and sequences shown separately. Ratios remain available;
large ratios for very short view calls are not large GPU bottlenecks. Unresolved
clock samples never enter the ranking. The saved
[coverage expansion report](reports/metal-coverage-expanded/NOTES.md) contains
full correctness evidence and fresh measurements of the added cases. Its timing
contract and expanded registry change the suite fingerprint, so older snapshots
remain historical evidence rather than compatible regression baselines.


### Concatenation and stacking axes

Six additional cases exercise distinct inputs at non-leading axes: cat axis 1
and stack axes 1/2, each with contiguous and transposed [4,7] fixtures. Together
with the existing axis-0 cases, these expose per-row submission costs without
requiring model workloads. The dense-join checkpoint has 138 declared cases: 135
supported Metal comparisons and 137 CPU comparisons, still covering 45/66 OpTags.
See the [join report](reports/metal-joins/NOTES.md) for matched before/after runs.


Four additional cat/stack cases use [6,9] fixtures narrowed to interior [4,7]
views, with and without transposition. Both workers construct these views before
timing. Reports distinguish offset and transposed-offset inputs. The strided-join checkpoint
has 142 declared cases: 139 supported Metal comparisons and 141 CPU comparisons,
still covering 45/66 OpTags. The
[strided join report](reports/metal-strided-joins/NOTES.md) records their measurements.


### Whole-reduction views and batching

Fourteen additional cases cover sum, mean, min, max, population variance and std
on interior offset views and larger transposed [128,127] fixtures, plus small
transposed sum/mean cases. Both workers prepare views before timing. The reduction-batching
checkpoint has 156 declared cases: 153 supported Metal and 155 CPU comparisons,
covering 45/66 isolated OpTags and retaining 14 sequence cases.

The [reduction report](reports/metal-reduction-batching/NOTES.md) includes matched
eager timings and a separate correctness-checked command/allocation probe for
40 scoped reductions. These diagnostics do not establish every shape or dtype,
or claim a matched PyTorch scoped-sequence timing.


### Parallel reduction boundaries

Thirty additional cases cover six whole f32 reductions at 17, 255, 256, 257 and
65,537 elements. The single-threadgroup checkpoint contains 186 declared cases: 183 supported
Metal comparisons and 185 CPU comparisons, still covering 45/66 isolated OpTags
and 14 sequences. Together with the offset/transposed cases, these check tiny
inputs, either side of parallel selection, partial final lane workloads and a
larger dense workload. The [parallel reduction report](reports/metal-parallel-reductions/NOTES.md)
records matched measurements and separate exceptional-value/accuracy checks.


### Whole-reduction scaling sweep

Fifty-four additional cases compare sum, mean, min, max, variance and std on dense,
transposed and interior-offset inputs. Logical shapes are [256,255], [1024,1023]
and [2048,2047], reaching 4,192,256 elements. Views are prepared outside timing;
operation-required packing and reduction workspace are included in timing.
That checkpoint has 240 declared cases, with 45/66 isolated OpTags and 14
sequences. All 237 supported Metal and 239 CPU comparisons pass. The sweep exposed
12 pre-existing CPU variance/std accuracy failures on large inputs; shared f64
accumulation for f32 inputs fixes them. The report preserves the failing baseline
and independent audit. No cases were excluded or given looser tolerances. See the
[multi-threadgroup report](reports/metal-multigroup-reductions/NOTES.md) for matched
before/after results and retained limitations. Exact selection boundaries and
uneven partial groups also have native correctness coverage.


### Axis-reduction scaling and batching

Thirty additional cases exercise sum, mean, min, max, population variance and std
on long inner/outer axes, rank-three inner/middle axes, and transposed inputs.
The registry now declares 270 cases: 267 supported Metal and 269 CPU comparisons
pass, covering 45/66 isolated OpTags and 14 sequence cases. These counts describe
API cases, not exhaustive implementation, dtype or shape coverage.

The [axis-reduction report](reports/metal-axis-reductions/NOTES.md) records 58
matched before/after cases, native tests around the 256-element parallel boundary,
480 exceptional-value comparisons, and separate command/allocation diagnostics.
Dense axis reductions participate in automatic graph batching. At that checkpoint, transposed inputs
still incurred packing outside the reduction invocation; the extension below
removes that submission boundary for supported f32 views. Full-matrix one-repeat runs verify correctness; use the matched targeted
runs for performance conclusions.


### Packed axis-reduction views

The [packed-axis report](reports/metal-axis-views/NOTES.md) reuses the same 58-case
axis sweep to isolate moving packing into the reduction invocation. It includes
separate eager/scoped command and allocation probes, native tests for offset,
stepped, broadcast and permuted views, and a physical-memory-budget drain test.
The benchmark registry and coverage counts remain unchanged; full Metal/CPU
smoke runs check regressions after the shared descriptor refactor.
