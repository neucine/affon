# Compute performance coverage

Model-independent, curated operation and sequence comparisons against PyTorch,
with correctness as a prerequisite. This suite lives at the compute API boundary;
no model files, HF adapter behavior or application requests are involved.


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

Save generated runs outside the repository. They are local measurements, not
universal performance guarantees or parity claims.

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



## Dispatch coverage

Use the case registry and generated coverage inventory to inspect shape/layout
and sequence coverage. Dispatch diagnostics distinguish instrumented implementation
choices; not every kernel has dispatch telemetry. Expand registry cases alongside
new supported paths. Current architecture is documented in docs/core.
