# CPU Program routing survey

This survey records which representative public CPU Programs compile to the
Cyna graph engine and which still fall back to Zig. It answers one narrow
question: where should the next portable Program-lowering effort go?

The survey compiles Programs without executing them. Set
`AFFON_TEST_CYNA_SESSION=1` so the public Session permits Cyna routing, then run:

```sh
AFFON_TEST_CYNA_SESSION=1 ./zig-out/bin/affon \
  test/benchmark/cyna-routing/route-representative.ts
```

The output is deterministic JSON. It counts Programs overall and by corpus,
groups Zig fallbacks by category, reason, operation, option, and corpus, and
marks the largest group that could be removed by portable lowering. A
compile-only result does not establish numeric agreement, runtime performance,
or complete model coverage.

## Current result

| Programs | Cyna | Zig | Cyna routing |
| ---: | ---: | ---: | ---: |
| 120 | 120 | 0 | 100% |

| Corpus | Programs | Cyna | Zig |
| --- | ---: | ---: | ---: |
| Authored operation and model-like Programs | 107 | 107 | 0 |
| Imported ONNX fixture Programs | 3 | 3 | 0 |
| Native model-family Programs | 10 | 10 | 0 |

The capture, invocation-timing, and kernel-timing Programs now route to Cyna.
Cyna emits bounded node observations and reports the requested synchronous
host timing scope. On CPU hardware metrics and NVTX profiler markers are
unavailable; Cyna reports that availability in its runtime evidence while
keeping execution on its own route.

The nondefault `searchLimits` Program also routes to Cyna. Affon passes all
seven public limits through the configured graph ABI. Cyna applies the
structural state, expansion, complete-evaluation, physical-work, ready-work,
peak-active-byte, and per-offer invocation-scratch budgets. It records the
applied values and truncation in `explain()` and runtime evidence schema 10.
The runtime report also separates preparation attempts and rejected offers
from session memory, prepared cache bytes, invocation and persistent scratch,
CPU pool capacity, and host readback. `highest_impact_portable_lowering` is
`null` because this corpus has no Zig fallback.

The public session cap is aggregate admission, not a limit on one allocation.
Before a new tensor or run is admitted, Cyna trims inactive cached CPU buffers
and counts the backend's total reserved pool plus prepared graph, persistent
scratch, invocation scratch, and staging demand. Runtime evidence keeps those
components separate.

CPU physical selection now has real alternatives. Generic elementwise work can
choose the `cpu_elementwise` implementation, dense conversion demand can choose
the portable `cpu_contiguous` configuration, and deterministic costs compare
transfers, synchronizations, dispatches, library calls, estimated work,
temporary bytes, and target score. A model ID keeps unrelated cost domains in
discovery order.

Preparation can reject an instruction or conversion and reselect up to four
times with a shared 131,072-check offer budget. Public `physicalWork` controls
combination exposure rather than that budget. Failed resources are released,
persistent scratch failure is terminal, and the prepared-cache key distinguishes
the executable's original selection from the graph that actually prepared.
The compiler differential compares 896 plans with zero mismatches.

## Corpus

The authored registry exercises public `affon:core` Programs across:

- arithmetic, comparisons, rank-zero values, singleton and broadcast shapes
- unary operations, activations, clamp, and softmax/log-softmax on negative axes
- whole and axis reductions, keep-dims, negative axes, extrema, variance, and
  standard deviation
- matmul, batched matmul, dot, quantized matmul, and f64/bf16 variants
- reshape, transpose, permutation, slice, squeeze, unsqueeze, and contiguous
- cat, stack, masked fill, where, indexing, scatter-add, top-k, and one-hot
- normalization, cross entropy, regression and classification losses
- fused model-like chains, including attention and a transformer block with
  layer norm, Q/K/V matmuls, attention, softmax, residual addition, and RMS norm

The imported corpus loads the generic, spatial, and Conv1d ONNX test fixtures
through the public importer. The model corpus authors Programs with the public
GPT-2, BERT, ViT, and Llama family constructors. It includes GPT-2 prefill,
BERT, and ViT in f32 and BF16 parameter storage, plus Llama prefill and decode
in both storage modes. The Programs are compiled, not executed.

Each Session is CPU-only. The runner reads the executable's internal routing
record, disposes the executable, and then disposes the Session. It never runs a
Program and does not count an unsupported Program as a Cyna success.

## Runtime parity

The bounded runtime check has 40 test cases. Each case runs the same Program
in two public CPU Sessions: one lowers to Cyna, while the other requests
`cpuExecution: "zig"`. The check requires the Cyna route to report `lowered`,
the comparison route to report `zig_requested`, and every output to have the same shape, dtype, and finite
values.

| Dtype path | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| f32 | `3e-4` | `3e-4` |
| f64 | `1e-12` | `1e-12` |
| BF16 output | `1/128` | `1/128` |

The BF16 tolerance represents one BF16 relative unit. The f64 result is tighter
than the f32 result because it has substantially more precision. A BF16-stored
GPT-2 model still uses the f32 tolerance because the model constructor casts its
parameters to f32 for arithmetic.

Run it with:

```sh
AFFON_TEST_CYNA_SESSION=1 ./zig-out/bin/affon \
  test test/benchmark/cyna-routing/runtime-parity.test.ts
```

Representative passing Programs are:

| Corpus | Program | Shapes | Output dtype |
| --- | --- | --- | --- |
| Selection | broadcast comparison, where, and masked fill | rank-two inputs with singleton axes | i64 or f32 |
| Indexing | embedding, index select, gather, scatter-add, top-k, and one-hot | small vectors and matrices with repeated or tied indices | f32 or i64 |
| Authored | transformer block with layer norm, attention, residual, and RMS norm | `[2,4,8]` with `[8,8]` weights | f32 |
| Imported ONNX | generic fixture | fixture-defined | fixture-defined |
| Imported ONNX | spatial fixture | fixture-defined | fixture-defined |
| Imported ONNX | Conv1d fixture | fixture-defined | fixture-defined |
| Dtype | exponential | `[2,3]` | f64 |
| Dtype | axis-one sum | `[2,3]` to `[2]` | f64 |
| Dtype | matrix multiplication | `[2,3] x [3,4]` to `[2,4]` | f64 |
| Dtype | exponential | `[2,3]` | BF16 |
| Dtype | axis-one sum | `[2,3]` to `[2]` | BF16 |
| Dtype | matrix multiplication | `[2,3] x [3,4]` to `[2,4]` | BF16 |
| GPT-2 | small f32 prefill | sequence 3, width 4 | f32 |
| GPT-2 | small BF16-stored prefill | sequence 3, width 4 | f32 |
| BERT | small f32 encoder | two tokens, width 4 | f32 |
| ViT | small f32 image encoder | one `3x2x2` image, width 4 | f32 |
| Llama | small f32 prefill | sequence 3, width 4 | f32 |
| Llama | small f32 cached decode | one new token, width 4 | f32 |

These are parity checks, not model-quality or performance claims. The focused
BF16 Programs exercise BF16 arithmetic. The BF16 GPT-2 case exercises BF16
parameter storage through public model constructors, while its arithmetic
remains f32. The models are tiny constructor fixtures, not trained checkpoints.
The ONNX evidence covers all three prepared importer fixtures, but not arbitrary
ONNX graphs. All comparisons use the CPU backend.

For a focused selection and indexing timing probe, run
`AFFON_TEST_CYNA_SESSION=1 ./zig-out/bin/affon test/benchmark/cyna-routing/selection-indexing.ts`
with a ReleaseFast Cyna-enabled build. It checks both routes before reporting
warm Session invocation times for comparison, `where`, embedding, scatter-add,
and top-k. The detailed audit and remaining gaps live in daydream.

## Llama performance

`run.py` builds two ReleaseFast Affon executables: one with Cyna enabled and one
with the normal Zig CPU path. Each executable runs `performance.ts` in its own
process. The benchmark creates a deterministic synthetic Llama, compiles prefill
and cached decode Programs, checks both routes, and compares their logits. Model
construction, Program creation, compilation, state initialization, parameter
creation, input creation, and warmup are outside the timed region. Each sample
times invocation through disposal of every output, with the same parameters and
inputs reused.

The measured workload uses f32 arithmetic with width 128, inner width 344, four
query heads, two KV heads, two layers, context length 256, vocabulary size 1024,
prefill length 64, and cached decode position 128. The weights are generated
deterministically from fixed seeds. This measures runtime speed and numerical
agreement, not trained-model quality.

Run the benchmark with:

```sh
python3 test/benchmark/cyna-routing/run.py
```

Set `AFFON_CYNA_BENCH_SELECTED_PLANS=1` to include each route's selected
physical node order in its runtime report. This adds Cyna source instruction
indices, implementation names, view flags, output byte counts, and exact kernel
counts alongside Zig's selected node origins and implementation names. Plan
capture happens after timing. `summary.json` compares source groups, selected
implementations, conversion counts, selected peak bytes, and observed execution
order, including fused source groups. It also checks that every selected Cyna
physical node has count evidence and publishes the exact aggregate under
`selected_plan_metrics`. Dropped observations stop the capture instead of
reporting a partial count as exact. The Cyna report includes selected peak bytes
under every optimization goal and memory selection search fields under the
memory goal.
Affon's lowerer maps Cyna step origins to Zig Program instruction IDs, including
multi-step operations such as paired top-k. The report retains the Cyna step IDs
as `cyna_step_origins`. Repeated execution of one Program instruction, such as
residual recomputation, still needs a separate comparison rule. Capturing execution order
enables runtime evidence throughout the run, including timed samples, so compare
timings only with runs using the same
capture setting.
Set `AFFON_CYNA_BENCH_OPTIMIZATION_GOAL=memory` to compare memory-goal plans;
the default is `balanced`.

The script prints the output directory and writes `summary.json`, the separate
runtime reports, and build/run logs. It requires local Cyna and compute-core
libraries; pass `--cyna-lib-dir` and `--compute-core-lib-dir` when they are not
in the default development locations.

For a focused f32 matrix check, run `transpose-matmul.ts` with a ReleaseFast
Affon executable and `AFFON_TEST_CYNA_SESSION=1`. It measures a 128×128 matrix
product with a transposed right operand. Set `AFFON_PROBE_TRANSPOSE=0` for the
dense control. Each route warms up, then runs seven samples of 20 invocations;
the script compares every output value and prints JSON timing samples. Set
`AFFON_PROBE_SAMPLES` to a larger count when collecting a CPU stack sample, or
`AFFON_PROBE_SIZE` to compare square matrix sizes. Set `AFFON_PROBE_BATCH` above
one to use a rank-three left input with the same right matrix in each batch.

On 2026-11-19 in New Zealand, an Apple M4 with 16 GB memory running macOS
26.6.2 produced these ReleaseFast results after two warmups and seven timed
samples. The Cyna and Zig routes ran in separate sequential processes with
selected-plan capture enabled. The Cyna route reused unchanged input tensors.

| Backend | Workload | Median (ms) | Standard deviation (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: |
| Cyna | prefill, 64 tokens | 3.852 | 0.012 | 16614.75 |
| Zig | prefill, 64 tokens | 31.365 | 0.026 | 2040.49 |
| Cyna | cached decode, 1 token | 0.372 | 0.003 | 2688.17 |
| Zig | cached decode, 1 token | 1.625 | 0.022 | 615.38 |

Cyna took 0.123 times as long as Zig for prefill and 0.229 times as long for
cached decode. Set `AFFON_CYNA_PROFILE=1` to print per invocation pack, input
creation, graph, and output times plus prepared cache hit and miss counts.
Set `AFFON_CYNA_RUN_PROFILE=1` to time the Cyna execution wrapper, or
`AFFON_CYNA_NATIVE_PROFILE=1` to time native input binding, execution, and
output wrapping. Run each profile switch separately: printing a profile line
inside another measured boundary adds time to that boundary.

The outputs agreed numerically: maximum absolute differences were `7.15e-7`
for prefill and `1.19e-6` for decode, with no values outside the f32 `3e-4`
absolute plus relative tolerance. The selected plans covered all 137 source
nodes, used the same seven dense conversions, reported the same peak active
bytes for both workloads, and had no semantic or source-order mismatches.

The separate selection and indexing probe ran seven samples of 20 invocations
per route. Embedding measured 13.45 µs for Cyna and 7.75 µs for Zig, a 1.74
ratio. Top-k measured 22.15 µs for Cyna and 21.80 µs for Zig, a 1.02 ratio.
Comparison, where, and scatter-add were faster on Cyna in that run. Warm
prepared-cache hits now bind dense inputs in one pass, and top-k executes one
physical node that emits both values and indices. These small fixtures isolate
different host and kernel costs from Llama, and their timings vary with host
activity.

The benchmark runner writes the full samples and build logs to its output
directory. This shape and host do not establish performance for other models,
dtypes, batch sizes, or CPU hardware.

## Decision

No portable Cyna lowering target appears in the authored, imported, or model
sample. Every added Program routes through Cyna. Adding another lowering
rule now would be driven by vocabulary shape rather than a measured fallback.

The bounded execution milestone is now met for the Programs above. The Llama
fixture is faster on Cyna for both measured workloads. Top-k is at parity in
the focused probe, and embedding has narrowed to 1.74 times Zig rather than a
model-level blocker. Compile coverage still does not imply complete numerical,
dtype, or model coverage. Keep looking for repeated lowering fallbacks as the
corpus grows, and extend runtime evidence before turning any compile-only route
into a broader support claim.

Keep the following limits explicit when comparing later runs:

- coverage means this registry, not every valid `affon:core` Program
- these are tiny constructor and importer fixtures, not complete trained models
- runtime parity covers a bounded operation matrix and does not imply complete model support
- one Llama measurement does not establish performance for other shapes, dtypes, or models
- CPU hardware metrics and profiler markers are unavailable rather than
  measured; Cyna kernel timing and kernel counts report per-node synchronous
  host execution
- non-CPU backends and disabled-Cyna configurations are outside this survey
