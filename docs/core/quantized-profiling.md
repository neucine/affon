# Native quantized inference: profiling and optimization

## Follow-up: packed matrix loader (version 4)

The controlled M128 comparison supersedes the version-3 M128 GPU numbers below.
Same M4, MLX 0.32.4, resident M128/N2048/K1024, 8 bits, group 64, F32 input,
BF16 metadata, F32 output. Five rounds rotate/reverse engine order, each with a
2,000 ms warmup and 500 retained samples. No other tests/builds run during this
comparison. The same native command-buffer timestamp interposer measures both
engines. Full QMM is evaluated; construction, compilation and host conversion
are excluded. Each child checks output values against an independent formula
before warmup. Raw samples, ordering and binary/source hashes are saved in
`test/benchmark/results/quantized-matched-v4-m4.json`.

| Engine | Median of round medians, ms | Round median range, ms | Relative MAD |
| --- | --- | --- | --- |
| Affon version 3 control | 0.55519 | 0.55260–0.57706 | 0.20% |
| Affon version 4 | 0.36244 | 0.36060–0.36406 | 0.31% |
| MLX compiled | 0.21454 | 0.21000–0.21500 | 0.21% |

GPU command time drops **34.7% (1.53×)**. The absolute gap shrinks from
0.34065 to 0.14790 ms, **56.6%**. MLX remains **1.69× faster**. These are operator
GPU command durations, not isolated instruction timings or end-to-end latency.
Individual samples still contain shared-system outliers; the round medians are
stable. No reduction in arithmetic precision is used to obtain the gain.

### What MLX does on this M4

Inspected the exact installed version's official source tag
[v0.32.4, commit 36969c0](https://github.com/ml-explore/mlx/tree/36969c094354086e15d2a327a9c50feb20b9cf11):

- [`QuantizedMatmul::eval_gpu`, `qmm_splitk`, `qmm`](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp)
  make matrix inputs row-contiguous. For this shape, split-K chooses two
  partitions from the 32×32 output grid to target 512 threadgroups. Each group
  uses 128 threads and K32 tiles. A second dispatch sums F32 partial outputs.
- [`QuantizedBlockLoader` and `qmm_t_impl`](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/quantized.h)
  load packed data in contiguous blocks, reuse metadata, stage row-major padded
  tiles, and transpose weights during SIMD matrix multiplication. Steel uses
  explicit per-lane matrix fragments.
- [`is_nax_available`](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/device.cpp)
  requires GPU generation >=17 (>=18 for architecture suffix p); this M4 is
  applegpu_g16g. Therefore its newer NAX kernels are not the relevant path here.
  Dispatch details above are inferred from the exact source and measured shape,
  rather than a captured MLX pipeline label.

### Retained changes

The complete-tile path is now `metal_quantized_matmul.matrix32_packed`, version 4:

1. Decode each U32 once, with one scale/bias load per word. Store adjacent
   decoded values into a row-major shared tile. All decoding is native Metal.
2. Use a 32×32×32 tile, 128 threads/four SIMD groups, four F32 accumulators per
   SIMD group, and direct device output stores. Shared storage falls from
   16,768 to 9,216 bytes. Matrix loads transpose weights. At M128 this gives
   256 output threadgroups instead of 64, with a smaller per-group footprint.
3. Prepare/cache both dense and strided PSOs. Runtime checks require every input
   to be row-contiguous and its element count to fit U32 before selecting checked
   32-bit dense indexing. The existing signed-stride and byte-offset path handles
   views. The cache key includes the dense flag. No contiguity copies, full-weight
   expansion, extra device scratch or extra dispatch is introduced.

The controlled result measures the combined change; it does not isolate each
change's contribution. Split-K was implemented and correctness-tested as an
experiment, but its extra partial-output/reduction work did not establish an
improvement in Affon's exploratory runs, so it was removed. Explicit fragment
loads were also tried and removed without an established improvement. Those
short experiments had system drift and are not evidence of a universal loss for
these strategies. The remaining 0.148 ms is not yet attributed to one mechanism;
matching MLX's dispatch shape alone was insufficient.

### Verification and reproduction

87 Program tests pass, including all 2/4/8-bit paths, dense/strided cache
selection, offsets, F32/BF16 values, and non-power-of-two group 48 across K tiles.
The checkpoint adapter's new `--matrix` mode validates **all 8,192 outputs** of
M128/N64/K1024 against an independent little-endian Python oracle, using 64
weight rows spread over the real 2,048-row checkpoint. CPU and Metal both pass
at atol/rtol 1e-4. Ordinary sampled checkpoint parity also passes. Type checking,
compute core tests and the non-Metal FFI stub syntax check pass.

```sh
python3 tools/benchmark-quantized-matched.py \
  --before /tmp/affon-qmm-v3 --rounds 5 --warmup-ms 2000 --samples 500 \
  --output test/benchmark/results/quantized-matched-v4-m4.json
python3 tools/test-quantized-checkpoint.py /path/to/model.safetensors --matrix
```

The control binary is a temporary snapshot of version 3. The report records its
hash; reproducing that control later requires preserving it or rebuilding the
version-3 source. The current checkout builds version 4.

## Earlier version-3 investigation (historical)

Measured on Apple M4, macOS 26.6.2, Zig 0.16.0 ReleaseFast, MLX 0.32.4,
2026-10-16. This report supersedes the earlier scalar/readback comparison.
Weights remain packed U32 throughout execution. All decoding and accumulation
remain in Zig/Metal; JavaScript only authors operations and requests results.

## Findings and implemented changes

1. **Host readback did unnecessary work for every element.** `scalar()` mapped
   the backend buffer and recomputed multidimensional coordinates using division
   for each output value. `to_array()` now maps once, carries element offsets
   recursively using strides, and decodes native values from that mapped span.
   Dtypes, noncontiguous views, offsets, rank-zero tensors, and item semantics
   are preserved. This accelerates ordinary tensor readback too.
2. **The GPU decode loop used a runtime 64-bit group division and dynamic dtype
   branches.** Preparation now specializes bit width, group size, input dtype,
   parameter dtype, and strategy using Metal function constants. A group of 64
   becomes a shift; dtype loads specialize to F32 or BF16 widening. PSOs are
   cached by the complete configuration and retained by the device. Compilation
   occurs during preparation; the generic native kernel remains a fallback.
3. **The 32×32 tile repeated decode/load work across too many output blocks.**
   A guarded 64×64×32 path doubles activation/weight reuse. Eight SIMD groups
   hold 16×32 output rectangles in register matrices and store directly to
   device output, avoiding the result staging tile. Padded threadgroup arrays
   use 16,768 bytes. Full M/N tiles and complete K tiles are required; the
   existing strided 32×32 path handles boundaries. Specialized pipeline resource
   limits are checked. No full-weight float expansion or device temporary is
   introduced.

The original 32×32 shared-memory padding was also tested independently; it did
not improve the measured workload and was not retained. The useful wider path
combines increased reuse and direct output stores, so their individual effects
are not claimed separately.

## Separating GPU execution from host work

Three complementary measurements are saved:

- `test/benchmark/quantized/profile-quantized.ts`: synchronous `Executable.run`, then host
  readback, then disposal, each timed separately. Native GPU timestamps are
  collected for these calls. The remaining run time includes validation,
  allocation, bridge/runtime work, driver encoding, queue/wait latency, and
  profiling costs. It is **not pure CPU execution time**.
- `tools/profile-quantized-timestamps.py`: the same profiling-only Objective-C
  interposer on both engines reads public `MTLCommandBuffer.GPUStartTime` and
  `GPUEndTime` after completion. It intercepts no tensor data and changes no
  shaders/encodings. Callback overhead is excluded from the separately run wall
  benchmark. These timings cover GPU command work, including any MLX casts;
  they exclude CPU graph building/encoding, queue wait, host readback and result
  conversion. They are an operator GPU-work measurement, not shader-instruction
  counter attribution.
- `tools/profile-quantized-gpu.py`: Instruments Metal System Trace supplies an
  independent compute-encoder view. Instrumentation changed timings materially,
  especially MLX matrix cases; those exploratory captures are not used as the
  primary cross-engine speed comparison. Only target-process aggregate numbers
  are saved, and raw system traces are removed automatically.

Profiling scripts warm each workload for at least 300 ms, then retain 100 GPU
samples. Host phases use five samples of 100 calls. Date.now resolves about
0.01 ms per averaged host phase; zeros mean below that resolution. All workloads
use resident synthetic `[2048,1024]` weights, 8 bits, group size 64, F32 input,
BF16 scales/offsets and F32 output. Sample distributions are included in the
JSON artifacts. GPU clock/cache and system contention remain variable on this
shared Mac; small-batch GPU results were particularly inconsistent across runs.

### GPU command work (matched timestamp interposer)

| M | Affon before, ms | Affon after, ms | MLX full QMM, ms |
| --- | --- | --- | --- |
| 1 | 0.274 | 0.320 | 0.051 |
| 8 | 0.385 | 0.271 | 0.178 |
| 32 | 0.496 | 0.285 | 0.159 |
| 128 | 1.497 | 0.567 | 0.218 |

M=128 GPU work falls from about 1.50 ms to 0.57 ms (roughly 2.6×), while MLX
remains faster at about 0.22 ms in this run. Low-batch GPU timing is noisy: the
matched interposer showed no M=1 improvement, while native in-phase telemetry
showed 0.119→0.072 ms. Do not claim a precise small-batch GPU speedup from these
runs. The substantial M=128 reduction appears in both methods.

### Affon host phase breakdown

Full-array readback conversion, excluding the completed run:

| M | Readback before, ms | Readback after, ms |
| --- | --- | --- |
| 1 | 0.03 | 0.02 |
| 8 | 0.34 | 0.11 |
| 32 | 1.53 | 0.54 |
| 128 | 6.16 | 1.90 |

At M=128, scalar-completion phase profiling measures run time 1.81→0.87 ms and
GPU work 1.494→0.556 ms. The non-GPU run residual stays near 0.32 ms. Thus the
synchronized-run improvement is mainly GPU work; the large full-readback gain
comes from removing repeated mapping/coordinate calculations. Further generic
host execution optimization was not established by these measurements.

## Matched end-to-end benchmark

`tools/benchmark-quantized-mlx.py` runs engines sequentially with the same data,
packing, dtypes and dimensions, for both 4 and 8 bits. Weights and inputs are
resident; compilation/construction are excluded. Each mode warms for at least
300 ms, then records five samples (30 full-readback calls or 100 scalar-completion
calls each). Affon telemetry is disabled for wall measurements; production code
can use `new Session({ device: 'metal', telemetry: false })`.

The full-readback mode materializes every result as nested host arrays/lists.
The scalar-completion mode evaluates the **entire MLX QMM before** taking the
scalar view outside `mx.compile`; Affon also executes the entire QMM. A slice
inside MLX compilation was considered as a possible measurement confounder.
Forcing complete evaluation did not materially explain the old latency gap;
slice-induced work reduction is not an established cause. The explicit full
materialization boundary removes the ambiguity from the benchmark.

Latest 8-bit medians, milliseconds per call; MLX uses `mx.compile`:

| M | Affon before full readback | Affon after full readback | MLX full readback | Affon after scalar completion | MLX scalar completion |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.667 | 0.533 | 0.293 | 0.550 | 0.267 |
| 8 | 1.867 | 1.200 | 0.759 | 0.610 | 0.543 |
| 32 | 5.067 | 0.933 | 1.437 | 0.490 | 0.535 |
| 128 | 12.667 | 4.767 | 5.776 | 0.980 | 0.740 |

Affon wins full host materialization at the larger sizes in this run, while MLX
retains a synchronized scalar advantage at M=1, 8 and 128. M=32 is close.
Raw before/after wall runs also show substantial system-state variation in the
MLX control, so a single headline end-to-end before/after speed ratio would be
misleading. Keep GPU work, CPU conversion, and total call latency separate.
There is no claim of competitive full-model TTS latency or audio quality.

## Reproduction and evidence

```sh
zig build install -Doptimize=ReleaseFast
# Environment prepared earlier in /tmp/affon-mlx-bench:
/tmp/affon-mlx-bench/bin/python tools/benchmark-quantized-mlx.py \
  --output test/benchmark/results/quantized-metal-profiled-vs-mlx-m4.json
python3 tools/profile-quantized-timestamps.py --engine affon \
  --executable zig-out/bin/affon --output /tmp/affon-gpu.json
python3 tools/profile-quantized-timestamps.py --engine mlx \
  --executable /tmp/affon-mlx-bench/bin/python --output /tmp/mlx-gpu.json
./zig-out/bin/affon test/benchmark/quantized/profile-quantized.ts
```

The timestamp profiler compiles a temporary dylib with xcrun/Clang and uses
DYLD_INSERT_LIBRARIES only for its child process. It does not modify MLX,
Framecept, the checkpoint, or the system Metal framework. Benchmark dependencies
are MLX 0.32.4 and NumPy 2.5.4 in an isolated temporary venv. The pre-change
executable was retained at `/tmp/affon-quantized-before-profile` for controls.

Artifacts under `test/benchmark/results/`:

- `quantized-host-phases-profile.json`: stage samples and native GPU observations.
- `quantized-timestamps-before-profile.json`, `quantized-timestamps-affon-profile.json`,
  `quantized-timestamps-mlx-profile.json`: matched command-buffer GPU samples.
- `quantized-metal-corrected-before-profile.json`,
  `quantized-metal-profiled-vs-mlx-m4.json`: wall samples, MLX eager/compiled
  results, correctness checks, environment, and compiler physical plans.
- `quantized-gpu-*-profile.json`: exploratory Instruments measurements; not the
  primary speed comparison.

Verification: 85 Program tests passed, including 2/4/8-bit SIMD/matrix paths,
partial tiles, complete 64×64 tiles, strided/offset views, BF16/F32 configuration
cache separation, and ordinary tensor readback across all five dtypes. Sampled
checkpoint parity passed on CPU and Metal. Type checking passed. The compute
core suite passed after the native backend changes.

Remaining work is primarily better low-batch vector scheduling and more efficient
large-matrix decode/load and matrix scheduling relative to MLX. Buffer allocation,
queue/wait and bridge work remain in the synchronous call; changing those should
be driven by a dedicated CPU profile rather than attributing all non-GPU wall
time to JavaScript. No decoded weight matrix cache, precision reduction, or
JavaScript decoding was introduced.
