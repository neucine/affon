# Quantized inference: affine packed weights

Status: first native implementation, 2026-10-16. `affon:ops.quantized_matmul`
supports evaluated and formal tensors on CPU and Metal. The CPU kernel is a
scalar reference. The Metal kernel fuses unpacking, affine decoding, and matrix
multiplication in one dispatch. Neither kernel constructs a floating weight
matrix. JavaScript only authors and validates the operation.

## Architecture and scope

Affon owns the public TypeScript API, Program authoring, checkpoint import, and
the JavaScript-to-native bridge. Its path dependency `../compute` owns tensor
specifications, semantic schemas, the compiler, runtime buffers, and backend
kernels. Implementing this as a distinct compute operation preserves packed
storage through compilation and execution and makes backend selection visible
in executable explanations. A composite cast/dequantize/matmul would require a
large floating intermediate unless the optimizer could prove and fuse the
whole expression.

Use ordinary `u32`, `bf16`, and `f32` tensors. A U32 tensor remains a tensor of
integers: ordinary `matmul`, `cast`, and checkpoint loading never reinterpret it
as a quantized weight. The explicit operation supplies the interpretation. No
model names, parameter-name conventions, or file-format heuristics enter the
universal compute core. Training and weight quantization are separate work.

The implementation touches Affon's `ops.ts`, formal authoring, native lowering,
and declaration docs, plus the shared compute schema, options/hash, compiler
selection, runtime invocation, CPU kernel, and Metal FFI/kernel. Both checkouts
must be included when moving this work to another machine; the compute changes
are not vendored into Affon.

## Representation and API

```ts
import { quantized_matmul, add } from 'affon:ops'
// All four operands are formal, or all four are evaluated in one Session.
const projected = quantized_matmul(x, packed, scales, biases, {
  bits: 8,
  group_size: 64,
})
// An optional ordinary linear output bias is a separate add.
const y = add(projected, output_bias)
```

The initial scheme is unsigned affine quantization in row-major logical
`[N,K]` weight order, with U32 words packing adjacent K values. It computes
`X @ decoded(W).T`; there is no implicit transpose of the physical packed
buffer. Inputs may be strided views; logical indices are mapped through their
element strides and byte offsets. Contiguous storage is the fast path to tune.

| Operand | Physical shape | Dtype | Meaning |
| --- | --- | --- | --- |
| input | `[M,K]` | f32 or bf16 | Activations |
| packed | `[N,K/P]` | u32 | Packed logical weight rows, `P=32/bits` |
| scales | `[N,K/group_size]` | f32 or bf16 | Group multipliers |
| biases | same as scales | same as scales | Group additive offsets |
| result | `[M,N]` | f32 | Native f32 accumulation |

For logical weight `[n,k]`, `word = packed[n,floor(k/P)]`,
`q = (word >> ((k % P) * bits)) & ((1 << bits) - 1)`, and
`weight = scales[n,floor(k/group_size)] * q + biases[n,floor(k/group_size)]`.
The first logical value occupies the least significant bits. U32 describes the
numeric word; checkpoint readers handle file byte order. On disk, the inspected
safetensors U32 values are little endian. Quantization biases are additive
offsets, not integer zero points and not the layer's output bias. Zero points
must be converted explicitly, using `offset = -scale * zero_point` with an
intentional metadata dtype/rounding decision.

Validation occurs in both the authoring API and native schema, before backend
execution. There must be four rank-two operands; M, N, and K must be positive;
bits must be 2, 4, or 8; group_size must be a positive integer; K must divide
into complete words and complete groups. Exact packed and parameter dimensions
are required, without broadcasting or a padded tail. Input and parameter
floating dtypes may differ, but scales and biases must share a dtype. Named
axes are omitted from the result because physical packed axes are not logical
feature axes. Values are not scanned for finite numbers during compilation.
Session ownership, disposed values, and formal/evaluated mixing use the normal
operation rules. The operation is inference-only; no derivative is registered.

Higher ranks can initially be flattened to `[M,K]` and reshaped back by the
caller. Batched weights, transpose modes, output dtype selection, empty tensors,
and partial groups are deliberately outside this first contract. They need
explicit shape rules and tests before extension.

## Checkpoint evidence and adapter boundary

Read-only inspection of the local Qwen3-TTS checkpoint found 904 tensors and:

```text
talker.code_predictor.lm_head.0.weight  U32  [2048,256]
talker.code_predictor.lm_head.0.scales  BF16 [2048,16]
talker.code_predictor.lm_head.0.biases  BF16 [2048,16]
```

At bits=8 and group_size=64 this is logical `[2048,1024]`. Packed weights use
2,097,152 bytes and the two BF16 parameter tensors use 131,072 bytes: 2,228,224
bytes total, versus 8,388,608 bytes for a fully expanded F32 matrix (3.76× less).
These are tensor-storage counts, not process memory or measured peak memory.

Packing and the affine formula were checked against the local primary MLX
source, `mlx/backend/cpu/quantized.cpp`, `_qmm_t` and SIMD extraction. The local
mlx-swift checkout is `4dccaeda1d83cf8697f235d2786c2d72ad4bb925`, containing
MLX commit `c215b6f88cf0fee0b0895623e4046cda797ef397`. The source lives beneath
`framecept-voice-eval/native-qwen-swift/.build/checkouts/mlx-swift/Source/Cmlx/mlx`.
It was read only. Source inspection establishes packing compatibility, not
bitwise output equivalence with all MLX execution paths.

A model adapter should obtain bits/group_size from checkpoint configuration,
bind the three existing ordinary tensors, and select quantized linear layers.
Names such as `.weight`, `.scales`, and `.biases` belong in that adapter. Header
shapes alone do not uniquely identify a quantization scheme. Do not silently
apply this operation to every U32 checkpoint tensor.

## CPU behavior and numerical contract

`compute/src/backend/cpu/quantized_matmul.zig` maps input buffers without copies,
validates buffer bounds/offsets, dispatches specialized 2/4/8-bit loops and dtype
pairs, loads group parameters, and decodes a scalar weight inside the dot
product. BF16 values are widened to F32 before arithmetic. The result uses
ordered F32 accumulation, with no BF16 rounding of decoded weights. There is
no per-run floating weight cache, temporary dequantized matrix, or JavaScript
array conversion. Input buffers remain immutable; the runtime owns output
allocation and lifetime.

This reference prioritizes an explicit numerical oracle and bounded memory.
It is single threaded and does not use BLAS or vector instructions explicitly.
Future CPU optimization can vectorize packed-word decoding, share group loads,
and tile activations/output columns while keeping this reference for testing.

## Metal execution and next performance work

The compiler selects a version-4 physical implementation with an explicit ID:

| Physical implementation | Selection | Work mapping |
| --- | --- | --- |
| `metal_quantized_matmul.simd1` | M < 4, complete groups of packed words | One SIMD group reduces each output dot product |
| `metal_quantized_matmul.simd4` | 4 <= M < 8, complete groups of packed words | One SIMD group shares decoded words across four activation rows |
| `metal_quantized_matmul.matrix32_packed` | M >= 32, M/N multiples of 32, K multiple of 32, supported pipeline | 32×32×32 tile, packed-word staging, register output stores |
| `metal_quantized_matmul.matrix32` | Other M >= 8, supported matrix pipelines | 32×32 output tile, K tile 64; SIMD matrix multiplication |
| `metal_quantized_matmul.scalar` | Groups crossing word boundaries, or unavailable optimized pipelines | Original strided scalar kernel |

The SIMD paths use 128 threads (four 32-lane SIMD groups) per threadgroup.
Each lane loads a U32 word, decodes its values, and contributes partial sums;
`simd_sum` completes each dot product. Bit width, group size, and input/parameter dtypes are Metal function constants.
Packing and group address calculations specialize during preparation; group sizes
such as 64 become shifts instead of runtime 64-bit division. Cached PSOs are owned
by the device and keyed by the complete scheme/dtype/strategy/dense-layout configuration. Group scales and
biases are loaded once per word. SIMD4 shares each decoded weight across four
rows in registers.

The matrix path uses 256 threads (eight SIMD groups). It stages a 32×64
activation tile and 64×32 decoded weight tile in threadgroup F32 memory,
performs 8×8 SIMD matrix multiply-accumulates, and stages the result for guarded
stores. Each SIMD group owns an 8×16 output rectangle. Activations and decoded
weights are reused across the 32×32 tile. Threadgroup storage is 20 KiB,
including the result tile; no temporary device buffer or full floating weight
matrix is allocated. Partial M/N/K tiles are zero-filled and guarded. Inputs
retain their explicit strides and offsets. The packed weights stay resident.

For complete output tiles, the packed 32×32×32 path uses four SIMD groups
(128 threads), each owning a 16×16 rectangle in four F32 matrices. A word is
loaded once and decoded into adjacent elements of a row-major weight tile;
SIMD matrix loads transpose that tile. Padded 32×36 activation and weight arrays
use 9,216 bytes in total. Registers store directly to the device output.
Dense resident inputs use an additional cached specialization with checked
32-bit element indexing; strided/offset inputs retain the generic specialization.
Both pipelines are prepared before execution; layout checks only select between
them at invocation time. No input is copied to force contiguity, and no temporary
device buffer or full decoded weight matrix is allocated. Accumulation remains
F32, including for BF16 input/metadata.

Optimized pipeline creation checks SIMD width, threadgroup size, and matrix
memory requirements. If the required pipelines are unavailable, selection uses
the SIMD or scalar path. BF16 loads widen U16 bits to F32. The existing command
batching and asynchronous buffer-lifetime path are retained. Metal candidates
are classified `reassociated`; `backend_equivalent` accepts them and
`exact_only` rejects them. CUDA remains unsupported.

The M=8 crossover was checked on the M4. The simple shape-based choice is a
first calibration, not a universal cost model. MLX remains faster in the
comparison below. Future work includes wider packed-word decode, fewer repeated
parameter/address calculations during tile loads, larger register tiles and
shape-specific scheduling, plus reducing Affon's host execution/readback cost.
A kernel-only comparison requires equivalent GPU profiling of both engines;
wall latency with output conversion must not be presented as kernel time.

## Correctness and reproducible checks

`test/e2e/program/operations/quantized-matmul.test.ts` exercises CPU and Metal: unsigned high
bits, low-bit-first packing, multiple groups and output rows, affine offsets,
mixed F32/BF16 operands, 2/4/8-bit decoding, malformed shapes/options, Session
ownership, Program inspection/native selection, and transposed/sliced views.
`compute/src/compute_test.zig` Q01 checks native validation independently of the
JavaScript checks. Type tests lock the public options and formal/evaluated result
types. No Qwen-specific data is required for these tests.

```sh
zig build install
./zig-out/bin/affon test test/e2e/program/operations/quantized-matmul.test.ts
bun x tsc -p test/types/tsconfig.json --noEmit
# In ../compute:
zig build test-compute -Dcompute-filter=Q01
```

Use Zig 0.16.0, as declared by the repository. The system Zig 0.17 build fails
in existing dependency build scripts because `std.Build.args` was removed.

Optional checkpoint parity reads three complete weight rows (0, 1, and 2047),
all 1024 input features, and their BF16 group parameters from the local file.
A Python oracle extracts little-endian bytes independently of the native U32
shift logic, uses deterministic synthetic inputs, and tests CPU and Metal with
`atol=1e-4, rtol=1e-4`. Generated tests live in an automatically deleted temporary
directory. It does not modify or copy the model into the repository.

```sh
python3 tools/test-quantized-checkpoint.py /path/to/model.safetensors
```

This establishes sampled affine-format parity, not a full model comparison or
an MLX runtime comparison. Before using a new optimized kernel, add randomized
native-reference comparisons across group/word/tile boundaries, cancellation,
zero/negative scales, larger M/N/K, dtype pairs, and strided views. Full-model
audio quality and latency remain adapter/application validation tasks.

## Benchmark protocol

`test/benchmark/quantized/quantized-matmul.ts` constructs resident synthetic 8-bit weights for
N=2048, K=1024, group_size=64, and M=1 or 8. It compiles once, warms three runs,
and measures five samples of ten runs on CPU and Metal. Each run includes
output readback as a completion boundary, output allocation, and JS invocation
overhead; construction and compilation are excluded. Date.now() makes each
sample's timer resolution about 0.1ms per reported iteration. It prints JSON
with samples and theoretical storage counts. Use ReleaseFast for meaningful CPU
measurements.

```sh
zig build install -Doptimize=ReleaseFast
./zig-out/bin/affon test/benchmark/quantized/quantized-matmul.ts
```

Initial measurements on Apple M4 (10 CPU cores, 16 GB), macOS 26.6.2,
Zig 0.16.0 ReleaseFast, with synchronization/readback included:

| M | CPU median | Metal median |
| --- | --- | --- |
| 1 | 1.4 ms | 1.4 ms |
| 8 | 11.1 ms | 5.8 ms |

These are baseline measurements, not an MLX comparison. The serial K loop and
small output grid limit the current Metal kernel, especially at M=1. Full
samples and source revisions are saved in
`test/benchmark/results/quantized-matmul-m4-baseline.json`.

Before claiming competitive performance, compare against resident dense F32
matmul and MLX quantized_matmul at M=1/8/32/128, actual model projection sizes,
4/8 bits, group sizes 32/64/128, and both activation dtypes. Record device/OS,
compiler/commits, warmup, sample distributions, synchronization, GPU timing,
native/host dispatch counts, transfers, and peak live bytes. Separate cold
pipeline compilation, first inference, warm decode, and prefill. Assert zero
host fallback and zero full-weight floating temporary bytes from the plan.

## Extending schemes

The operation's initial contract fixes unsigned U32 affine packing. Adding 3/6
bits requires defining values that cross word boundaries; the current
`32/bits` extraction must not be reused. Signed integer blocks, integer zero
points, codebooks, GGUF block structures, FP8/NF4, activation quantization, and
per-channel axes require explicit scheme/layout descriptors or separate
semantic schemas with their own validators and CPU references. Never infer
them from dtype alone.

A future nn-level immutable weight descriptor may bundle ordinary tensors plus
a versioned scheme/layout and logical dimensions for convenient reuse. It should
lower to explicit operands/options and preserve resource ownership through the
Session. Checkpoint adapters may repack once into a canonical layout, with
conversion memory/time reported separately. Unsupported schemes should fail
clearly; full dequantization should require an explicit caller choice.

## Historical M4 comparison with MLX

These measurements predate the profiling changes. See [Quantized inference profiling](./quantized-profiling.md) for the current matched GPU/host breakdown and benchmark results.

The repeatable harness is `tools/benchmark-quantized-mlx.py`, using
`test/benchmark/quantized/quantized-metal.ts`. It compares resident synthetic weights with
identical packing and values, N=2048, K=1024, group_size=64, 4/8 bits,
F32 inputs, BF16 parameters, and F32 outputs. Affon uses Zig 0.16.0 ReleaseFast;
MLX is 0.32.4 on Python 3.13. Affon and MLX run sequentially. Input generation,
compilation, and ten warmup calls are excluded. Both eager and `mx.compile`
MLX results are saved. The table uses compiled MLX.

Two timings avoid confusing output conversion with compute:

- Full readback: five samples of 30 calls, each converting the whole output to
  nested host arrays/lists and releasing it.
- Scalar completion: five samples of 100 calls, each computing the full QMM,
  taking a `[1,1]` view, reading its scalar to synchronize, and releasing it.
  Both engines author the QMM plus scalar slice. This still includes host call,
  allocation, view, and synchronization overhead; it is not kernel-only time.

Affon also records Metal command-buffer GPU timestamps as a diagnostic.
Backend timing is enabled during these Affon measurements. There is no matched
MLX GPU timestamp measurement, so those diagnostics are not used to claim a
kernel speed ratio. Affon's Date.now() samples resolve approximately 0.033 ms
per call for full readback and 0.01 ms for scalar completion; MLX uses
perf_counter_ns. All selected output checks matched the independent synthetic
oracle within 1e-4 (observed maximum error zero).

Measured 8-bit medians, milliseconds per call:

| M | Affon full readback | MLX full readback | Affon scalar completion | MLX scalar completion |
| --- | --- | --- | --- | --- |
| 1 | 0.400 | 0.251 | 0.340 | 0.228 |
| 8 | 1.033 | 0.505 | 0.700 | 0.283 |
| 32 | 2.500 | 1.335 | 0.730 | 0.301 |
| 128 | 8.667 | 5.005 | 1.870 | 0.463 |

The older scalar implementation's recorded 8-bit full-readback medians were
1.4 ms (M=1) and 5.8 ms (M=8). The new figures are lower, but the old run used
fewer warmups/iterations and lacked GPU timing; treat that as a historical
baseline, not a controlled speedup experiment. The current matched comparison
shows MLX ahead by about 1.5–4.0× on scalar completion across these sizes.
Full samples, candidate IDs, explanations, environment, and 4-bit results are
saved in `test/benchmark/results/quantized-metal-vs-mlx-m4.json`.

```sh
python3 -m venv /tmp/affon-mlx-bench
/tmp/affon-mlx-bench/bin/pip install mlx==0.32.4 numpy==2.5.4
zig build install -Doptimize=ReleaseFast
/tmp/affon-mlx-bench/bin/python tools/benchmark-quantized-mlx.py
```

The comparison uses synthetic resident weights with the checkpoint's
representative dimensions. Checkpoint parity is a separate read-only check;
this is not a full Qwen3-TTS latency or audio-quality evaluation.
