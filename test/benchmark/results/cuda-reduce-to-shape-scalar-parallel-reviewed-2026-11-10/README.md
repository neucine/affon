# CUDA f32 reduce-to-shape scalar and column dispatch

The reviewed Compute change adds two reassociated rank-two f32 reduction
kernels and routes rank-zero outputs through the rank-two path:

- `compute_reduce_shape2_f32_scalar`: one 256-thread block reduces every
  element of a single-output reduction and writes one f32 value.
- `compute_reduce_shape2_f32_columns`: one block per output column reduces the
  rows for `[rows, cols] -> [1, cols]` and `[rows, cols] -> [cols]`.
- Rank-zero f32 and i64 outputs of rank-two inputs now use
  `affon_compute_cuda_reduce_to_shape2` instead of the generic eight-axis
  kernel.

Both new kernels accumulate in f32 inputs, sum in double precision, and round
once to f32. This is a reassociated numerical class: it is more accurate than
the previous single-thread f32 loop for cancellation-heavy inputs, but it is
not bit-identical to sequential f32 addition. The long-axis row kernel is
unchanged and keeps exact original-order f32 accumulation.

The source changes are in the sibling Compute checkout's
`src/backend/cuda/reduce_all8_parallel.cu`, `src/backend/cuda/ffi.c`,
`src/backend/cuda/device.zig`, and generated
`src/backend/cuda/reduce_all8_parallel_ptx.h`.

## Contract

- Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`
- Control Affon binary (promoted row dispatch):
  `42a8c02d83f1ffcb4ddeac1c144f2330ec8a7ce96db672bb8fc89c9321e71b2a`
- Candidate Affon binary:
  `584dc37ea364cad3a19221adf3cf8bb17a5afff828c9ebf3f4e2308a9eef8953`
- Peer: PyTorch `2.11.0+cu128`
- Catalog SHA-256:
  `1bf5a09cb8d3a795d3695ec853c8d46437a8c630a0aa07e10ae766d6e628c483`
- Protocol: 10 iterations per sample, 5 samples, 3 independent rounds
- Catalog scope: 134 f32 `reduce_to_shape` cells
- Complete candidate batch: 134/134 numerical, operator, and Program checks
  passed.
- Complete-batch audit: `audit-scalar-parallel.json` binds the expected
  catalog, candidate binary, target, peer version, and measurement protocol.
- Regression comparison against the row-dispatch control:
  `../regression-cuda-reduce-to-shape-scalar-parallel-vs-rows-2026-11-10`;
  zero added or removed cells, zero changed case definitions, and zero
  correctness regressions.
- General CUDA checks: seed `20261110` passed 256/256 and seed `20261106`
  passed 256/256. Evidence is retained at
  `../cuda-reduce-shape-scalar-parallel-general-final-20261110` and
  `../cuda-reduce-shape-scalar-parallel-general-final-20261106`. The checker
  compares i64 exactly and f32 against the capability-probe tolerance around
  an exact `fsum` reference.

## Controlled effect

Seventy-one catalog cells use the new dispatch: the scalar output, column
reductions with at least eight rows, and single-output reductions with at
least 256 elements.

- Operator execution speedup: median 2.90x, maximum 24.12x, minimum 1.26x.
- Operator total latency speedup: median 2.32x, maximum 20.69x.
- Program execution speedup: median 2.36x, maximum 21.01x, minimum 1.17x.
- Program total latency speedup: median 2.03x, maximum 18.37x.
- `ragged/dense_r128_c129` operator execution fell from 0.7323 to 0.0304 ms;
  total latency fell from 0.7376 to 0.0357 ms.
- `outer_axis/dense_r128_c129` operator execution fell from 0.0416 to
  0.0117 ms; total latency fell from 0.0476 to 0.0179 ms.
- `scalar` operator execution fell from 0.0542 to 0.0110 ms; total latency
  fell from 0.0594 to 0.0162 ms.

The promoted status counts moved in the intended direction. Across the 134
`reduce_to_shape` cells, operator `candidate_gap` fell from 113 to 74 and
Program `candidate_gap` fell from 119 to 91. Across the complete aggregate,
operator `candidate_gap` fell from 3,123 to 3,081 and Program
`candidate_gap` fell from 4,111 to 4,082.

The single regression-report timing suspect is
`long_axis/offset_r33_c32` Program: Affon latency was effectively unchanged
while the PyTorch peer moved, the same peer-driven diagnostic retained by the
row-dispatch review.

The prior ledger entries and aggregate reports are retained in
`../archive-cuda-reduce-to-shape-before-scalar-parallel-2026-11-10/`.
