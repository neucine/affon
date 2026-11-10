# CUDA f32 reduce-to-shape row dispatch

The reviewed Compute change adds `compute_reduce_shape2_f32_rows` for f32
rank-two inputs reduced to `[rows, 1]` when the reduced extent is 32 through
4096. Each output row gets one 256-thread block. Threads stage the row in
shared memory, then thread 0 adds the staged values in original column order.
The ordering preserves the existing sequential f32 accumulation semantics.
Other layouts and extents retain the previous serial path.

The source changes are in the sibling Compute checkout's
`src/backend/cuda/ffi.c`, `src/backend/cuda/reduce_all8_parallel.cu`, and
generated `src/backend/cuda/reduce_all8_parallel_ptx.h`.

## Contract

- Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`
- Control Affon binary:
  `8bd3dd7d664236c3c2ded29331eac8920470ea20a659d1f0ca4e253432098d2d`
- Candidate Affon binary:
  `42a8c02d83f1ffcb4ddeac1c144f2330ec8a7ce96db672bb8fc89c9321e71b2a`
- Peer: PyTorch `2.11.0+cu128`
- Catalog SHA-256:
  `1bf5a09cb8d3a795d3695ec853c8d46437a8c630a0aa07e10ae766d6e628c483`
- Protocol: 10 iterations per sample, 5 samples, 3 independent rounds
- Catalog scope: 134 f32 `reduce_to_shape` cells
- Complete candidate batch: 134/134 numerical, operator, and Program checks
  passed.
- Frozen control batch:
  `../cuda-reduce-to-shape-frozen-baseline-2026-11-10`.
- Regression comparison:
  `../regression-cuda-reduce-to-shape-rows-parallel-2026-11-10`; zero added or
  removed cells, zero changed case definitions, and zero correctness
  regressions.
- General CUDA checks: seed `20261110` passed 256/256 and seed `20261106`
  passed 256/256. Evidence is retained at
  `../cuda-reduce-shape-general-20261110` and
  `../cuda-reduce-shape-general-20261106`. The checks cover dense, offset,
  strided, and transposed layouts, exact output equality, and no host fallback.

The frozen control versus the matching cells in the same-binary
`../parity-cuda-readback-full2-2026-11-10` batch had a median Affon timing
ratio of 0.999, p95 1.009, and maximum 1.016, establishing the reproducibility
envelope for this target.

## Controlled effect

The dispatch affected 24 catalog cells: long-axis reductions over 32, 65, or
129 columns, with 3, 16, 33, or 128 output rows, in dense and offset layouts.
All 24 improved in both execution and whole-operation Affon latency for both
operator and Program measurement.

- Operator execution speedup: median 1.380x, maximum 2.032x.
- Operator total latency speedup: median 1.212x, maximum 1.594x.
- Program execution speedup: median 1.259x, maximum 1.702x.
- Program total latency speedup: median 1.169x, maximum 1.472x.
- `long_axis/dense_r128_c129` operator execution fell from 0.02218 to
  0.01091 ms; total latency fell from 0.04215 to 0.03083 ms.
- `long_axis/dense_r128_c129` Program execution fell from 0.02690 to
  0.01581 ms; total latency fell from 0.04734 to 0.03598 ms.

Across all 134 cells, the candidate/control median Affon latency ratio was
0.9991 operator and 0.9988 Program. The largest Affon slowdowns were 1.7%
operator and 2.3% Program, in unaffected small cells. The regression report's
single timing suspect is a PyTorch peer shift in Program mode for
`long_axis/offset_r33_c32`; Affon's total latency improved 10% in that cell.

These are controlled per-call diagnostic measurements on this GPU, including
output materialization. They are not a universal model-level speedup claim.
Other reduce-to-shape regimes remain on their existing paths; in particular,
the `ragged` reduce-all cells are outside this dispatch and retain their prior
timing.
