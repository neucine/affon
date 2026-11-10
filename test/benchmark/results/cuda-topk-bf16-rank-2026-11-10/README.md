# CUDA BF16 top-k rank dispatch

The reviewed Compute change extends `compute_topk_rank_parallel` to BF16.
BF16 reductions over their last axis with extent at least 32 now use one
parallel rank kernel block per output row. The kernel preserves the exact BF16
output bits and the existing stable lower-index tie ordering. Rows containing
NaN retain the existing stable serial ranking behavior. All other BF16 axes
remain on the previous small-top-k path.

The source changes are in the sibling Compute checkout's
`src/backend/cuda/binary_small.cu`, `src/backend/cuda/ffi.c`,
`src/backend/cuda/device.zig`, and generated
`src/backend/cuda/binary_small_ptx.h`.

## Contract

- Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`
- Control Affon binary:
  `42a8c02d83f1ffcb4ddeac1c144f2330ec8a7ce96db672bb8fc89c9321e71b2a`
- Candidate Affon binary:
  `8ccafdefe9a0589da3b3536d32b8158601a1fcca73492cdeca010f1cf651b876`
- Peer: PyTorch `2.11.0+cu128`, stable sort plus narrow
- Catalog SHA-256:
  `eb1df9d2e1d4ac4b830ba78bcc9d421f4ceb86e81b0429dafa72d0c166fb1288`
- Protocol: 10 iterations per sample, 5 samples, 3 independent rounds
- Catalog scope: 520 CUDA top-k cells
- Complete candidate batch: 520/520 numerical, operator, and Program checks
  passed.
- General CUDA checks: seed `20261110` passed 464/464 and seed `20261106`
  passed 464/464, both with NaN cases included. Evidence is retained at
  `../cuda-topk-bf16-general-20261110-last-axis` and
  `../cuda-topk-bf16-general-20261106-last-axis`.
- Regression comparison: zero added or removed cells, zero changed case
  definitions, and zero correctness regressions. All operator cases also
  matched the exact reference output hash.

The earlier broad dispatch experiment, which also routed BF16 axis0 reductions
through the rank kernel, is retained in `candidate-broad` and
`regression-broad`. It showed material axis0 slowdowns and is not the reviewed
behavior. The corrected `candidate-last-axis` and `regression-last-axis`
directories contain the evidence for this change.

`audit-last-axis.json` binds the complete corrected batch to the expected
catalog, exact candidate binary, target, PyTorch version, and measurement
protocol. The 520 cells were promoted to the current aggregate after the audit
passed. The prior ledger entries and aggregate reports are retained in
`../archive-cuda-topk-before-bf16-rank-2026-11-10/`.

## Controlled effect

The dispatch affected 64 BF16 catalog cells whose reduction axis is the last
axis and whose reduced extent is at least 32. All 64 improved in both execution
and whole-operation Affon latency for both operator and Program measurement.

- Operator execution speedup: median 6.69x, maximum 17.20x.
- Operator total latency speedup: median 2.64x, maximum 8.00x.
- Program execution speedup: median 5.98x, maximum 15.77x.
- Program total latency speedup: median 2.55x, maximum 7.72x.
- `large/dense_axis1_r8_c257_k256_smallest` operator execution fell from
  0.45746 to 0.02660 ms; total latency fell from 0.49226 to 0.06152 ms.
- `large/dense_axis1_r8_c257_k256_smallest` Program execution fell from
  0.45977 to 0.02914 ms; total latency fell from 0.49442 to 0.06422 ms.

The 49 BF16 axis0 cells with reduced extent at least 32 (48 layout-specific
cases plus the aggregate case) remain on the previous path. Their operator
execution ratio had median 1.0016, p95 1.0110, and maximum 1.0132. Across all
456 catalog cells outside the dispatch, the operator execution ratio had
median 1.0022, p95 1.0126, and maximum 1.0250; the largest Program execution
ratio was 1.0146.

The promoted status counts moved in the intended direction. Across the 520
top-k cells, operator `candidate_gap` fell from 86 to 21 and Program
`candidate_gap` fell from 88 to 24. All 64 dispatched cells left
`candidate_gap`: operator finished at 63 `better_than_peer` and one
`within_diagnostic_target`; Program finished at 60 `better_than_peer` and four
`within_diagnostic_target`. The other-axis operator `candidate_gap` count was
unchanged, and the Program count changed by two cells within timing noise.
After promotion, the complete aggregate contains 3,123 operator and 4,111
Program `candidate_gap` cells.

The regression report has one diagnostic timing suspect:
`batched/dense_axis1_b4_r7_c5_k3`. Affon operator latency was effectively
unchanged, 0.032447 to 0.032427 ms, while the PyTorch peer fell from 0.061962
to 0.041260 ms. This is a peer-driven diagnostic, not a kernel regression.

These are controlled per-call diagnostic measurements on this GPU, including
output materialization. They are not a universal model-level speedup claim.
