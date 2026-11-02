# Operation e2e tests

These tests exercise public `affon:ops` behavior through evaluated tensors and
compiled Programs. Use independent expected values and keep each contract near
its operation family. Numerical comparisons against other frameworks belong in
`test/parity/`.

| File | Main contract |
| --- | --- |
| `elementwise-contracts.test.ts` | Row and column broadcasting, comparisons, threshold selection, unary boundaries, invalid shape recovery |
| `reduction-contracts.test.ts` | Exact values for eight reductions, negative axes, retained dimensions, repeated compiled runs, CPU and Metal |
| `layout-contracts.test.ts` | Nonzero slice offsets, transposed views, contiguous copies, batched matmul through views, rank changes |
| `selection-contracts.test.ts` | Per-row gather, repeated indices, one-hot, broadcast masks, invalid index and mask recovery |
| `normalization-loss-contracts.test.ts` | Stable softmax, normalization, indexed and binary losses, regression losses, invalid shape recovery |
| `matmul-dot-contracts.test.ts` | Independent batches, strided inputs, dot, mismatched dimensions |
| `topk.test.ts` | Largest and smallest, sorted and unsorted, outer and inner axes, invalid `k` recovery |
| Other focused files | Dtypes, quantized operations, scatter, dense losses, trigonometry, and backend layouts |

When adding an operation or changing its contract, assert the output values and
shape, a relevant invalid input, and a later valid call when failure can affect
reusable state. Run the same behavior on CPU and Metal when both support it.

Run the operation suite and its TypeScript contracts with:

```sh
./zig-out/bin/affon test test/e2e/program/operations
bun x tsc -p test/e2e/program/operations/tsconfig.json --noEmit
```

The gather bounds contract runs on CPU and Metal. It checks that an invalid
column index raises `InvalidCommand` and that the same compiled Program still
returns the expected values with valid indices.
