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
reusable state. Use `testDevices` from `test/e2e/devices.ts` for behavior shared
by devices, and `testOn` or `describeOn` for backend-specific contracts. The
helper always enables CPU and detects Metal and CUDA by opening a Session;
unavailable GPU backends are skipped. A backend that opens but fails a test still
fails the run.

The remaining Metal-only cases assert GPU residency, dispatch details, or
GPU-produced invalid-index recovery. CPU-only cases cover f64 precision,
checkpoint storage, and `one_hot` invalid-label rejection. Metal currently
accepts that invalid label, so its value contract runs on every device while
the rejection assertion stays explicit on CPU. The Metal f64-to-u32 overflow
assertion is also backend-specific because CPU currently does not throw.
On the pinned CUDA build, three invalid-index recovery contracts and the
compiled rank-three dense loss are registered as skipped CUDA cases: CUDA
either accepts the invalid index, reports a different error, or rejects the
loss. Other cases in those files still run on CUDA.

Set `AFFON_TEST_DEVICES` to a comma-separated GPU filter when narrowing a run,
for example `AFFON_TEST_DEVICES=cpu ./tools/test-runtime.sh`. With the variable
unset, the suite runs CPU and every supported GPU backend. CPU-specific
reference tests always run as baselines.

Run the operation suite and its TypeScript contracts with:

```sh
./zig-out/bin/affon test test/e2e/program/operations
bun x tsc -p test/e2e/program/operations/tsconfig.json --noEmit
```

The gather bounds contract runs on CPU and Metal. It checks that an invalid
column index raises `InvalidCommand` and that the same compiled Program still
returns the expected values with valid indices.
