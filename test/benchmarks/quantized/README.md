# Quantized matrix multiplication investigations

These Affon scripts reproduce focused CPU and Metal measurements. They are
diagnostic programs, separate from the parity catalogs and retained batch
reports in `test/parity/` and `test/benchmarks/results/`.

| Script | Purpose |
| --- | --- |
| `quantized-matmul.ts` | Basic CPU and Metal timing for 8-bit weights. |
| `quantized-metal.ts` | Metal correctness and timing across 4-bit and 8-bit cases. |
| `profile-quantized.ts` | Execution, readback, and disposal timing phases. |
| `trace-quantized-metal.ts` | Detailed Metal timing samples for tracing. |

Run them with `./zig-out/bin/affon test/benchmarks/quantized/<script>.ts`.
