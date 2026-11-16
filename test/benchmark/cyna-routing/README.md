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

The output is deterministic JSON. It counts Programs, groups Zig fallbacks by
category, reason, operation, and option, and marks the largest group that could
be removed by portable lowering. A compile-only result does not establish
numeric agreement, runtime performance, or model coverage.

## Current result

| Programs | Cyna | Zig | Cyna routing |
| ---: | ---: | ---: | ---: |
| 104 | 102 | 2 | 98.08% |

The two Zig fallbacks are:

| Program | Category | Reason |
| --- | --- | --- |
| `observability_evidence_capture` | observability | `evidence_capture` |
| `observability_compute_timing` | observability | `compute_evidence` |

Both Programs request telemetry behavior that still requires the Zig path.
`highest_impact_portable_lowering` is `null`.

## Corpus

The registry exercises public `affon:core` Programs across:

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

Each Session is CPU-only. The runner reads the executable's internal routing
record, disposes the executable, and then disposes the Session. It never runs a
Program and does not count an unsupported Program as a Cyna success.

## Decision

No portable Cyna lowering target appears in this authored sample. Adding another
lowering rule now would be driven by vocabulary shape rather than a measured
fallback.

The next lowering milestone should repeat this survey with bounded imported
models and additional model-like graphs. Imported Programs can expose operation,
dtype, shape, and layout combinations that handwritten fixtures miss. Only after
those Programs are classified should another Cyna lowering feature be selected.

Keep the following limits explicit when comparing later runs:

- coverage means this registry, not every valid `affon:core` Program
- compilation says nothing about numerical parity or runtime speed
- the centralized observability fallbacks are expected until their evidence
  semantics exist on the Cyna path
- non-CPU backends and disabled-Cyna configurations are outside this survey
