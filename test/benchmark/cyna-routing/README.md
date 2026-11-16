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
| 117 | 115 | 2 | 98.29% |

| Corpus | Programs | Cyna | Zig |
| --- | ---: | ---: | ---: |
| Authored operation and model-like Programs | 104 | 102 | 2 |
| Imported ONNX fixture Programs | 3 | 3 | 0 |
| Native model-family Programs | 10 | 10 | 0 |

The two Zig fallbacks are:

| Program | Category | Reason |
| --- | --- | --- |
| `observability_evidence_capture` | observability | `evidence_capture` |
| `observability_compute_timing` | observability | `compute_evidence` |

Both Programs request telemetry behavior that still requires the Zig path.
`highest_impact_portable_lowering` is `null`.

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

## Decision

No portable Cyna lowering target appears in the authored, imported, or model
sample. Every added Program routes through Cyna. The remaining fallbacks request
observability behavior outside this lowering question. Adding another lowering
rule now would be driven by vocabulary shape rather than a measured fallback.

The next useful milestone is execution evidence for a bounded subset of these
Cyna routes. Compile coverage does not show numerical agreement, parameter
initialization, or runtime performance. Keep looking for repeated lowering
fallbacks as the corpus grows, but require measured evidence before implementing
one.

Keep the following limits explicit when comparing later runs:

- coverage means this registry, not every valid `affon:core` Program
- these are tiny constructor and importer fixtures, not complete trained models
- compilation says nothing about numerical parity or runtime speed
- the centralized observability fallbacks are expected until their evidence
  semantics exist on the Cyna path
- non-CPU backends and disabled-Cyna configurations are outside this survey
