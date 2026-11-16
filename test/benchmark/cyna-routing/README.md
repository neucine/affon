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

## Runtime parity

The bounded runtime check has 11 test cases covering 16 Programs. Each case runs
the same Program in two public CPU Sessions: one lowers to Cyna, while the other
requests evidence capture and therefore executes on Zig. The check requires the
Cyna route to report `lowered`, the comparison route to report
`evidence_capture`, and every output to have the same shape, dtype, and finite
values.

| Dtype path | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| f32 | `3e-4` | `3e-4` |
| f64 | `1e-12` | `1e-12` |
| BF16 output | `1/128` | `1/128` |

The BF16 tolerance represents one BF16 relative unit. The f64 result is tighter
than the f32 result because it has substantially more precision. A BF16-stored
GPT-2 model still uses the f32 tolerance because the model constructor casts its
parameters to f32 for arithmetic.

Run it with:

```sh
AFFON_TEST_CYNA_SESSION=1 ./zig-out/bin/affon \
  test test/benchmark/cyna-routing/runtime-parity.test.ts
```

The passing Programs are:

| Corpus | Program | Shapes | Output dtype |
| --- | --- | --- | --- |
| Authored | transformer block with layer norm, attention, residual, and RMS norm | `[2,4,8]` with `[8,8]` weights | f32 |
| Imported ONNX | generic fixture | fixture-defined | fixture-defined |
| Imported ONNX | spatial fixture | fixture-defined | fixture-defined |
| Imported ONNX | Conv1d fixture | fixture-defined | fixture-defined |
| Dtype | exponential | `[2,3]` | f64 |
| Dtype | axis-one sum | `[2,3]` to `[2]` | f64 |
| Dtype | matrix multiplication | `[2,3] x [3,4]` to `[2,4]` | f64 |
| Dtype | exponential | `[2,3]` | BF16 |
| Dtype | axis-one sum | `[2,3]` to `[2]` | BF16 |
| Dtype | matrix multiplication | `[2,3] x [3,4]` to `[2,4]` | BF16 |
| GPT-2 | small f32 prefill | sequence 3, width 4 | f32 |
| GPT-2 | small BF16-stored prefill | sequence 3, width 4 | f32 |
| BERT | small f32 encoder | two tokens, width 4 | f32 |
| ViT | small f32 image encoder | one `3x2x2` image, width 4 | f32 |
| Llama | small f32 prefill | sequence 3, width 4 | f32 |
| Llama | small f32 cached decode | one new token, width 4 | f32 |

These are parity checks, not model-quality or performance claims. The focused
BF16 Programs exercise BF16 arithmetic. The BF16 GPT-2 case exercises BF16
parameter storage through public model constructors, while its arithmetic
remains f32. The models are tiny constructor fixtures, not trained checkpoints.
The ONNX evidence covers all three prepared importer fixtures, but not arbitrary
ONNX graphs. All comparisons use the CPU backend. No result in this directory
measures Cyna speed relative to Zig.

## Decision

No portable Cyna lowering target appears in the authored, imported, or model
sample. Every added Program routes through Cyna. The remaining fallbacks request
observability behavior outside this lowering question. Adding another lowering
rule now would be driven by vocabulary shape rather than a measured fallback.

The bounded execution milestone is now met for the nine Programs above.
Compile coverage still does not imply complete numerical, dtype, or model
coverage. Keep looking for repeated lowering fallbacks as the corpus grows, and
extend runtime evidence before turning any compile-only route into a broader
support claim.

Keep the following limits explicit when comparing later runs:

- coverage means this registry, not every valid `affon:core` Program
- these are tiny constructor and importer fixtures, not complete trained models
- runtime parity covers 16 representative Programs and does not imply complete model support
- compilation and parity say nothing about runtime speed
- the centralized observability fallbacks are expected until their evidence
  semantics exist on the Cyna path
- non-CPU backends and disabled-Cyna configurations are outside this survey
