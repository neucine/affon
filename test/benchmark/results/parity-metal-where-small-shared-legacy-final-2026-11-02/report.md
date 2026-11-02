# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `05e96dbc28ae1013ba0016407f82be563896523825748dc053fddb30403d51c2`  
Binary SHA-256: `05fcf9eb53831e8074b17d9e9205add5d013a4c4dcd00dc0514f36f0e1eebfa1`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 22
- Numerical check passed: 22
- Operator check and timing passed: 22
- Program check and timing passed: 22
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/where/scalar` | comparison | — | dense | correct | [passed](operator_peer/metal__f32__where__scalar.json) | 0.0085 / 0.2761 | [passed](program_peer/metal__f32__where__scalar.json) | 0.2043 / 0.2155 |
| `metal/f32/where/small_dense` | comparison | 2 × 4 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense.json) | 0.01 / 0.2412 | [passed](program_peer/metal__f32__where__small_dense.json) | 0.1977 / 0.2116 |
| `metal/f32/where/large_dense` | comparison | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__where__large_dense.json) | 0.0763 / 0.2496 | [passed](program_peer/metal__f32__where__large_dense.json) | 0.2723 / 0.2421 |
| `metal/f32/where/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast.json) | 0.0154 / 0.2493 | [passed](program_peer/metal__f32__where__broadcast.json) | 0.214 / 0.2164 |
| `metal/f32/where/strided` | comparison | 8 × 16 | strided | correct | [passed](operator_peer/metal__f32__where__strided.json) | 0.0157 / 0.2508 | [passed](program_peer/metal__f32__where__strided.json) | 0.2162 / 0.2277 |
| `metal/f32/where/offset` | comparison | 8 × 16 | offset | correct | [passed](operator_peer/metal__f32__where__offset.json) | 0.0157 / 0.2475 | [passed](program_peer/metal__f32__where__offset.json) | 0.2101 / 0.2207 |
| `metal/f32/where/ragged` | comparison | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__where__ragged.json) | 0.0462 / 0.2338 | [passed](program_peer/metal__f32__where__ragged.json) | 0.2426 / 0.2289 |
| `metal/bf16/where/scalar` | comparison | — | dense | correct | [passed](operator_peer/metal__bf16__where__scalar.json) | 0.0085 / 0.2295 | [passed](program_peer/metal__bf16__where__scalar.json) | 0.1877 / 0.217 |
| `metal/bf16/where/small_dense` | comparison | 2 × 4 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense.json) | 0.0098 / 0.2267 | [passed](program_peer/metal__bf16__where__small_dense.json) | 0.1891 / 0.215 |
| `metal/bf16/where/large_dense` | comparison | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__where__large_dense.json) | 0.0755 / 0.2515 | [passed](program_peer/metal__bf16__where__large_dense.json) | 0.3014 / 0.2412 |
| `metal/bf16/where/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast.json) | 0.0154 / 0.235 | [passed](program_peer/metal__bf16__where__broadcast.json) | 0.2512 / 0.2171 |
| `metal/bf16/where/strided` | comparison | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__where__strided.json) | 0.0158 / 0.2511 | [passed](program_peer/metal__bf16__where__strided.json) | 0.2509 / 0.2224 |
| `metal/bf16/where/offset` | comparison | 8 × 16 | offset | correct | [passed](operator_peer/metal__bf16__where__offset.json) | 0.0155 / 0.2263 | [passed](program_peer/metal__bf16__where__offset.json) | 0.2485 / 0.2265 |
| `metal/bf16/where/ragged` | comparison | 17 × 65 | dense | correct | [passed](operator_peer/metal__bf16__where__ragged.json) | 0.0459 / 0.2382 | [passed](program_peer/metal__bf16__where__ragged.json) | 0.2806 / 0.2235 |
| `metal/f32/where/broadcast/r32_c32` | comparison | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__r32_c32.json) | 0.0491 / 0.2382 | [passed](program_peer/metal__f32__where__broadcast__r32_c32.json) | 0.242 / 0.2289 |
| `metal/f32/where/broadcast/r64_c128` | comparison | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__r64_c128.json) | 0.3465 / 0.292 | [passed](program_peer/metal__f32__where__broadcast__r64_c128.json) | 0.3221 / 0.3003 |
| `metal/bf16/where/broadcast/r32_c32` | comparison | 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__r32_c32.json) | 0.0488 / 0.2528 | [passed](program_peer/metal__bf16__where__broadcast__r32_c32.json) | 0.2712 / 0.2314 |
| `metal/bf16/where/broadcast/r64_c128` | comparison | 64 × 128 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__r64_c128.json) | 0.3458 / 0.2999 | [passed](program_peer/metal__bf16__where__broadcast__r64_c128.json) | 0.3178 / 0.2979 |
| `metal/f32/where/offset/r32_c127` | comparison | 32 × 127 | offset | correct | [passed](operator_peer/metal__f32__where__offset__r32_c127.json) | 0.1299 / 0.2896 | [passed](program_peer/metal__f32__where__offset__r32_c127.json) | 0.304 / 0.267 |
| `metal/f32/where/strided/r32_c127` | comparison | 32 × 127 | strided | correct | [passed](operator_peer/metal__f32__where__strided__r32_c127.json) | 0.1301 / 0.2829 | [passed](program_peer/metal__f32__where__strided__r32_c127.json) | 0.3042 / 0.2635 |
| `metal/bf16/where/offset/r32_c127` | comparison | 32 × 127 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__r32_c127.json) | 0.1303 / 0.2652 | [passed](program_peer/metal__bf16__where__offset__r32_c127.json) | 0.3685 / 0.2612 |
| `metal/bf16/where/strided/r32_c127` | comparison | 32 × 127 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__r32_c127.json) | 0.1293 / 0.2804 | [passed](program_peer/metal__bf16__where__strided__r32_c127.json) | 0.32 / 0.2644 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
