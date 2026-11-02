# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `05e96dbc28ae1013ba0016407f82be563896523825748dc053fddb30403d51c2`  
Binary SHA-256: `d453dae33ce53f0a3a2099ae33b469ee9239b73696fcd0a02dbae903c3d9f881`  
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
| `metal/f32/where/scalar` | comparison | — | dense | correct | [passed](operator_peer/metal__f32__where__scalar.json) | 0.2269 / 0.2232 | [passed](program_peer/metal__f32__where__scalar.json) | 0.2348 / 0.2128 |
| `metal/f32/where/small_dense` | comparison | 2 × 4 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense.json) | 0.2348 / 0.2152 | [passed](program_peer/metal__f32__where__small_dense.json) | 0.2463 / 0.219 |
| `metal/f32/where/large_dense` | comparison | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__where__large_dense.json) | 0.2779 / 0.2385 | [passed](program_peer/metal__f32__where__large_dense.json) | 0.2924 / 0.2412 |
| `metal/f32/where/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast.json) | 0.2448 / 0.213 | [passed](program_peer/metal__f32__where__broadcast.json) | 0.25 / 0.2166 |
| `metal/f32/where/strided` | comparison | 8 × 16 | strided | correct | [passed](operator_peer/metal__f32__where__strided.json) | 0.2253 / 0.2135 | [passed](program_peer/metal__f32__where__strided.json) | 0.2509 / 0.2218 |
| `metal/f32/where/offset` | comparison | 8 × 16 | offset | correct | [passed](operator_peer/metal__f32__where__offset.json) | 0.2225 / 0.2124 | [passed](program_peer/metal__f32__where__offset.json) | 0.2474 / 0.2222 |
| `metal/f32/where/ragged` | comparison | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__where__ragged.json) | 0.2673 / 0.2304 | [passed](program_peer/metal__f32__where__ragged.json) | 0.264 / 0.2284 |
| `metal/bf16/where/scalar` | comparison | — | dense | correct | [passed](operator_peer/metal__bf16__where__scalar.json) | 0.181 / 0.2157 | [passed](program_peer/metal__bf16__where__scalar.json) | 0.2357 / 0.2153 |
| `metal/bf16/where/small_dense` | comparison | 2 × 4 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense.json) | 0.2396 / 0.2157 | [passed](program_peer/metal__bf16__where__small_dense.json) | 0.2557 / 0.2133 |
| `metal/bf16/where/large_dense` | comparison | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__where__large_dense.json) | 0.2791 / 0.2404 | [passed](program_peer/metal__bf16__where__large_dense.json) | 0.2754 / 0.2411 |
| `metal/bf16/where/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast.json) | 0.2439 / 0.2119 | [passed](program_peer/metal__bf16__where__broadcast.json) | 0.205 / 0.2151 |
| `metal/bf16/where/strided` | comparison | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__where__strided.json) | 0.2301 / 0.2201 | [passed](program_peer/metal__bf16__where__strided.json) | 0.2604 / 0.2221 |
| `metal/bf16/where/offset` | comparison | 8 × 16 | offset | correct | [passed](operator_peer/metal__bf16__where__offset.json) | 0.2468 / 0.2189 | [passed](program_peer/metal__bf16__where__offset.json) | 0.2583 / 0.2253 |
| `metal/bf16/where/ragged` | comparison | 17 × 65 | dense | correct | [passed](operator_peer/metal__bf16__where__ragged.json) | 0.2697 / 0.2314 | [passed](program_peer/metal__bf16__where__ragged.json) | 0.2683 / 0.2184 |
| `metal/f32/where/broadcast/r32_c32` | comparison | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__r32_c32.json) | 0.2672 / 0.2246 | [passed](program_peer/metal__f32__where__broadcast__r32_c32.json) | 0.282 / 0.232 |
| `metal/f32/where/broadcast/r64_c128` | comparison | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__r64_c128.json) | 0.3357 / 0.2975 | [passed](program_peer/metal__f32__where__broadcast__r64_c128.json) | 0.3509 / 0.2977 |
| `metal/bf16/where/broadcast/r32_c32` | comparison | 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__r32_c32.json) | 0.2643 / 0.2296 | [passed](program_peer/metal__bf16__where__broadcast__r32_c32.json) | 0.3099 / 0.225 |
| `metal/bf16/where/broadcast/r64_c128` | comparison | 64 × 128 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__r64_c128.json) | 0.3195 / 0.2947 | [passed](program_peer/metal__bf16__where__broadcast__r64_c128.json) | 0.3661 / 0.3063 |
| `metal/f32/where/offset/r32_c127` | comparison | 32 × 127 | offset | correct | [passed](operator_peer/metal__f32__where__offset__r32_c127.json) | 0.2982 / 0.2651 | [passed](program_peer/metal__f32__where__offset__r32_c127.json) | 0.3089 / 0.2676 |
| `metal/f32/where/strided/r32_c127` | comparison | 32 × 127 | strided | correct | [passed](operator_peer/metal__f32__where__strided__r32_c127.json) | 0.3023 / 0.264 | [passed](program_peer/metal__f32__where__strided__r32_c127.json) | 0.3108 / 0.2654 |
| `metal/bf16/where/offset/r32_c127` | comparison | 32 × 127 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__r32_c127.json) | 0.3092 / 0.265 | [passed](program_peer/metal__bf16__where__offset__r32_c127.json) | 0.3004 / 0.262 |
| `metal/bf16/where/strided/r32_c127` | comparison | 32 × 127 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__r32_c127.json) | 0.3 / 0.262 | [passed](program_peer/metal__bf16__where__strided__r32_c127.json) | 0.263 / 0.2632 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
