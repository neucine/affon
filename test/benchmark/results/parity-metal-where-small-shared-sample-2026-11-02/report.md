# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `68b9059f4b5bf040bd1a4f92c5fc29196fe3280eb52ab6e806e9b4db222a1b36`  
Binary SHA-256: `05fcf9eb53831e8074b17d9e9205add5d013a4c4dcd00dc0514f36f0e1eebfa1`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 16
- Numerical check passed: 16
- Operator check and timing passed: 16
- Program check and timing passed: 16
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.0151 / 0.2451 | [passed](program_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.2143 / 0.2193 |
| `metal/f32/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.0692 / 0.2594 | [passed](program_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2703 / 0.2446 |
| `metal/f32/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.015 / 0.2319 | [passed](program_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.1924 / 0.2162 |
| `metal/f32/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.0697 / 0.2523 | [passed](program_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2714 / 0.2395 |
| `metal/f32/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.0152 / 0.229 | [passed](program_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2152 / 0.223 |
| `metal/f32/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.0701 / 0.2492 | [passed](program_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2475 / 0.2484 |
| `metal/f32/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.0153 / 0.2241 | [passed](program_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.2145 / 0.2214 |
| `metal/f32/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.0704 / 0.2477 | [passed](program_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2689 / 0.2505 |
| `metal/bf16/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.0152 / 0.2221 | [passed](program_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2541 / 0.2164 |
| `metal/bf16/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.0693 / 0.249 | [passed](program_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.3158 / 0.2441 |
| `metal/bf16/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.0152 / 0.2214 | [passed](program_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2447 / 0.212 |
| `metal/bf16/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.0705 / 0.2397 | [passed](program_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.3016 / 0.2434 |
| `metal/bf16/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.0153 / 0.2368 | [passed](program_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2547 / 0.226 |
| `metal/bf16/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.0703 / 0.2528 | [passed](program_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.3052 / 0.245 |
| `metal/bf16/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.0154 / 0.2482 | [passed](program_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.2338 / 0.2186 |
| `metal/bf16/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.0695 / 0.2473 | [passed](program_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.3149 / 0.2447 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
