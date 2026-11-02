# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `68b9059f4b5bf040bd1a4f92c5fc29196fe3280eb52ab6e806e9b4db222a1b36`  
Binary SHA-256: `ca51b59b50b6275a76109c030cbb2c592906504c1660d3d01b19ed56d050be8e`  
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
| `metal/f32/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.2578 / 0.2176 | [passed](program_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.2481 / 0.219 |
| `metal/f32/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2795 / 0.236 | [passed](program_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2828 / 0.2406 |
| `metal/f32/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.243 / 0.2057 | [passed](program_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.1985 / 0.2143 |
| `metal/f32/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2818 / 0.24 | [passed](program_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2573 / 0.2419 |
| `metal/f32/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2473 / 0.2157 | [passed](program_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2503 / 0.2233 |
| `metal/f32/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2746 / 0.2444 | [passed](program_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2845 / 0.2469 |
| `metal/f32/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.25 / 0.2175 | [passed](program_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.2564 / 0.2194 |
| `metal/f32/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2801 / 0.2438 | [passed](program_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2388 / 0.247 |
| `metal/bf16/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2437 / 0.2144 | [passed](program_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2588 / 0.2087 |
| `metal/bf16/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.2663 / 0.239 | [passed](program_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.2761 / 0.2384 |
| `metal/bf16/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2434 / 0.2159 | [passed](program_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2626 / 0.219 |
| `metal/bf16/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.2807 / 0.2437 | [passed](program_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.3147 / 0.2393 |
| `metal/bf16/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2486 / 0.2159 | [passed](program_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.26 / 0.2219 |
| `metal/bf16/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.2792 / 0.2433 | [passed](program_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.2774 / 0.2473 |
| `metal/bf16/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.2471 / 0.2206 | [passed](program_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.2604 / 0.2185 |
| `metal/bf16/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.2566 / 0.2441 | [passed](program_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.253 / 0.2415 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
