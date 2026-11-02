# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `68b9059f4b5bf040bd1a4f92c5fc29196fe3280eb52ab6e806e9b4db222a1b36`  
Binary SHA-256: `921ae28ce07860990ec9e503afd1d4472af2c309c647eb11d084cd7d329cdb62`  
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
| `metal/f32/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.1812 / 0.2266 | [passed](program_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.1934 / 0.2106 |
| `metal/f32/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2762 / 0.244 | [passed](program_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2685 / 0.2404 |
| `metal/f32/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.2437 / 0.2079 | [passed](program_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.248 / 0.2195 |
| `metal/f32/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2795 / 0.2423 | [passed](program_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2447 / 0.2439 |
| `metal/f32/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2413 / 0.2163 | [passed](program_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.251 / 0.2225 |
| `metal/f32/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2602 / 0.2435 | [passed](program_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2775 / 0.2422 |
| `metal/f32/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.2393 / 0.2136 | [passed](program_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.2471 / 0.2236 |
| `metal/f32/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2808 / 0.2367 | [passed](program_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2824 / 0.2489 |
| `metal/bf16/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2439 / 0.2195 | [passed](program_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2377 / 0.2192 |
| `metal/bf16/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.2752 / 0.241 | [passed](program_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.3187 / 0.2419 |
| `metal/bf16/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2433 / 0.2168 | [passed](program_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2394 / 0.2168 |
| `metal/bf16/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.2362 / 0.2405 | [passed](program_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.2922 / 0.2428 |
| `metal/bf16/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2408 / 0.2234 | [passed](program_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2565 / 0.222 |
| `metal/bf16/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.279 / 0.247 | [passed](program_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.259 / 0.2483 |
| `metal/bf16/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.2405 / 0.2187 | [passed](program_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.23 / 0.2161 |
| `metal/bf16/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.2306 / 0.2471 | [passed](program_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.3102 / 0.2423 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
