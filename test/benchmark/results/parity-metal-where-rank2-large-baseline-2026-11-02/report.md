# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d8d46850cd9fba03b56c2b61310ccede1bd2dd6793fb1493673549050332e73d`  
Binary SHA-256: `98da19bc4ae5f89bbfd99df5feac070b6b71bcd035159ed20c8d5eb25c383646`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 8
- Numerical check passed: 8
- Operator check and timing passed: 8
- Program check and timing passed: 8
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2663 / 0.2287 | [passed](program_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2603 / 0.2504 |
| `metal/f32/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2558 / 0.2279 | [passed](program_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2224 / 0.2302 |
| `metal/f32/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2586 / 0.2321 | [passed](program_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2441 / 0.2363 |
| `metal/f32/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2559 / 0.2299 | [passed](program_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2278 / 0.2366 |
| `metal/bf16/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.262 / 0.2286 | [passed](program_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.2803 / 0.2364 |
| `metal/bf16/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.2603 / 0.2303 | [passed](program_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.3111 / 0.2415 |
| `metal/bf16/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.2585 / 0.237 | [passed](program_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.3142 / 0.242 |
| `metal/bf16/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.2523 / 0.2349 | [passed](program_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.3145 / 0.2462 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
