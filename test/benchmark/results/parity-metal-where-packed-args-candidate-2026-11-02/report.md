# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d8d46850cd9fba03b56c2b61310ccede1bd2dd6793fb1493673549050332e73d`  
Binary SHA-256: `05ba9cc4cee3edf3aaea32f241810cbea3bf379a5a82e21eb9f8c1b010b3fe3b`  
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
| `metal/f32/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2816 / 0.2421 | [passed](program_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2552 / 0.2394 |
| `metal/f32/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2781 / 0.2329 | [passed](program_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2815 / 0.2401 |
| `metal/f32/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2518 / 0.245 | [passed](program_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2872 / 0.2434 |
| `metal/f32/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.28 / 0.2447 | [passed](program_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2855 / 0.2427 |
| `metal/bf16/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.2702 / 0.241 | [passed](program_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.3166 / 0.2411 |
| `metal/bf16/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.2824 / 0.2383 | [passed](program_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.3174 / 0.2339 |
| `metal/bf16/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.2756 / 0.2439 | [passed](program_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.3156 / 0.2473 |
| `metal/bf16/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.2701 / 0.2433 | [passed](program_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.3139 / 0.2466 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
