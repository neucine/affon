# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d8d46850cd9fba03b56c2b61310ccede1bd2dd6793fb1493673549050332e73d`  
Binary SHA-256: `5330adb59964a8b3f2430f73e9b3a83a9ee0c958afae8e50e88276649e8e5b6f`  
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
| `metal/f32/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2473 / 0.2411 | [passed](program_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.2843 / 0.2302 |
| `metal/f32/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2767 / 0.2393 | [passed](program_peer/metal__f32__where__broadcast__grid_r16_c129.json) | 0.2851 / 0.2394 |
| `metal/f32/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2793 / 0.2389 | [passed](program_peer/metal__f32__where__strided__grid_r16_c129.json) | 0.2817 / 0.2453 |
| `metal/f32/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2575 / 0.2456 | [passed](program_peer/metal__f32__where__offset__grid_r16_c129.json) | 0.2767 / 0.2447 |
| `metal/bf16/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.2753 / 0.2402 | [passed](program_peer/metal__bf16__where__small_dense__grid_r16_c129.json) | 0.2611 / 0.2374 |
| `metal/bf16/where/broadcast/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.2792 / 0.2422 | [passed](program_peer/metal__bf16__where__broadcast__grid_r16_c129.json) | 0.302 / 0.2388 |
| `metal/bf16/where/strided/grid_r16_c129` | comparison | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.2816 / 0.2417 | [passed](program_peer/metal__bf16__where__strided__grid_r16_c129.json) | 0.3107 / 0.246 |
| `metal/bf16/where/offset/grid_r16_c129` | comparison | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.2781 / 0.2462 | [passed](program_peer/metal__bf16__where__offset__grid_r16_c129.json) | 0.3156 / 0.2444 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
