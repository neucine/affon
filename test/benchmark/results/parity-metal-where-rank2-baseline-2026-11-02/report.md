# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `6e5b221ae89cddf6cdb5713e72d8f6843f62c985c9ba8aace0abbce50909d540`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
Timing: 5 iterations × 5 samples × 3 rounds

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
| `metal/f32/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.2708 / 0.2469 | [passed](program_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.2518 / 0.234 |
| `metal/f32/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.2436 / 0.2005 | [passed](program_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.2518 / 0.2196 |
| `metal/f32/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2446 / 0.2248 | [passed](program_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2498 / 0.2196 |
| `metal/f32/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.2248 / 0.2202 | [passed](program_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.2488 / 0.2222 |
| `metal/bf16/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2466 / 0.2156 | [passed](program_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2584 / 0.2196 |
| `metal/bf16/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2464 / 0.2206 | [passed](program_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2586 / 0.2189 |
| `metal/bf16/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2454 / 0.2265 | [passed](program_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2352 / 0.2252 |
| `metal/bf16/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.242 / 0.2031 | [passed](program_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.2444 / 0.2186 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
