# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `6e5b221ae89cddf6cdb5713e72d8f6843f62c985c9ba8aace0abbce50909d540`  
Binary SHA-256: `cffe62b8e4546a85edf7b842a6d443b2953546d0d6de9192bafca166f70513f1`  
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
| `metal/f32/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.253 / 0.2203 | [passed](program_peer/metal__f32__where__small_dense__grid_r7_c17.json) | 0.2532 / 0.2336 |
| `metal/f32/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.2504 / 0.2199 | [passed](program_peer/metal__f32__where__broadcast__grid_r7_c17.json) | 0.249 / 0.2222 |
| `metal/f32/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2336 / 0.2263 | [passed](program_peer/metal__f32__where__strided__grid_r7_c17.json) | 0.2522 / 0.2319 |
| `metal/f32/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.199 / 0.2213 | [passed](program_peer/metal__f32__where__offset__grid_r7_c17.json) | 0.2526 / 0.2159 |
| `metal/bf16/where/small_dense/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.2496 / 0.2083 | [passed](program_peer/metal__bf16__where__small_dense__grid_r7_c17.json) | 0.246 / 0.2107 |
| `metal/bf16/where/broadcast/grid_r7_c17` | comparison | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2418 / 0.2172 | [passed](program_peer/metal__bf16__where__broadcast__grid_r7_c17.json) | 0.2604 / 0.2171 |
| `metal/bf16/where/strided/grid_r7_c17` | comparison | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2472 / 0.2349 | [passed](program_peer/metal__bf16__where__strided__grid_r7_c17.json) | 0.2606 / 0.2158 |
| `metal/bf16/where/offset/grid_r7_c17` | comparison | 7 × 17 | offset | correct | [passed](operator_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.2434 / 0.2196 | [passed](program_peer/metal__bf16__where__offset__grid_r7_c17.json) | 0.26 / 0.2244 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
