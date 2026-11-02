# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `63c9906193991766d04f38dbea0c09d3962ea8f0fd9a61a38ad1361803721dd0`  
Binary SHA-256: `7ae44a281e0e1aad23de1512f2dedc0ee4a81b938738f4166e2eab6964097553`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 1
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/scatter_add/invalid_index` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__invalid_index.json) | 0.1987 / 0.2167 | [passed](program_peer/metal__f32__scatter_add__invalid_index.json) | 0.2269 / 0.2261 |
| `metal/bf16/scatter_add/invalid_index` | indexing | 3 × 3 | dense | probe_failed | probe_not_correct | — | probe_not_correct | — |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
