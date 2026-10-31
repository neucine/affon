# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `70dd434ef2d870be793fd834b277211108acbb764f049a2c8ac3abfea826a596`  
Binary SHA-256: `94e3c1795a7cbcae6e6202b022ce83d5f4d5a69f03d81ab1e9de230af9c7bc4e`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 1
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/topk/large/offset_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | offset | correct | [passed](operator_peer/metal__bf16__topk__large__offset_axis1_r128_c257_k256_smallest.json) | 3.215 / 1.135 | [passed](program_peer/metal__bf16__topk__large__offset_axis1_r128_c257_k256_smallest.json) | 3.192 / 1.172 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
