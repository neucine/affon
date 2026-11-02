# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2d0fc117692e51ac37d5bbbc54a68bd5524b95f9820eaf72a820372b3869c287`  
Binary SHA-256: `164de0a2a31d0861b68c9ed0b40373fa5509390338cc05ee342db6d317df4f82`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `metal/f32/one_hot/small` | indexing | 1 × 1 | dense | correct | [passed](operator_peer/metal__f32__one_hot__small.json) | 0.0088 / 0.4626 | [passed](program_peer/metal__f32__one_hot__small.json) | 0.2348 / 0.4007 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
