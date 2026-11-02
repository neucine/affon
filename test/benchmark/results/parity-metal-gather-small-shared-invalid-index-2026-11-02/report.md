# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `6d447c4a14c06776b59061cdf79a7631db0f624c995283fed8d0627887d9fdb4`  
Binary SHA-256: `5be7468dbd2419582c4ea8118a0c9de0adb843bdf1d69efd17968bf64c02152c`  
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 2
- Numerical check passed: 0
- Operator check and timing passed: 0
- Program check and timing passed: 0
- Numerical failures: 2
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/gather/invalid_index` | indexing | 4 × 8 | dense | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/gather/invalid_index` | indexing | 4 × 8 | dense | probe_failed | probe_not_correct | — | probe_not_correct | — |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
