# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7c93b119395b06385764f29f6130de0d53d0a21aac09f7e4492cf2f23a30e567`  
Binary SHA-256: `5e934b3026f06481e7b89dd409dc51caa017fd00e07a31fa6d5bad76a3adc4d2`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/topk/small/dense_axis1_r16_c64_k33` | indexing | 16 × 64 | dense | correct | [passed](operator_peer/metal__f32__topk__small__dense_axis1_r16_c64_k33.json) | 0.4324 / 0.5104 | [passed](program_peer/metal__f32__topk__small__dense_axis1_r16_c64_k33.json) | 0.3146 / 0.4889 |
| `metal/f32/topk/small/dense_axis1_r33_c129_k65` | indexing | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__topk__small__dense_axis1_r33_c129_k65.json) | 0.438 / 0.5575 | [passed](program_peer/metal__f32__topk__small__dense_axis1_r33_c129_k65.json) | 0.4388 / 0.5151 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
