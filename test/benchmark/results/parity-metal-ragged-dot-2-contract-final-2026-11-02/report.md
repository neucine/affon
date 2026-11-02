# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `62eae18127ec27edc5c950099c91d36e5dd8717482a8d249326cb5acc94b0a85`  
Binary SHA-256: `3552c52bc08639089d28f433463f3fb1e1502ad44033b793a0d38ceebf3bb968`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/f32/dot/ragged` | matrix | 257 | dense | correct | [passed](operator_peer/metal__f32__dot__ragged.json) | 0.2135 / 0.1982 | [passed](program_peer/metal__f32__dot__ragged.json) | 0.1922 / 0.2079 |
| `metal/bf16/dot/ragged` | matrix | 259 | dense | correct | [passed](operator_peer/metal__bf16__dot__ragged.json) | 0.1864 / 0.194 | [passed](program_peer/metal__bf16__dot__ragged.json) | 0.1891 / 0.2052 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
