# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f5db72dfbec8cf224ff5de55c7d194a29f531b9fc4aab69a81fd69a8897b31c2`  
Binary SHA-256: `9b093ba558b2da5073efbbb5746f369bc12fc55578b54a47186c448e4b325d70`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/bf16/dot/ragged` | matrix | 259 | dense | correct | [passed](operator_peer/metal__bf16__dot__ragged.json) | 0.1965 / 0.1888 | [passed](program_peer/metal__bf16__dot__ragged.json) | 0.197 / 0.1938 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
