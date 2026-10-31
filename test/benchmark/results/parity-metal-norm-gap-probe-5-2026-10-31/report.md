# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `0c55c5161b0b4a432b6cbeee4a6fcbe26f194a91e1061b3e92d5b191cb9aded3`  
Binary SHA-256: `b9c4777916fad9e283cd39ab5416742527a8697eae0a087abbb1e8b7c6b97a0c`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 5
- Numerical check passed: 5
- Operator check and timing passed: 5
- Program check and timing passed: 5
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/rms_norm/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r256_c64.json) | 0.6126 / 0.5079 | [passed](program_peer/metal__bf16__rms_norm__long_row__r256_c64.json) | 0.6348 / 0.5477 |
| `metal/f32/rms_norm/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r256_c64.json) | 0.6167 / 0.414 | [passed](program_peer/metal__f32__rms_norm__long_row__r256_c64.json) | 0.6166 / 0.5147 |
| `metal/bf16/rms_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r64_c255.json) | 1.091 / 0.4358 | [passed](program_peer/metal__bf16__rms_norm__long_row__r64_c255.json) | 1.098 / 0.4417 |
| `metal/bf16/log_softmax/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r256_c64.json) | 0.521 / 0.4948 | [passed](program_peer/metal__bf16__log_softmax__long_row__r256_c64.json) | 0.4713 / 0.5117 |
| `metal/f32/log_softmax/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row__r128_c129.json) | 0.4189 / 0.4524 | [passed](program_peer/metal__f32__log_softmax__long_row__r128_c129.json) | 0.4185 / 0.465 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
