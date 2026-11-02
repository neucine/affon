# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `26bfd9c0142648eccc9f67093a142ba1e0f8f9676a1317332e8ec7e24f962b06`  
Binary SHA-256: `907c1479836f2e018c427ac7ddcc3b6c075774a5091eed1656b1315bb871a1c6`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/metal__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.0096 / 0.5483 | [passed](program_peer/metal__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.4824 / 0.7815 |
| `metal/bf16/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.036 / 0.5881 | [passed](program_peer/metal__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.5388 / 0.5315 |
| `metal/bf16/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/metal__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.1042 / 0.6125 | [passed](program_peer/metal__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.8755 / 0.9246 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
