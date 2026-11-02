# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7172fab929d1ccd402c0bc7a9659dfd2e241dd2bb8fa0564460b2217066456c7`  
Binary SHA-256: `f26e0cb5fae097320dbc2a16110f1de81a2065b8cddd0208ce927a2b94687b7f`  
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
| `metal/f32/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/metal__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.0095 / 0.247 | [passed](program_peer/metal__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.2285 / 0.2735 |
| `metal/f32/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/metal__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.0309 / 0.2577 | [passed](program_peer/metal__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.2689 / 0.2748 |
| `metal/f32/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/metal__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.0975 / 0.2753 | [passed](program_peer/metal__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.3188 / 0.2876 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
