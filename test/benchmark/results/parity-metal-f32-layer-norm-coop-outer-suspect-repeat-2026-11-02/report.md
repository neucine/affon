# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `9b3db55cc151e26af82155b1d18266d9e5f54d2b765837390e506afbecc53073`  
Binary SHA-256: `921ae28ce07860990ec9e503afd1d4472af2c309c647eb11d084cd7d329cdb62`  
Timing: 5 iterations × 3 samples × 2 rounds

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
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c257.json) | 0.4224 / 0.2785 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c257.json) | 0.3805 / 0.2571 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
