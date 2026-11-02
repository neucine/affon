# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `70187b96c80401d5f6576ed4dc2b1448a8bf8731ea2ac8edb0717506c7bb6ed2`  
Binary SHA-256: `dfea8b4892ade85ba02a87867f0ce1dab46e45622e6260b2b4e6a588277cceef`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/strided/inner_strided_r3_c5` | normalization | 3 × 5 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r3_c5.json) | 0.0097 / 0.2204 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r3_c5.json) | 0.1831 / 0.2402 |
| `metal/f32/layer_norm/strided/inner_strided_r33_c65` | normalization | 33 × 65 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r33_c65.json) | 0.053 / 0.2863 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r33_c65.json) | 0.2518 / 0.3251 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c5` | normalization | 3 × 5 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r3_c5.json) | 0.0095 / 0.2505 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r3_c5.json) | 0.1816 / 0.2008 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c129` | normalization | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r16_c129.json) | 0.0426 / 0.2128 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r16_c129.json) | 0.228 / 0.277 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
