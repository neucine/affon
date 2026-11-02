# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f31114741dc038aa2afa21918ea84a27fb2d49a0b369c0edc21d29a6bbcba782`  
Binary SHA-256: `f25caea82a00bafd045213e187e5aa8233a760c352d2f4df4d3079e60d2aaadd`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 8
- Numerical check passed: 8
- Operator check and timing passed: 8
- Program check and timing passed: 8
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/dot/strided/n1` | matrix | 1 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n1.json) | 0.1997 / 0.2066 | [passed](program_peer/metal__f32__dot__strided__n1.json) | 0.1744 / 0.2115 |
| `metal/f32/dot/strided/n9` | matrix | 9 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n9.json) | 0.1742 / 0.2076 | [passed](program_peer/metal__f32__dot__strided__n9.json) | 0.188 / 0.2135 |
| `metal/f32/dot/strided/n17` | matrix | 17 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n17.json) | 0.1819 / 0.2104 | [passed](program_peer/metal__f32__dot__strided__n17.json) | 0.187 / 0.2126 |
| `metal/f32/dot/strided/n31` | matrix | 31 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n31.json) | 0.1777 / 0.1914 | [passed](program_peer/metal__f32__dot__strided__n31.json) | 0.2141 / 0.1796 |
| `metal/f32/dot/dot_or_vector/offset_n1` | matrix | 1 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.1703 / 0.1898 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.1738 / 0.2115 |
| `metal/f32/dot/dot_or_vector/offset_n9` | matrix | 9 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.1676 / 0.2076 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.1794 / 0.2127 |
| `metal/f32/dot/dot_or_vector/offset_n31` | matrix | 31 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.1857 / 0.2084 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.1913 / 0.2127 |
| `metal/f32/dot/dot_or_vector/offset_n128` | matrix | 128 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.1879 / 0.2097 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.1802 / 0.2101 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
