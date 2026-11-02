# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f31114741dc038aa2afa21918ea84a27fb2d49a0b369c0edc21d29a6bbcba782`  
Binary SHA-256: `9b093ba558b2da5073efbbb5746f369bc12fc55578b54a47186c448e4b325d70`  
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
| `metal/f32/dot/strided/n1` | matrix | 1 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n1.json) | 0.1782 / 0.2059 | [passed](program_peer/metal__f32__dot__strided__n1.json) | 0.1765 / 0.2077 |
| `metal/f32/dot/strided/n9` | matrix | 9 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n9.json) | 0.1686 / 0.2058 | [passed](program_peer/metal__f32__dot__strided__n9.json) | 0.1604 / 0.2097 |
| `metal/f32/dot/strided/n17` | matrix | 17 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n17.json) | 0.1725 / 0.2061 | [passed](program_peer/metal__f32__dot__strided__n17.json) | 0.164 / 0.1848 |
| `metal/f32/dot/strided/n31` | matrix | 31 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n31.json) | 0.1824 / 0.1987 | [passed](program_peer/metal__f32__dot__strided__n31.json) | 0.2033 / 0.2122 |
| `metal/f32/dot/dot_or_vector/offset_n1` | matrix | 1 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.1838 / 0.2101 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.1767 / 0.2074 |
| `metal/f32/dot/dot_or_vector/offset_n9` | matrix | 9 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.1402 / 0.1989 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.1794 / 0.2099 |
| `metal/f32/dot/dot_or_vector/offset_n31` | matrix | 31 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.1685 / 0.1837 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.1956 / 0.2081 |
| `metal/f32/dot/dot_or_vector/offset_n128` | matrix | 128 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.2264 / 0.2046 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.2196 / 0.2072 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
