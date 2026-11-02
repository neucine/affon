# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2eb07191918dcfc279e8abb910c228764742345b969c6ab597692769fd70aa74`  
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
| `metal/f32/dot/dot_or_vector/n1` | matrix | 1 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.1786 / 0.2113 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.175 / 0.208 |
| `metal/f32/dot/dot_or_vector/n9` | matrix | 9 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.1758 / 0.2051 | [passed](program_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.1845 / 0.2055 |
| `metal/f32/dot/dot_or_vector/n17` | matrix | 17 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.1832 / 0.2053 | [passed](program_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.186 / 0.2052 |
| `metal/f32/dot/dot_or_vector/n25` | matrix | 25 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.1797 / 0.2048 | [passed](program_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.1813 / 0.2082 |
| `metal/f32/dot/dot_or_vector/n33` | matrix | 33 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.1861 / 0.2036 | [passed](program_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.1869 / 0.2076 |
| `metal/f32/dot/dot_or_vector/n41` | matrix | 41 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.1966 / 0.2063 | [passed](program_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.205 / 0.207 |
| `metal/f32/dot/dot_or_vector/n257` | matrix | 257 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.2458 / 0.2014 | [passed](program_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.2398 / 0.2057 |
| `metal/f32/dot/dot_or_vector/n2048` | matrix | 2048 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.4199 / 0.2063 | [passed](program_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.4354 / 0.2061 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
