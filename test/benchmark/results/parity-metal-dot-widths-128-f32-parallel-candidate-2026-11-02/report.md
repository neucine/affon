# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2eb07191918dcfc279e8abb910c228764742345b969c6ab597692769fd70aa74`  
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
| `metal/f32/dot/dot_or_vector/n1` | matrix | 1 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.1829 / 0.21 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.1896 / 0.2119 |
| `metal/f32/dot/dot_or_vector/n9` | matrix | 9 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.1736 / 0.2077 | [passed](program_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.1818 / 0.21 |
| `metal/f32/dot/dot_or_vector/n17` | matrix | 17 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.1781 / 0.209 | [passed](program_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.1812 / 0.2055 |
| `metal/f32/dot/dot_or_vector/n25` | matrix | 25 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.1854 / 0.2059 | [passed](program_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.2022 / 0.2101 |
| `metal/f32/dot/dot_or_vector/n33` | matrix | 33 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.1898 / 0.2064 | [passed](program_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.1818 / 0.2095 |
| `metal/f32/dot/dot_or_vector/n41` | matrix | 41 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.1857 / 0.2051 | [passed](program_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.1886 / 0.2093 |
| `metal/f32/dot/dot_or_vector/n257` | matrix | 257 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.1728 / 0.2078 | [passed](program_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.1813 / 0.2118 |
| `metal/f32/dot/dot_or_vector/n2048` | matrix | 2048 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.1896 / 0.2058 | [passed](program_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.1821 / 0.2083 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
