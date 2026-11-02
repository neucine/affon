# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2dac1bee20146631184e47a9095625546387435f20e131ac0a1ed44540471fc4`  
Binary SHA-256: `7ae44a281e0e1aad23de1512f2dedc0ee4a81b938738f4166e2eab6964097553`  
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
| `metal/bf16/dot/strided/n1` | matrix | 1 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n1.json) | 0.1833 / 0.2072 | [passed](program_peer/metal__bf16__dot__strided__n1.json) | 0.2173 / 0.207 |
| `metal/bf16/dot/strided/n9` | matrix | 9 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n9.json) | 0.178 / 0.2089 | [passed](program_peer/metal__bf16__dot__strided__n9.json) | 0.1855 / 0.2084 |
| `metal/bf16/dot/strided/n17` | matrix | 17 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n17.json) | 0.1916 / 0.2079 | [passed](program_peer/metal__bf16__dot__strided__n17.json) | 0.1862 / 0.2138 |
| `metal/bf16/dot/strided/n31` | matrix | 31 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n31.json) | 0.1945 / 0.2076 | [passed](program_peer/metal__bf16__dot__strided__n31.json) | 0.1927 / 0.2109 |
| `metal/bf16/dot/dot_or_vector/offset_n1` | matrix | 1 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.1817 / 0.206 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.1847 / 0.21 |
| `metal/bf16/dot/dot_or_vector/offset_n9` | matrix | 9 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.176 / 0.2077 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.2024 / 0.2129 |
| `metal/bf16/dot/dot_or_vector/offset_n31` | matrix | 31 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.1993 / 0.2069 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.212 / 0.2089 |
| `metal/bf16/dot/dot_or_vector/offset_n128` | matrix | 128 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.1845 / 0.2085 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.2 / 0.2091 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
