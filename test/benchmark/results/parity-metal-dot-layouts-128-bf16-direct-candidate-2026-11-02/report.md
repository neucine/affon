# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2dac1bee20146631184e47a9095625546387435f20e131ac0a1ed44540471fc4`  
Binary SHA-256: `4110cf5d4d771282557173983743bc2a5c26ea20d60537aaf7d81f9efd666935`  
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
| `metal/bf16/dot/strided/n1` | matrix | 1 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n1.json) | 0.1781 / 0.2089 | [passed](program_peer/metal__bf16__dot__strided__n1.json) | 0.1865 / 0.205 |
| `metal/bf16/dot/strided/n9` | matrix | 9 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n9.json) | 0.1723 / 0.2091 | [passed](program_peer/metal__bf16__dot__strided__n9.json) | 0.1866 / 0.2065 |
| `metal/bf16/dot/strided/n17` | matrix | 17 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n17.json) | 0.1858 / 0.2098 | [passed](program_peer/metal__bf16__dot__strided__n17.json) | 0.1876 / 0.2089 |
| `metal/bf16/dot/strided/n31` | matrix | 31 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n31.json) | 0.1859 / 0.2105 | [passed](program_peer/metal__bf16__dot__strided__n31.json) | 0.1961 / 0.21 |
| `metal/bf16/dot/dot_or_vector/offset_n1` | matrix | 1 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.1719 / 0.209 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.1824 / 0.2065 |
| `metal/bf16/dot/dot_or_vector/offset_n9` | matrix | 9 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.1742 / 0.2076 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.1754 / 0.21 |
| `metal/bf16/dot/dot_or_vector/offset_n31` | matrix | 31 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.1714 / 0.2042 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.183 / 0.2095 |
| `metal/bf16/dot/dot_or_vector/offset_n128` | matrix | 128 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.2274 / 0.2034 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.2156 / 0.2091 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
