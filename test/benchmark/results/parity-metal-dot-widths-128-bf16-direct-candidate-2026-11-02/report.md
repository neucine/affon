# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `e8c86177d5d6d4c78fd0d63f5189ea137a6d8ab32e2f00b4d14e1591f7a44fbf`  
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
| `metal/bf16/dot/dot_or_vector/n1` | matrix | 1 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.1852 / 0.208 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.1761 / 0.2072 |
| `metal/bf16/dot/dot_or_vector/n9` | matrix | 9 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.1772 / 0.2049 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.1712 / 0.2076 |
| `metal/bf16/dot/dot_or_vector/n17` | matrix | 17 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.1828 / 0.205 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.1845 / 0.2061 |
| `metal/bf16/dot/dot_or_vector/n25` | matrix | 25 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.1775 / 0.2048 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.1882 / 0.2067 |
| `metal/bf16/dot/dot_or_vector/n33` | matrix | 33 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.1956 / 0.2017 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.1845 / 0.2092 |
| `metal/bf16/dot/dot_or_vector/n41` | matrix | 41 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.181 / 0.205 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.1911 / 0.2063 |
| `metal/bf16/dot/dot_or_vector/n257` | matrix | 257 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.2286 / 0.2025 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.2446 / 0.2079 |
| `metal/bf16/dot/dot_or_vector/n2048` | matrix | 2048 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.3943 / 0.2005 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.3942 / 0.2071 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
