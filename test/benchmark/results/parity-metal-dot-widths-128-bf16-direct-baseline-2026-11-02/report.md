# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `e8c86177d5d6d4c78fd0d63f5189ea137a6d8ab32e2f00b4d14e1591f7a44fbf`  
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
| `metal/bf16/dot/dot_or_vector/n1` | matrix | 1 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.1816 / 0.1945 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.2043 / 0.2096 |
| `metal/bf16/dot/dot_or_vector/n9` | matrix | 9 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.1782 / 0.206 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.1873 / 0.1931 |
| `metal/bf16/dot/dot_or_vector/n17` | matrix | 17 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.1866 / 0.2035 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.1848 / 0.193 |
| `metal/bf16/dot/dot_or_vector/n25` | matrix | 25 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.1861 / 0.2044 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.2079 / 0.2092 |
| `metal/bf16/dot/dot_or_vector/n33` | matrix | 33 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.1973 / 0.2039 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.1912 / 0.2043 |
| `metal/bf16/dot/dot_or_vector/n41` | matrix | 41 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.1926 / 0.2063 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.2 / 0.2077 |
| `metal/bf16/dot/dot_or_vector/n257` | matrix | 257 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.2205 / 0.2045 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.1936 / 0.2071 |
| `metal/bf16/dot/dot_or_vector/n2048` | matrix | 2048 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.1989 / 0.1934 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.217 / 0.2083 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
