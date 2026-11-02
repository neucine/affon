# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `6922dbbeed2cf335e7462c0003a85fef0a420352d5f67cebed9602d8ec9269fd`  
Binary SHA-256: `a9a10dd4a4dd6ab577715a114b1261d475aeddc676e7750489ffff1b83de8cef`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 35
- Numerical check passed: 35
- Operator check and timing passed: 35
- Program check and timing passed: 35
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/dot_or_vector` | matrix | 1 × 64 × 1 | dense | correct | [passed](operator_peer/metal__bf16__matmul__dot_or_vector.json) | 0.2819 / 0.1993 | [passed](program_peer/metal__bf16__matmul__dot_or_vector.json) | 0.2967 / 0.2021 |
| `metal/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n64_default.json) | 0.2913 / 0.2017 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n64_default.json) | 0.3174 / 0.197 |
| `metal/bf16/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n64_dense.json) | 0.3089 / 0.2019 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n64_dense.json) | 0.3109 / 0.2072 |
| `metal/bf16/matmul/decode_m1/k64_n63_default` | matrix | 1 × 64 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n63_default.json) | 0.3089 / 0.1981 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n63_default.json) | 0.2985 / 0.2026 |
| `metal/bf16/matmul/decode_m1/k64_n63_dense` | matrix | 1 × 64 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n63_dense.json) | 0.3081 / 0.2034 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n63_dense.json) | 0.3153 / 0.2019 |
| `metal/bf16/matmul/decode_m1/k64_n64_default` | matrix | 1 × 64 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n64_default.json) | 0.3122 / 0.1987 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n64_default.json) | 0.3084 / 0.206 |
| `metal/bf16/matmul/decode_m1/k64_n64_dense` | matrix | 1 × 64 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n64_dense.json) | 0.3086 / 0.2019 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n64_dense.json) | 0.3153 / 0.205 |
| `metal/bf16/matmul/decode_m1/k65_n64_default` | matrix | 1 × 65 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n64_default.json) | 0.3109 / 0.1943 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n64_default.json) | 0.2722 / 0.1992 |
| `metal/bf16/matmul/decode_m1/k65_n64_dense` | matrix | 1 × 65 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n64_dense.json) | 0.3116 / 0.2011 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n64_dense.json) | 0.318 / 0.2053 |
| `metal/bf16/matmul/decode_m1/k64_n65_default` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n65_default.json) | 0.3099 / 0.1989 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n65_default.json) | 0.3208 / 0.2057 |
| `metal/bf16/matmul/decode_m1/k64_n65_dense` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n65_dense.json) | 0.3065 / 0.2022 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n65_dense.json) | 0.3196 / 0.2041 |
| `metal/bf16/matmul/decode_m1/k63_n65_default` | matrix | 1 × 63 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n65_default.json) | 0.3074 / 0.2063 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n65_default.json) | 0.3037 / 0.197 |
| `metal/bf16/matmul/decode_m1/k63_n65_dense` | matrix | 1 × 63 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n65_dense.json) | 0.306 / 0.1965 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n65_dense.json) | 0.2431 / 0.2008 |
| `metal/bf16/matmul/decode_m1/k65_n63_default` | matrix | 1 × 65 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n63_default.json) | 0.3067 / 0.2029 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n63_default.json) | 0.3204 / 0.2076 |
| `metal/bf16/matmul/decode_m1/k65_n63_dense` | matrix | 1 × 65 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n63_dense.json) | 0.3012 / 0.2013 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n63_dense.json) | 0.3165 / 0.2025 |
| `metal/bf16/matmul/decode_m1/k65_n65_default` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n65_default.json) | 0.3038 / 0.2055 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n65_default.json) | 0.2489 / 0.2074 |
| `metal/bf16/matmul/decode_m1/k65_n65_dense` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n65_dense.json) | 0.2546 / 0.2061 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n65_dense.json) | 0.2603 / 0.2046 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32_default` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_default.json) | 0.2692 / 0.1949 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_default.json) | 0.2975 / 0.2032 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32_dense` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_dense.json) | 0.264 / 0.2032 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_dense.json) | 0.2331 / 0.1999 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31_default` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_default.json) | 0.2636 / 0.2 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_default.json) | 0.2652 / 0.2018 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31_dense` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_dense.json) | 0.268 / 0.1968 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_dense.json) | 0.235 / 0.2009 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32_default` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_default.json) | 0.2603 / 0.1974 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_default.json) | 0.259 / 0.2011 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32_dense` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_dense.json) | 0.2744 / 0.2029 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_dense.json) | 0.2937 / 0.2037 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32_default` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_default.json) | 0.2319 / 0.2057 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_default.json) | 0.2964 / 0.2055 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32_dense` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_dense.json) | 0.2502 / 0.1993 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_dense.json) | 0.2919 / 0.1957 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33_default` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_default.json) | 0.2406 / 0.1998 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_default.json) | 0.2661 / 0.2036 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33_dense` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_dense.json) | 0.2648 / 0.1991 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_dense.json) | 0.259 / 0.2026 |
| `metal/bf16/matmul/decode_m1/m1_k127_n128_default` | matrix | 1 × 127 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_default.json) | 0.2584 / 0.1985 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_default.json) | 0.2992 / 0.2056 |
| `metal/bf16/matmul/decode_m1/m1_k127_n128_dense` | matrix | 1 × 127 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_dense.json) | 0.3358 / 0.2036 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_dense.json) | 0.2529 / 0.2017 |
| `metal/bf16/matmul/decode_m1/m1_k31_n127_default` | matrix | 1 × 31 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_default.json) | 0.2627 / 0.1992 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_default.json) | 0.273 / 0.2059 |
| `metal/bf16/matmul/decode_m1/m1_k31_n127_dense` | matrix | 1 × 31 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_dense.json) | 0.2759 / 0.2025 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_dense.json) | 0.2974 / 0.2 |
| `metal/bf16/matmul/decode_m1/m1_k32_n127_default` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_default.json) | 0.222 / 0.201 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_default.json) | 0.2444 / 0.2066 |
| `metal/bf16/matmul/decode_m1/m1_k32_n127_dense` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_dense.json) | 0.2627 / 0.2049 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_dense.json) | 0.264 / 0.2042 |
| `metal/bf16/matmul/decode_m1/m1_k127_n127_default` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_default.json) | 0.2754 / 0.1987 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_default.json) | 0.3379 / 0.1952 |
| `metal/bf16/matmul/decode_m1/m1_k127_n127_dense` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_dense.json) | 0.3361 / 0.2031 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_dense.json) | 0.3451 / 0.1926 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
