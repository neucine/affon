# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `6922dbbeed2cf335e7462c0003a85fef0a420352d5f67cebed9602d8ec9269fd`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
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
| `metal/bf16/matmul/dot_or_vector` | matrix | 1 × 64 × 1 | dense | correct | [passed](operator_peer/metal__bf16__matmul__dot_or_vector.json) | 0.2682 / 0.195 | [passed](program_peer/metal__bf16__matmul__dot_or_vector.json) | 0.2936 / 0.202 |
| `metal/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n64_default.json) | 0.292 / 0.1838 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n64_default.json) | 0.3182 / 0.1976 |
| `metal/bf16/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n64_dense.json) | 0.3058 / 0.2015 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n64_dense.json) | 0.2818 / 0.2005 |
| `metal/bf16/matmul/decode_m1/k64_n63_default` | matrix | 1 × 64 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n63_default.json) | 0.3099 / 0.2032 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n63_default.json) | 0.2801 / 0.1948 |
| `metal/bf16/matmul/decode_m1/k64_n63_dense` | matrix | 1 × 64 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n63_dense.json) | 0.3071 / 0.202 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n63_dense.json) | 0.3167 / 0.202 |
| `metal/bf16/matmul/decode_m1/k64_n64_default` | matrix | 1 × 64 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n64_default.json) | 0.308 / 0.2036 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n64_default.json) | 0.3192 / 0.2024 |
| `metal/bf16/matmul/decode_m1/k64_n64_dense` | matrix | 1 × 64 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n64_dense.json) | 0.311 / 0.2038 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n64_dense.json) | 0.2925 / 0.2067 |
| `metal/bf16/matmul/decode_m1/k65_n64_default` | matrix | 1 × 65 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n64_default.json) | 0.3075 / 0.1994 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n64_default.json) | 0.2943 / 0.2037 |
| `metal/bf16/matmul/decode_m1/k65_n64_dense` | matrix | 1 × 65 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n64_dense.json) | 0.3098 / 0.2017 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n64_dense.json) | 0.3212 / 0.2041 |
| `metal/bf16/matmul/decode_m1/k64_n65_default` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n65_default.json) | 0.2423 / 0.203 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n65_default.json) | 0.3172 / 0.2021 |
| `metal/bf16/matmul/decode_m1/k64_n65_dense` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n65_dense.json) | 0.3068 / 0.2039 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n65_dense.json) | 0.3181 / 0.2072 |
| `metal/bf16/matmul/decode_m1/k63_n65_default` | matrix | 1 × 63 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n65_default.json) | 0.3101 / 0.2014 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n65_default.json) | 0.2345 / 0.2035 |
| `metal/bf16/matmul/decode_m1/k63_n65_dense` | matrix | 1 × 63 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n65_dense.json) | 0.306 / 0.2006 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n65_dense.json) | 0.3019 / 0.2036 |
| `metal/bf16/matmul/decode_m1/k65_n63_default` | matrix | 1 × 65 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n63_default.json) | 0.3052 / 0.2014 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n63_default.json) | 0.3082 / 0.2029 |
| `metal/bf16/matmul/decode_m1/k65_n63_dense` | matrix | 1 × 65 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n63_dense.json) | 0.3058 / 0.2043 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n63_dense.json) | 0.3229 / 0.2024 |
| `metal/bf16/matmul/decode_m1/k65_n65_default` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n65_default.json) | 0.3109 / 0.2067 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n65_default.json) | 0.3189 / 0.2001 |
| `metal/bf16/matmul/decode_m1/k65_n65_dense` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n65_dense.json) | 0.3109 / 0.1945 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n65_dense.json) | 0.3211 / 0.2059 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32_default` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_default.json) | 0.264 / 0.2035 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_default.json) | 0.2912 / 0.1949 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32_dense` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_dense.json) | 0.2683 / 0.1973 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_dense.json) | 0.2681 / 0.198 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31_default` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_default.json) | 0.264 / 0.2031 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_default.json) | 0.2942 / 0.2048 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31_dense` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_dense.json) | 0.266 / 0.202 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_dense.json) | 0.2614 / 0.2072 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32_default` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_default.json) | 0.2682 / 0.2027 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_default.json) | 0.2789 / 0.2013 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32_dense` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_dense.json) | 0.2681 / 0.2003 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_dense.json) | 0.294 / 0.2022 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32_default` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_default.json) | 0.2785 / 0.2013 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_default.json) | 0.2553 / 0.2022 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32_dense` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_dense.json) | 0.2779 / 0.2011 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_dense.json) | 0.2955 / 0.1931 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33_default` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_default.json) | 0.2645 / 0.2011 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_default.json) | 0.2519 / 0.1987 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33_dense` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_dense.json) | 0.2535 / 0.2046 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_dense.json) | 0.2897 / 0.2029 |
| `metal/bf16/matmul/decode_m1/m1_k127_n128_default` | matrix | 1 × 127 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_default.json) | 0.3142 / 0.2034 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_default.json) | 0.2701 / 0.1986 |
| `metal/bf16/matmul/decode_m1/m1_k127_n128_dense` | matrix | 1 × 127 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_dense.json) | 0.3126 / 0.1981 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_dense.json) | 0.3057 / 0.2028 |
| `metal/bf16/matmul/decode_m1/m1_k31_n127_default` | matrix | 1 × 31 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_default.json) | 0.242 / 0.1952 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_default.json) | 0.2415 / 0.2071 |
| `metal/bf16/matmul/decode_m1/m1_k31_n127_dense` | matrix | 1 × 31 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_dense.json) | 0.2701 / 0.2009 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_dense.json) | 0.3008 / 0.2047 |
| `metal/bf16/matmul/decode_m1/m1_k32_n127_default` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_default.json) | 0.261 / 0.2054 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_default.json) | 0.2418 / 0.2053 |
| `metal/bf16/matmul/decode_m1/m1_k32_n127_dense` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_dense.json) | 0.2282 / 0.1944 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_dense.json) | 0.2935 / 0.201 |
| `metal/bf16/matmul/decode_m1/m1_k127_n127_default` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_default.json) | 0.2917 / 0.1869 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_default.json) | 0.3581 / 0.205 |
| `metal/bf16/matmul/decode_m1/m1_k127_n127_dense` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_dense.json) | 0.3517 / 0.1966 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_dense.json) | 0.2561 / 0.2083 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
