# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `6922dbbeed2cf335e7462c0003a85fef0a420352d5f67cebed9602d8ec9269fd`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/bf16/matmul/dot_or_vector` | matrix | 1 × 64 × 1 | dense | correct | [passed](operator_peer/metal__bf16__matmul__dot_or_vector.json) | 0.1928 / 0.2055 | [passed](program_peer/metal__bf16__matmul__dot_or_vector.json) | 0.2038 / 0.2032 |
| `metal/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n64_default.json) | 0.235 / 0.2013 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n64_default.json) | 0.2388 / 0.2057 |
| `metal/bf16/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n64_dense.json) | 0.2331 / 0.2052 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n64_dense.json) | 0.2402 / 0.2034 |
| `metal/bf16/matmul/decode_m1/k64_n63_default` | matrix | 1 × 64 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n63_default.json) | 0.2327 / 0.1999 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n63_default.json) | 0.2401 / 0.2029 |
| `metal/bf16/matmul/decode_m1/k64_n63_dense` | matrix | 1 × 64 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n63_dense.json) | 0.2348 / 0.2002 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n63_dense.json) | 0.2361 / 0.2024 |
| `metal/bf16/matmul/decode_m1/k64_n64_default` | matrix | 1 × 64 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n64_default.json) | 0.2333 / 0.2036 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n64_default.json) | 0.239 / 0.2031 |
| `metal/bf16/matmul/decode_m1/k64_n64_dense` | matrix | 1 × 64 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n64_dense.json) | 0.2325 / 0.2016 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n64_dense.json) | 0.2414 / 0.203 |
| `metal/bf16/matmul/decode_m1/k65_n64_default` | matrix | 1 × 65 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n64_default.json) | 0.2341 / 0.2046 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n64_default.json) | 0.2393 / 0.2003 |
| `metal/bf16/matmul/decode_m1/k65_n64_dense` | matrix | 1 × 65 × 64 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n64_dense.json) | 0.2364 / 0.2008 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n64_dense.json) | 0.241 / 0.2015 |
| `metal/bf16/matmul/decode_m1/k64_n65_default` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n65_default.json) | 0.235 / 0.2009 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n65_default.json) | 0.2399 / 0.2049 |
| `metal/bf16/matmul/decode_m1/k64_n65_dense` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k64_n65_dense.json) | 0.2327 / 0.2007 | [passed](program_peer/metal__bf16__matmul__decode_m1__k64_n65_dense.json) | 0.2381 / 0.2061 |
| `metal/bf16/matmul/decode_m1/k63_n65_default` | matrix | 1 × 63 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n65_default.json) | 0.2347 / 0.2011 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n65_default.json) | 0.2384 / 0.2044 |
| `metal/bf16/matmul/decode_m1/k63_n65_dense` | matrix | 1 × 63 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k63_n65_dense.json) | 0.2334 / 0.2037 | [passed](program_peer/metal__bf16__matmul__decode_m1__k63_n65_dense.json) | 0.2397 / 0.205 |
| `metal/bf16/matmul/decode_m1/k65_n63_default` | matrix | 1 × 65 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n63_default.json) | 0.2334 / 0.2028 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n63_default.json) | 0.2394 / 0.2043 |
| `metal/bf16/matmul/decode_m1/k65_n63_dense` | matrix | 1 × 65 × 63 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n63_dense.json) | 0.2326 / 0.2051 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n63_dense.json) | 0.2427 / 0.2037 |
| `metal/bf16/matmul/decode_m1/k65_n65_default` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n65_default.json) | 0.2297 / 0.2066 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n65_default.json) | 0.2373 / 0.202 |
| `metal/bf16/matmul/decode_m1/k65_n65_dense` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__k65_n65_dense.json) | 0.2318 / 0.201 | [passed](program_peer/metal__bf16__matmul__decode_m1__k65_n65_dense.json) | 0.2398 / 0.2052 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32_default` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_default.json) | 0.1985 / 0.2026 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_default.json) | 0.2293 / 0.2082 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32_dense` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_dense.json) | 0.2045 / 0.2022 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32_dense.json) | 0.2188 / 0.2017 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31_default` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_default.json) | 0.2037 / 0.2017 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_default.json) | 0.2169 / 0.1888 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31_dense` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_dense.json) | 0.2016 / 0.2071 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31_dense.json) | 0.2251 / 0.1953 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32_default` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_default.json) | 0.1962 / 0.2085 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_default.json) | 0.2335 / 0.202 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32_dense` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_dense.json) | 0.1962 / 0.2086 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32_dense.json) | 0.2256 / 0.1999 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32_default` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_default.json) | 0.2 / 0.206 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_default.json) | 0.2041 / 0.1868 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32_dense` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_dense.json) | 0.204 / 0.2074 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32_dense.json) | 0.2317 / 0.2051 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33_default` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_default.json) | 0.2003 / 0.2064 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_default.json) | 0.2329 / 0.2026 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33_dense` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_dense.json) | 0.2158 / 0.2083 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33_dense.json) | 0.2326 / 0.2044 |
| `metal/bf16/matmul/decode_m1/m1_k127_n128_default` | matrix | 1 × 127 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_default.json) | 0.2456 / 0.2027 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_default.json) | 0.2508 / 0.2024 |
| `metal/bf16/matmul/decode_m1/m1_k127_n128_dense` | matrix | 1 × 127 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_dense.json) | 0.2468 / 0.2004 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n128_dense.json) | 0.2562 / 0.2027 |
| `metal/bf16/matmul/decode_m1/m1_k31_n127_default` | matrix | 1 × 31 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_default.json) | 0.1987 / 0.2033 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_default.json) | 0.2118 / 0.188 |
| `metal/bf16/matmul/decode_m1/m1_k31_n127_dense` | matrix | 1 × 31 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_dense.json) | 0.1948 / 0.206 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n127_dense.json) | 0.2343 / 0.205 |
| `metal/bf16/matmul/decode_m1/m1_k32_n127_default` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_default.json) | 0.1951 / 0.2026 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_default.json) | 0.2158 / 0.2037 |
| `metal/bf16/matmul/decode_m1/m1_k32_n127_dense` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_dense.json) | 0.1913 / 0.1811 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n127_dense.json) | 0.214 / 0.2068 |
| `metal/bf16/matmul/decode_m1/m1_k127_n127_default` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_default.json) | 0.2477 / 0.2005 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_default.json) | 0.2504 / 0.2032 |
| `metal/bf16/matmul/decode_m1/m1_k127_n127_dense` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_dense.json) | 0.249 / 0.2002 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k127_n127_dense.json) | 0.2528 / 0.2069 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
