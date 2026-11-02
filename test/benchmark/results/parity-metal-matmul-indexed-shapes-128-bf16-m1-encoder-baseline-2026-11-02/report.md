# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `53c66178b0102715c92817fd8c3a2a3542baa50a3312e8852f7b500ab45d3d63`  
Binary SHA-256: `3552c52bc08639089d28f433463f3fb1e1502ad44033b793a0d38ceebf3bb968`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 16
- Numerical check passed: 16
- Operator check and timing passed: 16
- Program check and timing passed: 16
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2245 / 0.2031 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2331 / 0.2042 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2094 / 0.2059 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2343 / 0.2049 |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | matrix | 1 × 31 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2073 / 0.2021 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2322 / 0.2052 |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | matrix | 1 × 31 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2052 / 0.1924 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2287 / 0.2035 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2207 / 0.2068 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.236 / 0.2061 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.204 / 0.2076 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.2356 / 0.2087 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2144 / 0.2075 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2329 / 0.2048 |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | matrix | 1 × 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2 / 0.2064 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2358 / 0.1909 |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | matrix | 1 × 33 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.219 / 0.2036 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2357 / 0.2041 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2158 / 0.2042 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2331 / 0.2056 |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | matrix | 1 × 33 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.2072 / 0.209 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.2351 / 0.2044 |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | matrix | 1 × 33 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2103 / 0.2036 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2344 / 0.1997 |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | matrix | 1 × 64 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2383 / 0.1997 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2421 / 0.209 |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | matrix | 1 × 64 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2373 / 0.2021 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2447 / 0.2042 |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | matrix | 1 × 64 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.239 / 0.2021 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.2408 / 0.2017 |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2325 / 0.2011 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2413 / 0.2019 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
