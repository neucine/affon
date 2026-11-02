# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `53c66178b0102715c92817fd8c3a2a3542baa50a3312e8852f7b500ab45d3d63`  
Binary SHA-256: `f1619e3a440fa453af2a2db4760d964b700317b9e2cd142bced5156526dcf11a`  
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
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2003 / 0.2064 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2445 / 0.2045 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2058 / 0.2081 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2636 / 0.2097 |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | matrix | 1 × 31 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2355 / 0.203 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2412 / 0.2092 |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | matrix | 1 × 31 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.252 / 0.2173 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2246 / 0.2084 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2143 / 0.2057 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2605 / 0.2147 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.2461 / 0.2105 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.2615 / 0.2345 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.231 / 0.2155 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.252 / 0.2074 |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | matrix | 1 × 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2091 / 0.2037 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2529 / 0.2083 |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | matrix | 1 × 33 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2392 / 0.2108 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2364 / 0.2163 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2525 / 0.2087 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2392 / 0.2075 |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | matrix | 1 × 33 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.2374 / 0.2056 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.2586 / 0.211 |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | matrix | 1 × 33 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2468 / 0.2078 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2628 / 0.2051 |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | matrix | 1 × 64 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2766 / 0.2254 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2387 / 0.2044 |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | matrix | 1 × 64 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2749 / 0.2028 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2675 / 0.2032 |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | matrix | 1 × 64 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.269 / 0.217 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.298 / 0.2113 |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2475 / 0.2033 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2459 / 0.209 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
