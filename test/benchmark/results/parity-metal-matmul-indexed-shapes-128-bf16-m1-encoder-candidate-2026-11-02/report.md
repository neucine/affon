# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `53c66178b0102715c92817fd8c3a2a3542baa50a3312e8852f7b500ab45d3d63`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
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
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2138 / 0.2001 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2314 / 0.2061 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2061 / 0.203 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2244 / 0.2074 |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | matrix | 1 × 31 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.1934 / 0.2003 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.229 / 0.2048 |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | matrix | 1 × 31 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.1984 / 0.2052 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2161 / 0.2024 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.1985 / 0.2067 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2316 / 0.2004 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.1968 / 0.2064 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.23 / 0.1982 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2191 / 0.2056 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2253 / 0.2062 |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | matrix | 1 × 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.1976 / 0.2067 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2142 / 0.2041 |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | matrix | 1 × 33 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2062 / 0.1932 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2293 / 0.2044 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2061 / 0.1922 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2268 / 0.208 |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | matrix | 1 × 33 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.1968 / 0.2054 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.2262 / 0.2059 |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | matrix | 1 × 33 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.1994 / 0.2088 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2305 / 0.2089 |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | matrix | 1 × 64 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2276 / 0.2037 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2386 / 0.2051 |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | matrix | 1 × 64 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2319 / 0.2055 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2394 / 0.2046 |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | matrix | 1 × 64 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.2337 / 0.1968 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.242 / 0.2026 |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2339 / 0.1933 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2372 / 0.1998 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
