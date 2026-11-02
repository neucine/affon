# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `53c66178b0102715c92817fd8c3a2a3542baa50a3312e8852f7b500ab45d3d63`  
Binary SHA-256: `a9a10dd4a4dd6ab577715a114b1261d475aeddc676e7750489ffff1b83de8cef`  
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
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2086 / 0.2049 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2318 / 0.2038 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.205 / 0.199 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2307 / 0.2032 |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | matrix | 1 × 31 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2012 / 0.1936 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.224 / 0.206 |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | matrix | 1 × 31 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.1991 / 0.203 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2174 / 0.2059 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2 / 0.203 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2323 / 0.1989 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.1963 / 0.1997 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.2331 / 0.207 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2054 / 0.2001 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2323 / 0.2054 |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | matrix | 1 × 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2036 / 0.2023 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.226 / 0.2061 |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | matrix | 1 × 33 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2021 / 0.2051 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2296 / 0.2074 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.1981 / 0.2068 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2255 / 0.2042 |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | matrix | 1 × 33 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.196 / 0.2042 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.2257 / 0.2077 |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | matrix | 1 × 33 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2019 / 0.2053 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2229 / 0.2017 |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | matrix | 1 × 64 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2298 / 0.207 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2421 / 0.2035 |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | matrix | 1 × 64 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2375 / 0.2003 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2396 / 0.2026 |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | matrix | 1 × 64 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.237 / 0.203 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.2414 / 0.2039 |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2332 / 0.2001 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2386 / 0.2043 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
