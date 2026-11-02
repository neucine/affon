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
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2033 / 0.2048 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2329 / 0.2048 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.1951 / 0.2015 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2269 / 0.1997 |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | matrix | 1 × 31 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.1966 / 0.2055 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2122 / 0.1971 |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | matrix | 1 × 31 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.1936 / 0.2087 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2164 / 0.2028 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.1961 / 0.2058 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2341 / 0.2053 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.2028 / 0.2073 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.2276 / 0.2054 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.1939 / 0.2025 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2316 / 0.2088 |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | matrix | 1 × 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.202 / 0.2085 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2241 / 0.2074 |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | matrix | 1 × 33 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2219 / 0.2031 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2354 / 0.2041 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.1958 / 0.2082 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2289 / 0.2051 |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | matrix | 1 × 33 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.1986 / 0.2058 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.2277 / 0.2059 |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | matrix | 1 × 33 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.1963 / 0.2043 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2299 / 0.2066 |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | matrix | 1 × 64 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2312 / 0.205 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.241 / 0.2031 |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | matrix | 1 × 64 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.235 / 0.2017 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2408 / 0.2042 |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | matrix | 1 × 64 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.2312 / 0.2058 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.2393 / 0.1989 |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2303 / 0.1956 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2395 / 0.2063 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
