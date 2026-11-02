# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `53c66178b0102715c92817fd8c3a2a3542baa50a3312e8852f7b500ab45d3d63`  
Binary SHA-256: `165614e444d0927f60b4ff553be291135a2315f15d13f69574187782656e462d`  
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
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2583 / 0.2075 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n31.json) | 0.2378 / 0.2105 |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2432 / 0.2501 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n32.json) | 0.2205 / 0.2906 |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | matrix | 1 × 31 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2454 / 0.215 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n33.json) | 0.2579 / 0.2832 |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | matrix | 1 × 31 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2633 / 0.2609 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k31_n65.json) | 0.2894 / 0.3235 |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | matrix | 1 × 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2635 / 0.3586 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n31.json) | 0.2366 / 0.1932 |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | matrix | 1 × 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.1994 / 0.1947 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n32.json) | 0.2417 / 0.2262 |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.2524 / 0.234 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n33.json) | 0.246 / 0.2526 |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | matrix | 1 × 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.224 / 0.1957 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k32_n65.json) | 0.2608 / 0.2184 |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | matrix | 1 × 33 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2172 / 0.1856 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n31.json) | 0.2506 / 0.2181 |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2683 / 0.2611 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n32.json) | 0.2432 / 0.2088 |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | matrix | 1 × 33 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.238 / 0.2747 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n33.json) | 0.3801 / 0.2488 |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | matrix | 1 × 33 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.276 / 0.2477 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k33_n65.json) | 0.2645 / 0.2268 |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | matrix | 1 × 64 × 31 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2333 / 0.1952 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n31.json) | 0.2346 / 0.2214 |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | matrix | 1 × 64 × 32 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2634 / 0.2383 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n32.json) | 0.2679 / 0.2201 |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | matrix | 1 × 64 × 33 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.247 / 0.1963 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n33.json) | 0.2823 / 0.2383 |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2887 / 0.218 | [passed](program_peer/metal__bf16__matmul__decode_m1__m1_k64_n65.json) | 0.2538 / 0.2118 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
