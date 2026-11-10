# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `3b8f2eacd18d70de910d8bc34a9b60af5cad1379e2ef47efbb5d8b4ee4c42c62`  
Binary SHA-256: `4fafaf5b6ff353e6f47b3cff8f7c0e1899bfff0eb16a76fe6f16097df00dce27`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 26
- Numerical check passed: 26
- Operator check and timing passed: 26
- Program check and timing passed: 26
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/abs/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__abs__strided__r32_c127.json) | 0.101 / 0.05902 | [passed](program_peer/cuda__f32__abs__strided__r32_c127.json) | 0.1059 / 0.05136 |
| `cuda/f32/abs/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__abs__large_dense__r32_c32.json) | 0.04913 / 0.0233 | [passed](program_peer/cuda__f32__abs__large_dense__r32_c32.json) | 0.05449 / 0.0261 |
| `cuda/f32/clamp/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__clamp__strided__r32_c127.json) | 0.1016 / 0.04863 | [passed](program_peer/cuda__f32__clamp__strided__r32_c127.json) | 0.1062 / 0.06033 |
| `cuda/f32/clamp/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__clamp__large_dense__r32_c32.json) | 0.04952 / 0.0231 | [passed](program_peer/cuda__f32__clamp__large_dense__r32_c32.json) | 0.0544 / 0.02583 |
| `cuda/f32/erf/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__erf__strided__r32_c127.json) | 0.1023 / 0.04889 | [passed](program_peer/cuda__f32__erf__strided__r32_c127.json) | 0.1054 / 0.05813 |
| `cuda/f32/erf/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__erf__large_dense__r32_c32.json) | 0.04937 / 0.02241 | [passed](program_peer/cuda__f32__erf__large_dense__r32_c32.json) | 0.05473 / 0.02642 |
| `cuda/f32/exp/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__exp__strided__r128_c129.json) | 0.3272 / 0.1523 | [passed](program_peer/cuda__f32__exp__strided__r128_c129.json) | 0.3345 / 0.1583 |
| `cuda/f32/exp/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__exp__large_dense__r32_c32.json) | 0.04938 / 0.02252 | [passed](program_peer/cuda__f32__exp__large_dense__r32_c32.json) | 0.05449 / 0.02414 |
| `cuda/f32/gelu/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__gelu__strided__r32_c127.json) | 0.1024 / 0.04783 | [passed](program_peer/cuda__f32__gelu__strided__r32_c127.json) | 0.1059 / 0.05526 |
| `cuda/f32/gelu/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__gelu__large_dense__r32_c32.json) | 0.04927 / 0.02277 | [passed](program_peer/cuda__f32__gelu__large_dense__r32_c32.json) | 0.05472 / 0.02378 |
| `cuda/f32/log/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__log__strided__r128_c129.json) | 0.33 / 0.151 | [passed](program_peer/cuda__f32__log__strided__r128_c129.json) | 0.3342 / 0.1572 |
| `cuda/f32/log/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__log__large_dense__r32_c32.json) | 0.04895 / 0.02023 | [passed](program_peer/cuda__f32__log__large_dense__r32_c32.json) | 0.05472 / 0.0261 |
| `cuda/f32/neg/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__neg__strided__r32_c127.json) | 0.1026 / 0.05394 | [passed](program_peer/cuda__f32__neg__strided__r32_c127.json) | 0.106 / 0.05784 |
| `cuda/f32/neg/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__neg__large_dense__r32_c32.json) | 0.04938 / 0.02265 | [passed](program_peer/cuda__f32__neg__large_dense__r32_c32.json) | 0.05459 / 0.02541 |
| `cuda/f32/relu/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__relu__strided__r32_c127.json) | 0.1015 / 0.0575 | [passed](program_peer/cuda__f32__relu__strided__r32_c127.json) | 0.1063 / 0.06023 |
| `cuda/f32/relu/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__relu__large_dense__r32_c32.json) | 0.04948 / 0.02289 | [passed](program_peer/cuda__f32__relu__large_dense__r32_c32.json) | 0.05497 / 0.0267 |
| `cuda/f32/sigmoid/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__sigmoid__strided__r128_c129.json) | 0.3288 / 0.1534 | [passed](program_peer/cuda__f32__sigmoid__strided__r128_c129.json) | 0.335 / 0.1602 |
| `cuda/f32/sigmoid/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sigmoid__large_dense__r32_c32.json) | 0.04887 / 0.02233 | [passed](program_peer/cuda__f32__sigmoid__large_dense__r32_c32.json) | 0.05461 / 0.02614 |
| `cuda/f32/sign/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__sign__strided__r32_c127.json) | 0.1015 / 0.05806 | [passed](program_peer/cuda__f32__sign__strided__r32_c127.json) | 0.1061 / 0.06092 |
| `cuda/f32/sign/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sign__large_dense__r32_c32.json) | 0.0495 / 0.0225 | [passed](program_peer/cuda__f32__sign__large_dense__r32_c32.json) | 0.05421 / 0.02571 |
| `cuda/f32/silu/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__silu__strided__r32_c127.json) | 0.1015 / 0.05792 | [passed](program_peer/cuda__f32__silu__strided__r32_c127.json) | 0.1066 / 0.06003 |
| `cuda/f32/silu/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__silu__large_dense__r32_c32.json) | 0.04911 / 0.0226 | [passed](program_peer/cuda__f32__silu__large_dense__r32_c32.json) | 0.0547 / 0.02571 |
| `cuda/f32/sqrt/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__sqrt__strided__r128_c129.json) | 0.332 / 0.1554 | [passed](program_peer/cuda__f32__sqrt__strided__r128_c129.json) | 0.3344 / 0.1579 |
| `cuda/f32/sqrt/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sqrt__large_dense__r32_c32.json) | 0.04948 / 0.02287 | [passed](program_peer/cuda__f32__sqrt__large_dense__r32_c32.json) | 0.05464 / 0.02563 |
| `cuda/f32/tanh/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__tanh__strided__r128_c129.json) | 0.3282 / 0.1503 | [passed](program_peer/cuda__f32__tanh__strided__r128_c129.json) | 0.334 / 0.155 |
| `cuda/f32/tanh/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__tanh__large_dense__r32_c32.json) | 0.04946 / 0.02238 | [passed](program_peer/cuda__f32__tanh__large_dense__r32_c32.json) | 0.05492 / 0.02546 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
