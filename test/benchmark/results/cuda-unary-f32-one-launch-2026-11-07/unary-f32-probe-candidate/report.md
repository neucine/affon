# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `3b8f2eacd18d70de910d8bc34a9b60af5cad1379e2ef47efbb5d8b4ee4c42c62`  
Binary SHA-256: `de9f587b9528b946e0e3f046708cf7c80f0590b84574ce3e9be84c03d6c2ff57`  
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
| `cuda/f32/abs/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__abs__strided__r32_c127.json) | 0.09876 / 0.0575 | [passed](program_peer/cuda__f32__abs__strided__r32_c127.json) | 0.1011 / 0.05159 |
| `cuda/f32/abs/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__abs__large_dense__r32_c32.json) | 0.04462 / 0.02318 | [passed](program_peer/cuda__f32__abs__large_dense__r32_c32.json) | 0.04986 / 0.02372 |
| `cuda/f32/clamp/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__clamp__strided__r32_c127.json) | 0.09815 / 0.05611 | [passed](program_peer/cuda__f32__clamp__strided__r32_c127.json) | 0.1021 / 0.05214 |
| `cuda/f32/clamp/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__clamp__large_dense__r32_c32.json) | 0.04481 / 0.0232 | [passed](program_peer/cuda__f32__clamp__large_dense__r32_c32.json) | 0.05019 / 0.02661 |
| `cuda/f32/erf/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__erf__strided__r32_c127.json) | 0.0986 / 0.05495 | [passed](program_peer/cuda__f32__erf__strided__r32_c127.json) | 0.1018 / 0.05236 |
| `cuda/f32/erf/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__erf__large_dense__r32_c32.json) | 0.04474 / 0.0226 | [passed](program_peer/cuda__f32__erf__large_dense__r32_c32.json) | 0.05009 / 0.02373 |
| `cuda/f32/exp/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__exp__strided__r128_c129.json) | 0.3252 / 0.1516 | [passed](program_peer/cuda__f32__exp__strided__r128_c129.json) | 0.3314 / 0.1567 |
| `cuda/f32/exp/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__exp__large_dense__r32_c32.json) | 0.04455 / 0.02269 | [passed](program_peer/cuda__f32__exp__large_dense__r32_c32.json) | 0.04981 / 0.02564 |
| `cuda/f32/gelu/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__gelu__strided__r32_c127.json) | 0.09832 / 0.05545 | [passed](program_peer/cuda__f32__gelu__strided__r32_c127.json) | 0.1009 / 0.06038 |
| `cuda/f32/gelu/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__gelu__large_dense__r32_c32.json) | 0.04463 / 0.02266 | [passed](program_peer/cuda__f32__gelu__large_dense__r32_c32.json) | 0.05033 / 0.0258 |
| `cuda/f32/log/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__log__strided__r128_c129.json) | 0.3239 / 0.1522 | [passed](program_peer/cuda__f32__log__strided__r128_c129.json) | 0.3274 / 0.1596 |
| `cuda/f32/log/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__log__large_dense__r32_c32.json) | 0.04443 / 0.02262 | [passed](program_peer/cuda__f32__log__large_dense__r32_c32.json) | 0.04983 / 0.02544 |
| `cuda/f32/neg/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__neg__strided__r32_c127.json) | 0.09764 / 0.0488 | [passed](program_peer/cuda__f32__neg__strided__r32_c127.json) | 0.1017 / 0.05088 |
| `cuda/f32/neg/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__neg__large_dense__r32_c32.json) | 0.0442 / 0.01967 | [passed](program_peer/cuda__f32__neg__large_dense__r32_c32.json) | 0.0497 / 0.02568 |
| `cuda/f32/relu/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__relu__strided__r32_c127.json) | 0.09801 / 0.04875 | [passed](program_peer/cuda__f32__relu__strided__r32_c127.json) | 0.1017 / 0.05355 |
| `cuda/f32/relu/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__relu__large_dense__r32_c32.json) | 0.04486 / 0.02349 | [passed](program_peer/cuda__f32__relu__large_dense__r32_c32.json) | 0.04978 / 0.0262 |
| `cuda/f32/sigmoid/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__sigmoid__strided__r128_c129.json) | 0.3238 / 0.1518 | [passed](program_peer/cuda__f32__sigmoid__strided__r128_c129.json) | 0.3302 / 0.1586 |
| `cuda/f32/sigmoid/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sigmoid__large_dense__r32_c32.json) | 0.04479 / 0.02312 | [passed](program_peer/cuda__f32__sigmoid__large_dense__r32_c32.json) | 0.04973 / 0.02578 |
| `cuda/f32/sign/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__sign__strided__r32_c127.json) | 0.09813 / 0.05586 | [passed](program_peer/cuda__f32__sign__strided__r32_c127.json) | 0.1015 / 0.05188 |
| `cuda/f32/sign/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sign__large_dense__r32_c32.json) | 0.0447 / 0.0231 | [passed](program_peer/cuda__f32__sign__large_dense__r32_c32.json) | 0.04995 / 0.0254 |
| `cuda/f32/silu/strided/r32_c127` | elementwise_math | 32 × 127 | strided | correct | [passed](operator_peer/cuda__f32__silu__strided__r32_c127.json) | 0.09797 / 0.05164 | [passed](program_peer/cuda__f32__silu__strided__r32_c127.json) | 0.1011 / 0.05832 |
| `cuda/f32/silu/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__silu__large_dense__r32_c32.json) | 0.04463 / 0.0234 | [passed](program_peer/cuda__f32__silu__large_dense__r32_c32.json) | 0.04979 / 0.0263 |
| `cuda/f32/sqrt/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__sqrt__strided__r128_c129.json) | 0.3267 / 0.1508 | [passed](program_peer/cuda__f32__sqrt__strided__r128_c129.json) | 0.3311 / 0.1559 |
| `cuda/f32/sqrt/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sqrt__large_dense__r32_c32.json) | 0.04459 / 0.02277 | [passed](program_peer/cuda__f32__sqrt__large_dense__r32_c32.json) | 0.0498 / 0.02532 |
| `cuda/f32/tanh/strided/r128_c129` | elementwise_math | 128 × 129 | strided | correct | [passed](operator_peer/cuda__f32__tanh__strided__r128_c129.json) | 0.3272 / 0.1532 | [passed](program_peer/cuda__f32__tanh__strided__r128_c129.json) | 0.3329 / 0.1574 |
| `cuda/f32/tanh/large_dense/r32_c32` | elementwise_math | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__tanh__large_dense__r32_c32.json) | 0.04445 / 0.02279 | [passed](program_peer/cuda__f32__tanh__large_dense__r32_c32.json) | 0.05001 / 0.02541 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
