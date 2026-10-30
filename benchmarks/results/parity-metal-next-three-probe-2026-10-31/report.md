# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `9e1136e8a61f2760df2bb08a2a15391ddc717dddaed8d2718b52b47a1175c294`  
Binary SHA-256: `420e5315d554f3c9d08baf19a6f5517b0471f8b2a96377cacc0a6d30be0eda16`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/stack/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__bf16__stack__strided__r64_c127.json) | 0.4077 / 0.3861 | [passed](program_peer/metal__bf16__stack__strided__r64_c127.json) | 0.513 / 0.4262 |
| `metal/bf16/cat/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__bf16__cat__strided__r64_c127.json) | 0.3979 / 0.3705 | [passed](program_peer/metal__bf16__cat__strided__r64_c127.json) | 0.4199 / 0.4096 |
| `metal/f32/one_hot/invalid_index/b8_n33_c33` | indexing | 8 × 33 | dense | correct | [passed](operator_peer/metal__f32__one_hot__invalid_index__b8_n33_c33.json) | 0.4179 / 0.361 | [passed](program_peer/metal__f32__one_hot__invalid_index__b8_n33_c33.json) | 0.4118 / 0.3768 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
