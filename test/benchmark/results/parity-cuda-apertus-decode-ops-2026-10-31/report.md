# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `9665103ca2ffe4372efc4a2a5d04706d0e6dfca465d1dfc9f7e27ce014d98ec9`
Binary SHA-256: `a7016a051fc45cdb985aa05b4c27060cb979a41c6f1087266ecdf1a49ed78bca`
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 1
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/index_select/rope_table_320x128` | indexing | 320 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__rope_table_320x128.json) | 0.02055 / 0.01279 | [passed](program_peer/cuda__bf16__index_select__rope_table_320x128.json) | 0.02318 / 0.01583 |
| `cuda/f32/softmax/attention_rows_32x320` | normalization | 32 × 320 | dense | correct | [passed](operator_peer/cuda__f32__softmax__attention_rows_32x320.json) | 0.1927 / 0.09169 | [passed](program_peer/cuda__f32__softmax__attention_rows_32x320.json) | 0.1945 / 0.09729 |
| `cuda/bf16/rms_norm/head_rows_32x128` | normalization | 32 × 128 | dense | probe_failed | probe_not_correct | — | probe_not_correct | — |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
