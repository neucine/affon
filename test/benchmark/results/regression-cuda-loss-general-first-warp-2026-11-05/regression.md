# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 24
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 2
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `cuda/bf16/cross_entropy/batched/general_b2_c37_s5` | none | 0.40× | 0.40× |
| `cuda/bf16/cross_entropy/large/general_r11_c1031` | none | 0.67× | 0.66× |
| `cuda/bf16/cross_entropy/large/general_r1_c8193` | none | 2.67× | 2.63× |
| `cuda/bf16/cross_entropy/large/offset_general_r5_c97` | none | 0.51× | 0.52× |
| `cuda/bf16/cross_entropy/small/general_r1_c3` | none | 0.82× | 0.83× |
| `cuda/bf16/cross_entropy/strided/cols_general_r19_c257` | none | 0.40× | 0.40× |
| `cuda/bf16/log_softmax_nll/batched/general_b2_c37_s5` | none | 0.39× | 0.39× |
| `cuda/bf16/log_softmax_nll/large/general_r11_c1031` | none | 0.64× | 0.66× |
| `cuda/bf16/log_softmax_nll/large/general_r1_c8193` | none | 2.41× | 2.46× |
| `cuda/bf16/log_softmax_nll/large/offset_general_r5_c97` | none | 0.51× | 0.52× |
| `cuda/bf16/log_softmax_nll/small/general_r1_c3` | none | 0.84× | 0.85× |
| `cuda/bf16/log_softmax_nll/strided/cols_general_r19_c257` | none | 0.39× | 0.39× |
| `cuda/f32/cross_entropy/batched/general_b2_c37_s5` | none | 0.60× | 0.65× |
| `cuda/f32/cross_entropy/large/general_r11_c1031` | none | 0.10× | 0.11× |
| `cuda/f32/cross_entropy/large/general_r1_c8193` | none | 0.28× | 0.30× |
| `cuda/f32/cross_entropy/large/offset_general_r5_c97` | none | 0.38× | 0.42× |
| `cuda/f32/cross_entropy/small/general_r1_c3` | none | 1.13× | 1.13× |
| `cuda/f32/cross_entropy/strided/cols_general_r19_c257` | none | 0.12× | 0.12× |
| `cuda/f32/log_softmax_nll/batched/general_b2_c37_s5` | none | 0.50× | 0.64× |
| `cuda/f32/log_softmax_nll/large/general_r11_c1031` | none | 0.11× | 0.11× |
| `cuda/f32/log_softmax_nll/large/general_r1_c8193` | none | 0.27× | 0.29× |
| `cuda/f32/log_softmax_nll/large/offset_general_r5_c97` | none | 0.38× | 0.42× |
| `cuda/f32/log_softmax_nll/small/general_r1_c3` | none | 1.20× | 1.10× |
| `cuda/f32/log_softmax_nll/strided/cols_general_r19_c257` | none | 0.11× | 0.12× |
