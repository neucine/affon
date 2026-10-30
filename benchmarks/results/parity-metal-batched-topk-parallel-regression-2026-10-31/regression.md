# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 130
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 7
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/topk/batched` | none | 1.43× | 1.80× |
| `metal/bf16/topk/batched/dense_axis1_b2_r16_c129_k8` | none | 0.96× | 1.01× |
| `metal/bf16/topk/batched/dense_axis1_b2_r16_c17_k8` | none | 0.58× | 1.00× |
| `metal/bf16/topk/batched/dense_axis1_b2_r16_c5_k8` | none | 1.01× | 0.97× |
| `metal/bf16/topk/batched/dense_axis1_b2_r16_c65_k8` | none | 1.00× | 1.04× |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c129_k16` | none | 0.99× | 1.00× |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c17_k16` | none | 0.53× | 1.42× |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c5_k16` | none | 1.18× | 0.95× |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c65_k16` | none | 0.46× | 0.98× |
| `metal/bf16/topk/batched/dense_axis1_b2_r3_c129_k1` | none | 1.00× | 1.10× |
| `metal/bf16/topk/batched/dense_axis1_b2_r3_c17_k1` | none | 1.11× | 1.29× |
| `metal/bf16/topk/batched/dense_axis1_b2_r3_c5_k1` | none | 1.40× | 1.40× |
| `metal/bf16/topk/batched/dense_axis1_b2_r3_c65_k1` | none | 0.98× | 0.86× |
| `metal/bf16/topk/batched/dense_axis1_b2_r7_c129_k3` | none | 1.40× | 1.71× |
| `metal/bf16/topk/batched/dense_axis1_b2_r7_c17_k3` | none | 0.72× | 1.00× |
| `metal/bf16/topk/batched/dense_axis1_b2_r7_c5_k3` | none | 0.98× | 1.00× |
| `metal/bf16/topk/batched/dense_axis1_b2_r7_c65_k3` | none | 0.92× | 0.66× |
| `metal/bf16/topk/batched/dense_axis1_b4_r16_c129_k8` | none | 2.42× | 1.00× |
| `metal/bf16/topk/batched/dense_axis1_b4_r16_c17_k8` | none | 0.92× | 0.79× |
| `metal/bf16/topk/batched/dense_axis1_b4_r16_c5_k8` | none | 1.77× | 1.12× |
| `metal/bf16/topk/batched/dense_axis1_b4_r16_c65_k8` | none | 1.00× | 1.01× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c129_k16` | none | 0.99× | 0.94× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c17_k16` | none | 1.04× | 0.52× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c5_k16` | none | 0.63× | 1.15× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c65_k16` | none | 0.99× | 1.02× |
| `metal/bf16/topk/batched/dense_axis1_b4_r3_c129_k1` | none | 1.08× | 0.98× |
| `metal/bf16/topk/batched/dense_axis1_b4_r3_c17_k1` | none | 1.05× | 0.96× |
| `metal/bf16/topk/batched/dense_axis1_b4_r3_c5_k1` | none | 1.00× | 1.03× |
| `metal/bf16/topk/batched/dense_axis1_b4_r3_c65_k1` | none | 0.97× | 1.01× |
| `metal/bf16/topk/batched/dense_axis1_b4_r7_c129_k3` | none | 0.73× | 1.08× |
| `metal/bf16/topk/batched/dense_axis1_b4_r7_c17_k3` | none | 1.00× | 1.49× |
| `metal/bf16/topk/batched/dense_axis1_b4_r7_c5_k3` | none | 1.01× | 0.96× |
| `metal/bf16/topk/batched/dense_axis1_b4_r7_c65_k3` | none | 0.72× | 1.05× |
| `metal/bf16/topk/batched/dense_axis2_b2_r16_c129_k64` | none | 1.01× | 0.98× |
| `metal/bf16/topk/batched/dense_axis2_b2_r16_c17_k8` | none | 1.01× | 0.99× |
| `metal/bf16/topk/batched/dense_axis2_b2_r16_c5_k2` | none | 0.80× | 1.02× |
| `metal/bf16/topk/batched/dense_axis2_b2_r16_c65_k32` | none | 1.13× | 1.10× |
| `metal/bf16/topk/batched/dense_axis2_b2_r33_c129_k64` | none | 0.96× | 1.04× |
| `metal/bf16/topk/batched/dense_axis2_b2_r33_c17_k8` | none | 0.97× | 1.05× |
| `metal/bf16/topk/batched/dense_axis2_b2_r33_c5_k2` | none | 1.03× | 1.05× |
| `metal/bf16/topk/batched/dense_axis2_b2_r33_c65_k32` | none | 0.99× | 1.01× |
| `metal/bf16/topk/batched/dense_axis2_b2_r3_c129_k64` | none | 0.99× | 1.50× |
| `metal/bf16/topk/batched/dense_axis2_b2_r3_c17_k8` | none | 0.95× | 1.03× |
| `metal/bf16/topk/batched/dense_axis2_b2_r3_c5_k2` | none | 1.14× | 0.79× |
| `metal/bf16/topk/batched/dense_axis2_b2_r3_c65_k32` | none | 0.83× | 0.78× |
| `metal/bf16/topk/batched/dense_axis2_b2_r7_c129_k64` | none | 0.98× | 1.00× |
| `metal/bf16/topk/batched/dense_axis2_b2_r7_c17_k8` | none | 1.21× | 1.01× |
| `metal/bf16/topk/batched/dense_axis2_b2_r7_c5_k2` | none | 0.84× | 0.83× |
| `metal/bf16/topk/batched/dense_axis2_b2_r7_c65_k32` | none | 0.96× | 1.00× |
| `metal/bf16/topk/batched/dense_axis2_b4_r16_c129_k64` | none | 0.97× | 1.00× |
| `metal/bf16/topk/batched/dense_axis2_b4_r16_c17_k8` | none | 0.99× | 1.00× |
| `metal/bf16/topk/batched/dense_axis2_b4_r16_c5_k2` | none | 0.98× | 0.84× |
| `metal/bf16/topk/batched/dense_axis2_b4_r16_c65_k32` | none | 0.70× | 1.01× |
| `metal/bf16/topk/batched/dense_axis2_b4_r33_c129_k64` | none | 0.97× | 1.05× |
| `metal/bf16/topk/batched/dense_axis2_b4_r33_c17_k8` | none | 1.00× | 1.03× |
| `metal/bf16/topk/batched/dense_axis2_b4_r33_c5_k2` | none | 1.00× | 0.82× |
| `metal/bf16/topk/batched/dense_axis2_b4_r33_c65_k32` | none | 1.00× | 1.02× |
| `metal/bf16/topk/batched/dense_axis2_b4_r3_c129_k64` | none | 0.91× | 1.06× |
| `metal/bf16/topk/batched/dense_axis2_b4_r3_c17_k8` | none | 1.03× | 0.76× |
| `metal/bf16/topk/batched/dense_axis2_b4_r3_c5_k2` | none | 1.01× | 0.81× |
| `metal/bf16/topk/batched/dense_axis2_b4_r3_c65_k32` | none | 0.91× | 0.89× |
| `metal/bf16/topk/batched/dense_axis2_b4_r7_c129_k64` | none | 0.99× | 1.17× |
| `metal/bf16/topk/batched/dense_axis2_b4_r7_c17_k8` | none | 0.93× | 1.00× |
| `metal/bf16/topk/batched/dense_axis2_b4_r7_c5_k2` | none | 0.96× | 1.03× |
| `metal/bf16/topk/batched/dense_axis2_b4_r7_c65_k32` | none | 1.01× | 1.02× |
| `metal/f32/topk/batched` | none | 1.09× | 0.59× |
| `metal/f32/topk/batched/dense_axis1_b2_r16_c129_k8` | none | 0.97× | 0.98× |
| `metal/f32/topk/batched/dense_axis1_b2_r16_c17_k8` | none | 0.69× | 1.24× |
| `metal/f32/topk/batched/dense_axis1_b2_r16_c5_k8` | none | 0.88× | 1.33× |
| `metal/f32/topk/batched/dense_axis1_b2_r16_c65_k8` | none | 1.03× | 1.03× |
| `metal/f32/topk/batched/dense_axis1_b2_r33_c129_k16` | none | 0.31× | 0.32× |
| `metal/f32/topk/batched/dense_axis1_b2_r33_c17_k16` | none | 0.56× | 0.36× |
| `metal/f32/topk/batched/dense_axis1_b2_r33_c5_k16` | none | 0.53× | 0.58× |
| `metal/f32/topk/batched/dense_axis1_b2_r33_c65_k16` | none | 0.36× | 0.28× |
| `metal/f32/topk/batched/dense_axis1_b2_r3_c129_k1` | none | 0.98× | 1.15× |
| `metal/f32/topk/batched/dense_axis1_b2_r3_c17_k1` | none | 1.02× | 0.98× |
| `metal/f32/topk/batched/dense_axis1_b2_r3_c5_k1` | none | 1.08× | 0.86× |
| `metal/f32/topk/batched/dense_axis1_b2_r3_c65_k1` | none | 1.18× | 1.13× |
| `metal/f32/topk/batched/dense_axis1_b2_r7_c129_k3` | none | 0.58× | 1.08× |
| `metal/f32/topk/batched/dense_axis1_b2_r7_c17_k3` | none | 1.40× | 1.01× |
| `metal/f32/topk/batched/dense_axis1_b2_r7_c5_k3` | none | 1.37× | 0.80× |
| `metal/f32/topk/batched/dense_axis1_b2_r7_c65_k3` | none | 1.07× | 1.08× |
| `metal/f32/topk/batched/dense_axis1_b4_r16_c129_k8` | none | 1.00× | 1.00× |
| `metal/f32/topk/batched/dense_axis1_b4_r16_c17_k8` | none | 0.97× | 0.99× |
| `metal/f32/topk/batched/dense_axis1_b4_r16_c5_k8` | none | 0.59× | 0.97× |
| `metal/f32/topk/batched/dense_axis1_b4_r16_c65_k8` | none | 0.99× | 0.95× |
| `metal/f32/topk/batched/dense_axis1_b4_r33_c129_k16` | none | 0.45× | 0.47× |
| `metal/f32/topk/batched/dense_axis1_b4_r33_c17_k16` | none | 0.44× | 0.28× |
| `metal/f32/topk/batched/dense_axis1_b4_r33_c5_k16` | none | 0.86× | 0.54× |
| `metal/f32/topk/batched/dense_axis1_b4_r33_c65_k16` | none | 0.34× | 0.34× |
| `metal/f32/topk/batched/dense_axis1_b4_r3_c129_k1` | none | 1.05× | 0.95× |
| `metal/f32/topk/batched/dense_axis1_b4_r3_c17_k1` | none | 1.11× | 1.26× |
| `metal/f32/topk/batched/dense_axis1_b4_r3_c5_k1` | none | 1.12× | 1.13× |
| `metal/f32/topk/batched/dense_axis1_b4_r3_c65_k1` | none | 1.14× | 0.82× |
| `metal/f32/topk/batched/dense_axis1_b4_r7_c129_k3` | none | 0.72× | 0.82× |
| `metal/f32/topk/batched/dense_axis1_b4_r7_c17_k3` | none | 1.38× | 1.05× |
| `metal/f32/topk/batched/dense_axis1_b4_r7_c5_k3` | none | 1.32× | 0.93× |
| `metal/f32/topk/batched/dense_axis1_b4_r7_c65_k3` | none | 1.01× | 1.51× |
| `metal/f32/topk/batched/dense_axis2_b2_r16_c129_k64` | none | 0.03× | 0.03× |
| `metal/f32/topk/batched/dense_axis2_b2_r16_c17_k8` | none | 1.25× | 0.99× |
| `metal/f32/topk/batched/dense_axis2_b2_r16_c5_k2` | none | 1.23× | 1.36× |
| `metal/f32/topk/batched/dense_axis2_b2_r16_c65_k32` | none | 0.10× | 0.07× |
| `metal/f32/topk/batched/dense_axis2_b2_r33_c129_k64` | none | 0.03× | 0.03× |
| `metal/f32/topk/batched/dense_axis2_b2_r33_c17_k8` | none | 0.98× | 1.03× |
| `metal/f32/topk/batched/dense_axis2_b2_r33_c5_k2` | none | 1.34× | 1.29× |
| `metal/f32/topk/batched/dense_axis2_b2_r33_c65_k32` | none | 0.13× | 0.10× |
| `metal/f32/topk/batched/dense_axis2_b2_r3_c129_k64` | none | 0.05× | 0.05× |
| `metal/f32/topk/batched/dense_axis2_b2_r3_c17_k8` | none | 1.00× | 1.06× |
| `metal/f32/topk/batched/dense_axis2_b2_r3_c5_k2` | none | 1.25× | 0.95× |
| `metal/f32/topk/batched/dense_axis2_b2_r3_c65_k32` | none | 0.18× | 0.19× |
| `metal/f32/topk/batched/dense_axis2_b2_r7_c129_k64` | none | 0.03× | 0.03× |
| `metal/f32/topk/batched/dense_axis2_b2_r7_c17_k8` | none | 1.58× | 0.94× |
| `metal/f32/topk/batched/dense_axis2_b2_r7_c5_k2` | none | 1.26× | 1.32× |
| `metal/f32/topk/batched/dense_axis2_b2_r7_c65_k32` | none | 0.14× | 0.14× |
| `metal/f32/topk/batched/dense_axis2_b4_r16_c129_k64` | none | 0.03× | 0.04× |
| `metal/f32/topk/batched/dense_axis2_b4_r16_c17_k8` | none | 0.98× | 1.00× |
| `metal/f32/topk/batched/dense_axis2_b4_r16_c5_k2` | none | 1.11× | 1.19× |
| `metal/f32/topk/batched/dense_axis2_b4_r16_c65_k32` | none | 0.09× | 0.10× |
| `metal/f32/topk/batched/dense_axis2_b4_r33_c129_k64` | none | 0.06× | 0.06× |
| `metal/f32/topk/batched/dense_axis2_b4_r33_c17_k8` | none | 0.99× | 0.97× |
| `metal/f32/topk/batched/dense_axis2_b4_r33_c5_k2` | none | 0.76× | 1.29× |
| `metal/f32/topk/batched/dense_axis2_b4_r33_c65_k32` | none | 0.13× | 0.14× |
| `metal/f32/topk/batched/dense_axis2_b4_r3_c129_k64` | none | 0.03× | 0.03× |
| `metal/f32/topk/batched/dense_axis2_b4_r3_c17_k8` | none | 1.00× | 1.01× |
| `metal/f32/topk/batched/dense_axis2_b4_r3_c5_k2` | none | 1.09× | 1.13× |
| `metal/f32/topk/batched/dense_axis2_b4_r3_c65_k32` | none | 0.21× | 0.21× |
| `metal/f32/topk/batched/dense_axis2_b4_r7_c129_k64` | none | 0.03× | 0.03× |
| `metal/f32/topk/batched/dense_axis2_b4_r7_c17_k8` | none | 1.40× | 0.99× |
| `metal/f32/topk/batched/dense_axis2_b4_r7_c5_k2` | none | 1.37× | 1.37× |
| `metal/f32/topk/batched/dense_axis2_b4_r7_c65_k32` | none | 0.08× | 0.09× |
