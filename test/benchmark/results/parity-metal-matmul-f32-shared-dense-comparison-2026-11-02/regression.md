# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 128
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/matmul/large_matrix/grid_m128_k127_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k127_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k127_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k127_n7` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k33_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k33_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k33_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k33_n7` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k5_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k5_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k5_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k5_n7` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k65_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k65_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k65_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m128_k65_n7` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k127_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k127_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k127_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k127_n7` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k33_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k33_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k33_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k33_n7` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k5_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k5_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k5_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k5_n7` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k65_n129` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k65_n31` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k65_n63` | none | — | — |
| `metal/bf16/matmul/large_matrix/grid_m32_k65_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k127_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k127_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k127_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k127_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k33_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k33_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k33_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k33_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k5_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k5_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k5_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k5_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k65_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k65_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k65_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m2_k65_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k127_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k127_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k127_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k127_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k33_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k33_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k33_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k33_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k5_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k5_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k5_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k5_n7` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k65_n129` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k65_n31` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k65_n63` | none | — | — |
| `metal/bf16/matmul/small_batch/grid_m8_k65_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k127_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k127_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k127_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k127_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k33_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k33_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k33_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k33_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k5_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k5_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k5_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k5_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k65_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k65_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k65_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m128_k65_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k127_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k127_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k127_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k127_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k33_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k33_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k33_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k33_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k5_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k5_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k5_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k5_n7` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k65_n129` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k65_n31` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k65_n63` | none | — | — |
| `metal/f32/matmul/large_matrix/grid_m32_k65_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k127_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k127_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k127_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k127_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k33_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k33_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k33_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k33_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k5_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k5_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k5_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k5_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k65_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k65_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k65_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m2_k65_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k127_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k127_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k127_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k127_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k33_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k33_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k33_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k33_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k5_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k5_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k5_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k5_n7` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k65_n129` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k65_n31` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k65_n63` | none | — | — |
| `metal/f32/matmul/small_batch/grid_m8_k65_n7` | none | — | — |
