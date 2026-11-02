# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 16
- Added cells: 112
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | none | — | — |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | none | — | — |

## Added cells

- `metal/bf16/cross_entropy_indexed/large/b7_c15`
- `metal/bf16/cross_entropy_indexed/large/b7_c16`
- `metal/bf16/cross_entropy_indexed/large/b7_c17`
- `metal/bf16/cross_entropy_indexed/large/b7_c255`
- `metal/bf16/cross_entropy_indexed/large/b7_c256`
- `metal/bf16/cross_entropy_indexed/large/b7_c257`
- `metal/bf16/cross_entropy_indexed/large/b7_c512`
- `metal/bf16/cross_entropy_indexed/large/b7_c513`
- `metal/bf16/cross_entropy_indexed/large/b9_c15`
- `metal/bf16/cross_entropy_indexed/large/b9_c16`
- `metal/bf16/cross_entropy_indexed/large/b9_c17`
- `metal/bf16/cross_entropy_indexed/large/b9_c255`
- `metal/bf16/cross_entropy_indexed/large/b9_c256`
- `metal/bf16/cross_entropy_indexed/large/b9_c257`
- `metal/bf16/cross_entropy_indexed/large/b9_c512`
- `metal/bf16/cross_entropy_indexed/large/b9_c513`
- `metal/bf16/cross_entropy_indexed/small/b3_c15`
- `metal/bf16/cross_entropy_indexed/small/b3_c16`
- `metal/bf16/cross_entropy_indexed/small/b3_c17`
- `metal/bf16/cross_entropy_indexed/small/b3_c255`
- `metal/bf16/cross_entropy_indexed/small/b3_c256`
- `metal/bf16/cross_entropy_indexed/small/b3_c257`
- `metal/bf16/cross_entropy_indexed/small/b3_c512`
- `metal/bf16/cross_entropy_indexed/small/b3_c513`
- `metal/bf16/cross_entropy_indexed/small/b5_c15`
- `metal/bf16/cross_entropy_indexed/small/b5_c16`
- `metal/bf16/cross_entropy_indexed/small/b5_c17`
- `metal/bf16/cross_entropy_indexed/small/b5_c255`
- `metal/bf16/cross_entropy_indexed/small/b5_c256`
- `metal/bf16/cross_entropy_indexed/small/b5_c257`
- `metal/bf16/cross_entropy_indexed/small/b5_c512`
- `metal/bf16/cross_entropy_indexed/small/b5_c513`
- `metal/bf16/matmul/small_batch/m16_k31_n31`
- `metal/bf16/matmul/small_batch/m16_k31_n32`
- `metal/bf16/matmul/small_batch/m16_k31_n33`
- `metal/bf16/matmul/small_batch/m16_k31_n65`
- `metal/bf16/matmul/small_batch/m16_k32_n31`
- `metal/bf16/matmul/small_batch/m16_k32_n32`
- `metal/bf16/matmul/small_batch/m16_k32_n33`
- `metal/bf16/matmul/small_batch/m16_k32_n65`
- `metal/bf16/matmul/small_batch/m16_k33_n31`
- `metal/bf16/matmul/small_batch/m16_k33_n32`
- `metal/bf16/matmul/small_batch/m16_k33_n33`
- `metal/bf16/matmul/small_batch/m16_k33_n65`
- `metal/bf16/matmul/small_batch/m16_k64_n31`
- `metal/bf16/matmul/small_batch/m16_k64_n32`
- `metal/bf16/matmul/small_batch/m16_k64_n33`
- `metal/bf16/matmul/small_batch/m16_k64_n65`
- `metal/f32/cross_entropy_indexed/large/b7_c15`
- `metal/f32/cross_entropy_indexed/large/b7_c16`
- `metal/f32/cross_entropy_indexed/large/b7_c17`
- `metal/f32/cross_entropy_indexed/large/b7_c255`
- `metal/f32/cross_entropy_indexed/large/b7_c256`
- `metal/f32/cross_entropy_indexed/large/b7_c257`
- `metal/f32/cross_entropy_indexed/large/b7_c512`
- `metal/f32/cross_entropy_indexed/large/b7_c513`
- `metal/f32/cross_entropy_indexed/large/b9_c15`
- `metal/f32/cross_entropy_indexed/large/b9_c16`
- `metal/f32/cross_entropy_indexed/large/b9_c17`
- `metal/f32/cross_entropy_indexed/large/b9_c255`
- `metal/f32/cross_entropy_indexed/large/b9_c256`
- `metal/f32/cross_entropy_indexed/large/b9_c257`
- `metal/f32/cross_entropy_indexed/large/b9_c512`
- `metal/f32/cross_entropy_indexed/large/b9_c513`
- `metal/f32/cross_entropy_indexed/small/b3_c15`
- `metal/f32/cross_entropy_indexed/small/b3_c16`
- `metal/f32/cross_entropy_indexed/small/b3_c17`
- `metal/f32/cross_entropy_indexed/small/b3_c255`
- `metal/f32/cross_entropy_indexed/small/b3_c256`
- `metal/f32/cross_entropy_indexed/small/b3_c257`
- `metal/f32/cross_entropy_indexed/small/b3_c512`
- `metal/f32/cross_entropy_indexed/small/b3_c513`
- `metal/f32/cross_entropy_indexed/small/b5_c15`
- `metal/f32/cross_entropy_indexed/small/b5_c16`
- `metal/f32/cross_entropy_indexed/small/b5_c17`
- `metal/f32/cross_entropy_indexed/small/b5_c255`
- `metal/f32/cross_entropy_indexed/small/b5_c256`
- `metal/f32/cross_entropy_indexed/small/b5_c257`
- `metal/f32/cross_entropy_indexed/small/b5_c512`
- `metal/f32/cross_entropy_indexed/small/b5_c513`
- `metal/f32/matmul/decode_m1/m1_k31_n31`
- `metal/f32/matmul/decode_m1/m1_k31_n32`
- `metal/f32/matmul/decode_m1/m1_k31_n33`
- `metal/f32/matmul/decode_m1/m1_k31_n65`
- `metal/f32/matmul/decode_m1/m1_k32_n31`
- `metal/f32/matmul/decode_m1/m1_k32_n32`
- `metal/f32/matmul/decode_m1/m1_k32_n33`
- `metal/f32/matmul/decode_m1/m1_k32_n65`
- `metal/f32/matmul/decode_m1/m1_k33_n31`
- `metal/f32/matmul/decode_m1/m1_k33_n32`
- `metal/f32/matmul/decode_m1/m1_k33_n33`
- `metal/f32/matmul/decode_m1/m1_k33_n65`
- `metal/f32/matmul/decode_m1/m1_k64_n31`
- `metal/f32/matmul/decode_m1/m1_k64_n32`
- `metal/f32/matmul/decode_m1/m1_k64_n33`
- `metal/f32/matmul/decode_m1/m1_k64_n65`
- `metal/f32/matmul/small_batch/m16_k31_n31`
- `metal/f32/matmul/small_batch/m16_k31_n32`
- `metal/f32/matmul/small_batch/m16_k31_n33`
- `metal/f32/matmul/small_batch/m16_k31_n65`
- `metal/f32/matmul/small_batch/m16_k32_n31`
- `metal/f32/matmul/small_batch/m16_k32_n32`
- `metal/f32/matmul/small_batch/m16_k32_n33`
- `metal/f32/matmul/small_batch/m16_k32_n65`
- `metal/f32/matmul/small_batch/m16_k33_n31`
- `metal/f32/matmul/small_batch/m16_k33_n32`
- `metal/f32/matmul/small_batch/m16_k33_n33`
- `metal/f32/matmul/small_batch/m16_k33_n65`
- `metal/f32/matmul/small_batch/m16_k64_n31`
- `metal/f32/matmul/small_batch/m16_k64_n32`
- `metal/f32/matmul/small_batch/m16_k64_n33`
- `metal/f32/matmul/small_batch/m16_k64_n65`
