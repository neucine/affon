# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 128
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/where/broadcast/grid_r16_c129` | none | 0.53× | 0.54× |
| `metal/bf16/where/broadcast/grid_r16_c17` | none | 0.60× | 0.65× |
| `metal/bf16/where/broadcast/grid_r16_c3` | none | 0.58× | 0.67× |
| `metal/bf16/where/broadcast/grid_r16_c64` | none | 0.61× | 0.60× |
| `metal/bf16/where/broadcast/grid_r33_c129` | none | 0.37× | 0.43× |
| `metal/bf16/where/broadcast/grid_r33_c17` | none | 0.54× | 0.55× |
| `metal/bf16/where/broadcast/grid_r33_c3` | none | 0.64× | 0.66× |
| `metal/bf16/where/broadcast/grid_r33_c64` | none | 0.46× | 0.48× |
| `metal/bf16/where/broadcast/grid_r3_c129` | none | 0.65× | 0.63× |
| `metal/bf16/where/broadcast/grid_r3_c17` | none | 0.66× | 0.67× |
| `metal/bf16/where/broadcast/grid_r3_c3` | none | 0.64× | 0.68× |
| `metal/bf16/where/broadcast/grid_r3_c64` | none | 0.66× | 0.71× |
| `metal/bf16/where/broadcast/grid_r7_c129` | none | 0.57× | 0.58× |
| `metal/bf16/where/broadcast/grid_r7_c17` | none | 0.63× | 0.69× |
| `metal/bf16/where/broadcast/grid_r7_c3` | none | 0.68× | 0.61× |
| `metal/bf16/where/broadcast/grid_r7_c64` | none | 0.66× | 0.47× |
| `metal/bf16/where/offset/grid_r16_c129` | none | 0.46× | 0.56× |
| `metal/bf16/where/offset/grid_r16_c17` | none | 0.61× | 0.61× |
| `metal/bf16/where/offset/grid_r16_c3` | none | 0.64× | 0.66× |
| `metal/bf16/where/offset/grid_r16_c64` | none | 0.59× | 0.65× |
| `metal/bf16/where/offset/grid_r33_c129` | none | 0.39× | 0.41× |
| `metal/bf16/where/offset/grid_r33_c17` | none | 0.58× | 0.63× |
| `metal/bf16/where/offset/grid_r33_c3` | none | 0.69× | 0.66× |
| `metal/bf16/where/offset/grid_r33_c64` | none | 0.46× | 0.47× |
| `metal/bf16/where/offset/grid_r3_c129` | none | 0.64× | 0.53× |
| `metal/bf16/where/offset/grid_r3_c17` | none | 0.65× | 0.68× |
| `metal/bf16/where/offset/grid_r3_c3` | none | 0.67× | 0.82× |
| `metal/bf16/where/offset/grid_r3_c64` | none | 0.65× | 0.64× |
| `metal/bf16/where/offset/grid_r7_c129` | none | 0.59× | 0.59× |
| `metal/bf16/where/offset/grid_r7_c17` | none | 0.61× | 0.74× |
| `metal/bf16/where/offset/grid_r7_c3` | none | 0.63× | 0.73× |
| `metal/bf16/where/offset/grid_r7_c64` | none | 0.62× | 0.62× |
| `metal/bf16/where/small_dense/grid_r16_c129` | none | 0.52× | 0.52× |
| `metal/bf16/where/small_dense/grid_r16_c17` | none | 0.60× | 0.66× |
| `metal/bf16/where/small_dense/grid_r16_c3` | none | 0.71× | 0.69× |
| `metal/bf16/where/small_dense/grid_r16_c64` | none | 0.53× | 0.60× |
| `metal/bf16/where/small_dense/grid_r33_c129` | none | 0.40× | 0.39× |
| `metal/bf16/where/small_dense/grid_r33_c17` | none | 0.58× | 0.63× |
| `metal/bf16/where/small_dense/grid_r33_c3` | none | 0.64× | 0.69× |
| `metal/bf16/where/small_dense/grid_r33_c64` | none | 0.50× | 0.60× |
| `metal/bf16/where/small_dense/grid_r3_c129` | none | 0.62× | 0.62× |
| `metal/bf16/where/small_dense/grid_r3_c17` | none | 0.64× | 0.66× |
| `metal/bf16/where/small_dense/grid_r3_c3` | none | 0.71× | 0.65× |
| `metal/bf16/where/small_dense/grid_r3_c64` | none | 0.62× | 0.56× |
| `metal/bf16/where/small_dense/grid_r7_c129` | none | 0.58× | 0.54× |
| `metal/bf16/where/small_dense/grid_r7_c17` | none | 0.56× | 0.66× |
| `metal/bf16/where/small_dense/grid_r7_c3` | none | 0.68× | 0.72× |
| `metal/bf16/where/small_dense/grid_r7_c64` | none | 0.66× | 0.67× |
| `metal/bf16/where/strided/grid_r16_c129` | none | 0.53× | 0.47× |
| `metal/bf16/where/strided/grid_r16_c17` | none | 0.49× | 0.69× |
| `metal/bf16/where/strided/grid_r16_c3` | none | 0.65× | 0.68× |
| `metal/bf16/where/strided/grid_r16_c64` | none | 0.61× | 0.66× |
| `metal/bf16/where/strided/grid_r33_c129` | none | 0.37× | 0.44× |
| `metal/bf16/where/strided/grid_r33_c17` | none | 0.61× | 0.50× |
| `metal/bf16/where/strided/grid_r33_c3` | none | 0.65× | 0.62× |
| `metal/bf16/where/strided/grid_r33_c64` | none | 0.47× | 0.50× |
| `metal/bf16/where/strided/grid_r3_c129` | none | 0.62× | 0.52× |
| `metal/bf16/where/strided/grid_r3_c17` | none | 0.56× | 0.59× |
| `metal/bf16/where/strided/grid_r3_c3` | none | 0.66× | 0.65× |
| `metal/bf16/where/strided/grid_r3_c64` | none | 0.62× | 0.67× |
| `metal/bf16/where/strided/grid_r7_c129` | none | 0.53× | 0.59× |
| `metal/bf16/where/strided/grid_r7_c17` | none | 0.63× | 0.53× |
| `metal/bf16/where/strided/grid_r7_c3` | none | 0.70× | 0.59× |
| `metal/bf16/where/strided/grid_r7_c64` | none | 0.65× | 0.51× |
| `metal/f32/where/broadcast/grid_r16_c129` | none | 0.47× | 0.51× |
| `metal/f32/where/broadcast/grid_r16_c17` | none | 0.63× | 0.63× |
| `metal/f32/where/broadcast/grid_r16_c3` | none | 0.64× | 0.68× |
| `metal/f32/where/broadcast/grid_r16_c64` | none | 0.54× | 0.45× |
| `metal/f32/where/broadcast/grid_r33_c129` | none | 0.34× | 0.41× |
| `metal/f32/where/broadcast/grid_r33_c17` | none | 0.58× | 0.51× |
| `metal/f32/where/broadcast/grid_r33_c3` | none | 0.65× | 0.62× |
| `metal/f32/where/broadcast/grid_r33_c64` | none | 0.48× | 0.51× |
| `metal/f32/where/broadcast/grid_r3_c129` | none | 0.64× | 0.64× |
| `metal/f32/where/broadcast/grid_r3_c17` | none | 0.66× | 0.75× |
| `metal/f32/where/broadcast/grid_r3_c3` | none | 0.65× | 0.65× |
| `metal/f32/where/broadcast/grid_r3_c64` | none | 0.59× | 0.70× |
| `metal/f32/where/broadcast/grid_r7_c129` | none | 0.54× | 0.59× |
| `metal/f32/where/broadcast/grid_r7_c17` | none | 0.54× | 0.65× |
| `metal/f32/where/broadcast/grid_r7_c3` | none | 0.70× | 0.69× |
| `metal/f32/where/broadcast/grid_r7_c64` | none | 0.55× | 0.64× |
| `metal/f32/where/offset/grid_r16_c129` | none | 0.48× | 0.51× |
| `metal/f32/where/offset/grid_r16_c17` | none | 0.61× | 0.64× |
| `metal/f32/where/offset/grid_r16_c3` | none | 0.58× | 0.69× |
| `metal/f32/where/offset/grid_r16_c64` | none | 0.56× | 0.44× |
| `metal/f32/where/offset/grid_r33_c129` | none | 0.38× | 0.40× |
| `metal/f32/where/offset/grid_r33_c17` | none | 0.56× | 0.57× |
| `metal/f32/where/offset/grid_r33_c3` | none | 0.49× | 0.57× |
| `metal/f32/where/offset/grid_r33_c64` | none | 0.51× | 0.53× |
| `metal/f32/where/offset/grid_r3_c129` | none | 0.61× | 0.75× |
| `metal/f32/where/offset/grid_r3_c17` | none | 0.65× | 0.50× |
| `metal/f32/where/offset/grid_r3_c3` | none | 0.61× | 0.68× |
| `metal/f32/where/offset/grid_r3_c64` | none | 0.61× | 0.59× |
| `metal/f32/where/offset/grid_r7_c129` | none | 0.65× | 0.58× |
| `metal/f32/where/offset/grid_r7_c17` | none | 0.62× | 0.73× |
| `metal/f32/where/offset/grid_r7_c3` | none | 0.62× | 0.68× |
| `metal/f32/where/offset/grid_r7_c64` | none | 0.48× | 0.57× |
| `metal/f32/where/small_dense/grid_r16_c129` | none | 0.50× | 0.54× |
| `metal/f32/where/small_dense/grid_r16_c17` | none | 0.64× | 0.76× |
| `metal/f32/where/small_dense/grid_r16_c3` | none | 0.62× | 0.57× |
| `metal/f32/where/small_dense/grid_r16_c64` | none | 0.54× | 0.54× |
| `metal/f32/where/small_dense/grid_r33_c129` | none | 0.42× | 0.43× |
| `metal/f32/where/small_dense/grid_r33_c17` | none | 0.59× | 0.57× |
| `metal/f32/where/small_dense/grid_r33_c3` | none | 0.64× | 0.48× |
| `metal/f32/where/small_dense/grid_r33_c64` | none | 0.50× | 0.54× |
| `metal/f32/where/small_dense/grid_r3_c129` | none | 0.64× | 0.52× |
| `metal/f32/where/small_dense/grid_r3_c17` | none | 0.84× | 0.65× |
| `metal/f32/where/small_dense/grid_r3_c3` | none | 0.65× | 0.69× |
| `metal/f32/where/small_dense/grid_r3_c64` | none | 0.63× | 0.61× |
| `metal/f32/where/small_dense/grid_r7_c129` | none | 0.57× | 0.61× |
| `metal/f32/where/small_dense/grid_r7_c17` | none | 0.64× | 0.66× |
| `metal/f32/where/small_dense/grid_r7_c3` | none | 0.67× | 0.60× |
| `metal/f32/where/small_dense/grid_r7_c64` | none | 0.61× | 0.64× |
| `metal/f32/where/strided/grid_r16_c129` | none | 0.49× | 0.44× |
| `metal/f32/where/strided/grid_r16_c17` | none | 0.61× | 0.64× |
| `metal/f32/where/strided/grid_r16_c3` | none | 0.67× | 0.72× |
| `metal/f32/where/strided/grid_r16_c64` | none | 0.51× | 0.56× |
| `metal/f32/where/strided/grid_r33_c129` | none | 0.38× | 0.30× |
| `metal/f32/where/strided/grid_r33_c17` | none | 0.53× | 0.60× |
| `metal/f32/where/strided/grid_r33_c3` | none | 0.51× | 0.62× |
| `metal/f32/where/strided/grid_r33_c64` | none | 0.40× | 0.45× |
| `metal/f32/where/strided/grid_r3_c129` | none | 0.66× | 0.59× |
| `metal/f32/where/strided/grid_r3_c17` | none | 0.64× | 0.63× |
| `metal/f32/where/strided/grid_r3_c3` | none | 0.69× | 0.69× |
| `metal/f32/where/strided/grid_r3_c64` | none | 0.62× | 0.70× |
| `metal/f32/where/strided/grid_r7_c129` | none | 0.57× | 0.49× |
| `metal/f32/where/strided/grid_r7_c17` | none | 0.76× | 0.64× |
| `metal/f32/where/strided/grid_r7_c3` | none | 0.69× | 0.69× |
| `metal/f32/where/strided/grid_r7_c64` | none | 0.59× | 0.49× |
