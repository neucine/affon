# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 64
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/dot/dot_or_vector/offset_n1` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n10` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n11` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n12` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n127` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n128` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n13` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n14` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n15` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n16` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n17` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n18` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n19` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n2` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n20` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n21` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n22` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n23` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n24` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n3` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n31` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n32` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n33` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n4` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n5` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n6` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n63` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n64` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n65` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n7` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n8` | none | — | — |
| `metal/f32/dot/dot_or_vector/offset_n9` | none | — | — |
| `metal/f32/dot/strided` | none | — | — |
| `metal/f32/dot/strided/n1` | none | — | — |
| `metal/f32/dot/strided/n10` | none | — | — |
| `metal/f32/dot/strided/n11` | none | — | — |
| `metal/f32/dot/strided/n12` | none | — | — |
| `metal/f32/dot/strided/n127` | none | — | — |
| `metal/f32/dot/strided/n128` | none | — | — |
| `metal/f32/dot/strided/n13` | none | — | — |
| `metal/f32/dot/strided/n14` | none | — | — |
| `metal/f32/dot/strided/n15` | none | — | — |
| `metal/f32/dot/strided/n17` | none | — | — |
| `metal/f32/dot/strided/n18` | none | — | — |
| `metal/f32/dot/strided/n19` | none | — | — |
| `metal/f32/dot/strided/n2` | none | — | — |
| `metal/f32/dot/strided/n20` | none | — | — |
| `metal/f32/dot/strided/n21` | none | — | — |
| `metal/f32/dot/strided/n22` | none | — | — |
| `metal/f32/dot/strided/n23` | none | — | — |
| `metal/f32/dot/strided/n24` | none | — | — |
| `metal/f32/dot/strided/n3` | none | — | — |
| `metal/f32/dot/strided/n31` | none | — | — |
| `metal/f32/dot/strided/n32` | none | — | — |
| `metal/f32/dot/strided/n33` | none | — | — |
| `metal/f32/dot/strided/n4` | none | — | — |
| `metal/f32/dot/strided/n5` | none | — | — |
| `metal/f32/dot/strided/n6` | none | — | — |
| `metal/f32/dot/strided/n63` | none | — | — |
| `metal/f32/dot/strided/n64` | none | — | — |
| `metal/f32/dot/strided/n65` | none | — | — |
| `metal/f32/dot/strided/n7` | none | — | — |
| `metal/f32/dot/strided/n8` | none | — | — |
| `metal/f32/dot/strided/n9` | none | — | — |

## Removed cells

- `metal/bf16/dot/dot_or_vector/offset_n1`
- `metal/bf16/dot/dot_or_vector/offset_n10`
- `metal/bf16/dot/dot_or_vector/offset_n11`
- `metal/bf16/dot/dot_or_vector/offset_n12`
- `metal/bf16/dot/dot_or_vector/offset_n127`
- `metal/bf16/dot/dot_or_vector/offset_n128`
- `metal/bf16/dot/dot_or_vector/offset_n13`
- `metal/bf16/dot/dot_or_vector/offset_n14`
- `metal/bf16/dot/dot_or_vector/offset_n15`
- `metal/bf16/dot/dot_or_vector/offset_n16`
- `metal/bf16/dot/dot_or_vector/offset_n17`
- `metal/bf16/dot/dot_or_vector/offset_n18`
- `metal/bf16/dot/dot_or_vector/offset_n19`
- `metal/bf16/dot/dot_or_vector/offset_n2`
- `metal/bf16/dot/dot_or_vector/offset_n20`
- `metal/bf16/dot/dot_or_vector/offset_n21`
- `metal/bf16/dot/dot_or_vector/offset_n22`
- `metal/bf16/dot/dot_or_vector/offset_n23`
- `metal/bf16/dot/dot_or_vector/offset_n24`
- `metal/bf16/dot/dot_or_vector/offset_n3`
- `metal/bf16/dot/dot_or_vector/offset_n31`
- `metal/bf16/dot/dot_or_vector/offset_n32`
- `metal/bf16/dot/dot_or_vector/offset_n33`
- `metal/bf16/dot/dot_or_vector/offset_n4`
- `metal/bf16/dot/dot_or_vector/offset_n5`
- `metal/bf16/dot/dot_or_vector/offset_n6`
- `metal/bf16/dot/dot_or_vector/offset_n63`
- `metal/bf16/dot/dot_or_vector/offset_n64`
- `metal/bf16/dot/dot_or_vector/offset_n65`
- `metal/bf16/dot/dot_or_vector/offset_n7`
- `metal/bf16/dot/dot_or_vector/offset_n8`
- `metal/bf16/dot/dot_or_vector/offset_n9`
- `metal/bf16/dot/strided`
- `metal/bf16/dot/strided/n1`
- `metal/bf16/dot/strided/n10`
- `metal/bf16/dot/strided/n11`
- `metal/bf16/dot/strided/n12`
- `metal/bf16/dot/strided/n127`
- `metal/bf16/dot/strided/n128`
- `metal/bf16/dot/strided/n13`
- `metal/bf16/dot/strided/n14`
- `metal/bf16/dot/strided/n15`
- `metal/bf16/dot/strided/n17`
- `metal/bf16/dot/strided/n18`
- `metal/bf16/dot/strided/n19`
- `metal/bf16/dot/strided/n2`
- `metal/bf16/dot/strided/n20`
- `metal/bf16/dot/strided/n21`
- `metal/bf16/dot/strided/n22`
- `metal/bf16/dot/strided/n23`
- `metal/bf16/dot/strided/n24`
- `metal/bf16/dot/strided/n3`
- `metal/bf16/dot/strided/n31`
- `metal/bf16/dot/strided/n32`
- `metal/bf16/dot/strided/n33`
- `metal/bf16/dot/strided/n4`
- `metal/bf16/dot/strided/n5`
- `metal/bf16/dot/strided/n6`
- `metal/bf16/dot/strided/n63`
- `metal/bf16/dot/strided/n64`
- `metal/bf16/dot/strided/n65`
- `metal/bf16/dot/strided/n7`
- `metal/bf16/dot/strided/n8`
- `metal/bf16/dot/strided/n9`
