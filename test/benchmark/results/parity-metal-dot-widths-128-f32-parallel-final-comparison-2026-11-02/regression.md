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
| `metal/f32/dot/dot_or_vector` | none | — | — |
| `metal/f32/dot/dot_or_vector/n1` | none | — | — |
| `metal/f32/dot/dot_or_vector/n10` | none | — | — |
| `metal/f32/dot/dot_or_vector/n1023` | none | — | — |
| `metal/f32/dot/dot_or_vector/n1024` | none | — | — |
| `metal/f32/dot/dot_or_vector/n1025` | none | — | — |
| `metal/f32/dot/dot_or_vector/n11` | none | — | — |
| `metal/f32/dot/dot_or_vector/n12` | none | — | — |
| `metal/f32/dot/dot_or_vector/n127` | none | — | — |
| `metal/f32/dot/dot_or_vector/n128` | none | — | — |
| `metal/f32/dot/dot_or_vector/n129` | none | — | — |
| `metal/f32/dot/dot_or_vector/n13` | none | — | — |
| `metal/f32/dot/dot_or_vector/n14` | none | — | — |
| `metal/f32/dot/dot_or_vector/n15` | none | — | — |
| `metal/f32/dot/dot_or_vector/n17` | none | — | — |
| `metal/f32/dot/dot_or_vector/n18` | none | — | — |
| `metal/f32/dot/dot_or_vector/n19` | none | — | — |
| `metal/f32/dot/dot_or_vector/n2` | none | — | — |
| `metal/f32/dot/dot_or_vector/n20` | none | — | — |
| `metal/f32/dot/dot_or_vector/n2048` | none | — | — |
| `metal/f32/dot/dot_or_vector/n21` | none | — | — |
| `metal/f32/dot/dot_or_vector/n22` | none | — | — |
| `metal/f32/dot/dot_or_vector/n23` | none | — | — |
| `metal/f32/dot/dot_or_vector/n24` | none | — | — |
| `metal/f32/dot/dot_or_vector/n25` | none | — | — |
| `metal/f32/dot/dot_or_vector/n255` | none | — | — |
| `metal/f32/dot/dot_or_vector/n256` | none | — | — |
| `metal/f32/dot/dot_or_vector/n257` | none | — | — |
| `metal/f32/dot/dot_or_vector/n26` | none | — | — |
| `metal/f32/dot/dot_or_vector/n27` | none | — | — |
| `metal/f32/dot/dot_or_vector/n28` | none | — | — |
| `metal/f32/dot/dot_or_vector/n29` | none | — | — |
| `metal/f32/dot/dot_or_vector/n3` | none | — | — |
| `metal/f32/dot/dot_or_vector/n30` | none | — | — |
| `metal/f32/dot/dot_or_vector/n31` | none | — | — |
| `metal/f32/dot/dot_or_vector/n32` | none | — | — |
| `metal/f32/dot/dot_or_vector/n33` | none | — | — |
| `metal/f32/dot/dot_or_vector/n34` | none | — | — |
| `metal/f32/dot/dot_or_vector/n35` | none | — | — |
| `metal/f32/dot/dot_or_vector/n36` | none | — | — |
| `metal/f32/dot/dot_or_vector/n37` | none | — | — |
| `metal/f32/dot/dot_or_vector/n38` | none | — | — |
| `metal/f32/dot/dot_or_vector/n39` | none | — | — |
| `metal/f32/dot/dot_or_vector/n4` | none | — | — |
| `metal/f32/dot/dot_or_vector/n40` | none | — | — |
| `metal/f32/dot/dot_or_vector/n41` | none | — | — |
| `metal/f32/dot/dot_or_vector/n42` | none | — | — |
| `metal/f32/dot/dot_or_vector/n43` | none | — | — |
| `metal/f32/dot/dot_or_vector/n44` | none | — | — |
| `metal/f32/dot/dot_or_vector/n45` | none | — | — |
| `metal/f32/dot/dot_or_vector/n46` | none | — | — |
| `metal/f32/dot/dot_or_vector/n47` | none | — | — |
| `metal/f32/dot/dot_or_vector/n48` | none | — | — |
| `metal/f32/dot/dot_or_vector/n5` | none | — | — |
| `metal/f32/dot/dot_or_vector/n511` | none | — | — |
| `metal/f32/dot/dot_or_vector/n512` | none | — | — |
| `metal/f32/dot/dot_or_vector/n513` | none | — | — |
| `metal/f32/dot/dot_or_vector/n6` | none | — | — |
| `metal/f32/dot/dot_or_vector/n63` | none | — | — |
| `metal/f32/dot/dot_or_vector/n64` | none | — | — |
| `metal/f32/dot/dot_or_vector/n65` | none | — | — |
| `metal/f32/dot/dot_or_vector/n7` | none | — | — |
| `metal/f32/dot/dot_or_vector/n8` | none | — | — |
| `metal/f32/dot/dot_or_vector/n9` | none | — | — |

## Removed cells

- `metal/bf16/dot/dot_or_vector`
- `metal/bf16/dot/dot_or_vector/n1`
- `metal/bf16/dot/dot_or_vector/n10`
- `metal/bf16/dot/dot_or_vector/n1023`
- `metal/bf16/dot/dot_or_vector/n1024`
- `metal/bf16/dot/dot_or_vector/n1025`
- `metal/bf16/dot/dot_or_vector/n11`
- `metal/bf16/dot/dot_or_vector/n12`
- `metal/bf16/dot/dot_or_vector/n127`
- `metal/bf16/dot/dot_or_vector/n128`
- `metal/bf16/dot/dot_or_vector/n129`
- `metal/bf16/dot/dot_or_vector/n13`
- `metal/bf16/dot/dot_or_vector/n14`
- `metal/bf16/dot/dot_or_vector/n15`
- `metal/bf16/dot/dot_or_vector/n17`
- `metal/bf16/dot/dot_or_vector/n18`
- `metal/bf16/dot/dot_or_vector/n19`
- `metal/bf16/dot/dot_or_vector/n2`
- `metal/bf16/dot/dot_or_vector/n20`
- `metal/bf16/dot/dot_or_vector/n2048`
- `metal/bf16/dot/dot_or_vector/n21`
- `metal/bf16/dot/dot_or_vector/n22`
- `metal/bf16/dot/dot_or_vector/n23`
- `metal/bf16/dot/dot_or_vector/n24`
- `metal/bf16/dot/dot_or_vector/n25`
- `metal/bf16/dot/dot_or_vector/n255`
- `metal/bf16/dot/dot_or_vector/n256`
- `metal/bf16/dot/dot_or_vector/n257`
- `metal/bf16/dot/dot_or_vector/n26`
- `metal/bf16/dot/dot_or_vector/n27`
- `metal/bf16/dot/dot_or_vector/n28`
- `metal/bf16/dot/dot_or_vector/n29`
- `metal/bf16/dot/dot_or_vector/n3`
- `metal/bf16/dot/dot_or_vector/n30`
- `metal/bf16/dot/dot_or_vector/n31`
- `metal/bf16/dot/dot_or_vector/n32`
- `metal/bf16/dot/dot_or_vector/n33`
- `metal/bf16/dot/dot_or_vector/n34`
- `metal/bf16/dot/dot_or_vector/n35`
- `metal/bf16/dot/dot_or_vector/n36`
- `metal/bf16/dot/dot_or_vector/n37`
- `metal/bf16/dot/dot_or_vector/n38`
- `metal/bf16/dot/dot_or_vector/n39`
- `metal/bf16/dot/dot_or_vector/n4`
- `metal/bf16/dot/dot_or_vector/n40`
- `metal/bf16/dot/dot_or_vector/n41`
- `metal/bf16/dot/dot_or_vector/n42`
- `metal/bf16/dot/dot_or_vector/n43`
- `metal/bf16/dot/dot_or_vector/n44`
- `metal/bf16/dot/dot_or_vector/n45`
- `metal/bf16/dot/dot_or_vector/n46`
- `metal/bf16/dot/dot_or_vector/n47`
- `metal/bf16/dot/dot_or_vector/n48`
- `metal/bf16/dot/dot_or_vector/n5`
- `metal/bf16/dot/dot_or_vector/n511`
- `metal/bf16/dot/dot_or_vector/n512`
- `metal/bf16/dot/dot_or_vector/n513`
- `metal/bf16/dot/dot_or_vector/n6`
- `metal/bf16/dot/dot_or_vector/n63`
- `metal/bf16/dot/dot_or_vector/n64`
- `metal/bf16/dot/dot_or_vector/n65`
- `metal/bf16/dot/dot_or_vector/n7`
- `metal/bf16/dot/dot_or_vector/n8`
- `metal/bf16/dot/dot_or_vector/n9`
