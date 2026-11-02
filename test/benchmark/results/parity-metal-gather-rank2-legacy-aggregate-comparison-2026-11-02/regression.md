# Gather legacy aggregate comparison

The 192 legacy cases pass numerical, operator, and Program parity. Old and new raw reports match cell ID, kind, target, peer engine, shape, dtype, layout, input hash, iterations, and rounds.

October aggregate timings are faster than the November run across most cases. Same-session spot checks of eight representative cells show the rank-two candidate is roughly flat to faster than the generic kernel. The 192 old aggregate reports remain in place to avoid replacing their timing with a cross-session slowdown.

| Tier | Median November / October execution |
| --- | ---: |
| operator_peer | 1.57× |
| program_peer | 1.54× |

Same-session spot-check baselines: `parity-metal-gather-rank2-legacy-i5-s3-r2-fresh-baseline-2026-11-02` and `parity-metal-gather-rank2-legacy-i10-s5-r3-fresh-baseline-2026-11-02`. Candidate runs are `parity-metal-gather-rank2-legacy-*-final-2026-11-02`.

## Current-environment repeat

Both legacy protocol groups were rerun as `parity-metal-gather-rank2-legacy-i5-s3-r2-reviewed-2026-11-02` and `parity-metal-gather-rank2-legacy-i10-s5-r3-reviewed-2026-11-02`. All 192 cells again passed numerical, operator, and Program parity, with matching input hashes and protocols. Median execution relative to the first November candidate was 1.003× and 1.006× for operator, and 0.984× and 0.996× for Program, by the two respective groups. Relative to the October aggregate, medians remained roughly 1.5–1.6×. This reproducible cross-session difference is why the old legacy aggregate reports remain in place.
