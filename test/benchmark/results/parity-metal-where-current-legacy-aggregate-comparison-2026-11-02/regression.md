# Metal where legacy aggregate comparison

All 22 cells passed numerical, operator, and Program parity. Old and new raw reports match cell ID, kind, target, peer engine, shape, dtype, layout, input hash, iterations, and rounds.

| Tier | Median new / old execution |
| --- | ---: |
| operator_peer | 0.98× |
| program_peer | 1.00× |

4 cells retain their old aggregate reports after repeat timing review:
- `metal/f32/where/broadcast/r32_c32`
- `metal/f32/where/broadcast/r64_c128`
- `metal/f32/where/ragged`
- `metal/f32/where/scalar`

The 128-cell view catalog is compared separately in `parity-metal-where-current-full-comparison-2026-11-02`. Direct Metal command traces for 16×129 where and add measured approximately 0.023 ms GPU duration for each; the previously retained where view report had an approximately 0.205 ms readback phase, compared with approximately 0.027 ms in the fresh run. This is a cross-session measurement change, not a demonstrated where-kernel gain.
