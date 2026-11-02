# Metal where investigation, 2026-11-02

Target: Mac16,1 / macOS 26.6.2 / arm64. Both f32 and BF16 kernels remain at their prior implementation. All candidates passed numerical, operator, and Program parity, then were reverted because matched timing did not improve consistently.

| Candidate | Matched sample | Result |
| --- | --- | --- |
| Rank-two coordinate addressing | Eight 7×17 cells and eight 16×129 cells across dense, broadcast, strided, and offset layouts | No consistent execution gain; larger f32 Program timings were slower. |
| Packed shader parameters | Eight 16×129 cells | Flat to slower, especially f32 Program timing. |
| Shared Metal compute encoder | Sixteen 7×17 and 16×129 cells | Median operator ratios: BF16 1.014×, f32 1.018×. Median Program ratios: BF16 1.074×, f32 1.031×. |

The retained aggregate reports show roughly 0.3–0.35 ms Affon execution for small grids while peer execution is roughly 0.03–0.04 ms. The large dense cells have a smaller total ratio. Shader index arithmetic and individual encoder/parameter calls do not explain the small-grid gap based on these experiments. Next work should profile end-to-end command submission and synchronization for a single where operation against a comparable simple Metal operation, then change the measured bottleneck.

Evidence: `parity-metal-where-rank2-comparison-2026-11-02`, `parity-metal-where-rank2-large-comparison-2026-11-02`, `parity-metal-where-packed-args-comparison-2026-11-02`, and `parity-metal-where-shared-encoder-comparison-2026-11-02`.

## Current aggregate refresh

A direct `AFFON_METAL_COMMAND_TRACE=1` worker comparison of 16×129 f32 where and add measured approximately 0.023 ms GPU command duration for each. Their fresh worker readback phases were approximately 0.028 ms and 0.026 ms respectively. This points to the retained October view measurements as stale for the current runtime or environment, rather than a where shader bottleneck.

The full 128-cell where view rerun (`parity-metal-where-current-full-2026-11-02`) passed all three parity tiers under the matched 10×5×3 protocol. Against the October view manifest, median Affon execution ratios were 0.599× for f32 operator, 0.604× for f32 Program, 0.615× for BF16 operator, and 0.617× for BF16 Program. No cell was a timing suspect. The 128 reports were promoted with an archive. The both-tier gap count fell from 128/128 to 91/128 across these view cases. This is a measured cross-session improvement; no where-specific kernel change was retained.

The 22 legacy where cells also passed all parity tiers. Eighteen reports were promoted and four retained their old aggregate reports after repeat timing review. Their matched raw comparison is in `parity-metal-where-current-legacy-aggregate-comparison-2026-11-02`.

## Retained small shared-buffer path

The Metal backend now computes f32 and BF16 where on the CPU when the output has at most 4096 elements, all buffers use shared storage, no command is recording or pending, device timing and command tracing are inactive, and the output does not alias an input. The existing Metal kernel handles every other case. This removes command submission latency for small resident tensors. Source: `src/backend/metal/ffi.m` in the sibling Compute checkout.

A fresh 16-cell sample across both dtypes and four layouts passed numerical, operator, and Program parity. Median operator execution was about 0.12× the fresh GPU baseline. The full 128-cell view batch and 22-cell legacy batch passed all three parity tiers under their matched 10×5×3 protocol. Their regression comparisons reported no correctness or timing suspects. Against the fresh GPU batches, the combined both-tier gap count fell from 107/150 to 6/150. All 150 reports were promoted with separate archives, and both summaries, the parity index, and the Weiv report were regenerated.

A Metal `gt → where` program returned `[[2,4,4],[7,3,6]]`, exercising the GPU fallback when a producer command is pending. With `AFFON_METAL_COMMAND_TRACE=1`, a direct where worker still emitted completed GPU commands, confirming the trace guard takes the fallback. The CPU path is intended for the latency of small resident tensors; larger and pending-command workloads still use the GPU kernel.
