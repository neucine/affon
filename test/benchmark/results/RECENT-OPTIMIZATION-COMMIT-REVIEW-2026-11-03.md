# Optimization commit review since parity facility

## Scope

The review starts at Affon `eae71ed7e` (2026-10-31 08:54 NZDT), which introduced the permanent parity facility. It covers the subsequent committed Metal and CUDA optimization series in sibling Compute through `ace797f`, and the Affon parity and Apertus follow-up commits through `fe961af8a`. The large uncommitted backend changes from 2026-11-02 and 2026-11-03 are assessed separately in [the direction audit](METAL-OPTIMIZATION-DIRECTION-AUDIT-2026-11-02.md); they are not attributed to a commit here.

## Findings

| Compute commits | Change | Review status |
| --- | --- | --- |
| `62155aa`, `ffda5e5`, `33fe98f` | f32/BF16 top-k parallel and sort kernels | GPU paths; parity evidence retained in Affon. Pipeline choice is not named in raw cell reports. |
| `334365d`, `c5a07ac`, `61581d7`, `9d52859` | BF16 normalization cooperative kernel and three cutoff reductions | GPU path; correctness batches retained. Cutoffs need workload and boundary evidence. |
| `1c6cc08`, `46352ad` | Reduce-to-shape gather and scalar parallel kernels | GPU paths; no model acceptance result linked to these commits. |
| `50319e1`, `301d03f`, `5936295` | Full-tensor arg, variance/stddev, and sum/mean/min/max reductions | GPU paths; numerical and cell timing evidence retained. Review the exact selected pipeline before attributing a result. |
| `a6485a4`, `1148cc1`, `6690920`, `361303a`, `81d8962`, `2bbe4e7`, `9d343b7`, `3d256c9`, `ace797f` | CUDA reductions, indexed loss, softmax, concat, and BF16 arg reduction | Apertus phase and full-model evidence covers some resulting paths; RTX execution was not repeated in this review. |

Compute `4d36a2d` is a non-Linux CUDA stub for the concat kernel, not a performance change. Affon commits after `eae71ed7e` include parity refreshes for the Metal series and CUDA Apertus trace, parity, and model work. The file moves in `772efc4d6` and related commits reorganize the benchmark facility rather than changing a kernel.

### P2: Metal kernel selection is not observable in retained parity reports

The 12 committed Metal optimization changes add or select GPU paths for top-k, BF16 normalization, full reductions, arg reductions, and reduce-to-shape. The raw parity reports inspected for top-k record `selected_path` as `metal_primitive` with a configuration digest, but do not name the selected Metal pipeline. A passing or faster cell therefore cannot by itself prove which new kernel was exercised, especially across threshold changes such as BF16 normalization's 1,024 → 256 → 128 → 64 group cutoff. Record the pipeline/strategy identity and dispatch dimensions in the raw evidence or a trace before using a parity result to justify a selector boundary.

### P2: Metal optimization commits have cell evidence, but no model acceptance result

The Affon follow-up commits retain numerical, operator, and Program parity batches for the changed Metal families. Those checks establish correctness and isolated timing under their contracts. They do not include an accepted same-target Affon model baseline or a rerun of a traced model bottleneck. The repository's own benchmark contract requires that connection before claiming lasting workload improvement. Current aggregate timings are additionally historical after the 2026-11-03 GPU-default correction. Keep these GPU changes as candidates; prioritize representative model and boundary measurements before further selector tuning.

### CUDA Apertus work has stronger workload linkage, with a stated limit

The CUDA optimization series includes rank-one and full reductions, indexed loss, softmax, concat, and BF16 arg reduction. Affon's Apertus follow-up commits provide full-checkpoint logits, cached decode, phase timings, and Nsight kernel-family evidence. The [Apertus status](../../../docs/ml/apertus-8b.md) explicitly reports that the final softmax and host selection changes reduce phase/kernel cost without a proven whole-model token-rate gain. This is appropriately limited evidence. The CUDA kernels still need their own boundary and numerical review; this review did not execute the RTX 3090 target.

## Review conclusion

The committed Metal series consists of GPU kernel and dispatch work, unlike the later uncommitted shared-buffer CPU shortcuts. No confirmed correctness defect was found in the committed Metal series during this source review. The main review issue is evidence attribution: retained parity reports do not identify the actual Metal pipeline, and model-level improvement has not been established. The 2026-11-03 GPU-default correction addresses the separate backend-policy concern for the uncommitted shortcuts. No commits were reverted or rewritten.

The all-NaN edge case in Compute `50319e1` was checked against its predecessor: the parallel path explicitly returns index zero when the first logical input is NaN, matching the previous serial path for an all-NaN tensor. This is not a confirmed regression.

## Follow-up evidence, 2026-11-03

Compute's existing `AFFON_METAL_COMMAND_TRACE=1` command trace now annotates the selected Metal pipeline at the reviewed top-k, BF16 normalization, full reduction, arg reduction, and reduce-to-shape dispatch points. Separate traced sample runs recorded `normalize_bf16_cooperative:group=64:groups=16` and `topk_f32_sort:axis=64:k=16:rows=16`; the corresponding [normalization](metal-pipeline-trace-norm-sample-2026-11-03.log) and [top-k](metal-pipeline-trace-topk-sample-2026-11-03.log) logs are retained. Their compiler explanations still say `metal_primitive`, demonstrating why the runtime trace was needed. Tracing is kept separate from matched timing because it changes the measured path's overhead.

A fresh GPU-default [DistilGPT-2 Metal run](model-gpu-default-gpt2-metal-2026-11-03/) measured 435 ms median for 16 generated tokens, versus 195.603 ms for the matched PyTorch peer (2.224×). Its Affon samples ranged from 427 to 613 ms, so this is not a noise-calibrated self baseline. The provisional 1.25× peer gate failed. ViT remains blocked at Program parameter initialization by the 262,144-array-entry limit recorded in the [direction audit](METAL-OPTIMIZATION-DIRECTION-AUDIT-2026-11-02.md); no model acceptance result is claimed.
