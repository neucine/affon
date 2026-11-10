# Performance and parity results

**Backend policy and aggregate refresh (2026-11-03):** Metal uses GPU execution by default. The small shared-buffer CPU shortcuts require `AFFON_METAL_ENABLE_SHARED_CPU_SHORTCUTS=1`. The current Metal timing summaries and HTML report cover 11,946 valid timed cells, with 1,471 operator `candidate_gap` labels after the reviewed deferred GPU index-validation improvement. Two `reshape/view_only` cells were retired because their selected paths materialize the input; their prior reports remain in `archive-metal-pre-gpu-default-aggregate-2026-11-03/`.

Start with the [parity results index](PARITY-INDEX.md) to find named catalogs, retained batch reports, and the current aggregates. Open the [HTML report](performance-report.html) to filter operator and Program timings by device, operation, dtype, and status.

The reviewed [CUDA matmul Program metadata cache](cuda-matmul-program-cache-2026-11-07/README.md) covers 889 f32 and BF16 cells on the RTX 3090. Both matched full sweeps passed numerical, operator, and Program checks. Candidate Affon medians improved in 856 operator cells and 861 Program cells; the median reductions were 3.2% and 3.1%, respectively. Prior aggregate entries and reports are retained in [the replacement archive](archive-cuda-matmul-before-program-cache-2026-11-07/).

The reviewed [CUDA f32 index select kernel](cuda-index-select-f32-u32-2026-11-07/README.md) covers all 159 rank two f32 cells. Its full matched comparison passed every numerical, operator, and Program check. Operator medians improved in 150 cells and Program medians in 151; the median reductions were 2.0% and 1.7%. Prior entries and reports are retained in [the replacement archive](archive-cuda-index-select-before-f32-u32-2026-11-07/).

The reviewed [CUDA short-axis f32 reduction dispatch](cuda-reduction-axis-short-dispatch-2026-11-07/README.md) covers 528 cells across eight operations. Both full batches passed all numerical, operator, and Program checks; the candidate improved 512 cells in both timing modes. The median candidate/baseline ratios were 0.1372 operator and 0.1690 Program. That promotion brought the CUDA operator `candidate_gap` count to 7,065. Prior entries are retained in [the replacement archive](archive-cuda-reduction-before-short-dispatch-2026-11-07/).

The reviewed [CUDA f32 unary dispatch](cuda-unary-f32-one-launch-2026-11-07/README.md) covers 519 cells across 13 operations. Both complete audits passed all numerical, operator, and Program checks. The candidate improved 504 operator cells and 502 Program cells, with median candidate/baseline ratios of 0.9081 and 0.9209. The current CUDA operator `candidate_gap` count is 6,979. Prior entries are retained in [the replacement archive](archive-cuda-unary-before-f32-one-launch-2026-11-07/).

The reviewed CUDA top-k replacement covers 520 cells on the RTX 3090. The
[controlled comparison](cuda-topk-optimization-2026-11-06/controlled-before-after-nan-fast.md)
records the before and after execution and whole-operation timings; the
[complete audit](cuda-topk-optimization-2026-11-06/candidate-nan-fast-audit.json)
binds the replacement to its catalog, binary, and PyTorch version. The prior
aggregate reports are retained in
[the replacement archive](archive-cuda-topk-before-rank-nan-fast-2026-11-07/).
Top-k peer timings use PyTorch full stable sort then narrow to preserve
lower-index ties; they are not native `torch.topk` timings.

The reviewed [CUDA BF16 top-k row dispatch](cuda-topk-bf16-rank-2026-11-10/README.md)
replaces the 520-cell top-k aggregate with the audited RTX 3090 run. All 64
affected BF16 last-axis cells improved, with a median operator execution
speedup of 6.69x. The complete batch passed every numerical, operator, and
Program check. BF16 axis0 and other axes remain on their previous paths; their
worst operator execution change was within the measured noise envelope. On the
520 promoted top-k cells, operator `candidate_gap` fell from 86 to 21 and
Program from 88 to 24. Prior aggregate reports are retained in
[the replacement archive](archive-cuda-topk-before-bf16-rank-2026-11-10/).

The reviewed [CUDA f32 scatter-add replacement](cuda-scatter-f32-parallel-2026-11-07/README.md)
covers 197 cells. Its 132 changed rank-two cells all improved in controlled
operator and Program execution medians, while the complete candidate passed
all numerical and peer checks. The prior aggregate entries and reports are
retained in the replacement archive.

The reviewed
[CUDA f32 reduce-to-shape scalar and column dispatch](cuda-reduce-to-shape-scalar-parallel-reviewed-2026-11-10/README.md)
covers all 134 `reduce_to_shape` cells. The complete batch passed every
numerical, operator, and Program check, and the complete-batch audit binds the
catalog, candidate binary, target, and protocol. All 71 affected cells improved;
operator execution speedup had median 2.90x and maximum 24.12x. Across the 134
cells, operator `candidate_gap` fell from 113 to 74 and Program from 119 to 91.
Across the complete aggregate, operator `candidate_gap` fell from 3,123 to
3,081 and Program from 4,111 to 4,082. Prior ledger entries and aggregate
reports are retained in
[the replacement archive](archive-cuda-reduce-to-shape-before-scalar-parallel-2026-11-10/).

The reviewed [CUDA deferred index validation](cuda-bf16-indexing-deferred-candidate-2026-11-10/README.md)
covers all 384 BF16 `index_select` and `gather` cells. Small-index validation
moved from a per-call flag allocation and host synchronization into the device
synchronization that already precedes readback. The complete batch passed every
numerical, operator, and Program check, and the complete-batch audit binds the
catalog, candidate binary, target, and protocol. Every affected cell improved,
with median operator execution speedup 1.099x and Program 1.234x; no cell
regressed. Across the 384 cells, operator `candidate_gap` fell from 239 to 218
and Program from 288 to 218. Across the complete aggregate, operator
`candidate_gap` fell from 3,081 to 3,060 and Program from 4,082 to 4,012. Prior
ledger entries and aggregate reports are retained in
[the replacement archive](archive-cuda-bf16-index-deferred-before-2026-11-10/).

The reviewed [CUDA host readback materialization](parity-cuda-readback-full2-2026-11-10/README.md)
covers all 11,946 catalog cells. `Tensor.to_array()` now fills each innermost
row with one `JS_NewArrayFrom` call; on `[128,512]` f32, nested conversion fell
from about 1.05 ms to 0.38 ms while the device copy stayed near 0.05 ms. The
complete audit passed all numerical, operator, and Program checks. Against a
same-source 142-cell control on the same target, median ratios were 0.8388
operator and 0.8566 Program, with no cell regressing above 1.05. The promotion
reduced aggregate CUDA `candidate_gap` counts from 6,908 to 2,200 operator and
from 6,972 to 2,566 Program. Prior entries are retained in
[the replacement archive](archive-cuda-readback-before-full-fast-array-2026-11-10/).

Each `parity-*` directory is one immutable run. It contains the catalog snapshot, `capability.json`, raw `operator_peer` and `program_peer` timings, separate failure directories, `manifest.json`, `outcomes.json`, and `report.md`. Failed numerical probes and timings remain in their original batch directory.

`coverage-outcomes-current-mac-2026-10-29.json` is the reviewed aggregate coverage record. The `cell-peer-current-mac-2026-10-29` and `cell-program-peer-mac-2026-10-29` directories hold reviewed raw results; the adjacent summary JSON files derive from them. `performance-report.html` derives from those summaries and the coverage matrix. Existing file paths are stable because the aggregate record stores paths and SHA-256 hashes.

Run `python3 test/parity/index.py` after adding a retained batch. See [the parity facility guide](../../parity/README.md) for reruns and promotion.
