# Performance and parity results

**Backend policy and aggregate refresh (2026-11-03):** Metal uses GPU execution by default. The small shared-buffer CPU shortcuts require `AFFON_METAL_ENABLE_SHARED_CPU_SHORTCUTS=1`. The current Metal timing summaries and HTML report cover 11,946 valid timed cells, with 1,471 operator `candidate_gap` labels after the reviewed deferred GPU index-validation improvement. Two `reshape/view_only` cells were retired because their selected paths materialize the input; their prior reports remain in `archive-metal-pre-gpu-default-aggregate-2026-11-03/`.

Start with the [parity results index](PARITY-INDEX.md) to find named catalogs, retained batch reports, and the current aggregates. Open the [HTML report](performance-report.html) to filter operator and Program timings by device, operation, dtype, and status.

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
