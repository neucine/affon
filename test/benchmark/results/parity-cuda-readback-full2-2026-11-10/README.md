# CUDA host readback full parity batch

This complete batch replaces the nested-array host readback path for all 11,946
CUDA parity cells. `Tensor.to_array()` now fills each innermost row with one
`JS_NewArrayFrom` call instead of one property store per element. The device
copy is unchanged: on `[128,512]` f32, `to_bytes()` remains about 0.05 ms while
nested conversion fell from about 1.05 ms to 0.38 ms.

## Contract

- Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`
- Affon binary: `8bd3dd7d664236c3c2ded29331eac8920470ea20a659d1f0ca4e253432098d2d`
- Peer: PyTorch `2.11.0+cu128`
- Protocol: 10 iterations per sample, 5 samples, 3 independent rounds
- Catalog: `test/parity/catalogs/cuda-metal-parity.json` (11,946 cells)
- Audit: `AUDIT.json`, exact catalog and binary scope, all 11,946 numerical,
  operator, and Program checks passed.
- Profile: `readback-profile.json`, the median of 30 split measurements on a
  `[128,512]` f32 tensor. `to_bytes()` took 0.052 ms and nested `to_array()`
  took 0.365 ms; the script is retained as `readback-profile.ts`.

The sweep completed with 11,946/11,946 numerical probes, operator reports, and
Program reports, with no failure artifacts. Its target and catalog hashes match
the audited inputs exactly.

## Performance effect

A same-source control binary (`10eeb31a897f1c47b68318a7048f5138516cb7b0f23015e6c48c76df02a40c4f`)
was measured on the same target with the readback change reverted. Across the
142-cell smoke catalog, controlled candidate/control medians were 0.8388
operator and 0.8566 Program; 112 operator and 111 Program cells improved, none
regressed above 1.05, and the maximum ratios were 1.0151 and 1.0146. The
control batch is retained at `parity-cuda-readback-control-2026-11-09`.

The full promotion moved the aggregate CUDA `candidate_gap` counts from 6,908
to 2,200 operator and from 6,972 to 2,566 Program. These are diagnostic timing
labels, not a universal model-level performance claim. The machine kernel
string changed from `7.0.0-38` to `7.0.0-42` since the earlier aggregate, so
the controlled smoke comparison above is the causal evidence for the readback
change.

The independent general CUDA check also passed with this binary: seed 314159
produced 368 numerical passes and 16 expected rank rejections across 384 cases,
retained at `parity-cuda-readback-general-314159-8bd3-2026-11-10`.

Prior aggregate entries and reports are retained in
`archive-cuda-readback-before-full-fast-array-2026-11-10`. The 142-cell smoke
evidence remains at `parity-cuda-readback-smoke-full2-2026-11-10`.
