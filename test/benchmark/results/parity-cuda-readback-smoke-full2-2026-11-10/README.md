# CUDA host readback materialization

This batch replaces the nested-array host readback path for 142 CUDA smoke
cells. `Tensor.to_array()` now fills each innermost row with one
`JS_NewArrayFrom` call instead of one property store per element. The device
copy is unchanged: on `[128,512]` f32, `to_bytes()` remains about 0.05 ms while
nested conversion fell from about 1.05 ms to 0.38 ms.

## Contract

- Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`
- Affon binary: `8bd3dd7d664236c3c2ded29331eac8920470ea20a659d1f0ca4e253432098d2d`
- Peer: PyTorch `2.11.0+cu128`
- Protocol: 10 iterations per sample, 5 samples, 3 independent rounds
- Catalog: `test/parity/catalogs/cuda-metal-smoke.json` (142 cells)
- Audit: `AUDIT.json`, exact catalog and binary scope, all 142 numerical,
  operator, and Program checks passed.
- General check: `check-cuda-general.py` seed 314159 passed 368 cases with 16
  expected rank rejections (384 total); evidence is retained at
  `parity-cuda-readback-general-314159-8bd3-2026-11-10`.

The machine kernel string changed from `7.0.0-38` to `7.0.0-42` since the
prior aggregate. A same-source control binary
(`10eeb31a897f1c47b68318a7048f5138516cb7b0f23015e6c48c76df02a40c4f`)
was measured on the same target with the readback change reverted. Controlled
candidate/control medians were 0.8388 operator and 0.8566 Program; 112 operator
and 111 Program cells improved, none regressed above 1.05, and the maximum
ratios were 1.0151 and 1.0146. The control batch is retained at
`parity-cuda-readback-control-2026-11-09`.

Prior aggregate entries and reports are retained in
`archive-cuda-readback-before-fast-array-fixed-2026-11-10`. The first candidate
batch remains at `parity-cuda-readback-smoke-full-2026-11-09`; its promoted
evidence was replaced after the per-row scratch allocation was freed, and the
final audited batch is this directory. Aggregate CUDA `candidate_gap` counts
moved from 6,979 to 6,908 operator and from 7,039 to 6,972 Program. These are
diagnostic host-materialization timings; they do not
replace numerical parity or establish a universal model gain.
