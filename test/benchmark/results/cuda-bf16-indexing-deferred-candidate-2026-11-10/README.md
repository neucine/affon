# CUDA deferred index validation

The reviewed Compute change moves small-index validation out of each kernel
launch and into device synchronization:

- `affon_compute_cuda_index_small` no longer allocates, clears, synchronizes,
  copies back, and frees a validation flag on every call.
- The device owns one lazily allocated flag; the indexing kernel still records
  invalid indices in it.
- `affon_compute_cuda_synchronize` copies the flag back, clears it, and returns
  `-4` when an earlier launch recorded an invalid index.
- The flag is freed with the device.

The runtime synchronizes before every readback, so a deferred index error still
surfaces at run completion rather than at the transform call. This mirrors the
already-promoted Metal `pendingValidationFlags` behavior. The source change is
limited to the sibling Compute checkout's `src/backend/cuda/ffi.c`.

## Contract

- Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`
- Control Affon binary:
  `42a8c02d83f1ffcb4ddeac1c144f2330ec8a7ce96db672bb8fc89c9321e71b2a`
- Candidate Affon binary:
  `b3db0353778c60e0dbe45469cdc58ffb600e5050608b3c61598a2959470686d4`
- Peer: PyTorch `2.11.0+cu128`
- Catalog SHA-256:
  `da3fcf87c7c0fdab3ef8fb25135a2c5942b48317b3d3d3693eff59ebae44e6f7`
- Protocol: 10 iterations per sample, 5 samples, 3 independent rounds
- Catalog scope: all 384 BF16 `index_select` and `gather` cells
- Complete candidate batch: 384/384 numerical, operator, and Program checks
  passed.
- Complete-batch audit: `audit-index-deferred.json` binds the expected catalog,
  candidate binary, target, peer version, and measurement protocol.
- Matched control batch: `../cuda-bf16-indexing-deferred-control-2026-11-10`.
- Smoke and sample batches covering the same transform paths are retained at
  `../cuda-index-deferred-*2026-11-10`.

## Controlled effect

Every affected cell improved; none regressed.

- Operator execution speedup: median 1.099x, minimum 1.066x, maximum 1.243x.
- Operator total latency speedup: median 1.062x, minimum 1.011x.
- Program execution speedup: median 1.234x, minimum 1.201x, maximum 1.255x.
- Program total latency speedup: median 1.157x, minimum 1.051x.

Dispatch is unchanged: both binaries report the same selected implementations
for all 384 cells, so the change is measured as host-side launch and
synchronization overhead only.

The promoted status counts moved in the intended direction. Across the 384
promoted cells, operator `candidate_gap` fell from 239 to 218 and Program from
288 to 218. Across the complete aggregate, operator `candidate_gap` fell from
3,081 to 3,060 and Program from 4,082 to 4,012.

The matched control batch labelled 244 operator and 282 Program cells as
candidate gaps; the aggregate's previous labels for the same cells were 239 and
288. Both comparisons agree that the replacement removes gaps, and the ledger
deltas above use the aggregate's own prior reports.

No other dispatch is modified. The validation flag is allocated on the first
indexing launch for a device, and synchronization checks it only when it exists;
a process that never runs an indexing kernel performs no extra copy or clear.
The prior ledger entries and aggregate reports are retained in
`../archive-cuda-bf16-index-deferred-before-2026-11-10/`.
