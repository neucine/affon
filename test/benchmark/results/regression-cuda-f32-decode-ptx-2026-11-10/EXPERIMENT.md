# f32 one-row matmul PTX fallback: rejected

The RTX 3090 control used Compute's cuBLAS dispatch. The candidate changed only
`affon_compute_cuda_matmul_f32` in `compute/src/backend/cuda/ffi.c`: its cuBLAS
condition added `m != 1`, sending one-row f32 matmul to the existing PTX
fallback. The source change was reverted after measurement.

Both runs used `cuda-f32-decode-ptx-probe-8.json`, 10 iterations, 5 samples,
and 3 rounds on the same RTX 3090 target. Each passed all 8 numerical,
operator, and Program checks. The control binary SHA-256 was
`7e4c26cb930ec660505be0e0e0452a7b29207727b0582819b3a1ee8fc3537447`;
the candidate binary SHA-256 was
`dce57e5223002a1e77668a3869779c27ba584e739ae57afd35f16f0af3e25aa5`.

Matched operator medians were about 10–13% slower with PTX for K/N near 64,
about 35% slower for K/N 128, and nearly unchanged near 32. Program medians
showed the same direction. `regression.json` records zero correctness
regressions and one timing suspect. The candidate was not promoted because it
regresses the largest sampled shapes and does not reduce the target gaps.
