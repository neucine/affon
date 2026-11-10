# Rank two and three indexing specialization: rejected

The candidate added direct coordinate calculations for rank two and rank three
`index_select` and `gather` to `compute_index_small` in
`compute/src/backend/cuda/binary_small.cu`. It preserved bounds checking and
the existing generic kernel path for other ranks. The source change was
reverted after measurement.

Both RTX 3090 runs used `cuda-index-rank23-probe.json` with 10 iterations,
5 samples, and 3 rounds. All 11 cells passed numerical, operator, and Program
checks in each run. The matched regression reports zero changed cases, zero
correctness regressions, and zero timing suspects. The bf16 operator medians
improved by roughly 1–2%; f32 changes were smaller. These gains are too small
to clear the sampled candidate gaps, so the dispatch was not promoted.

The control binary was the prior one-row matmul PTX probe build
(`dce57e5223002a1e77668a3869779c27ba584e739ae57afd35f16f0af3e25aa5`).
The candidate binary (`92b13d04776a56d3c7b9d157d954e47dc5b2cf351a58b240e439541a8e140a3d`)
also restored the original f32 matmul dispatch. Neither change affects the
indexing operations in this catalog, but this is a probe rather than a
promotion quality isolated regression batch.
