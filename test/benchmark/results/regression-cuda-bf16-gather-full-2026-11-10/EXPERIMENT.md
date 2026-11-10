# CUDA bf16 gather host validation: promoted

The candidate checks host-current gather indices when the output has at most
256 elements. Valid indices bypass the CUDA validation flag read at
synchronization. Invalid or unavailable host indices retain the existing
device validation path.

The RTX 3090 control and candidate each covered all 160 bf16 gather cells in
`test/parity/catalogs/cuda-bf16-gather-full-160.json`, with 10 iterations,
5 samples, and 3 rounds. Both passed 160 numerical, operator, and Program
checks. The candidate's `AUDIT.json` binds catalog SHA
`c131e2713316dba0e6df219f08746d857c1e47e7fc075a5f659dfbe4ced76c73`
and binary SHA
`c33c7b3b4db3388ff9574262f1e0fb938beffc6dfd1921bb455b6a5f7ef892f2`.
The matched regression found zero changed cases, correctness regressions, or
timing suspects. Median Affon time was 0.916× control for operators and
0.924× for Programs; maximum Affon median ratios were 1.013 and 1.013.

Against the previously promoted aggregate, this replacement removed a net
40 CUDA operator gaps and 64 Program gaps. Current CUDA totals are 1,598
operator and 1,855 Program gaps. The prior reports and ledger entries are in
`archive-cuda-bf16-gather-before-host-validation-2026-11-10`.
