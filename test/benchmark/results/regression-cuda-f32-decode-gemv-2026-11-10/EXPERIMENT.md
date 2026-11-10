# f32 one-row warp GEMV probe

The candidate adds `compute_matmul_m1_f32` to Compute's `binary_small.cu` and
dispatches f32 two-dimensional matmul with M=1 through it. Each warp reduces
one output column across K. The source change was reverted after the
full-family run.

The RTX 3090 probe reused `cuda-f32-decode-ptx-probe-8.json` against the
original cuBLAS control binary SHA-256
`7e4c26cb930ec660505be0e0e0452a7b29207727b0582819b3a1ee8fc3537447`.
The candidate binary SHA-256 is
`010341c39969b2fba715807120e6b5b4c230b42005de0b05a99f990a3bfa268e`.
Both used 10 iterations, 5 samples, and 3 rounds. All eight candidate cells
passed numerical, operator, and Program checks; the matched regression found
zero changed cases, zero correctness regressions, and zero timing suspects.

Operator medians fell about 6–10%, and Program medians about 5–9% relative to
the cuBLAS control. A subsequent 60-cell run
(`cuda-f32-decode-gemv-family-candidate-2026-11-10`) passed all numerical,
operator, and Program checks, but classified 56 operator and all 60 Program
cases as `candidate_gap`. Against the reviewed aggregate it cleared two
operator gaps, added seven operator gaps, and cleared no Program gaps. Those
aggregate classifications compare runs taken at different times and are
diagnostic rather than a matched regression. The full-family result does not
meet the high-cell-impact promotion requirement, so the candidate was rejected.
