# Small f32 matmul kernel: rejected

The candidate added a dedicated rank-two f32 kernel for M ≤ 8, N ≤ 129,
and K ≤ 129. Each output thread accumulated K products with explicit FMA.
It replaced cuBLAS SGEMM for those shapes. The source change was reverted
after the probe.

The RTX 3090 control and candidate used the same 32-cell catalog,
`test/parity/catalogs/cuda-f32-matmul-dispatch-probe-32.json`, at 10 iterations,
5 samples, and 3 rounds. Both passed all 32 numerical, operator, and Program
checks. The matched regression found zero changed cases, zero correctness
regressions, and zero timing suspects. The candidate binary SHA-256 was
`a27e00d6407d867d4a8150a81875c7f723bff29a6f2f55d22e276cd329e1c144`;
the control was the promoted
`c33c7b3b4db3388ff9574262f1e0fb938beffc6dfd1921bb455b6a5f7ef892f2`.

Operator gaps fell from 30 to 17 in the probe. Program gaps fell from 30
to 25, including one new gap. Most improvements were strided cases; several
other shapes slowed. Median Affon time ratios were 0.937 for operators and
0.956 for Programs, while the maximum ratios were 1.126 and 1.095. This
does not justify a broad dispatch replacement against the goal's two counts.
