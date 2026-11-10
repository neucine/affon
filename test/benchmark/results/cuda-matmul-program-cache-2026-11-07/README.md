# CUDA matmul Program metadata cache

This reviewed change caches immutable Program argument names, output axes, and the gradient flag in `src/js/compute/program.ts`. It avoids repeated metadata inspection during execution. No CUDA kernel changed.

The 889-cell f32/BF16 catalog ran on an RTX 3090 with PyTorch 2.11.0+cu128. Baseline and candidate each used 10 iterations, 5 samples, and 3 rounds per cell. Both complete audits passed 889 numerical, 889 operator, and 889 Program checks, with no changed cases or correctness regressions. The binary hashes and full catalog hash are recorded in `baseline-audit.json` and `candidate-audit.json`.

Across matched cells, Affon operator medians improved in 856 of 889 cells, and Program medians improved in 861 of 889. Median candidate/baseline ratios were 0.9676 and 0.9684, respectively. The 19-cell controlled probe improved in both phases for every cell. The three regression-tool timing suspects arise from shifts in PyTorch peer timing; Affon's own medians improved by 3.2–5.1% in those cells. The largest Affon median slowdowns in the full run were 2.3% operator and 1.2% Program, among small minority cells. These comparisons are diagnostic timing evidence, not an application throughput claim.

`baseline-full/` and `candidate-full/` retain raw per-round reports. `regression/`, the audits, probe batches, and build logs retain the review trail. Exploratory small-kernel selection and allocation-pool candidates were rejected after probes and remain in their separate probe directories. The candidate was promoted into the current ledger, with prior entries archived in `../archive-cuda-matmul-before-program-cache-2026-11-07/`.
