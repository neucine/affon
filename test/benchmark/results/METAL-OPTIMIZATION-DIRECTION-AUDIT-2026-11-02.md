# Metal optimization direction audit

## Decision

Park new operator `candidate_gap` chasing. The 2,128 remaining Metal gaps and the reductions recorded in [the six-group audit](METAL-SIX-GROUP-AUDIT-2026-11-02.md) are diagnostic results for isolated calls. They are not an accepted measure of improvement to GPU-resident Programs or models. Preserve the correctness evidence and raw timings, but do not use the gap count as the optimization objective.

This follows the [benchmark contract](../README.md): start with a representative model case, trace its dominant phase to an operator, then rerun the model after a change. The cell timing scope includes synchronized full host output materialization. The parity Program cases are small compositions and do not establish model parity.

## Findings

| Recent work | Mechanism | What the evidence establishes | Capability still unproved | Disposition |
| --- | --- | --- | --- | --- |
| one_hot, quantized_matmul, where, f32 rms_norm | Small shared-buffer CPU execution inside the Metal backend | Correctness and lower isolated call/readback ratios for guarded cases | GPU-resident chain, private storage, larger sizes, and model latency | Park as performance claims; retain correctness evidence. Measure real use before further tuning. |
| f32/BF16 small matmul, f32 layer_norm, f32/BF16 scatter_add, gather | Mix of shared-buffer CPU shortcuts and GPU path changes | Correctness and matched cell timing; some GPU fallback and producer tests | Which part improves actual Program/model latency, and whether selection remains good across shapes | Separate shortcut and GPU effects in a workload run before deciding what to retain. |
| BF16 matmul M1/vector/tiled, dot parallel, gather addressing, fused scatter, reduction and normalization GPU changes | GPU kernels or addressing/dispatch changes | Plausible general GPU capability and cell-level correctness/timing where measured | Model benefit, dispatch boundary behavior, and regressions against an accepted same-target baseline | Prioritize workload validation; review selector ranges after tracing model shapes. |

The shared-buffer helpers require shared storage and no recording, pending, or submitted command buffer; they also reject active timing or tracing. For eligible isolated calls, they run on the CPU before the GPU pipeline selector. This is a runtime policy change, not a compiler-selected Metal kernel. Its thresholds are fixed in `compute/src/backend/metal/ffi.m`, and the current cell catalog does not justify them as general crossover points. The f32 rms_norm path, for example, uses rank two and 4,096-element limits, with a 3,000-element outer-axis limit.

The matched regression comparisons check parity and flag timing suspects between cell batches. They do not compare a complete model against an accepted Affon baseline. The reduction from 2,152 to 2,128 after the rms_norm promotion is valid for the aggregate cell diagnostic; its model effect is unknown.

## First model check, 2026-11-03

The cached DistilGPT-2 full-prefix Metal case completed with the pre-switch ReleaseFast binary. Its five measured 16-token generations had a 413 ms median; the matched PyTorch run had a 191.392 ms median. The peer ratio was 2.158, above the provisional 1.25 target. [Raw model evidence](model-direction-audit-gpt2-metal-2026-11-03/) includes correctness output, samples, binary hash, normalized reports, and the failed peer gate. This is a model result, not an accepted same-target self baseline; system activity and noise were not calibrated.

An audit-only environment switch, `AFFON_METAL_DISABLE_SHARED_SHORTCUTS=1`, now bypasses the recent small shared-buffer CPU paths in the Metal backend. With the same rebuilt binary and five measured generations per run, an enabled/disabled/disabled/enabled sequence produced run medians of 415/403/414/412 ms. Output text matched. The spreads overlap; these four runs do not establish a GPT-2 benefit from the shortcuts. A one_hot control cell confirmed the switch affects its intended path: Affon operator execution medians were about 0.0073 ms enabled and 0.2482 ms disabled, with numerical, operator, and Program parity passing in both runs. See the [enabled](parity-metal-shared-toggle-enabled-sample-2026-11-03/) and [disabled](parity-metal-shared-toggle-disabled-sample-2026-11-03/) control reports. The control is one tiny cell and cannot quantify model benefit.

The ViT Metal run failed during runtime setup before timing: `array tensor exceeds 262144 array entries` while initializing `VitProgramRuntime`. The [captured log](model-direction-audit-metal-2026-11-03/vit-metal.log) records the failure. ViT remains unmeasured in this audit. The full-prefix GPT-2 canary also does not cover cached decode, quantized inference, image models, or training.

## Backend policy correction, 2026-11-03

Metal GPU execution is now the default. The small shared-buffer CPU paths run only when `AFFON_METAL_ENABLE_SHARED_CPU_SHORTCUTS=1` is explicitly set. This keeps the experimental paths available for comparison while preventing a Metal selection from silently executing those operations on the CPU. The ReleaseFast build and 12-cell Metal smoke batch passed numerical, operator, and Program parity with the GPU default. A representative f32 rms_norm cell and one_hot cell also passed all three checks on the GPU default. A one_hot opt-in control passed as well: its Affon execution median was 0.0076 ms, versus 0.2456 ms on the default GPU path. This confirms that the opt-in changes the execution path; it does not establish model benefit.

The aggregate `candidate_gap` totals above were measured before this correction and are historical. They cannot describe current GPU-default performance until the affected catalogs are rerun. The model comparison above was made before the correction; its disabled-path runs approximate the new default, but it is not an accepted self baseline.

## Required evidence before more optimization

1. Select a canonical Metal workload from the registered model cases, with a fixed revision, input, correctness check, and timing boundaries. Include warm resident execution, synchronized graph completion, and output materialization separately.
2. Record an accepted same-target Affon baseline and matched peer run. Include prefill/decode or other actual phases, representative sizes, and raw repeated samples.
3. Trace the slow model phase and selected physical paths. Reproduce only its actual shapes and layouts in a focused benchmark.
4. Compare the current backend with the shared-buffer shortcuts disabled, then with the GPU changes disabled where they can be isolated. Keep the same build, target, and input contract. Include chained GPU producers and consumers plus sizes around dispatch thresholds.
5. Keep an optimization only when it improves the targeted workload without material regression in adjacent regimes. Record the measured effect and selector rationale; then rerun the canonical model case.

No causal model speedup is claimed in this audit. The existing gap reports remain useful for locating diagnostic cases, correctness coverage, and investigating a workload bottleneck once one is observed.
