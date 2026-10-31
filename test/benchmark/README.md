# Performance coverage and model onboarding

Focused quantized matrix multiplication investigations live in
[`quantized/`](quantized/README.md). Broad operator and Program comparisons use
the [parity facility](../parity/README.md); retained evidence is indexed in
[`results/`](results/README.md).

The [workload coverage facility](WORKLOAD-COVERAGE.md) generates a matrix from
the canonical operation and dtype vocabulary, then maps existing diagnostics
to it. Run `python3 tools/coverage-matrix.py` for the candidate matrix and
`python3 tools/workload-coverage.py` for the smaller diagnostic-link inventory.
Neither report turns an existing test into an accepted performance pass.

## Canonical facility contract

The registry is the source of truth for required target groups, workload
archetypes, case IDs, peers, and acceptance ratios. A canonical run has four
artifacts: immutable raw Affon and peer measurements; normalized reports with
identical workload contracts; an accepted, same-target Affon baseline; and the
coverage result. The result must account for every required archetype and case,
including missing ones. A tuning notebook or isolated kernel speedup is
diagnostic evidence, not a release result.

For each target, run the fixed registry on the same machine and software
configuration, record source and binary hashes, and keep all raw samples. The
normalizer must reject incorrect output or unequal timing boundaries before
speed is compared. The peer comparison answers whether Affon is competitive;
the self comparison answers whether it regressed. Neither substitutes for the
other. Accept a new baseline only after reviewing a complete run, correctness,
noise, peer gaps, and any intentional protocol change. Keep the previous
baseline and its report; never replace it automatically with the latest run.
Changing a model revision, input, precision, cache policy, timing boundary, or
hardware creates a new contract or target and requires a new baseline. A
performance failure should be traced from the full model to its dominant phase
and then to the relevant operator shape; rerun the canonical model case after
the optimization. This is the path from exploratory tuning to lasting evidence.

The benchmark suite is layered. A passing operator benchmark does not establish
model parity: graph construction, layout conversion, command submission,
synchronization, preprocessing, and output handling can dominate a real model.
Keep correctness and performance results together, and report an unsupported or
unmeasured case as such rather than as a pass.

## Current coverage

| Layer | Existing evidence | Remaining gap |
| --- | --- | --- |
| Compute Program and physical selection | [`../../../compute/benchmarks/README.md`](../../../compute/benchmarks/README.md): 22 cases at two optimization levels; repeatability and regression comparators; selected PyTorch gates | Representative Programs do not cover every model operator, dtype, layout, or model shape |
| Metal operator comparison | Compute's 64-case f32 PyTorch diagnostic; quantized matmul matched against MLX in [`../../docs/core/quantized-profiling.md`](../../docs/core/quantized-profiling.md) | The broad f32 comparison is diagnostic, not an accepted repeated gate; quantized coverage is narrow |
| Native model path | [`../../apps/hf-inference/benchmark/README.md`](../../apps/hf-inference/benchmark/README.md): DistilGPT-2 and ViT; Whisper diagnostic scripts | No uniform peer parity or regression gate across model families |
| Serving path | [`../../apps/vision-serving/CPU-PERFORMANCE.md`](../../apps/vision-serving/CPU-PERFORMANCE.md): MobileNetV2 CPU graph and HTTP comparisons with ONNX Runtime | One model, target, and serving configuration |

These are separate scopes and should not be combined into one speed ratio.
Existing uncommitted benchmark and backend work must be identified by source and
binary hashes when used as a baseline.

## Are the models representative?

No. The current three models are useful canaries, but model count is a poor
coverage measure. GPT-2 is a small decoder, ViT is an image transformer, and
MobileNetV2 is a CPU convolutional serving case. Their results cannot establish
performance for Affon's other supported execution paths. Track **workload
archetype × device × phase × shape regime × precision** instead. A model is a
representative only for the cells it actually exercises with a matched peer.

| Workload archetype | Candidate already in this repository | Critical phases and shape regimes | Current matched gate |
| --- | --- | --- | --- |
| Small legacy decoder | DistilGPT-2 | Prompt generation, short batch-one request | Metal; CUDA case registered but unmeasured |
| Modern cached decoder | SmolLM2 135M/360M/1.7B | Separate prefill and token decode; short/long context, batch, KV-cache growth, memory; quantized variant when supported | Missing. An Affon phase profiler exists, but no matched peer gate |
| Image transformer | ViT base | Batch and image resolution; model forward and output readback | Metal; CUDA case registered but unmeasured |
| Convolutional/depthwise vision | MobileNetV2 | CPU and GPU; batch and resolution; resident inference and HTTP request | CPU HTTP only |
| Encoder-decoder generation | Whisper tiny | Audio duration, encoder, first decode and cached decode, end-to-end transcription | Diagnostic peer script only; no unified gate |
| Audio encoder/classifier | AST speech commands | Spectrogram length, encoder, classification | Missing |
| Bidirectional text encoder | BERT implementation | Sequence length, batch, attention and classification/embedding readback | Missing model peer gate |
| Training | Supported training models | Forward, backward, optimizer update, memory | Compute evidence only; no model peer gate |

Prioritize the modern cached decoder and Whisper: they add execution patterns
that the current gate does not exercise. Keep GPT-2 as a small regression canary,
not a proxy for current LLMs. Add GPU MobileNet and an encoder case next. An
operator-only quantization result is not evidence for a quantized whole model.
Do not count a row as covered until correctness, equivalent timing boundaries,
peer performance, and a same-target baseline are all recorded. A new model
maps to an existing archetype only if it uses the same operators, cache behavior,
precision, shapes, and serving path; otherwise add a new archetype or cell.
Review this matrix against actual supported models and production usage on each
model release. That is how the portfolio remains representative as Affon grows.

## Required record for a new model

Create a model-specific benchmark manifest before optimization. Record:

1. Exact model revision and file hashes; preprocessing, tokenizer, weights,
   numerical tolerance, and output correctness checks against the peer.
2. Hardware, OS, backend, Affon and peer versions, build profile, thread limits,
   memory budget, and executable/source hashes.
3. A **workload matrix**: at least the intended production batch and input
   sizes, plus a small and a large case where applicable. For generation,
   measure prefill and one-token decode separately, and state whether each
   engine uses a KV cache. For training, include forward, backward, and update.
4. Matching timing boundaries: loading/compilation, warm resident inference,
   synchronized graph completion, output materialization, and complete request
   latency where applicable. Explicitly state which allocations and transfers
   are inside each boundary. Do not compare unlike boundaries.
5. Per-case warmups, raw samples, median and p95, peak process memory, output
   error, and status. Run peers sequentially on the same target; rotate order
   over at least three independent rounds. Persist the exact command and report.

Use the model report to identify the slow phase. Reproduce the dominant operation
with its actual shape, dtype, strides, and layout in the compute evidence suite.
Keep that microbenchmark linked to the original model case; a faster isolated
kernel is not an accepted model improvement until the model is rerun.

## Acceptance and regression policy

- **Correctness gate:** reject missing peer outputs, wrong shapes, or errors
  above the model's declared tolerance before comparing speed.
- **Self-regression gate:** compare each Affon case against a same-target,
  same-protocol baseline. Use repeated same-code runs to calibrate a target-wide
  noise budget, as the compute evidence suite does. Flag a case only when the
  slowdown exceeds both the relative threshold and calibrated absolute noise.
- **Peer gap gate:** report Affon/peer ratios for each matched scope and case.
  A peer gap is visible even when Affon has not regressed. Set a model-specific
  acceptance target before tuning; do not silently loosen it after measurement.
- **Coverage gate:** a new model is not characterized until its production
  workload, correctness check, peer comparison, and linked bottleneck evidence
  are recorded. Missing cases remain `unmeasured`.

Run a short deterministic correctness and benchmark smoke check on ordinary
changes. Run repeated performance gates on the same controlled target for
backend/compiler changes and model releases: this laptop for Metal and
`chaoyang@pc` for CUDA. Shared CI hardware is too noisy
for small latency regressions. Retain raw reports, protocols, and baseline hashes
so a later slowdown can be attributed to a model, runtime, compiler plan, or
backend kernel change.

## Normalized comparison gate

`tools/compare-model-performance.py` is the first shared gate. Model runners
currently have different output formats; adapters must emit this small envelope
before the gate is used:

```json
{
  "schema_version": 1,
  "suite": "affon_model_performance",
  "target": "apple-m4-local",
  "engine": "affon",
  "cases": [{
    "id": "model/revision/metal/batch1/prefill128",
    "correct": true,
    "contract": {
      "model_sha256": "same-on-both-engines",
      "shape": [1, 128], "dtype": "f32", "phase": "prefill",
      "timing_scope": "resident synchronized output materialization",
      "warmups": 5, "cache_policy": "warm", "threads": 1
    },
    "samples_ms": [1.2, 1.3, 1.25]
  }]
}
```

The contract must contain every attribute needed to establish equivalent work.
The target identifies a specific machine configuration, not just an architecture.
Case IDs and contracts must match exactly. A missing case, failed correctness
check, nonpositive sample, or changed contract rejects the comparison.

```sh
python3 tools/compare-model-performance.py --kind self \
  --current candidate.json --reference accepted-affon.json \
  --maximum-ratio 1.10 --absolute-noise-ms 0.05 --output self-gate.json
python3 tools/compare-model-performance.py --kind peer \
  --current candidate.json --reference matched-peer.json \
  --maximum-ratio 1.25 --output peer-gate.json
```

These illustrative limits are policy inputs, not validated defaults. Calibrate
the self noise allowance from repeated same-code measurements on the target.
The gate compares medians and retains each case ratio; it does not establish p95
latency from three samples. No new model baseline is claimed by this tool alone.

The existing MobileNetV2 HTTP report can now be normalized into both engine
reports. The adapter requires the runner's passing output-parity checks:

```sh
python3 tools/normalize-model-performance.py --source vision \
  --input /path/to/vision/report.json --target my-exact-machine \
  --output-dir /tmp/vision-normalized
python3 tools/compare-model-performance.py --kind peer \
  --current /tmp/vision-normalized/affon.json \
  --reference /tmp/vision-normalized/onnxruntime.json \
  --maximum-ratio 2
```

The existing GPT-2 and ViT Affon run directory can be normalized for a
same-target self-regression check:

```sh
python3 tools/normalize-model-performance.py --source hf \
  --input /path/to/hf-run-directory --target my-exact-machine \
  --output-dir /tmp/hf-normalized
```

GPT-2 and ViT now have a PyTorch peer runner through
`apps/hf-inference/benchmark/run.py --pytorch-python`; see that benchmark's
README for its exact contract. Metal peer runs have been recorded on this
laptop; CUDA remains unmeasured while the user's PC is unreachable. CI checks
comparison and adapter logic; timed gates use this laptop for Metal and the PC
for CUDA.

## Coverage registry

`model-coverage.json` lists the model cases currently expected to have both
self and peer evidence. Adding a supported model means adding its production
cases here; a missing report then appears as `unmeasured` and fails the gate.
It also declares required workload archetypes for each target. An archetype
without any registered case appears as `missing_case` and fails even inventory
mode (`--allow-unmeasured`). This prevents a short case list from appearing
complete merely because every listed case passes. Registered cases identify
their archetype; a model name alone does not imply coverage of every phase or
shape in that archetype. The current registry is deliberately incomplete.
The `mac` target group runs on the current Apple laptop; `cuda-pc` runs on the
user's Ubuntu NVIDIA PC over SSH. They have separate evidence directories and
acceptance results. The machine's actual hardware and runtime versions still
belong in each run's metadata.
The 1.25 PyTorch targets for GPT-2 and ViT are initial development targets,
not claims that those models have passed. MobileNet's 2.0 ONNX Runtime target
comes from its CPU audit. Targets can be revised through an explicit policy
change, never by silently omitting a slow case.

Use one evidence directory per target with `hf/`, `vision-t1/`, and
`vision-t4/` subdirectories. Each runner can write its normalized files beneath
its output directory; place an accepted same-target `baseline-affon.json` in
each suite directory. A baseline must come from a reviewed, complete run with
the same timing contract. Do not promote a candidate merely because it is the
latest result.

```sh
python3 tools/model-coverage.py --target-group mac \
  --evidence-dir /path/to/mac-evidence \
  --output /path/to/mac-evidence/coverage.json
```

The command exits nonzero for a missing archetype case, an unmeasured case, a
peer gap, a self-regression, or invalid evidence. `--allow-unmeasured` is only
for inventory: it tolerates missing reports but still fails missing archetype
cases, measured gaps, and regressions. Peer and self
statuses are shown independently so a missing baseline cannot hide a known
peer gap. The registry's current self limit is a provisional 10% median ratio
without a noise allowance; calibrate it from same-code target repeats before
using it as a release gate. This registry does not schedule hardware runs.

The first local M4 exploratory portfolio is recorded in
`results/model-portfolio-mac-2026-10-18.json`. Both HF Metal cases fail the
peer target while the two historical MobileNet CPU cases pass their 2.0 target.
All four lack accepted same-code baselines. The individual GPT-2 and ViT reports
retain raw samples and caveats. The CUDA PC was unavailable during this run,
so its target group remains unmeasured. Do not combine Mac and CUDA ratios or
promote this exploratory report as a baseline.
