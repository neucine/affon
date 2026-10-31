# Canonical performance facility: design proposal

Status: design for review; this document does not declare any current benchmark
or baseline accepted. The existing scripts are prototypes to reuse or replace
against the contracts below.

## Decision the facility must answer

For every supported production workload on a target machine:

1. Is the output correct under the declared numerical tolerance?
2. Has Affon become slower than its last accepted, comparable Affon reference?
3. How far is Affon from a matched peer doing equivalent work?
4. Is the workload represented at all, and which phases, shapes, devices, and
   precisions remain unmeasured?
5. If there is a gap, which model phase and compute operation explains it?

The facility reports these answers separately. A stable but slow implementation
is still a peer gap. A fast result with wrong output is a correctness failure.
An unmeasured cell is never a pass. A single overall speed score must not hide
either a failed case or missing coverage.

## Scope and portfolio

The unit of coverage is a **cell**: workload archetype × target device × model
phase × shape regime × precision × execution path. Model names alone do not
establish coverage. Each supported model maps to cells; if its operator mix,
cache policy, precision, shape range, or serving path is materially different,
it introduces new cells. The manifest records why a candidate represents each
cell, and a reviewer checks that mapping when a model is added or changed.

Initial portfolio, based on current repository paths:

| Archetype | Candidate | Required phase/shape evidence | Present status |
| --- | --- | --- | --- |
| Small decoder canary | DistilGPT-2 | Batch 1, short prompt, complete generation | Metal peer prototype; CUDA unmeasured |
| Cached decoder | SmolLM2 135M and 1.7B | Prefill and token decode separately; short/long context; KV-cache and memory behavior | Affon profiler exists; no matched peer gate |
| Image transformer | ViT base | Resident forward, readback, batch 1 and larger batch | Metal peer prototype; CUDA unmeasured |
| Convolutional vision | MobileNetV2 | Resident forward and HTTP; CPU and accelerator; image/batch sweep | CPU HTTP comparison only |
| Encoder-decoder audio | Whisper tiny | Feature extraction, encoder, first token, later tokens, full transcription; short/long audio | Diagnostic comparison only |
| Audio encoder | AST speech commands | Feature extraction, encoder, output; short/long input | No unified gate |
| Bidirectional text encoder | BERT path | Sequence length and batch sweep | No unified gate |
| Training, where supported | A maintained model training path | Forward, backward, update, memory | Compute-level evidence only |

Start with a small **required core** on each applicable target: cached decoder,
ViT, convolutional vision, and Whisper. Keep GPT-2 as a fast canary. Add AST,
BERT, training, quantized whole-model, and serving concurrency as soon as those
paths are release-supported. The manifest states whether a cell is required,
experimental, or unsupported on each target and why. Unsupported must be based
on an actual capability limit, not on a missing runner. Review the portfolio
against shipped models and representative production shapes at each release;
record both covered and missing cells in the result.

## Targets and execution

- Metal: the current laptop, identified by a stable logical target ID plus a
  captured hardware/software fingerprint for each run. No dedicated Mac is
  assumed.
- CUDA: the Ubuntu NVIDIA PC reached by `ssh chaoyang@pc`; runs and cached model
  files reside there. Its results and baselines are independent of the laptop.
- CPU: an explicit target on the machine where the CPU case is run. CPU results
  never inherit a Metal or CUDA baseline merely because they share a host.

The controller can run locally or invoke the same worker CLI over SSH. The
worker has no network dependency during measurement: model revisions, peers,
toolchains, and fixtures are pinned and cached beforehand. If a host is
unreachable, the run is `infrastructure_blocked`, preserving the last result;
it is never reported as a performance pass or failure. Serialize competing
performance jobs on each host with a lock. Capture power state, thermal and
clock information when available; flag a noisy or throttled run as invalid.
Other user activity on the laptop is a source of noise, so repeatability checks
are mandatory before accepting a small change.

## Manifest and measurement contract

Use one versioned, reviewable manifest as the source of truth for case IDs,
target support, model revision and weight hashes, fixtures, peer engine/version,
correctness tolerance, phases, shapes, dtype, cache policy, timing boundary,
warmups, samples, and performance policy. A case ID is stable only while these
semantics are stable. A protocol hash includes the complete case contract;
changing it starts a new baseline lineage instead of comparing unlike runs.

Every runner emits the same versioned raw envelope. The envelope contains:

- Case and protocol IDs; target fingerprint; repository commits and dirty state;
  executable and source hashes; build flags; OS, driver, compiler, runtime,
  peer and dependency versions; model and fixture hashes.
- Correctness result, output summaries and error metrics, with an explicit
  comparison of Affon and peer outputs. Random inputs are seeded and retained
  or regenerable from a pinned fixture specification.
- Warmup and every timed sample, run order, synchronization points, peak memory,
  allocations/dispatches where available, and timestamps for the major phases.
  Summaries are derived from raw samples, never substituted for them.
- Exact commands and worker logs, with secrets and private paths removed before
  sharing. The artifact is content-addressed and immutable after collection.

Define timing scopes explicitly: cold load/compile, warm resident model phase,
output materialization, and complete request. Compare a peer only within the
same scope. Generation separates prefill from subsequent one-token decode and
uses the same prompt, token budget, stopping rule, and KV-cache policy. Where
engines cannot share a cache policy, report both production paths honestly and
mark a direct phase ratio `not_comparable`; do not force a misleading ratio.
For GPU timings, synchronize at the declared sample boundary. Track latency,
throughput, and memory as separate metrics. p95 requires enough requests to
support a tail estimate; a three-sample median run cannot claim p95 confidence.

## Three references, three decisions

| Reference | Purpose | Rule |
| --- | --- | --- |
| Correctness oracle | Protect semantics | Output error within the case tolerance before timing is accepted |
| Accepted Affon baseline | Detect regressions | Same target and protocol lineage; candidate slowdown exceeds both a case-relative limit and calibrated absolute noise |
| Matched external peer | Expose competitive gaps | Same work and timing scope; case-specific ratio target and absolute latency/throughput shown |

The peer can be PyTorch, MLX, ONNX Runtime, or a vendor implementation where
appropriate. It is a reference implementation, not proof of a global optimum.
Kernel diagnostics may additionally compare alternate Affon implementations,
vendor libraries, and device bandwidth/compute limits. Those comparisons help
explain a gap but do not replace model-level acceptance.

Baseline promotion is explicit and auditable. A candidate run must be complete,
correct, reproducible within the calibrated noise budget, and reviewed with
its exact protocol and target fingerprint. Preserve the old baseline and a
promotion record naming both artifact hashes and the reason. An existing peer
gap does **not** prevent recording a trustworthy Affon regression baseline:
otherwise a slow starting point could never gain regression protection. The
peer gap remains an open, separately failing performance objective. Never
auto-promote the latest run or relax a target because a candidate missed it.
When the hardware, model, peer contract, or timing protocol changes, establish
a new lineage with an overlap run where possible.

Calibrate noise with repeated same-code runs on each target, then use paired
candidate/reference runs when a change is near the threshold. Rotate run order
to reduce thermal or temporal bias. Report confidence as observed variance and
sample count; use `inconclusive` when the effect is within noise. Avoid
aggregate-only acceptance: every required case must have an individual status.
Geometric mean speedup may be shown as a summary after per-case gates.

## Gate and diagnosis behavior

Each required cell resolves to one of: `pass`, `correctness_failure`,
`self_regression`, `peer_gap`, `missing_case`, `unmeasured`, `invalid_evidence`,
`inconclusive`, or `infrastructure_blocked`. Multiple findings can coexist;
for example a peer gap and a self regression. The machine-readable report lists
both underlying comparisons and the final policy decision. CI checks schemas,
fixtures, runner logic, and a quick correctness smoke; it does not use noisy
shared CI timing to accept a small latency change.

When a model case misses its target, profile its exact input and separate model
load, preprocessing, host dispatch, compile/cache behavior, device execution,
sync/readback, and output handling. Link the dominant phase to actual operator
shapes, strides, dtype, layout, and selected backend implementation. Reproduce
that operation in the compute evidence suite, optimize it, then rerun the
original model case and the full required portfolio. Keep before/after raw
artifacts; a faster isolated kernel is not a model-level success by itself.

## Artifact layout and interfaces

Proposed CLI contract, rather than a requirement to keep the current scripts:

```text
benchctl list --target mac-metal
benchctl prepare --target mac-metal
benchctl run --target mac-metal --tier core --output RUN_DIR
benchctl run --target cuda-pc --tier core --output RUN_DIR
benchctl compare --run RUN_DIR --baselines BASELINE_INDEX
benchctl diagnose --run RUN_DIR --case CASE_ID
benchctl propose-baseline --run RUN_DIR --output PROPOSAL_DIR
```

`prepare` checks pins and caches but does not run timing. `run` never changes
baselines. `compare` is pure and repeatable from retained artifacts.
`propose-baseline` creates a reviewable proposal, not an accepted baseline.
The baseline index maps target + protocol hash + case ID to an immutable run
artifact. Store small manifests, summaries, and promotion records in the repo;
keep large raw logs and model weights in a retained artifact store with hashes
and documented access. The suite must reproduce a result from the manifest and
cached inputs without relying on an analyst's shell history.

## Cadence and rollout

Proposed cadence: quick correctness checks on ordinary changes; a targeted
same-host performance run for backend/compiler/model changes; the complete core
portfolio on the laptop and PC at a regular interval and before a release.
The exact timed schedule should be chosen after measuring run cost and host
availability. Missing scheduled runs are reported as stale evidence, never
silently carried forward as a new pass.

1. **Contract first:** freeze schema, manifest semantics, target fingerprints,
   correctness fixtures, status vocabulary, and artifact retention. Migrate
   existing GPT-2, ViT, and MobileNet runners; mark exploratory reports as such.
2. **Reliable baseline lifecycle:** measure same-code noise on the laptop and
   PC, establish reviewed Affon baselines, implement immutable promotion and
   per-case self gates. Validate rejection of changed protocols and missing
   cells with fixtures.
3. **Representative peer portfolio:** add matched SmolLM2 phase runs and
   Whisper, then GPU convolution and encoder cases. Record all existing gaps;
   do not define the first green run by weakening targets.
4. **Operations:** add host serialization, SSH worker execution, retained raw
   artifacts, scheduled runs, trend view, and case-to-operator diagnostic links.

Facility acceptance criteria: a clean run is reproducible from its manifest;
all required cells are accounted for; missing/incorrect/unmatched work fails;
an intentional slowdown is detected on each target; a stable but slow case
still reports a peer gap; an old baseline survives promotion; and an
unreachable PC reports blocked evidence without inventing CUDA results.

## Current repository mapping

`test/benchmark/model-coverage.json`, `tools/model-coverage.py`,
`tools/compare-model-performance.py`, and `tools/normalize-model-performance.py`
are initial registry, gate, and adapter pieces. `apps/hf-inference/benchmark`
contains model runners and peer prototypes. `../compute/benchmarks` has
operator evidence schemas, repeatability work, and baseline examples. None of
these yet implements the complete contract above, and current Metal model peer
gaps and missing CUDA/model-family cases remain open.
