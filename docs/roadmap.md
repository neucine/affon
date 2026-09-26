# Roadmap: application-oriented ML

Affon is an experimental TypeScript compute runtime. Its working direction is
application-embedded inference and lightweight learning, with a small deployment
runtime and no Python environment required in the deployed application. This is
a differentiation hypothesis to validate, not a demonstrated advantage yet.

The intended user builds and ships applications. General academic/research
framework coverage is not the objective. Training and inference should serve
complete application workflows, rather than grow as independent API inventories.

## Direction and boundaries

- **CPU and Metal are first-class validation targets.** Prioritize correctness,
  usability, memory, and performance on hardware we can regularly exercise.
  This priority does not imply every operation is implemented or verified.
- **CUDA is a secondary validation target.** Existing support remains useful,
  but testing is less complete. Report CUDA evidence separately; CPU/Metal
  success does not establish CUDA correctness or performance.
- **Implemented is not the same as validated.** Distributed-training capability
  does not establish large-scale readiness. Without access to representative
  infrastructure, large-scale distributed training is outside the roadmap's
  validation commitments.
- **Design for Affon's applications, not PyTorch compatibility.** Reuse sound
  ideas and mathematical semantics, but do not inherit API conventions solely
  for compatibility. Full PyTorch API parity is not a goal.
- **Hugging Face is a verification resource.** Selected pretrained models expose
  gaps in operators, loading, preprocessing, and numerical behavior. Broad HF
  integration, model-count coverage, and ecosystem membership are not goals.
- **Let application evidence justify core work.** An operator, importer, or new
  backend feature should unblock a chosen workflow or fix a measured defect.

## Delivery sequence

| Stage | Work | Exit evidence |
| --- | --- | --- |
| 0 — Establish correctness | Retain training regressions and selected pretrained inference audits | Reproducible references, explicit tolerances, and recorded failures on CPU/Metal |
| 1 — Prove an application | Select one workflow combining pretrained inference and lightweight local learning | A runnable application with useful task behavior and a measured reference implementation |
| 2 — Make it shippable | Bundle runtime, model assets, processors, and entry point; document requirements | Reproducible installation and execution without Python at deployment |
| 3 — Improve practical execution | Address the application's measured loading, memory, dtype, or kernel bottlenecks | Measured footprint, startup, latency, and training/update cost on target hardware |
| 4 — Test reuse | Add a second application with different input or execution needs | Shared abstractions work without per-application core special cases |

An initial candidate is local image organization: a pretrained encoder produces
features and a small classifier learns user-provided labels. This is a candidate,
not a committed product. Choose the application before expanding model families.

Evaluate a bounded graph-import experiment (for example ONNX) against the
existing reference models before committing to more handwritten architectures.
An importer is worthwhile if it simplifies the chosen applications; full format
coverage is not a prerequisite or a goal in itself.

For each experiment, record the hypothesis, reference baseline, target hardware,
correctness criteria, and practical measurements. Smaller footprint, simpler
distribution, or easier local adaptation must be demonstrated rather than assumed.
If a feature serves none of these experiments, defer it.

## What counts as support

Track support per **workload/configuration + processor + artifact format/dtype
+ backend**. A passing tiny checkpoint does not establish an architecture family,
a device, or production readiness. Distinguish:

- Implemented: a code path exists.
- Verified: a stated workload passed reproducible checks on a stated backend.
- Untested or partially verified: evidence is missing or bounded; retain failures.

For pretrained probes, record immutable checkpoint revisions and independent
reference versions; preprocessing, intermediate and final output comparisons;
input/configuration validation; offline regressions; and device/dtype/size limits.
Use synchronized measurements for performance claims. Python frameworks may
provide preparation and independent oracles without becoming deployment dependencies.

## Candidate verification probes

The GPT-2, BERT, and ViT audits are existing evidence, with their limitations
recorded below. Additional probes are selected only when applications need them:

| Application need | Possible probe | What it would investigate |
| --- | --- | --- |
| Text features or classification | BERT/DistilBERT or a sentence embedding model | Padding, pooling, task heads, local classifier updates |
| Image features or classification | ViT or a CNN | Image decoding, resizing, convolution, feature extraction |
| Image-text retrieval | CLIP | Paired processors and normalized similarity |
| Local speech processing | Whisper or Wav2Vec2 | Resampling, spectral features, temporal operations and decoding |
| Text generation | A small decoder | State/cache behavior, stopping, streaming, memory limits |

These are possible tests, not promises to support each family. Larger models,
diffusion, video, and broad multimodal coverage require an application case before
becoming roadmap commitments.

## Package boundaries

- `@affon/huggingface` owns HF configuration, weight mapping, model/task
  dispatch, processor integration, and pinned public Hub snapshots with verified
  offline caching through Hao native HTTP streaming. Shards, authentication,
  and removing remaining cache utility dependencies remain work.
- Architecture packages own reusable blocks. Existing `@affon/transformers`,
  `@affon/lm`, `@affon/cnn`, and `@affon/vision` are starting points; add
  audio/multimodal packages when used.
- Tokenizers and domain processors own input/output compatibility. Do not
  encode an assumption that every model consumes text IDs or generates tokens.
- `affon:compute` and `affon:nn` own primitives justified by application workloads and their audits.
- `affon:checkpoint` remains the low-level tensor/training persistence surface;
  Hub resolution and architecture dispatch live above it.
- Complete audits and runnable inference workflows live in `apps/`.

The audited GPT-2, BERT, and ViT implementations now share the experimental
`@affon/huggingface` package. Audit machinery stays app-owned. Unsupported
architectures/configurations must fail explicitly.

## Current work

The [HF inference audit](../apps/hf-inference/README.md) is a verification experiment.
See its [gap report](ml/inference-gaps.md) for measured findings, limitations,
and the next concrete probes.
The [expanded audit](ml/expanded-inference-audit.md) now includes production-sized
DistilGPT-2, BERT embeddings, ViT classification, and memory samples. The tested
BERT and RGB resize pipelines now pass; strict ViT hidden-state parity remains
open, with rounding-sensitivity evidence recorded. These results establish only the tested configurations; broad family
support is not the roadmap objective.

The [bounded ONNX graph-import experiment](../apps/hf-inference/onnx/README.md)
now compares a converted static ViT graph with the existing native adapter.
The second probe now runs [MobileNetV2](../apps/hf-inference/onnx/reports/mobilenet/summary.md)
through the same importer using generic spatial operators, with CPU/Metal parity
and a passing ViT regression. This provides bounded reuse evidence across a
transformer and a CNN; it does not establish broad ONNX or HF coverage.

The tested runtime and converter now live in experimental `@affon/onnx`.
HF local loading can select this backend for image classification and AST audio classification; native
architecture adapters remain a parallel path. Hub graph acquisition and
additional task bindings remain future work.

Reference task contracts: [BERT](https://huggingface.co/docs/transformers/model_doc/bert),
[ViT](https://huggingface.co/docs/transformers/model_doc/vit),
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper), and
[Diffusers pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview).

The third ONNX probe now runs [AST speech commands](../apps/hf-inference/onnx/reports/audio/summary.md)
with CPU/Metal reference parity. WAV decoding, stereo downmix, anti-alias
resampling and AST filterbank processing live in the HF package; the playground
adds upload and playback. Graph execution reused existing operators with a
static Unsqueeze-to-Reshape conversion. Codec breadth, streaming audio,
accelerated DSP, additional audio architectures and multilabel task semantics
remain gaps; this checkpoint is limited to 35 command words.

The [Whisper milestone](../apps/hf-inference/onnx/reports/whisper/summary.md) now
adds bounded English speech-to-text with CPU/Metal token and logit parity, native
Whisper features, and request-local self/cross-attention caches. The playground
serves WAV transcription with 16 MiB / 30-second uploads and 254 generated tokens.
Profiling found quadratic work in generic Metal softmax; dense last-axis tensors
now reuse the row kernel. The warm encoder fell from 8.67 s to 0.54 s, with HF
token/logit parity preserved. Native telemetry now profiles cached decoding without lost records. Batching
Metal concatenation blits cuts median decoder graph time from 1.43 s to 1.19 s
across three runs; see the [telemetry report](../apps/hf-inference/benchmarks/reports/whisper-telemetry/README.md).
Shape telemetry identified long cross-attention value products. A cooperative-load
Metal kernel lowers their measured time from 104 ms to 43 ms with strict HF parity.
Short-clip total improves slightly; longer-clip totals remain noisy and unchanged
(see the [matmul report](../apps/hf-inference/benchmarks/reports/whisper-matmul/README.md)).
The UI separates preprocessing, encoder and decoder timings. Native Erf now
replaces twenty eager operations per node, removing 836 decoder dispatches on the
short reference clip. Fresh total-latency medians improve 1.64→1.44 s (short) and
6.49→5.57 s (long), with strict CPU/Metal HF parity; see the
[native Erf report](../apps/hf-inference/benchmarks/reports/whisper-erf/README.md).
Dense last-axis Metal layer normalization now computes statistics once per row,
reducing fresh short/long total medians from 1.423→1.283 s and 5.538→5.318 s.
Strict HF parity is preserved. Generic affine graph fusion is available, but the
ONNX opt-in stays off by default because its isolated end-to-end gain is negligible;
see the [normalization report](../apps/hf-inference/benchmarks/reports/whisper-normalization/README.md).
Next investigate small-operation batching or larger graph fusion in the compute
core, gated on strict parity and end-to-end gains.

A [matched PyTorch MPS comparison](../apps/hf-inference/benchmarks/reports/whisper-pytorch/README.md)
now establishes a substantial remaining Whisper performance gap: Affon totals
are about 11–12× slower than HF eager and 14–19× slower than SDPA on two clips,
with exact token/text parity. Excluding preprocessing, execution remains roughly
9–14× slower. Next isolate dispatch/synchronization and host overhead before
selecting another compute optimization; track audio DSP separately. These are
bounded Whisper results, not a general comparison of model coverage or training.

The [Whisper overhead investigation](../apps/hf-inference/benchmarks/reports/whisper-overhead/README.md)
now separates Metal command GPU time from host commit/wait time using opt-in
compute telemetry. The decoder issues 209 synchronous buffers per step; about
65% of decoder graph wall time lies in commit/wait outside GPU execution.
Host token selection adds 0.10/0.51 s on the short/long clips. Batching and asynchronous scheduling are deferred architecture proposals,
requiring careful design of lifetimes, synchronization, errors and execution
boundaries. The immediate work is finer CPU/Metal attribution and isolated
experiments; native suppressed argmax remains a separate candidate. Encoder time is
mostly GPU execution and needs a different investigation.

[Finer submission timing](../apps/hf-inference/benchmarks/reports/whisper-submission/README.md)
shows that narrow command creation/encoding and host commit calls are small; the
large completion-wait residual still needs attribution. Batching/async remains
deferred. A localized score-first suppression check in host token selection
reduces short/long total medians from 1.277→1.192 s and 5.356→4.839 s, with exact
output parity and unchanged compute execution semantics.

[Whisper-shaped operator benchmarks](../apps/hf-inference/benchmarks/reports/whisper-operators/README.md)
now confirm a GPU-side matmul/attention gap on representative encoder shapes:
Affon GPU duration alone exceeds PyTorch synchronized wall time. The current
layout-aware dispatch uses generic strided kernels and bypasses the existing MPS
helper; removing a size-one batch dimension does not change that route. Next
consider a localized eligible-layout backend-selection prototype, with strict
numerical/model parity and full transcription gates. Async/batching remains deferred.

The [dense MPS selection prototype](../apps/hf-inference/benchmarks/reports/whisper-layout-mps/README.md)
now accelerates eligible encoder matrices by roughly 10× in GPU time and passes
Whisper, ViT and AST parity. Short total improves 1.189→1.114 s, but two long-clip
comparisons show no total benefit despite faster encoding. Keep
`AFFON_METAL_LAYOUT_MPS=1` opt-in and scheduling unchanged; investigate attention
kernel selection and latency repeatability before broad default enablement.

[Automatic Metal matmul selection](../apps/hf-inference/benchmarks/reports/whisper-auto-mps/README.md)
replaces the experimental flag. Compatible larger dense/transpose/offset/batched
products use MPS, while small and unsupported products keep their existing kernels.
Several representative encoder matmuls are now close to PyTorch MPS wall time;
Whisper's measured encoder is 106–111 ms. Full requests remain about 9–17× slower
than PyTorch due to remaining preprocessing, attention and decoding costs. Strict
Whisper/ViT/AST parity and 82 regression checks pass. Scheduling remains unchanged.

[Native spectral preprocessing](../apps/hf-inference/benchmarks/reports/whisper-dsp/README.md)
replaces Whisper's TypeScript FFT/filterbank loops with reusable CPU compute
primitives. HF retains framing, window/filter construction and log normalization.
Long-clip feature processing falls from 1,322 to 30 ms; current preprocessing
including WAV decoding is 13/60 ms for short/long clips. Strict feature/logit/token
parity and 85 regression checks pass. Full request medians are 0.785/3.354 s;
decoding remains the largest gap. Scheduling is unchanged.

The [single-row Metal matmul follow-up](../apps/hf-inference/benchmarks/reports/whisper-single-row/README.md)
extends automatic MPS selection to compatible products with K >= 256 and
K*N >= 65,536. Strict Whisper parity and 87 regression checks pass. Fresh local
request medians improve from 805 to 743 ms (short) and 3,368 to 3,071 ms (long),
with unchanged tokens. Short-reduction attention scores retain existing kernels
after a measured MPS regression. Next investigate short-reduction decoder value
kernels; completion overhead remains unresolved and async/batching stays deferred.

The [short decoder reduction follow-up](../apps/hf-inference/benchmarks/reports/whisper-short-reduction/README.md)
extends the existing order-preserving SIMD prefetch kernel to K >= 64, N <= 128.
The representative K=257 value product drops from 0.348 to 0.089 ms GPU time;
strict Whisper parity and 88 tests pass. Full-request medians move only 1–2%,
within overlapping sample ranges, so no reliable end-to-end speedup is established.
Next reattribute decoder time with Affon telemetry; async/batching remains deferred.

The [fresh Affon telemetry profile](../apps/hf-inference/benchmarks/reports/whisper-reattribution/README.md)
measures long decoder graph wall/GPU time at 2,686/264.5 ms, still with 209 command
buffers per step. Completion waits account for about 2,173 ms (overlapping GPU
work); the counters cannot attribute the non-GPU portion to specific causes.
Next capture a Metal System Trace to examine submission, GPU execution and CPU
resumption before selecting another optimization. Async/batching remains deferred.

The [Metal System Trace capture](../apps/hf-inference/benchmarks/reports/whisper-system-trace/README.md)
shows median encoding-end-to-GPU and GPU-end-to-completion-event intervals of
101 and 83 µs, versus a 5.9 µs GPU span in the selected window. The capture includes
non-decoder work and adds overhead; these are not estimates of recoverable time.
Next correlate CPU context-switch/wakeup events to distinguish driver/completion
work from CPU scheduling. Execution strategy and deferred async/batching stay unchanged.

The [combined CPU/Metal capture](../apps/hf-inference/benchmarks/reports/whisper-cpu-scheduling/README.md)
finds only 12.9 ms runnable-but-unscheduled main-thread time in a 3.222 s active
window. Metal completion-to-running is 4.6 µs median; the larger matched GPU-end-to-
completion interval is 91.7 µs. This narrows the observed bottleneck toward the
submission/completion path, not main-thread CPU starvation. Next audit decoder
layout materializations and small operations for avoidable commands, preserving
synchronous semantics. Instrumentation overhead and incomplete Metal event joins
prevent interpreting these timings as recoverable production latency.

The [singleton-layout correction](../apps/hf-inference/benchmarks/reports/whisper-singleton-layout/README.md)
fixes compute-core contiguity checks to ignore strides on size-one axes. This
avoids 16 copies/commands and allocations per decoder step (209 → 193 commands),
without changing scheduling. Fresh full-request medians improve 741 → 697 ms
(short) and 3,073 → 2,867 ms (long). Strict Whisper parity, 91 regressions and native
compute tests pass. Continue auditing remaining materializations and pointwise
chains using measured evidence; async/batching stays deferred.

The [decoder command audit](../apps/hf-inference/benchmarks/reports/whisper-command-audit/README.md)
accounts for all 193 remaining commands and promotes existing normalization affine
fusion to the automatic Metal ONNX path (180 commands per step). Two reversed-order
rounds show approximately 5% faster full requests; strict Whisper/AST parity, 91
regressions and CPU ONNX checks pass. The old affine opt-in switch is removed.
Next consider the eight concatenations that each encode two input-copy commands
within one operation; cross-operation batching/async remains deferred.

The completed inference baseline is committed across Affon `becafce4`, compute
`0dfaf41`, and Hao `af9af07`. The next architecture step is the
[bounded Metal execution-scope proposal](core/metal-execution-scopes.md): begin with
internal graph scopes that retain resources, enforce host-access barriers and
complete before returning. General asynchronous eager tensors remain a later phase.
Next implement the ownership/barrier audit and eligibility inventory before a
private prototype; no scheduler change has been made by the proposal.
