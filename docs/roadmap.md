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
- `@affon/models` owns model families and shared components. The former `lm`
  and `transformers` packages are consolidated. Native GPT-2/BERT/ViT execution
  is separated from HF loading into family modules. Add families within models as needed.
- Tokenizers and domain processors own input/output compatibility. Do not
  encode an assumption that every model consumes text IDs or generates tokens.
- `affon:compute` and `affon:nn` own primitives justified by application workloads and their audits.
- `affon:checkpoint` remains the low-level tensor/training persistence surface;
  Hub resolution and architecture dispatch live above it.
- Complete audits and runnable inference workflows live in `apps/`.

The audited GPT-2, BERT, and ViT implementations now share the experimental
`@affon/huggingface` package. Audit machinery stays app-owned. Unsupported
architectures/configurations must fail explicitly.

## Documentation ownership

Accepted cross-repository decisions and curated investigations are maintained in
`affon-arch`. Current API and implementation contracts remain with their code.
Generated measurements, progress checkpoints, and local handoff notes are not
maintained as public roadmap entries. Use the benchmark and audit tools to obtain
current evidence for the supported configurations.
