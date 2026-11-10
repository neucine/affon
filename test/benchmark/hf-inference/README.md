# Native inference benchmarks

This directory contains the Python harness, runnable Affon TypeScript programs,
and benchmark telemetry helper. Application inference code remains under
`apps/hf-inference/src/inference/`.

Run from the repository root with a ReleaseFast Affon build and verified cached
DistilGPT-2/ViT snapshots:

```sh
python3 test/benchmark/hf-inference/run.py \
  --affon /tmp/affon-hf-release/bin/affon \
  --cache /tmp/affon-hub-cache \
  --output /tmp/affon-bench-results \
  --build-profile ReleaseFast --iterations 5
```

To run the matched PyTorch peer gate on a stable Apple machine, use a Python
environment with PyTorch and Transformers plus the same offline cache:

```sh
python3 test/benchmark/hf-inference/run.py \
  --affon /tmp/affon-hf-release/bin/affon \
  --cache /tmp/affon-hub-cache --output /tmp/affon-model-gate \
  --build-profile ReleaseFast --iterations 5 --devices metal \
  --target-fingerprint apple-m4-local \
  --pytorch-python /path/to/peer-venv/bin/python \
  --maximum-peer-ratio 1.25
```

The command writes `normalized/affon.json`, `normalized/pytorch.json`, and
`normalized/peer-gate.json`; it exits nonzero for missing models, absent MPS,
failed output parity, changed workload contracts, or a peer ratio above the
declared target. This ratio is an example policy choice, not an accepted target
for these models. The peer uses the exact preprocessed Affon ViT tensor and
checks all logits. GPT-2 runs full-prefix greedy inference without a KV cache
on both engines and checks the generated text. Both models stay resident;
construction is outside the timed phase. This protocol differs from older GPT-2
reports, which included per-iteration Affon runtime initialization.

On the CUDA PC, run the same command from its Affon checkout with
`--devices cuda`, a target fingerprint for that PC, and a CUDA-enabled PyTorch
environment. The peer refuses CPU fallback. The Mac and CUDA reports must not
share a baseline or target fingerprint.

The Metal peer gate was first measured on this laptop on 2026-10-18; both
GPT-2 and ViT missed the initial 1.25 target. The result is exploratory because
each case has only three samples and no accepted self-baseline. For regression
confidence, keep an accepted normalized Affon report
for each machine and compare it to later runs with
`tools/compare-model-performance.py --kind self`.

The Python harness only starts native Affon processes and collects OS metadata.
All loading, processing and inference run in `test/benchmark/hf-inference/inference.ts`.
There are four fresh processes, run sequentially: GPT-2 CPU, GPT-2 Metal,
ViT CPU, ViT Metal. Each loads one model, performs one warmup, then measures
five iterations by default. Missing snapshots fail rather than download.

## Joined Apertus decode profile

`profile-apertus.py` captures one warmed, selected cached-decode step. It runs
Affon with invocation-scoped backend timing, bridge timing, NVTX markers, and a
bounded runtime evidence buffer. Kernel counts are recorded independently for
every invocation. When Nsight Systems is available, it also records CUDA and
NVTX activity, exports SQLite, and joins CUDA API calls, kernels, and copies to
the matching Affon run, step, invocation, and readback ranges:

```sh
python3 test/benchmark/hf-inference/profile-apertus.py \
  --affon /tmp/affon-apertus-release/bin/affon \
  --checkpoint /path/to/verified/apertus/snapshot \
  --output /tmp/apertus-profile
```

Pass `--nsys /path/to/nsys` when it is not on `PATH`, or use `--no-nsys` to
capture only Affon telemetry and runtime evidence. `--workload` selects
`decode` by default or `prefill`; prefill captures one warmed prefill execution
with the same identity and counting checks. `--prompt`, `--capacity`, and
`--budget` change the fixed-cache workload. Decode budget must cover prefill,
one decode warmup, and the selected token; prefill only needs the selected
prefill token. `--capture-records` bounds
Compute's per-Session evidence buffer (default 262,144; maximum 4,194,304).
`--telemetry-records` bounds the process-wide telemetry console ring (default
16,384; maximum 1,048,576) by setting `RUNTIME_TELEMETRY_TRACE_RECORDS`. The
console ring must cover every record produced between polls or the collector
reports `missed: true`. The output directory contains:

- `apertus-profile.json`: the selected step result, application metrics,
  Compute's runtime-evidence aggregate, and the decode `Executable.explain()`.
- `telemetry-traces.json`: the collected `std:telemetry` records and the
  collector's `missed` flag.
- `capture.json`: invocation details and completeness metadata.
- With Nsight Systems: `apertus-profile.nsys-rep`, `profile.sqlite`, and
  `compute-timeline.json`, including per-run and per-step activity, device gaps,
  readbacks, memcpy bytes, and unmatched activity counts.

The runner exits nonzero if telemetry missed records, runtime evidence dropped
events, or any required span is absent. The required host evidence is
`apertus.cached_decode.selected`, `affon.executable.run`, `affon.bridge.run`,
`affon.bridge.readback`, `compute.execution`, `inference.generation.inputs`,
`inference.generation.input_cleanup`, `inference.generation.selection`,
`inference.generation.cache_release`, and `inference.generation.cleanup`.
Inspect `dropped_event_count`, `missed`, `missing_spans`,
`unmatched_device_activities`, and the per-step timing and memory fields before
using the artifact to choose an optimization.

`capture.json` also reports the direct-child generation phases inside the
selected trace, their unioned coverage, and the remaining unattributed wall
time. Nested bridge and readback spans are not added to that total, so this is a
non-overlapping host time account rather than a sum of every recorded span.

With Nsight enabled, the runner also verifies that the selected telemetry and
external timeline name the same Compute run and executable, and that every
invocation is assigned to the same physical step. Per-step invocation counts
must match Compute's runtime-evidence aggregate. `capture.json` records this
identity, the correlated invocation counts, and any mismatch count; a mismatch
fails the capture.

Compute IDs are 64-bit. JavaScript cannot represent every 64-bit integer
exactly, so joins use the decimal `*_id_text` attributes and fields whenever
they are present. This includes run, executable, invocation, submission, first
invocation, last invocation, Program, and aggregate executable identities. The
numeric forms are display mirrors and are not authoritative above
`Number.MAX_SAFE_INTEGER`.

This is an attribution measurement. NVTX ranges and per-invocation evidence add
host and GPU overhead, so do not use its elapsed time as a throughput result.
Measure throughput separately with observation disabled, using the same
checkpoint, prompt, cache capacity, generated IDs, and build.

## Deterministic inference Program profile

`profile-inference-program.py` captures one warmed fixed-shape tiny GPT-2
inference Program with deterministic f32 weights. It exercises embedding,
normalization, GELU, masked attention, softmax, matmuls, logit readback, and
token selection without requiring a pretrained checkpoint:

```sh
python3 test/benchmark/hf-inference/profile-inference-program.py \
  --affon /tmp/affon-release/bin/affon \
  --device cuda \
  --output /tmp/affon-inference-profile
```

On CUDA the runner joins Affon telemetry, Compute schema-3 runtime evidence,
and Nsight device/API activity. On Metal or with `--no-nsys`, it validates
internal telemetry, executable and invocation identity, exact counts, dropped
events, and selected trace phase coverage. The output contains
`inference-profile.json`, `telemetry-traces.json`, `capture.json`, and, when
enabled, the Nsight report, SQLite export, and correlated timeline.

This is an accounting probe for a fixed transformer-like Program. It does not
measure pretrained-model quality or throughput.

To run one benchmark without Python:

```sh
AFFON_DEVICE=metal AFFON_BENCH_FAMILY=vit \
AFFON_HF_CACHE=/tmp/affon-hub-cache AFFON_AUDIT_BUILD=ReleaseFast \
AFFON_BENCH_REPORT=/tmp/vit-metal-benchmark.json \
/tmp/affon-hf-release/bin/affon test/benchmark/hf-inference/inference.ts
```

## Measurements

- Offline snapshot integrity verification, processor construction, model
  loading, and resident runtime setup are timed separately. File caches are not flushed; these are fresh
  processes, not guaranteed cold-disk measurements.
- DistilGPT-2 uses a five-token prompt and requests 16 new greedy tokens, without
  KV caching. Its initialized runtime is reused across measured runs.
  Tokenization is averaged over 1,000 calls for timer resolution.
  Generation includes native token selection/readback; decoding is separate.
  Tokens/second uses the actual number of generated tokens.
- ViT uses the deterministic 320×256 RGB8 audit pattern. Input construction is
  excluded. Resize/normalize/placement, forward, and top-five ranking are timed
  separately. The native ViT runtime is initialized once before warmup.
  Preprocessing includes import into that runtime and an extra readback;
  forward includes logit readback. Their sum is not an HTTP/UI latency metric.
- Output checks reject a changed known text prefix or ViT top-1 class.
- Timings use a millisecond clock. Zero values for short operations mean below
  useful timer resolution. Reports retain every sample, min/median/mean/max,
  and the separate first-run timing; five samples do not establish tail latency.
- Runtime memory is sampled after loading, warmup and each measured iteration.
  Tensor-storage peak metrics include loading and are not reset between phases.
  Stable samples alone do not prove leak freedom.
- On macOS `/usr/bin/time -l` supplies process-lifetime peak RSS in bytes,
  covering startup, cache validation, loading and inference. This includes CPU
  and GPU-related process allocations; it is not a dedicated GPU-memory metric.
  On other hosts OS peak RSS is left null. Choose available backends with
  `--devices`; the default four-process run requires Metal.
- `run.json` records host information and the binary SHA-256. Background machine
  activity and the running playground are not controlled. Do not extrapolate
  these local measurements to other hardware or large batches.

## Full-prefix generation

The GPT-2 benchmark executes explicit full-prefix Programs with one-token output
windows; it does not claim a KV cache. Use fresh output directories outside the
repository. Regenerate summaries with `summarize.py /path/to/results`. Compare
matching prompts, token budgets, devices, builds, and warmup protocols; preserve
environment metadata with each local run.

Operation and sequence correctness is covered by the canonical Program suites
under `test/e2e/program` and the model-family tests under `packages/@affon/models`.
Model timings supplement that coverage rather than replacing it.
