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
