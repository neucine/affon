# Native inference benchmarks

Run from the repository root with a ReleaseFast Affon build and verified cached
DistilGPT-2/ViT snapshots:

```sh
python3 apps/hf-inference/benchmarks/run.py \
  --affon /tmp/affon-hf-release/bin/affon \
  --cache /tmp/affon-hub-cache \
  --output /tmp/affon-bench-results \
  --build-profile ReleaseFast --iterations 5
```

The Python harness only starts native Affon processes and collects OS metadata.
All loading, processing and inference run in `src/benchmark/inference.ts`.
There are four fresh processes, run sequentially: GPT-2 CPU, GPT-2 Metal,
ViT CPU, ViT Metal. Each loads one model, performs one warmup, then measures
five iterations by default. Missing snapshots fail rather than download.

To run one benchmark without Python:

```sh
AFFON_DEVICE=metal AFFON_BENCH_FAMILY=vit \
AFFON_HF_CACHE=/tmp/affon-hub-cache AFFON_AUDIT_BUILD=ReleaseFast \
AFFON_BENCH_REPORT=/tmp/vit-metal-benchmark.json \
/tmp/affon-hf-release/bin/affon apps/hf-inference/src/benchmark/inference.ts
```

## Measurements

- Offline snapshot integrity verification, processor construction, and model
  loading are timed separately. File caches are not flushed; these are fresh
  processes, not guaranteed cold-disk measurements.
- DistilGPT-2 uses a five-token prompt and requests 16 new greedy tokens, without
  KV caching. Tokenization is averaged over 1,000 calls for timer resolution.
  Generation includes native token selection/readback; decoding is separate.
  Tokens/second uses the actual number of generated tokens.
- ViT uses the deterministic 320×256 RGB8 audit pattern. Input construction is
  excluded. Resize/normalize/placement, forward, and top-five ranking are timed
  separately. Preprocessing includes an extra readback to complete GPU work;
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
  On other hosts OS peak RSS is left null. Metal support is required for the
  four-process harness; use the single-process command for CPU-only hosts.
- `run.json` records host information and the binary SHA-256. Background machine
  activity and the running playground are not controlled. Do not extrapolate
  these local measurements to other hardware or large batches.

## Cached generation

Pass `--families gpt2 --kv-cache` to the runner to measure cached generation.
Use fresh output directories outside the repository. Regenerate summaries with
`summarize.py /path/to/results`. Compare matching prompts, token budgets, devices,
builds, and warmup protocols; preserve environment metadata with each local run.

For operation and sequence coverage use [test/benchmarks](../../../test/benchmarks/README.md).
Model timings supplement that coverage rather than replacing it.
