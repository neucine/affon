# Native inference benchmarks

Run from the repository root with a ReleaseFast Affon build and verified cached
DistilGPT-2/ViT snapshots:

```sh
python3 apps/hf-inference/benchmarks/run.py \
  --affon /tmp/affon-hf-release/bin/affon \
  --cache /tmp/affon-hub-cache \
  --output apps/hf-inference/benchmarks/reports/local-release \
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

## Saved local baseline

[Apple M4 / 16 GiB / ReleaseFast results](reports/local-release/summary.md):

| Workload | CPU median | Metal median | CPU / Metal ratio |
| --- | ---: | ---: | ---: |
| DistilGPT-2, 16 generated tokens | 10.607 s | 1.698 s | 6.25× |
| ViT forward, excluding preprocessing | 12.788 s | 0.584 s | 21.90× |

ViT preprocessing with readback takes 234 ms on CPU and 231 ms on Metal.
OS peak RSS ranges from 689 to 777 MiB across the four processes, including
loading. Live tensor storage stays at 312.5 MiB for GPT-2 and 332.5 MiB for
ViT at each warmup/measured sample. This short run does not establish long-term
memory stability. This saved baseline explicitly disables KV caching. Cached generation can be
measured separately with the command below.

The runner generates a summary automatically. Regenerate it from saved data:

```sh
python3 apps/hf-inference/benchmarks/summarize.py \
  apps/hf-inference/benchmarks/reports/local-release
```

## KV-cache comparison

The benchmark deliberately defaults to uncached generation to preserve the
original baseline. `--kv-cache` enables caching; `--families gpt2` skips ViT.

```sh
python3 apps/hf-inference/benchmarks/run.py \
  --affon /tmp/affon-hf-release/bin/affon --cache /tmp/affon-hub-cache \
  --output apps/hf-inference/benchmarks/reports/kv-cache \
  --build-profile ReleaseFast --families gpt2 --kv-cache
```

For single-process runs, set `AFFON_BENCH_KV_CACHE=1`.
[Cached measurements](reports/kv-cache/summary.md) use the same prompt,
16-token budget, warmup and five-run protocol as the uncached baseline.

| Backend | Uncached median | Cached median | Speedup |
| --- | ---: | ---: | ---: |
| CPU | 10.607 s | 1.152 s | 9.21× |
| Metal | 1.698 s | 1.184 s | 1.43× |

Both backends produce the same complete benchmark text as the saved baseline.
Cached CPU and Metal performance is similar for this small workload; these
measurements do not predict longer prompts or larger models. Metal process peak
RSS increased from 777.1 to 881.9 MiB; CPU peak RSS was 618.1 MiB. These are
whole-process peaks, not cache payload measurements. Live tensor storage returned
to 312.5 MiB after every measured generation on both backends.

The adapter stores sequence-major K/V and appends by copying the growing cache.
Generation owns and releases a separate cache per request; it does not share
conversation state. In-place cache updates remain a potential optimization.

The expanded reference audit passes 23/23 checks on each backend, including
cached chunk logits and hidden states against the saved PyTorch reference and
exact greedy token sequences:
[CPU](../audit/reports/distilgpt2/kv-cache-cpu.json),
[Metal](../audit/reports/distilgpt2/kv-cache-metal.json).
Three focused regression tests also pass on each backend, covering chunking,
session isolation/reset, validation, context limits and EOS stopping.

## Longer prompts and decoding

Run a separate fixed-length cache scaling benchmark (16/128/512 prompt tokens,
16/64 new tokens) with prefill and decode measured separately:

```sh
python3 apps/hf-inference/benchmarks/cache-scaling.py \
  --affon /tmp/affon-hf-release/bin/affon --cache /tmp/affon-hub-native \
  --output apps/hf-inference/benchmarks/reports/cache-scaling
```

[Saved scaling measurements](reports/cache-scaling/summary.md) use one warmup and
three measured runs per scenario. The harness checks deterministic generation,
CPU/Metal token agreement, exact retained cache size, and release on reset.
It deliberately ignores EOS to keep generation lengths fixed. Prompt IDs repeat
an encoded prose seed to reach exact lengths. This measures the session API and
includes readback/greedy selection, excluding tokenization and between-phase
memory sampling; it is not an end-to-end HTTP latency measurement.

On the saved M4 ReleaseFast run, 512 prompt tokens plus 16 new tokens took
30.079 s on CPU and 3.498 s on Metal. Most of the CPU time was prefill
(29.001 s versus 1.805 s on Metal). Conversely, generating the remaining 63
tokens in the 128/64 scenario took 4.272 s on CPU versus 6.475 s on Metal.
This identifies Metal single-token decoding as a profiling target; these timings
alone do not distinguish kernel launch, cache copying, readback, or math costs.

Retained KV storage matched 36 KiB per processed token in every run, reaching
18.53 MiB for the 512/16 scenario, and returned to the model-only baseline after
reset. Full generated token sequences agreed across both backends for all four
scenarios. This is cross-backend agreement, not an additional PyTorch oracle
comparison for these longer inputs.

Regenerate the summary without running inference:

```sh
python3 apps/hf-inference/benchmarks/cache-scaling.py --summarize-only \
  --output apps/hf-inference/benchmarks/reports/cache-scaling
```

## Metal decode profiling and native normalization

[Optimization results](reports/metal-decode/summary.md) compare the cached baseline
with GPT-2 using Affon's native `layer_norm` tensor operation. The 128-token
prompt / 64-token generation case improved from 6.762 s to 4.389 s on Metal;
retained cache size and generated tokens were unchanged.

To reproduce operation attribution with a prepared local model directory:

```sh
python3 apps/hf-inference/benchmarks/profile-decode.py \
  --affon /tmp/affon-hf-decode/bin/affon \
  --model-dir /tmp/affon-hf-distilgpt2 \
  --output /tmp/metal-decode-profile.json
```

The profiler creates temporary instrumented TypeScript copies and removes them
on exit. It measures a 128-token prefill followed by 32 forced single-token
steps, with full logit readback. API wall times include synchronous Metal work
and host overhead; they are not GPU timestamps. `no_grad` is an inclusive scope.
Use the regular short and scaling benchmarks for uninstrumented comparisons.

## Model-independent compute coverage

Operator and batched-sequence comparisons now have a separate shared-fixture suite
in [test/benchmarks](../../../test/benchmarks/README.md). It reports live
Compute OpTag coverage, uncovered operations, Affon/PyTorch ratios and optional
Affon-versus-baseline regression checks. Model measurements here remain the
end-to-end complement to that suite.
