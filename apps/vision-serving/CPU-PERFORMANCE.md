# MobileNetV2 CPU performance audit

This audit fixes the existing `google/mobilenet_v2_1.0_224` CPU workload. It does
not cover other models, GPU execution, throughput under concurrency, or a cluster.
The model revision is `75e607b00aeae1297cc89d026a118bce012f5c5a`; the
ONNX model SHA-256 is
`8319a6d3db8f282c9108ad25585c9d40e9727242bc8bb50c0db48e2c39b4448d`.
Measurements used macOS arm64, ReleaseFast Affon, ONNX Runtime 1.30.0, and a
320×260 RGB8 synthetic image. The numerical parity check also covers 224×224
and 260×320 inputs. All services run one sequential request at a time.

## Acceptance result

The CPU target was median Affon time at most 2× ONNX Runtime for both isolated
graph execution and complete HTTP requests, at one and four configured threads.
The final measured results were:

| Measure | Threads | Affon | ONNX Runtime | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Graph, median of three fresh runs | 1 | 6.7 ms | 4.95 ms | 1.35× |
| Graph, median of three fresh runs | 4 | 4.45 ms | 2.25 ms | 1.98× |
| HTTP, median of 50 requests | 1 | 27.1 ms | 22.1 ms | 1.23× |
| HTTP, median of 50 requests | 4 | 24.1 ms | 19.5 ms | 1.23× |

Every graph run passed the 2× threshold; the worst individual graph ratio was
1.98×. The initial four-thread HTTP median was 686.5 ms for Affon versus 20.0 ms
for ONNX Runtime, or about 34×. The final four-thread HTTP median above is about
28× faster for Affon than that initial result. The two HTTP runs used the same
model and input contract, but they were separate runs on an uncontrolled machine.
Median client latency includes request upload, server processing, response
transfer, and JSON decoding; input serialization is excluded.

The final service checks passed all 22 boundary cases on both engines. Three
input shapes matched ONNX Runtime logits with maximum absolute error below
`7.3e-6` and matching top-1 classes. This is numerical execution parity on
synthetic images, not a real-image accuracy study.

The 500-request follow-up sampled Affon RSS every 50 requests. It fluctuated
between about 200 and 292 MiB after the initial measured set and ended at
257 MiB, versus 253 MiB at the beginning of the probe. There was no monotonic
growth in this run. This sampled RSS result does not establish long-term memory
stability; the working set is also appreciably larger than ONNX Runtime's.

## What the profiles found

- The original graph spent most time in Conv and explicit Pad. CPU matmul now
  uses Accelerate SGEMM for compatible f32 layouts on macOS. CPU broadcast
  kernels iterate contiguous rows rather than recomputing coordinates for every
  element.
- The MobileNetV2 CPU graph now uses direct pointwise convolution, a packed
  general convolution path, SIMD depthwise convolution, and safe fusions of
  Conv→Clip and Pad→depthwise Conv. Fusions only occur when the intermediate
  value has a single consumer and is not exposed as a graph output.
- Native RGB resize, crop, and normalization removed the earlier per-pixel
  JavaScript work. The service then removed another bottleneck by sending the
  flat RGB request array directly to that native processor. Before that last
  change, isolated graph execution passed, but HTTP median remained 58 ms
  versus 21 ms, with 31 ms inside the Affon preprocessing path.
- ReleaseFast uses a production allocator for the Affon process and telemetry
  span handles. Debug and ReleaseSafe retain their diagnostic allocator.

The native RGB implementation reproduces the baseline's Pillow bilinear
quantization for the supported processor configuration. New tests cover its
nested and flat input forms, custom normalization, invalid pixels, spatial
border cases, graph output visibility during fusion, dtype/layout variants, and
SIMD tails. The CPU specialization is private to the current implementation;
other types and layouts use the existing paths.

## Reproduce

Prepare the model as described in [README.md](README.md#prepare-the-model), then
build Affon in ReleaseFast mode. Run from the repository root:

```sh
zig build -Doptimize=ReleaseFast -p /tmp/affon-cpu-profile-build
python apps/vision-serving/run-cpu-audit.py \
  --model-dir /path/to/model --affon /tmp/affon-cpu-profile-build/bin/affon \
  --output apps/vision-serving/reports/profile/final
python apps/vision-serving/benchmark.py \
  --model-dir /path/to/model --affon /tmp/affon-cpu-profile-build/bin/affon \
  --output apps/vision-serving/reports/profile/serving-final-t4 \
  --threads 4 --iterations 50 --warmups 5 --memory-iterations 500
python apps/vision-serving/benchmark.py \
  --model-dir /path/to/model --affon /tmp/affon-cpu-profile-build/bin/affon \
  --output apps/vision-serving/reports/profile/serving-final-t1 \
  --threads 1 --iterations 50 --warmups 5
```

`run-cpu-audit.py` launches fresh processes for three repeats per thread setting
and checks the graph ratio and logit parity. `benchmark.py` checks HTTP behavior,
parity, timing, and sampled RSS. `AFFON_CPU_THREADS` controls the native CPU
depthwise work partition on macOS; the benchmark runner sets it together with
the ONNX Runtime and math-library thread limits. These are requested limits,
not proof that both engines use identical CPU resources. The service's own
phase timers have millisecond resolution, so use client time for the HTTP gate.

The Accelerate SGEMM and dispatch-based depthwise parallelism are macOS paths.
The updated Linux arm64 image built and passed its 22 HTTP boundary checks and
logit parity smoke test. Linux builds retain the portable fallback, and this
audit does not claim Linux performance parity. Recheck performance separately
for each production machine, model, input shape, concurrency pattern, and
runtime build.
