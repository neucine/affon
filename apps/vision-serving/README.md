# Vision serving experiment

This app tests Affon as an inference service inside existing serving infrastructure.
It implements a bounded JSON subset of the [Open Inference Protocol V2](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol)
and includes a [KServe ServingRuntime](https://kserve.github.io/website/docs/concepts/resources/servingruntime)
example. Model packaging is unchanged: ordinary exported ONNX and the existing
Affon converter supply the artifacts.

The workload is `google/mobilenet_v2_1.0_224` at revision
`75e607b00aeae1297cc89d026a118bce012f5c5a`. Native TypeScript preprocessing performs
bilinear shortest-edge resize, center crop, and normalization before the prepared
graph executes on Affon. The baseline uses Pillow/NumPy and ONNX Runtime.

## Result

After CPU profiling and optimization, the existing MobileNetV2 workload is within
2× ONNX Runtime in both graph execution and sequential HTTP latency on the tested
macOS arm64 machine. The 50-request median HTTP measurements were 27.1 vs 22.1 ms
at one thread and 24.1 vs 19.5 ms at four threads (Affon vs ONNX Runtime).
The largest graph ratio in three fresh runs at each thread setting was 1.98×.
All three image cases retained logit parity. See [CPU performance audit](CPU-PERFORMANCE.md)
for methods, optimizations, memory observations, and limits.

### Initial run, before CPU optimization

The first macOS arm64 CPU run passes end-to-end logit parity on square, landscape,
and portrait synthetic images, with maximum absolute error `9.06e-6` and matching
top-1 classes. Each server passed 22 HTTP/boundary checks. These are deterministic
synthetic-image execution checks, not a validation of real-image accuracy.

| Measurement | Affon | ONNX Runtime baseline |
| --- | ---: | ---: |
| Process start to ready, including load and probe | 540 ms | 262 ms |
| Median client latency | 686.5 ms | 20.0 ms |
| Observed p95, 20 samples | 816.3 ms | 21.0 ms |
| Serial throughput | 1.40 requests/s | 49.55 requests/s |
| Median preprocessing | 347.5 ms | 3.74 ms |
| Median inference + logit readback | 318.0 ms | 2.64 ms |
| RSS after warmup | 97.7 MiB | 112.3 MiB |
| RSS after measured requests | 113.1 MiB | 112.3 MiB |

Affon is approximately 34 times slower at median latency in this experiment.
Both preprocessing and graph execution contribute. Lower initial RSS did not
produce a sustained memory advantage: Affon's RSS grew during this short run.
This observation alone does not diagnose a leak or establish memory stability.
The evidence supports API integration and numerical compatibility; it does not
support a performance advantage or justify a new orchestration layer.

Measurements used ReleaseFast Affon, ONNX Runtime 1.30.0, CPU only, four requested
CPU threads, three correctness cases, and two additional warmups before 20
sequential measured requests. Both services process one request at a time. The
measured input is 320×260 RGB8, serialized once as 891,312 bytes of JSON. Client
latency includes a fresh loopback HTTP connection, upload, server processing,
response transfer, and JSON decoding. Input serialization is excluded. Server
phase timers exclude JSON parsing/serialization and use millisecond resolution
in Affon. Backend thread usage is not guaranteed identical.

Startup is one process-start sample per engine with warm filesystem caches.
RSS is sampled process memory, not peak memory. Engines run sequentially on an
uncontrolled developer machine. These samples do not establish production tail
latency, concurrency scaling, GPU performance, or long-term memory behavior.
Affon's separate offline conversion took 270 ms in this environment; it is
excluded from service startup. Prepared graph and weights occupy about 14 MB.

## Prepare the model

Use the existing [MobileNet exporter](../hf-inference/onnx/README.md#second-architecture-mobilenetv2)
to produce `model.onnx` and `source/preprocessor_config.json`. Preparation needs
the exporter dependencies described there. The comparison environment needs:

```sh
python -m pip install -r apps/vision-serving/requirements.txt
python packages/@affon/onnx/tools/convert.py /path/to/model/model.onnx /path/to/model
```

Keep the source processor configuration beside the graph:

```text
model/
  model.onnx                    # Used by the baseline
  graph.json                    # Used by Affon
  weights.safetensors           # Used by Affon
  source/preprocessor_config.json
```

The locally validated artifacts are at `/tmp/affon-onnx-mobilenet`. The native
binary is `/tmp/affon-model-package-build/bin/affon`. These paths are local
conveniences and are not required by the implementation.

## Run locally

From the repository root, with an up-to-date Affon:

```sh
MODEL_DIR=/path/to/model AFFON_DEVICE=cpu PORT=8080 \
  affon apps/vision-serving/src/server.ts

# In a separate terminal, or after stopping Affon:
MODEL_DIR=/path/to/model PORT=8081 \
  python apps/vision-serving/baseline.py
```

Native serving requires no Python process. The baseline requires the Python
dependencies above. Both bind to loopback by default; `HOST=0.0.0.0` enables
container networking. Model loading and one probe inference complete before
the listener opens. Unsupported processor types and graph input/output contracts
fail startup. Changing to another architecture therefore requires explicit
processor/contract work, even if its ONNX operators are supported.

The HTTP surface is:

- `GET /v2`: server metadata.
- `GET /v2/health/live`, `/v2/health/ready`: health.
- `GET /v2/models/vision`, `/v2/models/vision/ready`: model metadata and readiness.
- `POST /v2/models/vision/infer`: batch-one RGB8 inference.

`MODEL_NAME` changes `vision` in the routes and responses. A request is:

```json
{
  "id": "sample",
  "inputs": [{
    "name": "rgb",
    "datatype": "UINT8",
    "shape": [1, 1, 1, 3],
    "data": [255, 0, 0]
  }]
}
```

Inputs are flat row-major RGB bytes, with height/width 1–512 and aspect ratio at
most 4. Requests over 4 MiB are rejected. Responses echo `id` and return a flat
FP32 `logits` tensor with its shape, plus diagnostic phase timings in `parameters`.
Optional output selection supports only `[{"name":"logits"}]`. Custom parameters,
binary tensor transport, gRPC, model versioning, and dynamic batching are outside
this subset. Image file decoding and label rendering belong to the caller.

Preprocessing and inference run in the same process. Affon's synchronous execution
blocks its event loop, including health checks; this service has no worker pool,
admission queue, or independent preprocessing scaling. An oversized upload that
continues after an early rejection can observe a connection reset. The protocol
checks test header-based early 413 rejection explicitly.

## Reproduce the comparison

```sh
python apps/vision-serving/benchmark.py \
  --model-dir /path/to/model --affon /path/to/current/affon \
  --output apps/vision-serving/reports/cpu --iterations 20
```

The runner freshly converts ONNX into a temporary Affon directory, starts each
server on a free loopback port, checks rejection behavior and subsequent health,
compares three inputs, and records timings and sampled RSS. Both processes are
stopped in cleanup. It requires `ps` and permission to listen on loopback.
The output `report.json` records versions, source/model/binary hashes, samples,
correctness results, and memory readings. Raw logs and generated models are
excluded from Git.

## Container and KServe integration

The Linux arm64 image `affon-vision:experiment` was built and locally tested.
It passed all 22 HTTP/boundary checks and matched the landscape-image baseline
within `9.06e-6` maximum absolute logit error while running as non-root with a
read-only filesystem and model mount. The test container was removed afterward.
The image remains available locally; amd64 has not been tested. Results and the
image ID are recorded in `reports/cpu/container-smoke.json`.

The image builds the current Affon checkout and its sibling `hao`, `compute`,
and `zig-libs` sources on Linux. The context generator copies only explicit source
directories; it excludes model weights, build caches, and Git directories.
Affon's Linux build explicitly links the Rust transpiler archive because ELF
linking does not resolve that archive when nested inside `hao_runtime`.

```sh
python3 apps/vision-serving/deploy/prepare-context.py apps/vision-serving/build/context
docker build -t affon-vision:experiment apps/vision-serving/build/context
python apps/vision-serving/deploy/smoke-container.py \
  --image affon-vision:experiment --model-dir /path/to/model \
  --output apps/vision-serving/reports/cpu
```

Use a new context path or refresh your generated context after changing source.
The image uses Rust 1.94, checksum-pinned Zig 0.16.0, and locked Cargo/Zig library
dependencies. Base images and OS packages are not pinned by digest. Builds require
network access. The runtime image runs as UID/GID 65532 and expects read access
to mounted model files, including Safetensors files that converters may create
with owner-only permissions. The smoke runner makes a separate readable copy.

`deploy/kserve.yaml` defines the custom runtime and an `InferenceService`. It
expects an existing KServe installation, an image reference you publish, and a
PVC named `vision-models` with a `mobilenet/` directory matching the layout above.
The model is mounted separately from the runtime image. Its CPU and memory
requests are initial example settings, not measurements from a Kubernetes node.
Standard deployment mode and one minimum replica are specified; no autoscaling
or multi-node performance is established by the local experiment.

No registry push or Kubernetes deployment is performed by the scripts. Cluster
admission, service routing, rollout behavior, and autoscaling need separate
validation on the target KServe version and cluster.
