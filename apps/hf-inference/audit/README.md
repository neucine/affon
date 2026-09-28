# Hugging Face inference audit

Reference preparation and comparison tools for model-family implementations,
HF processors, and checkpoint adapters. Generated evidence belongs outside the
repository. Curated cross-repository findings are maintained in affon-arch.

## Prepare a reference

Use Python 3.11 or newer with a supported PyTorch wheel:

```sh
python3 -m venv /tmp/affon-hf-venv
/tmp/affon-hf-venv/bin/pip install -r apps/hf-inference/audit/requirements.txt
HF_HOME=/tmp/affon-hf-cache /tmp/affon-hf-venv/bin/python \
  apps/hf-inference/audit/prepare-reference.py \
  --model sshleifer/tiny-gpt2 \
  --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be \
  --output /tmp/affon-hf-tiny-gpt2
```

Preparation uses HF/PyTorch on CPU in evaluation mode. It exports f32
SafeTensors with HF weight names, tokenizer JSON, and reference activations,
logits, token IDs, and uncached greedy sequences. This checkpoint's original
weights are converted by the reference tool; arbitrary source-format loading
is not established by this experiment. Direct Hub loading is tested separately
below with the original DistilGPT-2 and ViT SafeTensors. The manifest
records the exact commit, dependency versions, and artifact SHA-256 hashes.
The tiny checkpoint tests interoperability, not meaningful language quality.

## Run native inference checks

From the repository root:

```sh
zig build install
AFFON_DEVICE=cpu AFFON_HF_MODEL_DIR=/tmp/affon-hf-tiny-gpt2 \
  AFFON_AUDIT_BUILD=Debug ./zig-out/bin/affon apps/hf-inference/audit/audit.ts
```

Set `AFFON_DEVICE=metal` or `cuda` on a suitable host to exercise those backends.
Reports default to `<model-dir>/audit-<device>.json`; `AFFON_HF_REPORT` overrides
the output path. Any failed check exits unsuccessfully after writing the report.
Loader/setup failures terminate before the report is produced.

Checks cover four prompts (including Unicode, whitespace, and a special token),
all hidden states exposed by the reference, all prompt logits, four greedy
decoding steps, decoded output, and input/context validation. Logits and hidden
states use elementwise `abs(error) <= 1e-4 + 1e-4 * abs(reference)`; token IDs
and decoded strings require exact agreement. Forward checks use reference IDs
so a tokenizer mismatch does not conceal a separate model result.

`load_gpt2(directory, device)` in `@affon/huggingface` also provides `forward(ids)` and
`generate(ids, max_new_tokens)` for local experiments. The latter returns the
prompt plus generated IDs and stops at EOS. The adapter uses `no_grad`, constant
weights, and deterministic evaluation semantics (no dropout).

## Scope and limits

The following limits describe the GPT-2 path; encoder/vision probes are below.

Run offline regression tests with:

```sh
./zig-out/bin/affon test apps/hf-inference
./zig-out/bin/affon test packages/@affon/tokenizers/test
./zig-out/bin/affon test test/e2e/checkpoint
```

## Wider and production-sized decoder probes

Use the same pinned tiny-model tokenizer with seeded random weights (width 64,
four heads, two layers, seed 1729) and an additional longer prompt:

```sh
HF_HOME=/tmp/affon-hf-cache /tmp/affon-hf-venv/bin/python \
  apps/hf-inference/audit/prepare-reference.py --seeded-wide \
  --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be \
  --output /tmp/affon-hf-wide-gpt2
```

Prepare production-sized DistilGPT-2 (about 328 MB decimal of f32 weights):

```sh
HF_HOME=/tmp/affon-hf-cache /tmp/affon-hf-venv/bin/python \
  apps/hf-inference/audit/prepare-reference.py --model distilbert/distilgpt2 \
  --revision 2290a62682d06624634c1f46a6ad5be0f47f38aa \
  --output /tmp/affon-hf-distilgpt2
AFFON_DEVICE=cpu AFFON_AUDIT_BUILD=Debug \
  AFFON_HF_MODEL_DIR=/tmp/affon-hf-distilgpt2 \
  ./zig-out/bin/affon apps/hf-inference/audit/audit.ts
```

Point `AFFON_HF_MODEL_DIR` at the wide fixture to run that probe. Change
`AFFON_DEVICE` to `metal` or `cuda` only on hosts supporting those backends.
The loader streams selected weights; the former 500 MiB file cap is removed.
Indexed shards are covered by synthetic offline tests. Production coverage is
limited to the explicitly tested models.

Reports now include before/after-load and per-case memory telemetry. This samples
process footprint and tensor allocation counters; it does not measure sustained
memory stability or isolate model memory from the reference/comparison overhead.

## Encoder and vision probes

Prepare the pinned BERT base-encoder and ViT classifier references:

```sh
HF_HOME=/tmp/affon-hf-cache /tmp/affon-hf-venv/bin/python \
  apps/hf-inference/audit/prepare-domain-reference.py --family bert \
  --model prajjwal1/bert-tiny \
  --revision 6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837 \
  --output /tmp/affon-hf-bert
HF_HOME=/tmp/affon-hf-cache /tmp/affon-hf-venv/bin/python \
  apps/hf-inference/audit/prepare-domain-reference.py --family vit \
  --model google/vit-base-patch16-224 \
  --revision 3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3 \
  --output /tmp/affon-hf-vit
AFFON_DEVICE=cpu AFFON_AUDIT_BUILD=Debug AFFON_HF_MODEL_DIR=/tmp/affon-hf-bert \
  ./zig-out/bin/affon apps/hf-inference/audit/audit-domain.ts
```

Use an isolated optimized build for the CPU vision audit:

```sh
zig build -Doptimize=ReleaseFast --prefix /tmp/affon-hf-release install
AFFON_DEVICE=cpu AFFON_AUDIT_BUILD=ReleaseFast AFFON_HF_MODEL_DIR=/tmp/affon-hf-vit \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/audit-domain.ts
```

`audit-domain.ts` separates processor and forward checks. Passing native
processor outputs feed forward runs; a failed processor check falls back to
reference inputs for diagnosis, recorded in `input_source`. BERT probes single and padded two-row batches,
encoder hidden states, the model pooler, and masked-mean/L2 pooling. ViT probes
two synthetic RGB images, pixel-value parity, hidden states, logits, and top-1
classification. The package accepts RGB8 arrays and implements Pillow-compatible
bilinear resizing, rescaling, and normalization. It does not decode image files
or support arbitrary image processors.

The BERT audit passes all seven grouped checks on CPU and Metal, using native
BERT normalization, WordPiece, serialized single/pair templates, and right
padding. ViT passes both processor checks and both cases' logits/top-1 checks,
but **still exits unsuccessfully** because some hidden-state comparisons fail.
A passing classifier output is not a passing model audit. In `checks`,
entry zero is the final output, followed by hidden states in reference order,
then BERT's mean-pooled and pooler outputs when present.

Both encoder adapters use an explicitly approximate erf-GELU composition from
existing primitives. Its independent PyTorch fixture can be tested offline:

```sh
./zig-out/bin/affon test apps/hf-inference/tests/encoder-ops.test.ts
```

The fixture records `torch.nn.functional.gelu(approximate="none")` on 257 evenly
spaced f32 values from -8 to 8. ViT patch convolution is expressed as patches
plus matmul; general convolution support is not established by this adapter.

`processors.test.ts` covers 21 independent tokenizer and resize fixtures. To
regenerate them with the pinned Python dependencies, run
`apps/hf-inference/audit/generate-processor-references.py`. These synthetic fixtures
require no downloaded model weights.

## Diagnose ViT numerical drift

Capture individual PyTorch operation inputs and outputs, then run each native
operation with the exact reference input:

```sh
/tmp/affon-hf-venv/bin/python apps/hf-inference/audit/prepare-vit-diagnostics.py \
  --directory /tmp/affon-hf-vit
AFFON_DEVICE=cpu AFFON_HF_MODEL_DIR=/tmp/affon-hf-vit \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/diagnose-vit.ts
```

All 35 captured operations pass on CPU and Metal at the unchanged tolerance.
To measure propagation of the initial embedding difference through PyTorch:

```sh
AFFON_DEVICE=cpu AFFON_HF_MODEL_DIR=/tmp/affon-hf-vit \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/capture-vit-native.ts
/tmp/affon-hf-venv/bin/python apps/hf-inference/audit/probe-vit-sensitivity.py \
  --directory /tmp/affon-hf-vit --device cpu
```

## Direct Hub inference without Python

`hub-smoke.ts` downloads the original pinned DistilGPT-2 or ViT artifacts through
`@affon/huggingface`, selects the processor, and runs native inference. It needs
Hao native HTTP streaming and system shasum, mkdir, mv, and rm for cache
maintenance; it does not use curl or the Python environment.

```sh
AFFON_DEVICE=cpu AFFON_HF_CACHE=/tmp/affon-hub-cache \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/hub-smoke.ts
AFFON_DEVICE=cpu AFFON_HF_FAMILY=vit AFFON_HF_CACHE=/tmp/affon-hub-cache \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/hub-smoke.ts
```

Repeat with `AFFON_HF_OFFLINE=1` to verify cache-only loading. Text generates four
tokens; vision classifies a deterministic synthetic RGB array. These are smoke
checks, not expanded accuracy claims.

For independent parity checks on downloaded artifacts, set `AFFON_HF_MODEL_DIR`
to the returned snapshot and `AFFON_HF_REFERENCE_DIR` to an existing prepared
reference directory when running `audit.ts` or `audit-domain.ts`. Python remains
an optional oracle-generation dependency, not a model loading dependency.

## Calibrate ViT patch-projection sensitivity

Capture all reference cases on each native backend (the case-zero aliases remain
available for the older sensitivity probe), then run the PyTorch interventions:

```sh
AFFON_DEVICE=cpu AFFON_HF_MODEL_DIR=/tmp/affon-hf-vit \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/capture-vit-native.ts
AFFON_DEVICE=metal AFFON_HF_MODEL_DIR=/tmp/affon-hf-vit \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/capture-vit-native.ts
/tmp/affon-hf-audit-venv/bin/python apps/hf-inference/audit/probe-vit-projection.py \
  --directory /tmp/affon-hf-vit \
  --output /tmp/affon-hf-vit/projection-calibration.json
```

## Real-image calibration across PyTorch execution paths

The real-image preparation tool reuses the pinned local ViT weights and downloads
three public reference images. It saves original image hashes and decoded RGB
alongside CPU/eager, CPU/SDPA and, when available, MPS/eager reference outputs.
Use the checked-in manifest to reject changed sample bytes on reproduction:

```sh
/tmp/affon-hf-audit-venv/bin/python apps/hf-inference/audit/prepare-vit-real-images.py \
  --model-directory /tmp/affon-hf-vit --output /tmp/affon-hf-vit-real \
  --expected-manifest /tmp/affon-hf-vit-real/reference.json
AFFON_DEVICE=cpu AFFON_HF_MODEL_DIR=/tmp/affon-hf-vit-real \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/capture-vit-native.ts
AFFON_DEVICE=metal AFFON_HF_MODEL_DIR=/tmp/affon-hf-vit-real \
  /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/capture-vit-native.ts
/tmp/affon-hf-audit-venv/bin/python apps/hf-inference/audit/compare-vit-real-images.py \
  --directory /tmp/affon-hf-vit-real \
  --output /tmp/affon-hf-vit-real/calibration.json
```

## SmolLM2

Download and smoke-test the original pinned BF16 checkpoint without Python:

```sh
AFFON_DEVICE=metal /tmp/affon-smollm-release/bin/affon apps/hf-inference/audit/hub-smollm2.ts
```

Requires a current optimized build with BF16 checkpoint widening. Repeat with
`AFFON_HF_OFFLINE=1` to verify cache-only operation. Prepare independent references
from the downloaded snapshot using the audit Python environment:

```sh
python apps/hf-inference/audit/prepare-smollm2.py \
  --directory /tmp/affon-hub-cache/models--HuggingFaceTB--SmolLM2-135M-Instruct/12fd25f77366fa6b3b4b768ec3050bf629380bac
AFFON_DEVICE=cpu AFFON_SMOLLM2_DIR=/path/to/snapshot \
  /tmp/affon-smollm-release/bin/affon apps/hf-inference/audit/check-smollm2.ts
```

Repeat the final command with `AFFON_DEVICE=metal`. The oracle uses PyTorch f32,
eager attention, evaluation mode, and uncached greedy generation. Four prompts
cover basic questions, rewriting, digits, Unicode, and whitespace. Checks require
exact chat formatting, input IDs, eight-step cached/uncached output IDs, and
decoded completions. All hidden states, full logits, and chunked cached logits
use `abs(error) <= 5e-4 + 1e-4 * abs(reference)`. This model-specific absolute
tolerance is looser than the GPT-2 audit's `1e-4`; strict parity at that older
threshold is not claimed. Reports and model artifacts stay outside the repository.

The offline tiny random Llama oracle uses a tighter `1e-5` absolute threshold,
with two layers, grouped-query attention and nonzero rotary positions. Regenerate
it with `packages/@affon/huggingface/test/generate-llama-reference.py`; run
`affon test packages/@affon/huggingface/test/llama.test.ts` on CPU and Metal.
These bounded checks do not establish general Llama-family or long-context support.

### Larger SmolLM2 models and comparable benchmarks

`hub-smollm2.ts` accepts `AFFON_SMOLLM2_SIZE=135M|360M|1.7B` and uses the
pinned catalog in `src/inference/models.ts`. Pass the matching `--size` to
`prepare-smollm2.py`; `check-smollm2.ts` reads the selected snapshot/reference
from `AFFON_SMOLLM2_DIR`. References require a complete verified snapshot.

Run one model per process, with a warmup and two measured 16-token requests:

```sh
AFFON_DEVICE=metal AFFON_SMOLLM2_SIZE=360M \
  AFFON_HF_CACHE=/tmp/affon-hub-cache AFFON_BENCH_REPORT=/tmp/smollm360-metal.json \
  /usr/bin/time -l /tmp/affon-smollm-release/bin/affon \
  apps/hf-inference/src/benchmark/smollm2.ts
```

Repeat with `1.7B`. On Linux use `AFFON_DEVICE=cuda`, a CUDA-enabled Affon build,
and `/usr/bin/time -v` instead of `-l`. The command uses verified cached snapshots
by default; set `AFFON_HF_OFFLINE=0` to allow native downloads. Reports separate
prefill from subsequent decoding and include sampled telemetry. OS peak RSS
includes runtime and temporary allocations; neither metric is just weight size.
Loading time includes cache hashing and tokenizer preparation. The short fixed
workload does not establish long-context speed, memory stability, or model quality.
BF16 files are widened to f32; quantized execution is not implemented here.

Measured on an M4 MacBook Pro with 16 GB unified memory, ReleaseFast, Metal,
2026-09-28 (40 prompt tokens, 16 generated tokens; average of two post-warmup runs):

| Model | Prefill | Subsequent decode | OS peak physical footprint |
| --- | ---: | ---: | ---: |
| SmolLM2-360M-Instruct | 569 ms | 2.39 tokens/s | 1.71 GiB |
| SmolLM2-1.7B-Instruct | 484 ms | 2.54 tokens/s | 6.74 GiB |

These are observations for this workload, not a general model-size speed ranking.
The 360M CPU and Metal oracles pass. For 1.7B, all four prompts' tokenization,
eight-token cached/uncached greedy outputs, full logits, and cached chunk logits
pass. Six late-layer hidden-state comparisons contain 22 out-of-tolerance values;
the audit remains **failing**, with no tolerance relaxation. Its report records
all numerical failures before exiting unsuccessfully. CUDA has not been run on
this Mac and requires a separate Linux GPU validation.

The 1.7B text-only playground was also verified end to end: “What is the capital
of France?” returned “The capital of France is Paris.” in 5.88 seconds for eight
generated tokens on a warm request. A request during the concurrent CPU audit
took 55.98 seconds, and the first request after it exited took 16.93 seconds.
Desktop latency is therefore sensitive to other work and differs from isolated
benchmark throughput; the timings do not establish a guaranteed response time.
