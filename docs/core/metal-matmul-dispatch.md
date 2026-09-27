# Metal matmul dispatch

Ordinary f32 matmul now has one implementation-selection path in Compute. The
contiguous, offset, offset-batched, strided and strided-batched FFI entry points
normalize to `affon_metal_matmul_strided_many_f32`. They preserve their signatures
but no longer own separate MPS/custom policies or encoding implementations.

The common dispatcher supplies one ordered batch of matrices, byte offsets,
input row/column strides and M/N/K dimensions to the existing layout-aware policy:

1. The size policy prefers MPS for multi-row products with M×N×K >= 1,048,576,
   or single-row products with K >= 256 and N×K >= 65,536.
2. MPS eligibility checks require a representable row-major or transposed layout,
   aligned offsets and sufficient buffer capacity for its descriptors.
3. Eligible preferred products use MPS. Other products use the custom strided
   matrix-vector pipeline when its existing shape/SIMD requirements hold, or the
   generic strided matrix pipeline otherwise.

Compatibility and size policy retain distinct checks and diagnostic reasons.
No kernel-selection policy lives in HF, ONNX or benchmark code.

The old contiguous MPS-first implementation, duplicated custom contiguous/tiled
encoding helpers and unused shader pipelines are removed. `AFFON_METAL_MATMUL`
no longer forces a separate path. MPS failures propagate from the common path;
there is no legacy retry through a custom kernel after an MPS execution error.
This is an intentional behavior change for the old contiguous entry point.
Previously custom-only offset entries can now choose MPS under the same policy.

This consolidation covers ordinary f32 matmul. Integer matmul and fused
matmul+bias/GELU have distinct implementations and semantics; they have not been
rewritten as unfused calls. The benchmark suite includes ordinary f32 calls and compiled fused epilogue chains.
Ordinary f32 MPS and custom matmul now use the shared kernel encoding context.
They join an active graph scope, while ordinary eager calls still complete before
returning. F32 matmul+bias and matmul+bias+GELU also encode through that context.
Integer matmul and integer matmul+bias retain synchronous completion.

## Validation

A native integration test directly calls all five normalized entry points for
small generic, custom matrix-vector and MPS-sized products. Poisoned input
prefixes catch incorrect offsets, output sentinels catch unintended writes, and
all results are checked with independent CPU dot products. Multi-matrix calls
exercise shared input offsets and distinct output regions.

The benchmark harness is unchanged from the dispatch-visibility baseline, allowing
normal baseline compatibility checks. Its native dispatch events test the actual
observed routes, rather than duplicating the selector in the harness. Strict model
reference checks cover the graph/fusion paths as well as the isolated operator
suite. See the [consolidation report](../../test/benchmarks/reports/metal-unified/NOTES.md).


## Batched MPS ownership

All ordinary f32 wrappers declare their input/output Storage through typed context
bindings. Host offset arrays are consumed while encoding; shaders copy scalar
metadata. The MPS adapter retains its operation, descriptors and matrix wrappers
in the command scope. Completed chunks release those objects after waiting;
poison/abort releases them with the discarded, unsubmitted command. Ordinary
calls finish before local objects retire. Storage leases protect the input/output
buffers independently of Objective-C wrapper lifetime.

The adapter allocates no pooled tensor scratch or host-readable status buffer.
MPS-internal driver resources are not exposed as Storage, so Storage metrics and
the scope byte target do not bound all driver allocation. The existing 64 MiB
lease target permits a single oversized operation, which completes alone.
The new native test exercises both MPS and custom chains through chunk splits,
early Tensor destruction, abort, injected completion failure and recovery.

See the [batching report](../../test/benchmarks/reports/metal-matmul-batching/NOTES.md)
for isolated chains, full model correctness, request timings and measured memory.
