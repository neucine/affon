# Shared kernel execution contract

Metal kernel invocation now goes through `pipeline/backend/metal/kernel.zig`.
The unary, binary, broadcast, layer normalization, RMS normalization, residual-add
normalization, suffix affine, ordinary f32 matmul, fused f32 matmul epilogues,
cat/stack, whole-tensor and axis-reduction wrappers use this contract instead of
separately calling lease admission, poisoning and completion functions. Kernel dispatch still
selects the implementation by operation/dtype; batching policy contains no list of
operation names or kernel symbols.

## Kernel-facing API

`Context.encode(function, bindings)` accepts a synchronous encoding implementation.
Metal buffer arguments must use `kernel.read(storage)`, `kernel.write(storage)`, or
a binding returned by `context.scratch(bytes)`. The invocation derives its entire
Storage lease set from those arguments and converts bindings to FFI handles.
Passing an unbound raw pointer to an opaque Metal buffer parameter is a compile-time
error. Scalar and copied metadata arguments pass through unchanged. Writable
bindings reject immutable Storage.

```zig
var context = kernel.Context{ .allocator = allocator };
defer context.deinit();
const temporary = try context.scratch(bytes);
try context.encode(first_kernel, .{ kernel.read(input), temporary, len });
try context.encode(second_kernel, .{
    kernel.read(temporary.storage), kernel.write(output), len,
});
```

Scratch is allocated with the existing Metal pool and workspace metadata. The
context owns its preparation lifetime. Encoding registers it with the enclosing
graph's lease registry, which owns its pending-GPU lifetime and includes its actual
buffer capacity in the chunk budget. Destroying the context after encoding cannot
return a pending scratch buffer to the ordinary pool.

The native slot-planned runner owns physical slots separately from the eager lease
registry. Its existing scratch-free execution retains slot reuse. If a context
uses scratch under that owner, it conservatively completes each invocation before
scratch can retire. This avoids treating slot generations as eager leases, which
would prevent the planner from reassigning slots.

`Context.synchronous(function, bindings)` drains pending work and invokes a legacy,
host-dependent or status-checking implementation. `Context.synchronize()` provides
a fallible boundary between kernel invocations and retires completed leases. Host
read/write barriers remain in the existing common buffer APIs. CPU and CUDA paths
are unchanged; this is the Metal implementation of the contract.

## Driver boundary

All Metal helpers use the same command-buffer creator. The old special
`new_scoped_command` entry is removed. The creator batches only while a shared
invocation has encoding permission and a graph scope is active. Outside that
invocation, even a helper called from inside a graph is a synchronous barrier.
Ordinary eager calls continue to complete before returning.

Permission is installed only for the duration of the FFI call and is cleared on
both success and returned failure. Nested invocation is rejected. Resource
admission, physical-capacity limits, standalone oversized work, error poisoning
and completion live in the shared context and lease owner, not in kernel wrappers.

Encoded implementations use the common command creator and encoded-success finish
helper. They must use bound resources and context-managed scratch, with no hidden
pool allocation or direct submission. Common legacy direct-wait and status-finish
paths reject an encoded-contract violation before submitting the graph-owned
buffer or reading its pending status. The failed invocation poisons/discards pending
work and cannot publish successful output.

MPS matrix multiplication now explicitly retains adapter objects in the command
scope and binds all tensor buffers through the context. Other opaque MPS resources
and legacy private workspaces have not been silently admitted.
Existing unmigrated wrappers remain synchronous. When migrating another kernel,
its actual buffer arguments and temporary allocations must follow this contract;
there is no second resource list or operation-name admission table to update.
Replacing a synchronous call with `encode` without moving hidden resources is not
a valid migration.

## Ownership boundaries

| Component | Owns |
|---|---|
| Kernel wrapper | Implementation selection, layout/scalar arguments, scratch requirements |
| Kernel context | Typed resource bindings, scratch preparation, invocation mode, errors |
| Graph lease owner | Pending Storage references, deduplication, chunk/memory limits, completion |
| Native slot owner | Planned physical slots and reusable logical generations |
| Metal driver adapter | Command creation/encoding, submission/wait, invocation permission |

The public ONNX loader now supplies one trusted synchronous graph boundary for
Metal execution. Profiling bypasses batching. HF and Hao have no batching policy,
and graph outputs are complete before return. The source-only validation loader
can disable the scope to retain an ordinary-execution oracle; no public knob is
exported. The existing running playground has not been restarted.

Validation is recorded in the
[shared-context report](../../apps/hf-inference/benchmarks/reports/kernel-context/README.md).


## Normalization and affine migration

Layer normalization (dense last-axis and generic ND), RMS normalization,
add-layer normalization and suffix multiply-add now bind every input and output
through the context. Their pipelines are cached; shape/stride/scalar arguments are
copied with `setBytes`. These implementations have no private scratch, status
readback or MPS objects. Normalization now ends through the common encoded-success
finish helper; suffix affine already did. Numerical kernels and dispatch selection
are unchanged.

The native test exercises all three normalization operations on both axes of a
2×3 input, followed by vector-scale/scalar-bias affine. Forty dependent encodings
cross the 32-operation chunk boundary while intermediate Tensor owners retire.
Results are checked against independently calculated normalization and affine
values, including repeated input bindings in add-layer normalization.

Matched request measurements and the rollout decision are recorded in the
[normalization report](../../apps/hf-inference/benchmarks/reports/graph-scope-normalization/README.md).

## Matmul dispatch diagnostics

Implementation selection is still separate from the execution contract. Optional
`AFFON_METAL_DISPATCH_TRACE=1` records native f32 matmul selection/entry events under
`compute.execution/metal_dispatch_*`: layout eligibility, size/stride/bounds
reasons for custom selection, generic versus vector custom pipelines, and legacy
contiguous/direct-offset/tiled entries and MPS-error fallback. This diagnostic
reports existing choices; it does not unify or change dispatch policy.

The [compute benchmark suite](../../test/benchmarks/README.md) captures these events
in a separate correctness-checked process and attaches per-case deltas to reports.
Timed workers explicitly disable tracing. Missing Metal matmul events fail the
probe; other operations remain explicitly uninstrumented. Entry/decision counts
are not command counts or proof that an attempted implementation completed.


Ordinary f32 matmul dispatch is now consolidated: all five entry points normalize
to the same layout-aware dispatcher. The legacy contiguous/direct-offset/tiled
implementation events above describe the earlier diagnostic checkpoint; those
implementations and their separate override/retry policy are removed. Current
layout selection and custom strided pipeline events remain unchanged. See
[Metal matmul dispatch](metal-matmul-dispatch.md) for scope and validation.


## Fused matmul epilogues

F32 matmul+bias and matmul+bias+GELU bind all three input buffers and the output
through the same context, including offset entry points. Their existing custom
kernels already use the common command creator and encoded-success finish helper.
They copy scalar metadata and allocate no hidden scratch or status buffer. Kernel
math and fusion selection are unchanged; integer matmul+bias remains synchronous.

A native test checks both epilogues against independent CPU calculations, poisoned
prefixes, output sentinels, nonzero offsets and early intermediate destruction.
Thirty-four encodings cross the chunk boundary in two submissions. Compiled-chain
benchmarks additionally require a fusion hit for every step. See the
[epilogue report](../../test/benchmarks/reports/metal-epilogue-batching/NOTES.md).


## Variable-length input bindings and joins

`kernel.readMany(storages)` binds a variable-length input list to an FFI handle
array. The context derives and admits every Storage lease from that list, including
normal deduplication of repeated inputs. Temporary handle arrays live only through
encoding; implementations must consume them synchronously. Raw opaque handle arrays
are rejected just like unbound scalar handles. Fixed-argument kernels retain the
existing stack resource array and do not acquire new heap allocations.

Metal cat and stack now share `join.zig`, which validates a dense row-copy plan and
binds all inputs plus the output in one invocation. One blit encoder handles all
inputs and outer rows, followed by the common finish helper. There is no private
scratch, GPU metadata buffer or status readback. Eager calls complete once; graph
calls participate in the existing lease/chunk contract. Input packing remains for unsupported layouts; the direct view path below
handles supported strided inputs.
CPU/CUDA execution and public APIs are unchanged.

The [join report](../../test/benchmarks/reports/metal-joins/NOTES.md) records native
lifetime checks, expanded axis benchmarks and before/after measurements.


## Direct strided join inputs

Metal join capability now accepts nonnegative strides (including zero/broadcast
strides) and offsets. Execution passes logical shape/strides/offset plus Storage
through a backend view descriptor. Both cat and stack use the same preparation
helper and Metal encoder. CPU/CUDA capabilities are unchanged; negative strides
retain the existing packing fallback.

The backend validates ranks, non-concatenated dimensions, source bounds and output
size before encoding. Complete dense buffers use the existing blit path. Other
views use a cached bit-copy shader (32-bit or 64-bit payloads) that maps each logical
input element directly to its final output location. Contiguous prefixes of larger
Storage are treated as views, not whole buffers. One compute encoder handles all
inputs. Descriptor data is copied with `setBytes`; no temporary dense Storage or
GPU scratch is allocated. Typed `readMany` bindings retain the original inputs.
The direct shader bounds each input to at most UINT32_MAX elements and rank eight.

See the [strided join report](../../test/benchmarks/reports/metal-strided-joins/NOTES.md)
for correctness, command/allocation diagnostics and matched performance results.


## Whole-tensor reductions

Whole-tensor f32 reductions and the existing supported dense i64 reductions now
bind inputs and outputs through the shared context. Dense inputs encode the
existing reduction kernel directly. For f32 views with nonnegative strides and
offsets, the backend allocates context-managed scratch, encodes the existing
contiguous pack and reduction into one command buffer, and finishes once. Both
encoders are one invocation for chunk admission. Packing remains visible in
contiguity telemetry; it has not been eliminated.

This preserves logical traversal order, floating-point math and first-index tie
behavior. A direct serial strided experiment was slower and was rejected. The
final implementation shares the existing packing and reduction encoder helpers;
it adds no shader, public knob or operation whitelist. Shape, strides, source
bounds and output size are validated before encoding. The view adapter supports
rank one through eight and the existing pack kernel's 32-bit index range.
Negative strides and other dtypes retain their existing layout fallback. CPU and
CUDA dispatch are unchanged. Axis reductions were migrated separately; see the axis-reduction section below.

Scratch and source buffers stay leased until their chunk completes, with physical
capacity counted toward the existing memory limit. Scoped execution can therefore
retain more scratch than eager execution; this is not a memory-reduction claim.
At this batching checkpoint, serial reduction math still limited large-input
performance relative to PyTorch. See the [reduction report](../../test/benchmarks/reports/metal-reduction-batching/NOTES.md)
for measured performance and lifetime validation.


## Parallel whole-tensor f32 reductions

The shared reduction encoder selects a parallel implementation for sum, mean,
min, max, population variance and std at 256 or more elements. Dense and packed
views use this same selector. Smaller inputs, arg reductions and i64 reductions
retain their serial implementations. Batching policy is unchanged.

One power-of-two threadgroup (at most 256 lanes, capped by the pipeline limit)
reads coalesced elements and combines lane partials through a shared tree. Sum
and mean reuse the existing parallel entry points. All six parallel shaders use
one implementation helper; the former separate whole-parallel command helper is
removed. Min/max add a parallel index pass when the result is zero to retain the
serial implementation's last-equal-zero sign. The tree uses threadgroup memory, without additional Storage, pool
leases or submissions. This checkpoint uses a single threadgroup. The later large-input path below
adds multiple threadgroups without replacing this smaller-input implementation.

Sum/mean and variance accumulation are reassociated, so bitwise agreement with
serial summation is not promised. Variance uses a two-pass calculation centered
on one input value to retain accuracy for small variation around a large common
offset. Volatile centered intermediates prevent Metal fast-math from undoing
that subtraction. Tests compare with f64 references at the dispatch boundary,
odd lengths and larger inputs. Existing logical packing order and arg-index
semantics are unchanged. See the
[parallel reduction report](../../test/benchmarks/reports/metal-parallel-reductions/NOTES.md).


## Multiple threadgroups for large whole reductions

For the same six f32 operations at 131,072 or more elements, `reduction_all.zig`
selects up to 256 partial groups, targeting roughly 4,096 elements per group.
The first encoder reduces contiguous chunks using the shared arithmetic helper;
the second combines their partials. Sum/mean combine sums. Min/max combine extrema
and retain the last-equal-zero sign. Variance/std combine centered local means
and second moments with population weights, including an uneven final chunk.
The global anchor stays separate from the centered mean to preserve small
variation around large offsets.

The wrapper allocates at most 2 KiB of logical partial storage through
`Context.scratch`; the graph owner accounts for its actual pool capacity. Views
retain a separate pack buffer. Keeping these allocations separate avoids
promoting a power-of-two pack allocation to a larger pool class. Both reduction
passes (plus packing for a view) remain one context invocation and one eager
command buffer. No operation-name batching policy, public API or host readback
is added. Dense large calls gain one scratch allocation; large views gain one
in addition to their existing pack. Leases remain alive until chunk completion.

Smaller cases continue through the prior shared encoder. Arg reductions, i64,
axis reductions and CUDA execution were unchanged at this checkpoint. See the
[multi-threadgroup report](../../test/benchmarks/reports/metal-multigroup-reductions/NOTES.md)
for the size sweep, numerical checks and memory/submission evidence.


The expanded sweep also exposed pre-existing CPU whole-variance/std errors on
million-element f32 inputs. Those two CPU wrappers now share a two-pass helper
that accumulates in f64 and casts the final output back to the original dtype.
The f64 input path keeps f64 accumulation. This is a separate numerical correction;
it changes neither Metal selection nor the graph execution contract. Analytic
million-element tests cover both zero-centered data and variation around one
million. No benchmark tolerance or supported-case declaration is weakened.


## Axis reductions

Rank-two and ND Metal axis reductions now use the typed kernel context for f32
and supported i64 operations. Input/output leases follow the same automatic
scope rules as other migrated kernels; no operation whitelist or public API is
added. Metadata copied into the encoder does not outlive stack storage. Dense
40-call probes submit two graph chunks instead of 40 commands, with the same
number of Storage allocations.

One shared selector uses the parallel shader for f32 sum, mean, min, max,
population variance and std when the reduced axis has at least 256 elements.
A group handles each output, with up to 256 lanes and at most 1 KiB of threadgroup
memory. An axis length plus the product of trailing dimensions describes both
rank-two and ND dense indexing. The shader reuses the whole-reduction arithmetic
helper with strided loads, including centered variance and signed-zero handling.
The existing ND infinity seed versus rank-two first-element seed for extrema is
preserved, including their differing all-NaN results. Smaller axes, arg reductions
and i64 keep their existing arithmetic but now participate in batching.

No temporary Storage is added by this algorithm. Pending input/output leases
still retain physical pool capacity until a chunk completes, subject to the
existing invocation and byte limits. At the initial axis checkpoint, layout preparation remained outside the
axis invocation and drained the scope. The packed-view extension below removes
that boundary for supported f32 views. CUDA and public reduction semantics are
unchanged.

See the [axis-reduction report](../../test/benchmarks/reports/metal-axis-reductions/NOTES.md)
for matched timings, all-element numerical checks, scope retirement tests,
command/allocation measurements and remaining view overhead.


## Packed axis-reduction views

Metal f32 axis reductions accept nonnegative-stride views through the backend
view entry point. Whole and axis reductions share `reduction_view.zig` for
checked shape/stride/offset metadata. The axis selector is shared by dense and
packed inputs. Existing packing and serial/parallel reduction encoder helpers
are composed in one kernel-context invocation, with one command finish. No new
shader or public API is needed; rank-two versus ND extrema semantics and logical
arg-index order are retained.

The context declares the source, output and packed scratch buffer together.
Eager execution completes both encoders in one submission. In a leased graph
scope, the packed buffer stays alive until chunk completion; its physical pool
capacity contributes to the existing 64 MiB budget. Packing still occurs, but
it no longer creates a submission boundary. Slot-planned graphs retain the
existing conservative flush for scratch-using invocations. Integer inputs and
negative-stride fallback preparation retain their prior behavior.

See the [packed-axis report](../../test/benchmarks/reports/metal-axis-views/NOTES.md)
for matched performance, command counts, allocation retention and validation.
