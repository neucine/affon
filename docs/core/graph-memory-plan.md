# Graph storage reuse within bounded execution scopes

Status: compute now has the opt-in backend-neutral planner and a private native
execution prototype with slot reuse and bounded Metal command batching, plus a
Metal ONNX boundary with conservative resource leases, now enabled by default.
Native graph slot reuse remains opt-in; ordinary eager calls are unchanged. This implements a narrow part
of [bounded execution scopes](metal-execution-scopes.md), not asynchronous host
execution or a public batching API.

Kernel invocation now uses the [shared execution contract](kernel-execution-context.md):
typed arguments declare resources, the context owns scratch, and one driver boundary
controls batching permission. Individual wrappers no longer manage admission.

## Implemented native execution boundary

`compute.pipeline.execution.graph.reusable.execute` is the explicit internal entry.
It fixes the unfused schedule before encoding; candidate fusion regions are not
executed. Only inference with grad disabled, nonempty dense f32 values, same-shape
elementwise arithmetic/unary forward operations, and reshape/squeeze/unsqueeze views
qualify. Inputs must match their planned dense representation. Constants, CUDA,
training, broadcasts, packing, MPS, reductions, indexing, mutation and other operations
take the ordinary runner before any scoped work starts. Native nested scope entry
is rejected, including entry that would otherwise take the fallback.

The prototype bounds the whole invocation to 256 steps and 64 MiB of retained
physical input/slot/output capacity. Metal capacity is queried from the actual
MTLBuffer, including pool rounding. Outputs are preallocated separately and remain
owned after the scope returns. The default chunk holds 32 encoded operations;
submission immediately waits and checks completion. The internal test entry can
vary chunk length, with no environment or product-facing setting. CPU uses the
same slot ownership with synchronous kernels; CUDA keeps the ordinary path.

`execution/graph/slots.zig` owns physical allocations. `Storage.borrowed_slot`
represents a logical generation and retains its owner, without independently
returning the block or double-counting its bytes in Storage metrics. A new loan
requires the previous logical Storage and all its views to be gone. The runner
retires expired roots, including unused values, while the physical owners remain
alive until submission/completion or unsubmitted-command discard.

The native slot runner uses the existing dense f32 binary/unary FFI helpers.
The leased ONNX path also admits the audited f32 broadcast helper.
Their finish result means encoded success while a scope is active; the graph's
fallible scope end establishes completion. Other command paths drain first; status
reset and host read/write also drain. Raw pointer exports return null during a scope.
An encode/native error aborts unsubmitted work before releasing resources. A failed
completion prevents a successful Result. Already completed chunks are not replayed
or rolled back. No pending tensor escapes this API.

The implementation lives in compute's `pipeline/execution/graph/reusable.zig`,
`slots.zig`, eager output-storage binding, Storage loan ownership, and Metal FFI.
`pipeline/backend/metal/scope.zig` owns the backend begin/finish/abort bridge;
graph execution does not call Objective-C FFI entry points directly.
It adds no HF/Hao policy and does not alter the running playground. ONNX does not
call this slot-planned entry.

## Private ONNX validation boundary

`load_graph_for_scope_validation` is exported only from ONNX's source runtime,
not its package index. It encloses one synchronous Metal forward call in the
native `$with_graph_execution` bridge. CPU and profiling calls use the existing
synchronous body. The bridge requires disabled gradients, rejects nested owners
and returned thenables, preserves callback exceptions, and completes pending GPU
work before returning outputs. This is a trusted internal body contract, not a
general callback API; returned-thenable detection cannot make arbitrary async code
safe inside a scope.

Compute's `backend/metal/leased_scope.zig` owns the graph lease registry and chunk
budget. `backend/metal/kernel.zig` derives leases from typed buffer arguments and
owns scratch allocation. Dense f32 unary/binary and f32 broadcast dispatch use that
contract. Retaining the Metal buffer alone would not prevent
pool reuse. The backend has no dependency on the execution layer. The existing
compiled affine runner can execute under this boundary without opening another
scope; unconverted kernels, MPS and host access drain first.

Broadcast admission covers its existing nonnegative-stride and offset contract.
Both sources and the destination are leased before encoding; repeated Storage
inputs are deduplicated. Shape/stride metadata is copied by `setBytes`, and the
helper has no pool-backed scratch or host status buffer. Negative-stride packing
and other helpers retain their existing synchronous boundaries. This expands
command accumulation only; the native slot planner still excludes broadcasts.

Each chunk permits at most 32 admitted operations and 64 MiB of retained physical
capacity. A single operation larger than that memory target is executed and
completed alone; its peak is reported rather than hidden. Completed chunks release
leases on the next admission or scope exit. The budget covers pending admitted
resources, not the full graph's live memory or opaque backend scratch. Submission
still waits synchronously, with at most one chunk in flight.

This path deliberately allocates ordinary eager outputs. It does not reuse graph
slots: the TypeScript eager loop has no resolved compute GraphPlan. Native slot
reuse and ONNX command accumulation are separate implementations of the same
completion boundary. Whole-ONNX slot reuse still requires lowering to compute IR.

The ONNX tests cover model-fixture parity, output retention across forwards,
profiling bypass, exceptions, thenables, nested entry and recovery. Compute tests
also cover immediate Tensor destruction, allocation/completion failures, command
limits and memory-driven drains with multi-megabyte intermediates. Real model and
performance evidence is recorded in the
[initial integration report](../../apps/hf-inference/benchmarks/reports/graph-scope/README.md)
and [broadcast admission report](../../apps/hf-inference/benchmarks/reports/graph-scope-broadcast/README.md).
The subsequent [context refactor report](../../apps/hf-inference/benchmarks/reports/kernel-context/README.md)
checks scratch ownership, contract violations and preservation of the same batching.

## Ownership boundaries

| Layer | Responsibility | Must not do |
|---|---|---|
| TensorSpec / semantic inference | Shapes, dtype, device, layout and view semantics | Carry physical buffer handles or Metal policy |
| Existing GraphPlan | Ordered operation steps, eager plans and candidate fusion regions | Own runtime allocations or decide GPU readiness |
| `pipeline/plan/memory.zig` | Analyze backing lifetimes and assign reusable storage slots | Allocate device buffers, submit commands or assume a GPU operation has completed |
| Graph execution / bounded context | Bind logical values to slot generations, own resources, enforce barriers and error cleanup | Return pending tensors or release a pending slot to the ordinary pool |
| Metal backend | Encode resource dependencies and certify reuse for the actual command schedule | Infer graph liveness from reference counts alone |
| Ordinary memory pool | Supply/reclaim physical allocations when execution ownership allows it | Offer pending allocations to unrelated operations |
| ONNX | Supply a trusted synchronous graph boundary; eventually lower a complete supported graph to compute IR | Implement its own allocator, slot planner or Metal scheduling policy |
| HF / Hao | Invoke inference and serve requests | Know about slots, command buffers or reuse fences |

The implementation is an explicit `memory.create(allocator, graph, execution_plan,
options)` analysis, alongside `pipeline.plan.graph`. Its `Plan` owns only metadata
and is explicitly deinitialized. It is not automatically attached to every eager or
graph call, so the current inference path pays no new planning cost. There is no
new Tensor, ONNX or end-user batching API.

## What is implemented

The planner consumes existing `GraphPlan.steps`, each step's eager output byte sizes,
and graph value IDs. It does not repeat shape inference or discover another graph.
It analyzes the **unfused step schedule**, even if the existing plan has candidate
fusion regions. Explicit fused steps are rejected. Candidate regions are not a
resolved execution schedule: the current runner can select a fusion or fall back.

Each allocation root records its first producing step, last consuming step, aligned
size, whether it escapes, and optional slot assignment. Views resolve to a canonical
backing root. Reading a view extends that root's lifetime. Returning a view pins
the whole backing as an output; it is not a separate allocation.

Inputs and constants are externally owned and never selected for scratch reuse.
Graph-owned outputs and backing roots of output views receive dedicated allocation
requirements. Non-escaping roots receive stable best-fit slots of sufficient capacity
and alignment on the same device. Dtypes may share raw storage when capacity and
alignment permit. Multiple outputs of one operation overlap and cannot share a slot.

Reuse requires `previous.last_step < next.first_step`. Equality means that the
operation is still reading the old value while producing the new one. The planner
does not assume in-place kernels or use CPU submission/completion time as liveness.

For `input → A → B → C → output`, equally sized private A and C share a slot, while
B uses a second slot and the returned output has dedicated storage. A residual read
of A at the output step extends its lifetime and prevents that reuse.

The first physical model is **separate whole-buffer slots**, not one suballocated
arena. `slot_bytes` is their summed aligned capacity; `output_bytes` is the dedicated
graph-owned output capacity. Neither includes externally owned inputs/constants,
packing temporaries, backend scratch, allocator bucket slack beyond the supplied
alignment, or opaque MPS allocations. These numbers are not a total-memory bound
or a claim of an optimal packing. A best-fit planner can reserve more than the
theoretical peak simultaneously live bytes.

This choice fits existing Storage handles and avoids adding base offsets to every
Metal binding. A future arena would require backend alignment constraints, base-offset
propagation, alias dependency handling and output ownership rules across all paths.
It is not a prerequisite for within-batch reuse.

## Runtime integration contract and remaining expansion

The metadata is deliberately insufficient to authorize reuse on its own. Before
connecting a broader path to allocation, graph execution must resolve the **actual resource
schedule**, including fusion choice, packing, workspaces, secondary outputs and
fallback boundaries. Start with an explicitly eligible unfused native graph segment.
Do not apply unfused step lifetimes to a dynamically chosen fused execution path.
Either freeze the selected path before planning or fall back to ordinary allocations.

The runtime allocation owner holds one physical allocation per slot. Logical
values borrow a particular generation of that slot; views borrow the same generation.
The current `Storage.Handle.managed` owns and releases an `mm.Block`, so constructing
multiple ordinary owning Storage objects around the same block would be wrong.
A distinct internal `borrowed_slot` ownership representation now implements this.
Tensor destruction retires logical ownership, not the physical allocation. The slot
owner releases the block only after its final submitted GPU use completes.

Track two identities: logical backing/generation for alias and lifetime validation,
and physical slot for resource retention and dependency tracking. Merely keeping an
old Storage alive or retaining its MTLBuffer does not certify byte reuse. Likewise,
the ordinary pool must never receive a pending slot just because a graph value died.

When two generations share a slot, the backend must order the next write after all
earlier reads/writes to that physical resource. Initially keep complete encoder
boundaries and supported tracked resources; verify read-after-write and
write-after-read dependencies with actual Metal tests. Unsupported resource/MPS
paths drain or use separate allocations. No CPU wait is needed *between eligible
operations* once those GPU dependencies are established; submission at the chunk
boundary still immediately waits. This remains command batching, not asynchronous
host execution.

Host reads/writes, status resets, raw pointer access and user callbacks retain the
audit's restrictions. A host-initialized next generation cannot overwrite a pending
slot via memcpy; it must drain first or use an independently safe upload path.
Training, arbitrary in-place operations and externally visible aliases remain out.

Plan reuse must be tied to the graph revision, specialization (shape, dtype, device,
layout), resolved execution path and backend resource requirements. Do not cache
solely by operation names or byte sizes. A mismatch invalidates the assignment.

## Bounded allocation and graph integration

Slot capacity is planned in advance, so an eligible segment can acquire its buffers
before encoding, reusing the ordinary pool only for completed allocations. Account
actual returned capacities and scratch, not just logical sizes. Deduplicate retained
physical slots in the scope budget even when many logical generations use them.

If the planned segment exceeds the internal budget, shorten the segment before
encoding or take the synchronous fallback. Values live across a chunk boundary
remain pinned; submission/completion does not make a still-needed value dead. An
individual operation or required live set may exceed the budget and needs the design's
standalone synchronous path. Do not promise that chunking makes every workload fit
an arbitrary memory cap.

Native compiled graphs already provide value IDs and use sites. ONNX currently
executes a TypeScript eager loop, so a scope boundary alone cannot supply this
static plan. First integrate reuse in native graph segments. ONNX can initially
batch with conservative leases; nested affine calls join the outer context but
cannot independently recycle a returned value that the outer loop still uses.
Whole-ONNX reuse requires lowering its supported graph into the existing compute IR
or a reviewed execution resource schedule in compute. Do not add last-use/free-slot
policy to the TypeScript loop. Preserve its existing semantics until that separate
lowering work is complete.

## Implementation and verification sequence

1. **Done:** opt-in allocation analysis at
   `/Users/chao.yang/Private/compute/src/pipeline/plan/memory.zig`, exposed through
   compute's existing pipeline namespace. Tests use real GraphPlan construction.
2. **Done for the narrow native path:** freeze an eligible unfused graph schedule,
   with slot-owner/borrowed-generation storage and dedicated escaping outputs.
3. **Done for dense f32 binary/unary helpers:** bounded Metal context, fallible
   completion and planned slots. The ordinary unfused runner is the correctness
   oracle. Unsupported schedules retain ordinary execution.
4. **Native core checks passed; broader coverage remains:** test actual GPU
   producer/consumer chains, residuals, offset/signed views, temporary
   destruction, pool pressure, host barriers, chunk splitting, errors and alias
   generations. Verify both peak allocation reduction and exact outputs before any
   whole-model/default-policy change.
5. Integrate supported ONNX lowering and broader fusion/MPS schedules separately.
   Re-run strict model parity, then compare request latency, command count and actual
   peak memory. Static allocation estimates are not performance evidence.

`zig build test-memory-plan` is a device-independent test target, also included in
`zig build test`. It checks linear reuse, residual lifetimes, escaping and nonescaping
views, alignment, device separation, simultaneous outputs, unequal buffer sizes,
invalid schedules and allocation-failure cleanup. Runtime lease and GPU dependency
tests belong to the later execution integration; these planning tests do not replace
them.

The `test-reusable-graph` target exercises actual CPU and Metal execution. Its
29 tests pass, including direct output checks, residuals/escaping views, host
read/write barriers, raw-pointer rejection, reentry rejection, allocation failures,
abort before and after chunk submission, and test-build-only injected completion
failure/recovery. A 73-add dependency chain produces exact expected values using
two intermediate slots, 70 reassignments and three submitted chunks. These are
correctness/counter checks, not a latency or full-model performance measurement.

The oracle uses the ordinary runner with fusion regions removed to match the frozen
schedule. The first residual/view oracle attempted ordinary fusion and hit an
existing `unaryStage` error before scoped execution; that unrelated fusion path was
not changed. The tests also compare hand-calculated values, rather than relying only
on agreement between the two runners. Broader fusion integration needs its own gate.

The full compute `zig build test` run passed with **45 passed, 8 skipped** across
53 tests (including 10 planning and 29 scope tests). All scope tests executed on
the host, including the Metal cases. The eight skipped cases belong to the existing
general test artifact. Builds used isolated `/private/tmp` caches and normal Metal
compiler/device access. No playground process or default execution route was changed.

Strict Whisper/ViT/AST parity now passes through the separate private ONNX leased
boundary. Broadcast admission reduces decoder submissions from 180 to 136; default
adoption remains subject to repeatable request-performance and memory evidence.
Offset/signed layouts and opaque MPS scratch are excluded from slot reuse; these
tests make no claim about reusing their storage within a batch.


### Default ONNX activation

Following normalization and affine validation, the public Metal ONNX loader now
uses the bounded leased scope. This enables command batching, not whole-ONNX
static slot reuse. Profiling remains synchronous per operation; graph outputs are
complete on return. See the [activation report](../../apps/hf-inference/benchmarks/reports/graph-scope-default/README.md).
Earlier checkpoint measurements above describe their original private rollout.
