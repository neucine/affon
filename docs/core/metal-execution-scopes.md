# Proposal: bounded Metal execution scopes

Status: bounded batching is enabled at the public Metal ONNX graph boundary.
Each graph call completes before returning; profiling callbacks bypass batching.
Ordinary eager calls outside that boundary retain synchronous completion. Planned
slot reuse remains an opt-in native graph prototype, separate from ONNX leases.
There is no public scope API or asynchronous tensor behavior.
The [shared kernel context](kernel-execution-context.md) now centralizes bindings,
scratch ownership and invocation mode, replacing per-wrapper admission code.

The [storage-lifetime and host-access audit](metal-execution-scopes-audit.md)
is complete against compute `0dfaf41` and Affon `f0b465e0`. It inventories all
85 command-creation call sites and defines the first prototype's allowlist and
synchronous boundaries. That inventory describes the audited checkpoint, not the
subsequent prototype's shifted code locations. Prototype validation is recorded in
the graph storage reuse plan below.

The subsequent [graph storage reuse plan](graph-memory-plan.md) separates logical
value lifetime from physical buffer lifetime. Its opt-in allocation-analysis
component and private native execution path are now implemented in compute. They
use reusable whole-buffer slots before considering a larger arena, with graph
outputs and live aliases protected. The prototype admits only an explicitly fixed,
unfused dense-f32 schedule and rejects nested entry. The separate ONNX validation
boundary covers the eager loop and existing compiled affine calls without enabling
whole-ONNX slot reuse. See the
[integration report](../../apps/hf-inference/benchmarks/reports/graph-scope/README.md).
The subsequent audited broadcast expansion reduces decoder command submissions
from 180 to 136; its correctness, memory and latency evidence is in the
[broadcast report](../../apps/hf-inference/benchmarks/reports/graph-scope-broadcast/README.md).
The subsequent normalization and fused-affine migration reduces submissions to
110. Its [matched validation report](../../apps/hf-inference/benchmarks/reports/graph-scope-normalization/README.md)
records 23.0% lower long-request median latency, a 2.20 MiB tracked Storage peak
increase. The boundary is now enabled by default; see the
[activation checks](../../apps/hf-inference/benchmarks/reports/graph-scope-default/README.md).

## Decision and scope

Start with **internal, bounded command accumulation around one graph execution**.
The graph call still completes before returning. Outside that boundary, eager
operations retain their current completion semantics. General asynchronous eager
execution is a separate later design, not a prerequisite for this experiment.

This answers “batch until finished or until the CPU needs a tensor?” precisely:
within an eligible scope, accumulate until a host-access boundary, an unsupported
operation, a resource limit, or graph exit. At each boundary, submit and complete
pending work. Never accumulate indefinitely across unrelated calls or requests.
A decoder graph call is a useful initial scope; a whole autoregressive generation
is not, because token selection already requires a host result each step.

The objective is to measure whether removing repeated command lifecycle overhead
improves full inference. It is not to overlap GPU-dependent operations that must
remain ordered. There is no promised recovery of the trace's non-GPU interval.

## Evidence and committed baseline

- Affon `becafce4`: HF integration, native playground, correctness fixtures and reports.
- compute `0dfaf41`: Metal kernels, DSP, command telemetry and layout fixes.
- Hao `af9af07`: streaming HTTP and native server support.
- Current decoder: 180 command buffers per step with automatic affine fusion.
- Latest reversed-order long-request medians: 2,917 ms ordinary affine path versus
  2,782 ms fused affine path; compare future changes against the fused baseline.
- In the instrumented CPU/Metal sample, main-thread completion-to-running was
  4.6 µs median, while GPU-end-to-completion was approximately 92 µs. Runnable
  main-thread time was only 12.9 ms in the 3.222-second observed activity window.

The [command audit](../../apps/hf-inference/benchmarks/reports/whisper-command-audit/README.md)
and [scheduling report](../../apps/hf-inference/benchmarks/reports/whisper-cpu-scheduling/README.md)
document sample sizes, incomplete trace joins and instrumentation overhead.
Those measurements motivate an experiment; they do not establish its speedup.

## Why start with a scope rather than asynchronous tensors

| Choice | Completion visible to caller | Main new responsibilities |
|---|---|---|
| Current execution | Each operation completes | Existing synchronous behavior |
| Proposed first phase | Graph completes before return; intermediate boundaries may also complete | Encoding ownership, retention, bounded memory, error draining |
| General asynchronous eager execution | Tensor handles may outlive pending GPU work | Storage readiness, mutation hazards, cross-call ordering, delayed errors, cancellation and multi-context synchronization |

The first phase keeps readiness from escaping the graph call. It does change
internal execution, failure timing and buffer lifetimes, so it must not be treated
as merely deleting `waitUntilCompleted` calls.

## Current code that must change together

The following paths are relative to the sibling compute repository:

- `src/pipeline/backend/metal/ffi.m`: `affon_metal_new_command_buffer` and
  `affon_metal_commit_wait` centralize most command creation/completion. Kernel
  functions still own individual encoders and may allocate scratch resources.
- `src/shared/types/tensor/storage.zig`: `Storage.release` can immediately return
  backing storage to memory management; graph intermediates can disappear before
  a deferred command completes. `writeFromHost` and `copyToHost` cross the host boundary.
- `src/pipeline/backend/metal/ffi.m`: buffer read/write and contents functions expose
  shared memory through direct `memcpy` or pointers. Shared memory does not remove
  ordering requirements.
- `src/pipeline/execution/graph/runner.zig`: a native compiled graph has a natural
  entry/exit boundary, but the HF ONNX executor is currently a TypeScript loop of
  eager operations in `packages/@affon/onnx/src/runtime.ts` in the Affon repository.
  Scoping only the native graph runner would not cover the whole Whisper graph.
- Eager preparation, graph fusion, transfers, blits and MPS paths must all obey the
  same scope ownership and barrier rules. An audit must also locate any direct
  command commits or raw buffer accesses outside the helpers.

The compute core owns scheduling and lifetime rules. The ONNX executor should only
identify a graph-call boundary through a private runtime hook. HF adapters and Hao's
HTTP layer should have no Metal scheduling policy.

## First-phase contract

1. A synchronous graph invocation creates an execution context. Its context is
   explicit in compute execution state; the FFI bridge may use a narrowly bounded
   thread-local pointer only while that context is active. No process-global
   “current batch” shared between requests.
2. Encoded GPU operations execute in their existing order on the existing queue.
   Each operation still closes its encoder before the next begins. Initially do
   not reuse encoders or change barriers, kernel math, MPS selection or fusion.
3. Nested internal compiled calls join the enclosing context. Only the outer scope
   owns completion, except for explicit internal barriers. This matters for ONNX's
   compiled affine expression inside its eager graph loop.
4. The entry/exit bridge is private and synchronous. No user callback, Promise,
   `await`, event-loop yield or reentrant request is allowed to suspend a scope.
   A public JavaScript batching API is not part of phase one.
5. Successful scope exit closes encoders, submits any pending commands, checks
   completion and releases retained resources before returning ready outputs.
6. CPU and CUDA retain their existing behavior. The private hook is a no-op there;
   that does not claim asynchronous CUDA support.
7. Only pure inference operations are eligible initially. Training, optimizer
   mutation and externally visible in-place writes stay on the existing path.
8. ONNX calls with a per-node `profile` callback remain entirely synchronous in
   the first prototype. `no_grad` is not itself a scope boundary. The private
   bridge runs only the trusted synchronous graph body; nested native affine
   execution joins it, while unrelated/reentrant graph entry is rejected.

## State machine and boundaries

A context progresses through `idle → encoding → submitted → completed` for each
chunk. A completed chunk may return to `idle` and encode another. Failures enter
`failed`; cleanup must drain any submitted work before resources can be reclaimed.
The first version permits **at most one submitted chunk in flight per context**.
Chunk submission immediately waits, deliberately postponing pipelining across chunks.

| Trigger inside a scope | Required action |
|---|---|
| Another eligible GPU operation | Encode after prior operations; register resource use |
| Host read: `item`, `to_array`, checkpoint serialization, GPU→CPU transfer | Close, submit and complete pending work before exposing bytes |
| Host write or raw mutable contents access | Complete pending reads/writes that might alias the storage; conservative full-scope drain initially |
| CPU fallback or unsupported encoder path | Drain first, execute synchronously, then permit a fresh chunk |
| Command/retained-memory budget reached | Complete a chunk before admitting the next operation |
| Nested graph exit | Leave completion ownership with the outer scope |
| Outermost graph exit | Complete pending work before returning |
| Exception or native error | Stop encoding; discard unsubmitted work where safe; drain submitted work; report failure |

A metadata-only shape query does not require a barrier. A reshape/view inherits the
underlying storage dependency. New host-initialized storage can avoid a drain only
when proven disjoint from pending accesses; phase one may conservatively drain.
Exporting raw pointers outside a scope is forbidden unless their storage is ready.

## Resource ownership and memory limits

Before encoding, retain each unique backing storage used by the chunk, including
inputs, outputs and temporary/workspace buffers. Record views by backing identity,
not Tensor identity. Hold the lease until completion, not merely until submission.
The allocator must not recycle a pooled block while such a lease exists. Retaining
an Objective-C MTLBuffer alone is insufficient if a pool can reuse its contents.

This excludes returning pending resources to the **ordinary** pool. Planned reuse
inside a scope is a separate, explicit mechanism: a scope-owned physical slot can
hold successive logical values after the prior value's last GPU use is ordered
before the next write. See the graph storage reuse plan for alias lifetimes, output
ownership and backend dependency requirements. The conservative retain-all path
remains the execution oracle until that mechanism is implemented and validated.

Scratch buffers created below Storage need equivalent ownership through the context.
Copied scalar metadata passed with encoder APIs and MPS object lifetimes must be
audited against the actual encoding path. Keep a stable context/command-buffer
owner across existing per-operation autorelease pools.

The audit confirms that both Metal `.pooled` and `.scratch` allocations can reuse
the same size-bucket pool. Retain Storage at the actual backend call, including
prepared inputs and fusion temporaries; retaining only graph inputs is insufficient.
Count the allocation's bucket capacity, not just `Storage.bytes`, in the memory
budget. Buffer destruction also marks its resource purgeable, so an Objective-C
reference alone does not prevent every premature-release hazard.

Status-checking kernels share one `ctx.statusBuffer`: its CPU reset occurs before
command creation and its CPU read occurs after completion. Keep every such call
synchronous, draining **before reset**, until a separate status-slot design exists.
Raw pointer exports must remain unavailable during an active scope initially;
draining at pointer acquisition alone cannot constrain a pointer used later.

Track pending leased bytes separately from live Tensor bytes. Start with internal
prototype limits for encoded operation count and unique retained backing bytes;
benchmark a small sweep before choosing defaults. A proposed operation that alone
exceeds the limit executes as a synchronous standalone chunk after draining the
previous one. Do not expose tuning environment variables as the product interface.

Phase one uses tracked, ordered resources. Untracked heaps, aliased workspace reuse
and concurrent mutation require an explicit dependency mechanism and are excluded
until that mechanism is designed and tested. Returning a Tensor or freeing a graph
map entry must not release a backing still used by a queued encoder.

## Errors, reentrancy and threading

Existing FFI functions often inspect `command.error` immediately after their
completion helper. Scoped encoding must distinguish “encoded successfully” from
“completed successfully”; defer completion-error checks to the context barrier.
Simply changing the helper to return early would give those call sites a false
success signal.

GPU execution errors can be detected only when the chunk completes. Associate each
chunk with graph/node ranges for diagnostics; report the backend error plus the
range rather than claiming a precise failing operation when unavailable. Never
silently retry a partially executed chunk. Already-completed chunks are not rolled
back. Failed output tensors must not escape as successful results.

On a host exception, drain work already submitted; unsubmitted pure-output work can
be discarded with its resource leases. Preserve the original exception and attach
cleanup failures without replacing it. Always clear the active context, including
on allocation failure or error while constructing the scope.

Initially reject unsupported concurrent/reentrant use of one context rather than
merging scopes. Independent contexts sharing a queue still need a documented order
and storage-sharing policy before parallel host execution is enabled. Command queue
ordering is not a substitute for preventing host writes to storage still in use.

## Telemetry changes required before measurement

Keep operation counts separate from submitted command-buffer counts. Under a scope,
per-operation GPU time is no longer derivable from a whole-command timestamp.
Add chunk operation count, unique leased bytes/peak, forced-drain reason, encode
CPU time, submit time, completion wait, GPU duration and error counters.

Continue checking valid GPU timestamp counts and `submit + wait == command wall`.
Do not attribute a chunk's complete GPU duration to each constituent operation.
Report end-to-end timings with diagnostic instrumentation disabled; use traces and
counters separately to explain the outcome.

## Implementation and validation sequence

1. **Ownership/barrier audit:** enumerate every command creation, direct host access,
   workspace lifetime and pool-return path. Deliver an eligibility list and explicit
   synchronous fallback boundaries before enabling accumulation. Completed by the
   linked audit; next is its bounded compute-only prototype with a default-deny
   operation admission guard, Storage leases, fallible drains and failure tests.
2. **Private prototype:** add a compute execution context with resource leases and
   one chunk in flight. Start with dense pure f32 elementwise chains, then blits and
   MPS. Keep existing execution as a test oracle. No broad default change yet.
3. **Integrate a graph boundary:** cover the ONNX executor's eager loop and nested
   compiled affine calls. Verify readback, fallback and exception boundaries.
4. **Correctness stress:** compare ordinary and scoped execution for views, offsets,
   singleton axes, broadcasts, MPS/generic paths, temporary destruction, deliberate
   pool reuse pressure, host reads/writes, nested calls and forced chunk boundaries.
   Inject encoding/allocation/completion failures. Check cleanup and stale-output
   behavior. Unsupported training/mutation paths must retain their old semantics.
5. **Model validation:** unchanged strict Whisper per-step logits and exact short/long
   tokens; ViT/AST reference checks; CPU/CUDA behavior unchanged. Test both tiny and
   multi-megabyte intermediates, including chunk limits that split dependencies.
6. **Matched performance gate:** compare the committed fused baseline against the
   prototype with fixed fixtures, warmed models, reversed/interleaved run order,
   no competing work, and at least ten measured requests per condition. Report all
   samples, median/p90, command count, CPU/GPU timing and peak leased/total memory.
   Include small workloads to catch batching overhead regressions.
7. **Default policy decision:** adopt only if full-request improvement is repeatable,
   correctness holds and memory/latency costs are acceptable. A provisional useful
   target is at least 10% median long-request improvement without a material p90
   regression; this is a decision criterion, not a predicted speedup. Record the
   actual memory tradeoff and decide limits from evidence, not command count alone.

If phase one provides little benefit, preserve the findings and reconsider the next
step rather than automatically escalating to full async execution.

## Deferred general async design

Allowing pending tensors to escape graph calls would require storage-level readiness
fences, last-writer and outstanding-reader tracking, host-mutation ordering,
cross-context/queue synchronization, explicit error observation and shutdown/drain
semantics. Public synchronization APIs and interoperability would then need a stable
contract. Cancellation cannot undo already-submitted writes. Optimizer state and
autograd lifetimes need a separate review.

Those responsibilities belong to a later proposal. This document deliberately does
not make “all eager operations return immediately” the new default strategy.
