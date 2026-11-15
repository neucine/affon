# Node ownership experiment

This is a narrow Node-API binding to the real Affon compute Session and CPU tensor
storage. It measures native storage release when tensors are disposed explicitly
versus when JavaScript references are dropped. It does not implement Program
execution or the public `@affon/core` package.

On macOS with Zig 0.17 and Node 22:

```sh
zig build -Dnode-include="$(dirname "$(dirname "$(command -v node)")")/include/node"
node --expose-gc run.cjs
node package/test.mjs
```

The test prints live tensor count, active native storage bytes, and process RSS.
`after-explicit` must show zero live tensors and active bytes. `after-drop-before-gc`
shows what remains when JS references disappear without a collection checkpoint.
`after-forced-gc-and-turns` shows whether native finalizers eventually release them.
The forced GC run is diagnostic only; application code cannot depend on its timing.

The bridge keeps the Session alive while child tensors exist and makes explicit
tensor and Session release idempotent. This mirrors the current Affon ownership
contract. It uses N-API as a portability probe, not as a proposed complete API.

## Scoped `@affon/core` prototype

The private package in `package/` provides a small Node import surface:

```js
import { withSession } from '@affon/core'

withSession(session => {
  for (let i = 0; i < 500; i++) {
    session.run(() => {
      session.zeros(262144)
    })
  }
})
```

`session.run` tracks tensors created inside its callback and disposes them on
normal return, throw, or async completion. A tensor returned by the callback
stays alive for its caller to dispose. The explicit `session.scope` and
`scope.transfer(tensor)` remain available when ownership must be directed
manually. The package currently creates only one dimensional f32 CPU tensors;
it is an ownership prototype, not the full Affon compute API.

## Initial result

On macOS arm64, Zig 0.17.0 and Node 22.22.1, 500 tensors of 1 MiB each gave:

| Checkpoint | Live tensors | Active native bytes |
| --- | ---: | ---: |
| After explicit disposal in a 500 iteration loop | 0 | 0 |
| After dropping 500 references, before GC | 500 | 524,288,000 |
| After three forced GC passes and event-loop turns | 0 | 0 |

This confirms that the Node binding can preserve deterministic release through
`dispose`, while ordinary reference loss does not release Affon storage at a
predictable point. The experiment does not test GPU tensors. RSS is reported
separately because the compute allocator can retain pooled storage after tensor
disposal.

## Hao control

Run `AFFON_DEVICE=cpu ./zig-out/bin/affon run experiments/node-ownership/hao.ts`
from the Affon repository root. Affon 0.4.0 built against Hao's `legacy` branch
reports the following result. With
the same 500 one MiB tensors, `compute.storage/live_bytes` was 524,288,000
inside the allocation function and 67,108,864 immediately after it returned.
An explicit `dispose()` control also left 67,108,864 bytes, indicating that
the residual is retained storage rather than live tensor wrappers. This is
evidence of prompt automatic finalization on the tested Hao binary.

For this run, the Hao dependency path in `build.zig.zon` was temporarily set to
`../hao-legacy` and restored afterward. The adjacent `../hao` checkout is on
Hao's Cyna-based main branch and lacks the Zig `hao` module expected by Affon.
The observed 0.4.0 behavior matches the previously available 0.3.0 binary.
