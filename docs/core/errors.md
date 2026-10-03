# Error Handling

Native runtime failures are reported as `AffonError`, a JavaScript `Error` with
a machine-readable `code`.

## `AffonError` Class

Runtime errors extend the standard JavaScript `Error` but add first-class properties for machine-readable codes and native stack traces.

```typescript
import { Session } from 'affon:compute'
import { reshape } from 'affon:ops'

const session = new Session({ device: 'cpu' })

try {
  const value = session.tensor([1, 2, 3])
  reshape(value, [2, 5])
} catch (error) {
  if (error instanceof AffonError) {
    console.error(error.code, error.message)
    if (error.nativeStack) console.error(error.nativeStack)
  } else {
    throw error
  }
}
```

Do not match human-readable messages. Branch on `code` only when an application
can recover meaningfully; otherwise preserve the original error and stack.

## Properties

| Property | Type | Meaning |
| --- | --- | --- |
| `name` | `'AffonError'` | Stable class name |
| `message` | `string` | Human-readable context |
| `code` | `AffonErrorCode` | Stable error category |
| `stack` | `string` or `undefined` | JavaScript stack |
| `nativeStack` | `string` or `undefined` | Optional Zig backtrace |

Common codes include `invalid_arg`, `missing_arg`, `shape_mismatch`,
`invalid_shape`, `invalid_dtype`, `device_mismatch`, `out_of_memory`, and
`internal`. The complete union is defined in
[the global declarations](../../packages/@types/affon/globals.d.ts).

Normal JavaScript authoring mistakes can still raise `TypeError`, `RangeError`,
or package-specific errors. `AffonError` is the boundary for structured native
runtime failures, not a replacement for every JavaScript error.

## Native stacks

Set `AFFON_NATIVE_STACK_TRACE=1` when diagnosing native failures. When a native
backtrace is available, Affon attaches it as `nativeStack`. Availability and
symbol detail depend on the build and platform, so applications must not require
it for normal error handling.

## Resource cleanup

`Tensor`, `Executable`, `ExecutionState`, and `Session` implement idempotent
`dispose()` and `Symbol.dispose`. Long-running processes should release owned
resources deterministically. Disposing a Session prevents new work through it;
child objects retain their storage until they are also disposed.
