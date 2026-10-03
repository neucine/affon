# Error Handling in AFFON

AFFON provides a structured error handling system that bridges the Zig native runtime and JavaScript. All runtime errors are instances of the `AffonError` class.

## `AffonError` Class

Runtime errors extend the standard JavaScript `Error` but add first-class properties for machine-readable codes and native stack traces.

```typescript
try {
  const t = session.tensor([1, 2, 3])
  reshape(t, [2, 5]) // Throws shape_mismatch
} catch (err) {
  if (err instanceof AffonError) {
    console.log(err.code)         // "shape_mismatch"
    console.log(err.message)      // "reshape: 3 elems cannot become shape [2,5]"
    console.log(err.nativeStack)  // Optional Zig backtrace (debug builds)
  }
}
```

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `'AffonError'` | Fixed string to identify the class. |
| `message` | `string` | Human-readable description of the error. |
| `code` | `AffonErrorCode` | Machine-readable union of predefined constants. |
| `stack` | `string` | Standard JavaScript V8/JSC stack trace. |
| `nativeStack` | `string` \| `undefined` | Zig native backtrace from the runtime (available in debug builds). |

## Error Codes

The `code` property uses a union type `AffonErrorCode` for type-safe handling:

- `invalid_arg`: Function argument error (e.g. wrong type or value).
- `missing_arg`: Required argument was undefined or null.
- `shape_mismatch`: Dimension conflict (e.g. matmul shape mismatch).
- `invalid_shape`: Requested shape is impossible (e.g. empty dimension).
- `invalid_dtype`: Unsupported data type for the requested operation.
- `device_mismatch`: Tensors are on different devices (e.g. CPU vs Metal).
- `out_of_memory`: Native allocation failed.
- `internal`: Unexpected runtime bug.

For the full list, see the [global type definitions](../../packages/@types/affon/globals.d.ts).

## Native Stack Traces

In debug builds (`zig build`), AFFON captures the native Zig call stack at the moment an error occurs and attaches it to the JavaScript error object. This is invaluable for debugging errors that originate deep within the native kernels.

The `.nativeStack` property is a string containing the Zig backtrace symbols and addresses.

## Performance

AFFON's error system is designed for high-performance reporting:
- **Zero-Copy**: Error strings are generated using stack-allocated buffers into the JSC context, avoiding heap allocations and syscalls on the common error path.
- **Cached Prototypes**: The `AffonError` class and constructor are evaluated once at startup and cached in the native bridge to minimize overhead when throwing.
