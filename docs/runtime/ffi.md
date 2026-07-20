# FFI

Affon currently exposes one native interop module with two abstraction levels:

- `std:ffi` for raw C ABI calls
- the `c` export on `std:ffi` for constrained, more ergonomic C-library binding on top of raw FFI

Use `std:ffi` root exports when you want exact low-level control. Use the `c` export on `std:ffi` when you are writing a normal wrapper for a C library and want declaration parsing, opaque handles, and common policy-driven behavior.

## `std:ffi`

`std:ffi` currently exposes:

- `dlopen(name, symbols)`
- `c.decl(name, declarations, opts?)`
- `Pointer`
- `FFIType`

Example:

```ts
import { dlopen } from 'std:ffi'

const lib = dlopen('libm', {
  sqrt: { args: ['f64'], returns: 'f64' },
})

console.log(lib.sqrt(16))
lib.close()
```

### Supported Argument And Return Types

`FFIType` currently supports:

- `void`
- `bool`
- `i8`, `i16`, `i32`, `i64`
- `u8`, `u16`, `u32`, `u64`
- `f32`, `f64`
- `ptr`
- `cstring`
- `buffer`

### Type Mapping

| FFI type | JS/TS value |
| --- | --- |
| `void` | `void` |
| `bool` | `boolean` |
| integer and float types | `number` |
| `cstring` | `string` |
| `ptr` | `Pointer` |
| `buffer` | `Pointer` |

`Pointer` is an opaque handle. Treat it as a native pointer value, not as a normal JavaScript object or array.

### Raw FFI Limits

`std:ffi` is intentionally narrow. It does not promise:

- struct layout support
- callbacks from native code into JavaScript
- variadic function support
- automatic ownership management for native memory
- safe typed views over arbitrary native buffers
- C++ ABI compatibility

Users are responsible for matching the native ABI correctly and for native memory lifetime unless a higher-level wrapper handles it.

## `std:ffi` `c` Export

The `c` export on `std:ffi` is a constrained wrapper-author layer that builds on top of raw `std:ffi`.

At the TypeScript level, the `c` export on `std:ffi` uses the declaration string plus explicit
function policies to infer:

- known bound symbol names
- direct scalar and C-string return types
- direct scalar and C-string argument types
- exact visible argument arity from the declaration string
- hidden out params and hidden buffer length params removed from the JS call surface
- handle-pointer params such as `sqlite3* db`
- buffer-policy pointer params as element-aware typed-array or `ArrayBuffer` inputs
- POD struct parameter shapes inferred in TypeScript from inline `typedef struct { ... } Name;` declarations
- opaque handle return types for `kind: "handle"`
- lifted opaque-handle out returns such as `sqlite3** out_db`

It currently supports:

- C-like declaration strings
- function prototypes
- `typedef struct Name Name;` opaque handle declarations
- `typedef struct { ... } Name;` declarations in the parser
- direct numeric and string arguments
- pointer-plus-length buffer policies for caller-provided typed arrays
- opaque-handle wrapping for pointer returns
- hidden out-parameter lifting for `T** out`-style APIs
- status/null/sentinel error policies
- explicit search strategy and search-path overrides for library loading
- pointer-to-POD-struct marshaling from JS objects, including nested named POD structs whose fields are themselves primitive-only POD shapes

Example:

```ts
import { c } from 'std:ffi'

const sqlite = c.decl('sqlite3', `
  typedef struct sqlite3 sqlite3;
  int32_t sqlite3_open(const char* filename, sqlite3** out_db);
  int32_t sqlite3_close(sqlite3* db);
  const char* sqlite3_errmsg(sqlite3* db);
`, {
  handles: {
    sqlite3: { close: 'sqlite3_close' },
  },
  functions: {
    sqlite3_open: {
      returns: { out: 'out_db' },
      errors: {
        kind: 'status',
        ok: 0,
        code: 'io_error',
        message: { from: 'sqlite3_errmsg', arg: 'out_db' },
      },
    },
  },
})

const db = sqlite.sqlite3_open(':memory:')
db.close()
sqlite.close()
```

Buffer-policy example:

```ts
import { c } from 'std:ffi'

const libc = c.decl('c', `
  void* memcpy(void* dest, const void* src, size_t n);
`, {
  functions: {
    memcpy: {
      buffers: {
        dest: { length: 'n', element: 'u8', direction: 'out' },
        src: { length: 'n', element: 'u8' },
      },
    },
  },
})

const dest = new Uint8Array(4)
const src = new Uint8Array([10, 20, 30, 40])
libc.memcpy(dest, src)
```

POD struct pointer example:

```ts
import { c } from 'std:ffi'

const libc = c.decl('c', `
  typedef struct {
    int64_t tv_sec;
    int32_t tv_usec;
  } timeval;

  int32_t gettimeofday(timeval* tv, void* tz);
`)

const tv = { tv_sec: 0, tv_usec: 0 }
libc.gettimeofday(tv, 0)
console.log(tv.tv_sec, tv.tv_usec)
```

Search override example:

```ts
import { c } from 'std:ffi'

const lib = c.decl('System.B', `
  size_t strlen(const char* s);
`, {
  search: {
    strategy: 'relative-first',
    paths: ['/usr/lib'],
  },
})

console.log(lib.strlen('hello'))
```

### `c` Export Grammar Boundary

The current declaration subset is intentionally narrow. Supported input includes:

- function prototypes
- `typedef struct Name Name;`
- `typedef struct { ... } Name;`
- fixed-width integer names and `size_t`
- pointer depth such as `*` and `**`

The current parser explicitly rejects:

- preprocessor directives such as `#include`
- variadic functions
- function-pointer parameters
- unions
- platform-width integer names such as `int` and `long`

### Current Limits

The `c` export on `std:ffi` is still a constrained C layer, not general native integration. It does not yet promise:

- arbitrary header compatibility
- callback-heavy APIs
- C++ ABI support
- macro-driven binding generation at runtime
- runtime execution support for named by-value POD struct params and returns
- runtime struct marshaling beyond primitive fields and nested named POD structs
- native-owned output buffer lifetimes or automatic free policies
- general-purpose buffer/view ergonomics beyond the current typed-array policy model
