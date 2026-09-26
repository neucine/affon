declare function setDevice(device: "cpu" | "metal" | "cuda" | `cuda:${number}`): void

interface Console {
  log(...args: unknown[]): void
  error(...args: unknown[]): void
  warn(...args: unknown[]): void
}

declare var console: Console

type RuntimeErrorCode =
  | "invalid_arg"
  | "missing_arg"
  | "shape_mismatch"
  | "invalid_shape"
  | "invalid_dtype"
  | "out_of_memory"
  | "device_mismatch"
  | "device_error"
  | "grad_error"
  | "io_error"
  | "cancelled"
  | "invalid_state"
  | "not_implemented"
  | "assertion"
  | "unsupported_lowering"
  | "internal"
  | "thread_pool_unavailable"
  | "metal_unavailable"

type AffonErrorCode = RuntimeErrorCode

declare class RuntimeError extends Error {
  readonly name: "RuntimeError"
  readonly code: RuntimeErrorCode
  readonly nativeStack?: string

  constructor(code: RuntimeErrorCode, message: string)
}

declare class AffonError extends Error {
  readonly name: "AffonError"
  readonly code: AffonErrorCode
  readonly nativeStack?: string
  constructor(code: AffonErrorCode, message: string)
}

/** Schedule a callback after the given delay in milliseconds. */
declare function setTimeout(callback: () => void, delay?: number): number
/** Cancel a pending timer. */
declare function clearTimeout(id: number): void
