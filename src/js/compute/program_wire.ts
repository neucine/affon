import type {
  NativeCompileOptions,
  NativeDType,
  NativeProgram,
  NativeProgramNode,
  NativeSessionOptions,
} from "affon:core/native"
import type {
  CompileOptions,
  ProgramDType,
  ProgramInspection,
  ProgramNode,
  SessionOptions,
  SessionTelemetry,
} from "affon:_internal/compute/program"

const dtypeValues: Readonly<Record<ProgramDType, NativeDType>> = Object.freeze({
  f32: 1,
  f64: 2,
  i64: 3,
  u32: 4,
  bf16: 5,
})

const operationValues: Readonly<Record<string, number>> = Object.freeze({
  add: 1,
  sub: 2,
  mul: 3,
  div: 4,
  matmul: 5,
  abs: 6,
  neg: 7,
  sign: 8,
  relu: 9,
  sum_all: 10,
  mean_all: 11,
  sum_axis: 12,
  mean_axis: 13,
  min_all: 14,
  max_all: 15,
  argmin_all: 16,
  argmax_all: 17,
  min_axis: 18,
  max_axis: 19,
  argmin_axis: 20,
  argmax_axis: 21,
  variance_all: 22,
  std_all: 23,
  variance_axis: 24,
  std_axis: 25,
  reshape: 26,
  permute: 27,
  transpose: 28,
  squeeze: 29,
  unsqueeze: 30,
  contiguous: 31,
  slice: 32,
  cast: 33,
  scatter_add: 34,
  gather: 35,
  index_select: 36,
  embedding: 37,
  one_hot: 38,
  topk_values: 39,
  topk_indices: 40,
  top_k_values: 39,
  top_k_indices: 40,
  eq: 41,
  lt: 42,
  gt: 43,
  where: 44,
  masked_fill: 45,
  cat: 46,
  stack: 47,
  clamp: 48,
  exp: 49,
  log: 50,
  sqrt: 51,
  sigmoid: 52,
  silu: 53,
  tanh: 54,
  erf: 55,
  gelu: 56,
  sin: 57,
  cos: 58,
  softmax: 59,
  log_softmax: 60,
  cross_entropy: 61,
  cross_entropy_indexed: 62,
  layer_norm: 63,
  rms_norm: 64,
  quantized_matmul: 65,
  log_softmax_nll: 66,
  dot: 67,
  reduce_to_shape: 68,
})

const reductionOperations = new Set(["sum", "mean", "min", "max", "variance", "std", "argmin", "argmax"])

function optionNumber(node: ProgramNode, name: string): number | undefined {
  const value = node.options?.[name]
  return typeof value === "number" ? value : undefined
}

function optionBoolean(node: ProgramNode, name: string, fallback: boolean): boolean {
  const value = node.options?.[name]
  return typeof value === "boolean" ? value : fallback
}

function optionNumbers(node: ProgramNode, name: string): readonly number[] | undefined {
  const value = node.options?.[name]
  if (!Array.isArray(value)) return undefined
  return Object.freeze(value.map(item => {
    if (typeof item !== "number" || !Number.isSafeInteger(item)) throw new TypeError(`Program option ${name} must contain integers`)
    return item
  }))
}

function nativeOperation(node: ProgramNode): number | undefined {
  const operation = node.op
  if (operation === undefined) return undefined
  if (reductionOperations.has(operation)) {
    const axis = optionNumber(node, "axis")
    const suffix = axis === undefined ? "_all" : "_axis"
    return operationValues[`${operation}${suffix}`]
  }
  if (operation === "dense_cross_entropy") return operationValues.cross_entropy
  if (operation === "cross_entropy") return operationValues.cross_entropy_indexed
  return operationValues[operation]
}

function elementCount(shape: readonly number[]): number {
  let count = 1
  for (const dimension of shape) count *= dimension
  return count
}

function constantBytes(value: readonly unknown[], dtype: ProgramDType): ArrayBuffer {
  const variableDtype = dtype as ProgramDType
  const width = { bf16: 2, f32: 4, f64: 8, u32: 4, i64: 8 }[variableDtype]
  const flat: number[] = []
  const visit = (item: unknown): void => {
    if (typeof item === "number") flat.push(item)
    else if (Array.isArray(item)) for (const child of item) visit(child)
    else throw new TypeError("Program constant contains a non-numeric value")
  }
  visit(value)
  const buffer = new ArrayBuffer(flat.length * width)
  const view = new DataView(buffer)
  let offset = 0
  for (const item of flat) {
    switch (variableDtype) {
      case "f32":
        view.setFloat32(offset, item, true)
        break
      case "f64":
        view.setFloat64(offset, item, true)
        break
      case "i64":
        if (!Number.isSafeInteger(item)) throw new TypeError("Program i64 constant must contain safe integers")
        view.setBigInt64(offset, BigInt(item), true)
        break
      case "u32":
        if (!Number.isInteger(item) || item < 0 || item > 0xffffffff) throw new TypeError("Program u32 constant is out of range")
        view.setUint32(offset, item, true)
        break
      case "bf16": {
        const f32 = new Float32Array(1)
        f32[0] = item
        view.setUint16(offset, new Uint32Array(f32.buffer)[0] >>> 16, true)
        break
      }
    }
    offset += width
  }
  return buffer
}

function nativeNode(node: ProgramNode, provenance: ReadonlyMap<string, number>): NativeProgramNode {
  let kind: NativeProgramNode["kind"]
  switch (node.kind) {
    case "argument": kind = 0; break
    case "parameter": kind = 1; break
    case "state": kind = 2; break
    case "constant": kind = 3; break
    case "operation": kind = 4; break
    case "gradient": kind = 5; break
    default: throw new TypeError(`unsupported Program node kind during native compile: ${node.kind}`)
  }
  const result: {
    -readonly [K in keyof NativeProgramNode]: NativeProgramNode[K]
  } = {
    id: node.id,
    kind,
    dtype: dtypeValues[node.spec.dtype],
    shape: Object.freeze([...node.spec.shape]),
  }
  if (node.kind === "operation" || node.kind === "gradient") {
    result.operands = Object.freeze([...(node.operands ?? [])])
  }
  if (node.kind === "operation") {
    const operation = nativeOperation(node)
    if (operation === undefined) throw new TypeError(`unsupported Program operation during native compile: ${node.op}`)
    result.op = operation
  }
  const axis = optionNumber(node, "axis")
  if (axis !== undefined) result.axis = axis
  if (reductionOperations.has(node.op ?? "")) result.keep_dims = optionBoolean(node, "keep_dims", false) ? 1 : 0
  const minimum = optionNumber(node, "min")
  if (minimum !== undefined) result.min = minimum
  const maximum = optionNumber(node, "max")
  if (maximum !== undefined) result.max = maximum
  const fillValue = optionNumber(node, "value")
  if (fillValue !== undefined) result.fill_value = fillValue
  const epsilon = optionNumber(node, "epsilon")
  if (epsilon !== undefined) result.epsilon = epsilon
  const k = optionNumber(node, "k")
  if (k !== undefined) result.k = k
  const numClasses = optionNumber(node, "num_classes")
  if (numClasses !== undefined) result.num_classes = numClasses
  const classAxis = optionNumber(node, "class_axis")
  if (classAxis !== undefined) result.class_axis = classAxis
  const bits = optionNumber(node, "bits")
  if (bits !== undefined) result.bits = bits
  const groupSize = optionNumber(node, "group_size")
  if (groupSize !== undefined) result.group_size = groupSize
  const axes = optionNumbers(node, "axes")
  if (axes !== undefined) result.axes = axes
  const permutation = optionNumbers(node, "permutation")
  if (permutation !== undefined) result.permutation = permutation
  const rawRanges = node.options?.ranges
  if (Array.isArray(rawRanges)) {
    result.ranges = Object.freeze(rawRanges.map(range => {
      if (!range || typeof range !== "object" || Array.isArray(range)) throw new TypeError("Program slice range must be an object")
      const value = range as Readonly<Record<string, unknown>>
      if (typeof value.start !== "number" || typeof value.stop !== "number") throw new TypeError("Program slice range requires numeric start and stop")
      const step = typeof value.step === "number" ? value.step : 1
      return Object.freeze({ start: value.start, stop: value.stop, step })
    }))
  }
  if (node.op === "topk_values" || node.op === "topk_indices") {
    result.largest = optionBoolean(node, "largest", true) ? 1 : 0
    result.sorted = optionBoolean(node, "sorted", true) ? 1 : 0
  }
  if (node.kind === "constant") {
    if (typeof node.value === "number") result.constant = node.value
    else if (Array.isArray(node.value)) {
      result.constant_bytes = constantBytes(node.value, node.spec.dtype)
      if (result.constant_bytes.byteLength !== elementCount(node.spec.shape) * ({ bf16: 2, f32: 4, f64: 8, u32: 4, i64: 8 }[node.spec.dtype])) {
        throw new TypeError(`Program constant byte count does not match node ${node.id}`)
      }
    } else throw new TypeError(`Program constant node ${node.id} has no value`)
  }
  if (node.kind === "gradient") {
    const target = node.options?.with_respect_to
    if (typeof target !== "string") throw new TypeError(`Program gradient node ${node.id} has no target`)
    const targetId = provenance.get(target)
    if (targetId === undefined) throw new TypeError(`Program gradient node ${node.id} targets an unknown value`)
    result.gradient_target = targetId
  }
  return Object.freeze(result)
}

export function nativeProgram(inspection: ProgramInspection): NativeProgram {
  if (inspection.kind !== "authored" && inspection.kind !== "gradient") {
    throw new TypeError(`unsupported Program kind during native compile: ${inspection.kind}`)
  }
  const provenance = new Map<string, number>()
  for (const node of inspection.nodes) if (node.provenance !== undefined) provenance.set(node.provenance, node.id)
  return Object.freeze({
    schema_version: 2,
    kind: inspection.kind === "authored" ? 0 : 1,
    nodes: Object.freeze(inspection.nodes.map(node => nativeNode(node, provenance))),
    outputs: Object.freeze([...inspection.outputs]),
  })
}

function searchLimit(options: Required<CompileOptions>, name: keyof NonNullable<CompileOptions["searchLimits"]>): number | undefined {
  return options.searchLimits[name]
}

export function nativeCompileOptions(options: Required<CompileOptions>): NativeCompileOptions {
  const result: {
    -readonly [K in keyof NativeCompileOptions]: NativeCompileOptions[K]
  } = {
    schema_version: 1,
    optimization_level: options.optimizationLevel === "none" ? 0 : 1,
    optimization_goal: {
      memory: 1,
      balanced: 2,
      latency: 3,
      throughput: 4,
    }[options.optimizationGoal],
    numerical_policy: {
      exact_only: 0,
      backend_equivalent: 1,
      approximate: 2,
    }[options.numericalPolicy],
    explanation_level: options.explanationLevel === "summary" ? 0 : 1,
    dense_inputs: options.denseInputs ? 1 : 0,
    residual_policy: {
      retain: 0,
      automatic: 1,
      recompute: 2,
    }[options.residualPolicy],
  } as typeof result
  const fields: readonly [keyof typeof result, keyof NonNullable<CompileOptions["searchLimits"]>][] = [
    ["structural_states", "structuralStates"],
    ["structural_expansions", "structuralExpansions"],
    ["complete_evaluations", "completeEvaluations"],
    ["physical_work", "physicalWork"],
    ["ready_work", "readyWork"],
    ["max_peak_active_bytes", "maxPeakActiveBytes"],
    ["max_invocation_scratch_bytes", "maxInvocationScratchBytes"],
  ]
  for (const [nativeName, optionName] of fields) {
    const value = searchLimit(options, optionName)
    if (value !== undefined) (result as any)[nativeName] = value
  }
  return Object.freeze(result)
}

export function compileOptionsKey(options: Required<CompileOptions>): string {
  const limits = options.searchLimits
  return [
    options.optimizationLevel,
    options.numericalPolicy,
    options.optimizationGoal,
    options.explanationLevel,
    options.residualPolicy,
    options.denseInputs ? 1 : 0,
    limits.structuralStates ?? "-",
    limits.structuralExpansions ?? "-",
    limits.completeEvaluations ?? "-",
    limits.physicalWork ?? "-",
    limits.readyWork ?? "-",
    limits.maxPeakActiveBytes ?? "-",
    limits.maxInvocationScratchBytes ?? "-",
  ].join(":")
}

export function nativeSessionOptions(
  determinism: "strict" | "allow_nondeterministic",
  telemetry: false | SessionTelemetry,
): NativeSessionOptions {
  if (telemetry === false) {
    return Object.freeze({
      schema_version: 1,
      determinism: determinism === "strict" ? 0 : 1,
      telemetry_enabled: 0,
      compute_timing: 0,
      hardware_metrics: 0,
      profiler_markers: 0,
      bridge_timing: 0,
    })
  }
  const result: {
    -readonly [K in keyof NativeSessionOptions]: NativeSessionOptions[K]
  } = {
    schema_version: 1,
    determinism: determinism === "strict" ? 0 : 1,
    telemetry_enabled: 1,
    compute_timing: telemetry.computeTiming === "invocation" ? 1 : telemetry.computeTiming === "kernel" ? 2 : 0,
    hardware_metrics: telemetry.hardwareMetrics ? 1 : 0,
    profiler_markers: telemetry.profilerMarkers ? 1 : 0,
    bridge_timing: telemetry.bridgeTiming ? 1 : 0,
  }
  if (telemetry.captureRecords !== null) result.capture_records = telemetry.captureRecords
  return Object.freeze(result)
}
