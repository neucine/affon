import native from "affon:compute/native"
import type { LRSchedule, Optimizer } from "affon:optim"

export type ProgramDType = "f32" | "f64" | "i64"
export type Device = "cpu" | "metal" | "cuda" | `cuda:${number}`
export type ProgramShape = readonly number[]
export type TensorData = number | readonly TensorData[]
export type TensorInitializerValue = TensorData | Readonly<{
  shape: readonly number[]
  dtype: ProgramDType
  to_array(): TensorData
}>
export type ProgramArguments = Readonly<Record<string, Tensor>>
export type SliceRange = Readonly<{ start: number; stop: number; step?: number }>
export type TensorSpec = Readonly<{
  dtype: ProgramDType
  shape: ProgramShape
  axes?: readonly string[]
}>

export type TensorValueOptions = Readonly<{
  dtype?: ProgramDType
  axes?: readonly string[]
}>

export interface Tensor {
  readonly shape: readonly number[]
  readonly axes?: readonly string[]
  readonly ndim: number
  readonly dtype: "f32" | "f64" | "i64"
  readonly device: Device
  readonly disposed: boolean
  item(): number
  to_array(): TensorData
  toString(): string
  repr(): string
  dispose(): void
  [Symbol.dispose](): void
}

type FormalRole = "argument" | "parameter" | "state" | "constant" | "intermediate"
type ProgramValue = TensorData
type NodeKind = FormalRole | "operation" | "composition" | "gradient"
export type ProgramPathSegment = Readonly<{
  program: string
  instance: string
}>
export type ProgramPath = readonly ProgramPathSegment[]

type ProgramNode = Readonly<{
  id: number
  kind: NodeKind
  path: ProgramPath
  role?: Exclude<FormalRole, "intermediate">
  name?: string
  provenance?: string
  spec: TensorSpec
  op?: string
  operands?: readonly number[]
  options?: Readonly<Record<string, unknown>>
  value?: ProgramValue
}>

const EMPTY_PROGRAM_PATH: ProgramPath = Object.freeze([])

function composedPath(program: string, instance: string, path: ProgramPath): ProgramPath {
  return Object.freeze([Object.freeze({ program, instance }), ...path])
}

export type ProgramFormal = Readonly<{
  name: string
  role: "argument" | "parameter" | "state" | "constant"
  spec: TensorSpec
  provenance: string
}>

export type Initializer =
  | Readonly<{ kind: "zeros" | "ones" }>
  | Readonly<{ kind: "constant"; value: number }>
  | Readonly<{ kind: "uniform"; min: number; max: number }>
  | Readonly<{ kind: "normal"; mean?: number; standard_deviation?: number }>
  | Readonly<{ kind: "xavier_uniform" | "xavier_normal" }>

export type ProgramInspection = Readonly<{
  name: string
  provenance: string
  kind: string
  arguments: readonly ProgramFormal[]
  parameters: readonly ProgramFormal[]
  state: readonly ProgramFormal[]
  constants: readonly ProgramFormal[]
  nodes: readonly ProgramNode[]
  outputs: readonly number[]
  transitions: readonly Readonly<{ kind: "optimize"; optimizer: Optimizer; parameters: readonly string[] }>[]
}>

const PROGRAM = Symbol.for("affon.compute.program")
const FORMAL = Symbol.for("affon.compute.formal_tensor")
const BUILDER = Symbol.for("affon.compute.program_builder")
const EMIT = Symbol("affon.compute.emit_operation")
const FORMAL_OPERATION = Symbol("affon.compute.formal_operation")
const FORMAL_SCALAR = Symbol("affon.compute.formal_scalar")
const parameterUpdateSources = new WeakMap<Program, {
  source: Program<Record<string, FormalTensor>, FormalTensor>
  gradients: Program<Record<string, FormalTensor>, FormalTensor | readonly FormalTensor[]>
  optimizer: Optimizer
  parameters: readonly string[]
}>()

function freezeSpec(dtype: ProgramDType, shape: readonly number[], axes?: readonly string[]): TensorSpec {
  if (dtype !== "f32" && dtype !== "f64" && dtype !== "i64") throw new TypeError(`unsupported TensorSpec dtype: ${dtype}`)
  if (!Array.isArray(shape) || shape.some(value => !Number.isSafeInteger(value) || value < 0)) {
    throw new TypeError("TensorSpec shape must contain non-negative integers")
  }
  let elements = 1
  for (const size of shape) {
    elements *= size
    if (!Number.isSafeInteger(elements)) throw new TypeError("TensorSpec element count exceeds the safe integer range")
  }
  if (axes !== undefined && axes.length !== shape.length) throw new TypeError("TensorSpec axes must match shape rank")
  if (axes?.some(axis => typeof axis !== "string" || axis.length === 0)) throw new TypeError("TensorSpec axes must be non-empty strings")
  return Object.freeze({ dtype, shape: Object.freeze([...shape]), ...(axes ? { axes: Object.freeze([...axes]) } : {}) })
}

let defaultSessionValue: Session | undefined
const SESSION_TENSOR = Symbol("affon.compute.session_tensor")
const SESSION_FULL = Symbol("affon.compute.session_full")

function defaultSession(): Session {
  if (!defaultSessionValue) defaultSessionValue = new Session({ device: native.defaultDevice() })
  return defaultSessionValue
}

function checkedTensorShape(shape: readonly number[]): readonly number[] {
  if (!Array.isArray(shape) || shape.some(size => !Number.isSafeInteger(size) || size < 0)) {
    throw new TypeError("Tensor shape must contain non-negative integers")
  }
  let elements = 1
  for (const size of shape) {
    elements *= size
    if (!Number.isSafeInteger(elements)) throw new TypeError("Tensor element count exceeds the safe integer range")
  }
  return shape
}

export const Tensor = Object.freeze({
  spec(dtype: ProgramDType, shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec {
    return freezeSpec(dtype, shape, options.axes)
  },
  f32(shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec { return freezeSpec("f32", shape, options.axes) },
  f64(shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec { return freezeSpec("f64", shape, options.axes) },
  i64(shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec { return freezeSpec("i64", shape, options.axes) },
  from(values: TensorData, options: TensorValueOptions = {}): Tensor {
    return defaultSession().tensor(values, options)
  },
  full(shape: readonly number[], value: number, options: TensorValueOptions = {}): Tensor {
    if (!Number.isFinite(value)) throw new TypeError("Tensor.full value must be finite")
    if (options.dtype === "i64" && !Number.isSafeInteger(value)) throw new TypeError("Tensor.full i64 value must be a safe integer")
    return defaultSession()[SESSION_FULL](checkedTensorShape(shape), value, options)
  },
  zeros(shape: readonly number[], options: TensorValueOptions = {}): Tensor {
    return defaultSession()[SESSION_FULL](checkedTensorShape(shape), 0, options)
  },
  ones(shape: readonly number[], options: TensorValueOptions = {}): Tensor {
    return defaultSession()[SESSION_FULL](checkedTensorShape(shape), 1, options)
  },
  arange(start: number, end?: number, step = 1, options: TensorValueOptions = {}): Tensor {
    const first = end === undefined ? 0 : start
    const stop = end === undefined ? start : end
    if (![first, stop, step].every(Number.isFinite) || step === 0) throw new TypeError("Tensor.arange requires finite bounds and a non-zero step")
    const length = Math.max(0, Math.ceil((stop - first) / step))
    return defaultSession().tensor(Array.from({ length }, (_, index) => first + index * step), options)
  },
  linspace(start: number, end: number, steps = 100, options: TensorValueOptions = {}): Tensor {
    if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isSafeInteger(steps) || steps <= 0) throw new TypeError("Tensor.linspace requires finite bounds and positive steps")
    const values = steps === 1 ? [start] : Array.from({ length: steps }, (_, index) => start + (end - start) * index / (steps - 1))
    return defaultSession().tensor(values, options)
  },
  rand(shape: readonly number[], options: TensorValueOptions & { seed?: number } = {}): Tensor {
    const { seed = 0, ...tensorOptions } = options
    const spec = freezeSpec(tensorOptions.dtype ?? "f32", checkedTensorShape(shape), tensorOptions.axes)
    if (spec.dtype === "i64") throw new TypeError("Tensor.rand requires a floating-point dtype")
    return defaultSession().tensor(initializedValue(spec, { kind: "uniform", min: 0, max: 1 }, seededRandom(seed)), tensorOptions)
  },
  randn(shape: readonly number[], options: TensorValueOptions & { seed?: number } = {}): Tensor {
    const { seed = 0, ...tensorOptions } = options
    const spec = freezeSpec(tensorOptions.dtype ?? "f32", checkedTensorShape(shape), tensorOptions.axes)
    if (spec.dtype === "i64") throw new TypeError("Tensor.randn requires a floating-point dtype")
    return defaultSession().tensor(initializedValue(spec, { kind: "normal" }, seededRandom(seed)), tensorOptions)
  },
})

function sameSpec(left: TensorSpec, right: TensorSpec): boolean {
  return left.dtype === right.dtype
    && left.shape.length === right.shape.length
    && left.shape.every((value, index) => value === right.shape[index])
    && (left.axes === undefined) === (right.axes === undefined)
    && (!left.axes || left.axes.every((value, index) => value === right.axes![index]))
}

function scalarSpec(dtype: ProgramDType): TensorSpec { return freezeSpec(dtype, [1]) }

function elementCount(shape: readonly number[]): number { return shape.reduce((product, size) => product * size, 1) }

function freezeMetadata<T>(value: T): T {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value
  if (Array.isArray(value)) return Object.freeze(value.map(item => freezeMetadata(item))) as T
  const entries = Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, freezeMetadata(item)])
  return Object.freeze(Object.fromEntries(entries)) as T
}

function snapshotTensorData(value: unknown, options: { label: string; finite?: boolean; integer?: boolean }): { data: TensorData; shape: readonly number[] } {
  if (typeof value === "number") {
    if (options.finite && !Number.isFinite(value)) throw new TypeError(`${options.label} must contain finite numbers`)
    if (options.integer && !Number.isSafeInteger(value)) throw new TypeError(`${options.label} i64 values must be safe integers`)
    return { data: value, shape: Object.freeze([]) }
  }
  if (!Array.isArray(value)) throw new TypeError(`${options.label} must be a rectangular numeric value`)
  const children = value.map(item => snapshotTensorData(item, options))
  const childShape = children[0]?.shape ?? []
  if (children.some(child => child.shape.length !== childShape.length || child.shape.some((size, index) => size !== childShape[index]))) {
    throw new TypeError(`${options.label} must be a rectangular numeric value`)
  }
  return {
    data: Object.freeze(children.map(child => child.data)),
    shape: Object.freeze([children.length, ...childShape]),
  }
}

function broadcastShape(left: readonly number[], right: readonly number[]): number[] {
  const rank = Math.max(left.length, right.length)
  const shape = Array<number>(rank)
  for (let offset = 1; offset <= rank; offset++) {
    const a = left.at(-offset) ?? 1
    const b = right.at(-offset) ?? 1
    if (a !== b && a !== 1 && b !== 1) throw new TypeError(`cannot broadcast shapes [${left}] and [${right}]`)
    shape[rank - offset] = Math.max(a, b)
  }
  return shape
}

function broadcastAxes(left: TensorSpec, right: TensorSpec, shape: readonly number[]): readonly string[] | undefined {
  if (!left.axes || !right.axes) return undefined
  const rank = shape.length
  const axes = Array<string>(rank)
  for (let offset = 1; offset <= rank; offset++) {
    const aSize = left.shape.at(-offset) ?? 1
    const bSize = right.shape.at(-offset) ?? 1
    const a = left.axes.at(-offset)
    const b = right.axes.at(-offset)
    if (a && b && aSize !== 1 && bSize !== 1 && a !== b) throw new TypeError(`broadcast axis mismatch: ${a} and ${b}`)
    axes[rank - offset] = (aSize === 1 ? b : bSize === 1 ? a : a ?? b) ?? `axis${rank - offset}`
  }
  return axes
}

function arithmeticSpec(left: TensorSpec, right: TensorSpec): TensorSpec {
  if (left.dtype !== right.dtype) throw new TypeError(`operation dtype mismatch: ${left.dtype} and ${right.dtype}`)
  const shape = broadcastShape(left.shape, right.shape)
  return freezeSpec(left.dtype, shape, broadcastAxes(left, right, shape))
}

function matmulSpec(left: TensorSpec, right: TensorSpec): TensorSpec {
  if (left.dtype !== right.dtype) throw new TypeError(`matmul dtype mismatch: ${left.dtype} and ${right.dtype}`)
  if (left.shape.length < 2 || right.shape.length < 2) throw new TypeError("matmul requires tensors with rank at least two")
  if (left.shape[left.shape.length - 1] !== right.shape[right.shape.length - 2]) throw new TypeError("matmul contracting dimensions must match")
  const batch = broadcastShape(left.shape.slice(0, -2), right.shape.slice(0, -2))
  const shape = [...batch, left.shape.at(-2)!, right.shape.at(-1)!]
  let axes: readonly string[] | undefined
  if (left.axes && right.axes) {
    const batchSpecLeft = freezeSpec(left.dtype, left.shape.slice(0, -2), left.axes.slice(0, -2))
    const batchSpecRight = freezeSpec(right.dtype, right.shape.slice(0, -2), right.axes.slice(0, -2))
    axes = [...(broadcastAxes(batchSpecLeft, batchSpecRight, batch) ?? []), left.axes.at(-2)!, right.axes.at(-1)!]
  }
  return freezeSpec(left.dtype, shape, axes)
}

export class FormalTensor<S extends TensorSpec = TensorSpec> {
  readonly [FORMAL] = true
  readonly spec: S
  readonly id: number
  readonly name?: string
  readonly role: FormalRole
  private readonly owner: ProgramBuilder

  constructor(owner: ProgramBuilder, id: number, spec: S, role: FormalRole, name?: string) {
    this.owner = owner
    this.id = id
    this.spec = spec
    this.role = role
    this.name = name
    Object.freeze(this)
  }

  private binary(op: string, other: FormalTensor, result: TensorSpec = this.spec): FormalTensor {
    return this.owner[EMIT](op, [this, other], result)
  }

  [FORMAL_OPERATION](op: string, operands: readonly FormalTensor[], options: readonly unknown[]): FormalTensor {
    switch (op) {
      case "add": case "sub": case "mul": case "div": case "eq": case "lt": case "gt": case "matmul": case "dot": case "embedding": case "gather": return this.dispatchBinary(op, operands[1], options)
      case "where": return this.#where(operands[1], operands[2])
      case "stack": return this.#stack(operands, options[0] as number)
      case "one_hot": return this.#oneHot(options[0] as number)
      case "cross_entropy": return this.#cross_entropy(operands[1])
      case "masked_fill": return this.#masked_fill(operands[1], options[0] as number)
      case "index_select": return this.#index_select(options[0] as number, operands[1])
      case "softmax": return this.#softmax(options[0] as number)
      case "cast": return this.#cast(options[0] as ProgramDType)
      case "sum": return this.#sum(options[0] as number | undefined, options[1] as boolean | undefined)
      case "mean": return this.#mean(options[0] as number | undefined, options[1] as boolean | undefined)
      case "min": case "max": case "variance": case "std": case "argmin": case "argmax": return this.owner.reduction(op, this, options[0] as number | undefined, options[1] as boolean | undefined)
      case "reshape": return this.#reshape(options[0] as readonly number[], options[1] as readonly string[] | undefined)
      case "transpose": return this.#transpose(options[0] as readonly number[] | undefined)
      case "slice": return this.#slice(options[0] as readonly SliceRange[])
      case "squeeze": return this.#squeeze(options[0] as number | undefined)
      case "unsqueeze": return this.#unsqueeze(options[0] as number, options[1] as string | undefined)
      case "layer_norm": return this.#layer_norm(options[0] as number, options[1] as number | undefined)
      case "neg": return this.#neg(); case "abs": return this.#abs(); case "exp": return this.#exp(); case "log": return this.#log(); case "sign": return this.#sign()
      case "clamp": return this.#clamp(options[0] as number, options[1] as number)
      case "sqrt": return this.#sqrt(); case "relu": return this.#relu(); case "sigmoid": return this.#sigmoid(); case "silu": return this.#silu()
      case "tanh": return this.#tanh(); case "erf": return this.#erf(); case "gelu": return this.#gelu(); case "contiguous": return this.#contiguous()
      default: throw new TypeError(`unsupported formal operation: ${op}`)
    }
  }

  private dispatchBinary(op: string, other: FormalTensor, options: readonly unknown[] = []): FormalTensor {
    switch (op) {
      case "add": return this.#add(other); case "sub": return this.#sub(other); case "mul": return this.#mul(other); case "div": return this.#div(other)
      case "eq": case "lt": case "gt": return this.#compare(op, other)
      case "matmul": return this.#matmul(other); case "dot": return this.#dot(other); case "embedding": return this.#embedding(other)
      case "gather": return this.#gather(options[0] as number, other)
      default: throw new TypeError(`unsupported binary formal operation: ${op}`)
    }
  }

  #add(other: FormalTensor): FormalTensor { return this.binary("add", other, arithmeticSpec(this.spec, other.spec)) }
  #sub(other: FormalTensor): FormalTensor { return this.binary("sub", other, arithmeticSpec(this.spec, other.spec)) }
  #mul(other: FormalTensor): FormalTensor { return this.binary("mul", other, arithmeticSpec(this.spec, other.spec)) }
  #div(other: FormalTensor): FormalTensor { return this.binary("div", other, arithmeticSpec(this.spec, other.spec)) }
  #compare(op: "eq" | "lt" | "gt", other: FormalTensor): FormalTensor {
    const result = arithmeticSpec(this.spec, other.spec)
    return this.binary(op, other, freezeSpec("i64", result.shape, result.axes))
  }
  #matmul(other: FormalTensor): FormalTensor { return this.binary("matmul", other, matmulSpec(this.spec, other.spec)) }
  #dot(other: FormalTensor): FormalTensor {
    if (!sameSpec(this.spec, other.spec)) throw new TypeError("dot requires identical tensor specs")
    return this.binary("dot", other, scalarSpec(this.spec.dtype))
  }
  #neg(): FormalTensor { return this.owner[EMIT]("neg", [this], this.spec) }
  #abs(): FormalTensor { return this.owner[EMIT]("abs", [this], this.spec) }
  #exp(): FormalTensor { return this.owner[EMIT]("exp", [this], this.spec) }
  #log(): FormalTensor { return this.owner[EMIT]("log", [this], this.spec) }
  #sqrt(): FormalTensor { return this.owner[EMIT]("sqrt", [this], this.spec) }
  #sign(): FormalTensor { return this.owner[EMIT]("sign", [this], this.spec) }
  #clamp(minimum: number, maximum: number): FormalTensor {
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum) throw new TypeError("clamp requires finite min <= max")
    return this.owner[EMIT]("clamp", [this], this.spec, { min: minimum, max: maximum })
  }
  #relu(): FormalTensor { return this.owner[EMIT]("relu", [this], this.spec) }
  #sigmoid(): FormalTensor { return this.owner[EMIT]("sigmoid", [this], this.spec) }
  #silu(): FormalTensor { return this.owner[EMIT]("silu", [this], this.spec) }
  #tanh(): FormalTensor { return this.owner[EMIT]("tanh", [this], this.spec) }
  #erf(): FormalTensor { return this.owner[EMIT]("erf", [this], this.spec) }
  #gelu(): FormalTensor { return this.owner[EMIT]("gelu", [this], this.spec) }
  #softmax(axis: number): FormalTensor {
    const normalized = axis < 0 ? this.spec.shape.length + axis : axis
    if (!Number.isInteger(normalized) || normalized < 0 || normalized >= this.spec.shape.length) throw new TypeError("softmax axis is out of range")
    return this.owner[EMIT]("softmax", [this], this.spec, { axis: normalized })
  }
  #sum(axis?: number, keep_dims = false): FormalTensor { return this.owner.reduction("sum", this, axis, keep_dims) }
  #mean(axis?: number, keep_dims = false): FormalTensor { return this.owner.reduction("mean", this, axis, keep_dims) }
  #reshape(shape: readonly number[], axes?: readonly string[]): FormalTensor {
    const result = freezeSpec(this.spec.dtype, shape, axes)
    if (elementCount(result.shape) !== elementCount(this.spec.shape)) throw new TypeError("reshape must preserve element count")
    return this.owner[EMIT]("reshape", [this], result, { shape: [...shape] })
  }
  #contiguous(): FormalTensor { return this.owner[EMIT]("contiguous", [this], this.spec) }
  #transpose(permutation?: readonly number[]): FormalTensor {
    const order = permutation ?? [...this.spec.shape.keys()].reverse()
    if (order.length !== this.spec.shape.length) throw new TypeError("transpose permutation must match tensor rank")
    if ([...order].sort((a, b) => a - b).some((axis, index) => axis !== index)) throw new TypeError("transpose permutation must contain every axis exactly once")
    return this.owner[EMIT]("transpose", [this], freezeSpec(this.spec.dtype, order.map(index => this.spec.shape[index]), this.spec.axes && order.map(index => this.spec.axes![index])), { permutation: [...order] })
  }
  #cast(dtype: ProgramDType): FormalTensor { return this.owner[EMIT]("cast", [this], freezeSpec(dtype, this.spec.shape, this.spec.axes), { dtype }) }
  #slice(ranges: readonly SliceRange[]): FormalTensor {
    if (ranges.length !== this.spec.shape.length) throw new TypeError("slice ranges must match tensor rank")
    const shape = ranges.map((range, axis) => {
      const step = range.step ?? 1
      if (!Number.isSafeInteger(range.start) || !Number.isSafeInteger(range.stop) || !Number.isSafeInteger(step) || step <= 0 || range.start < 0 || range.stop < range.start || range.stop > this.spec.shape[axis]) throw new TypeError("invalid slice range")
      const width = range.stop - range.start
      return width === 0 ? 0 : 1 + Math.floor((width - 1) / step)
    })
    return this.owner[EMIT]("slice", [this], freezeSpec(this.spec.dtype, shape, this.spec.axes), { ranges: ranges.map(range => ({ ...range, step: range.step ?? 1 })) })
  }
  #squeeze(axis?: number): FormalTensor {
    const normalized = axis === undefined ? undefined : axis < 0 ? this.spec.shape.length + axis : axis
    if (normalized !== undefined && (!Number.isInteger(normalized) || normalized < 0 || normalized >= this.spec.shape.length || this.spec.shape[normalized] !== 1)) throw new TypeError("squeeze axis must select a dimension of size one")
    const keep = this.spec.shape.map((_, index) => index).filter(index => this.spec.shape[index] !== 1 || (normalized !== undefined && index !== normalized))
    return this.owner[EMIT]("squeeze", [this], freezeSpec(this.spec.dtype, keep.map(index => this.spec.shape[index]), this.spec.axes && keep.map(index => this.spec.axes![index])), normalized === undefined ? {} : { axis: normalized })
  }
  #unsqueeze(axis: number, name?: string): FormalTensor {
    const normalized = axis < 0 ? this.spec.shape.length + axis + 1 : axis
    if (!Number.isInteger(normalized) || normalized < 0 || normalized > this.spec.shape.length) throw new TypeError("unsqueeze axis is out of range")
    const shape = [...this.spec.shape]
    shape.splice(normalized, 0, 1)
    const axes = this.spec.axes ? [...this.spec.axes] : undefined
    if (axes) axes.splice(normalized, 0, name ?? `axis${normalized}`)
    return this.owner[EMIT]("unsqueeze", [this], freezeSpec(this.spec.dtype, shape, axes), { axis: normalized })
  }
  #masked_fill(mask: FormalTensor, value: number): FormalTensor {
    if (mask.spec.dtype !== "i64") throw new TypeError("masked_fill mask must have i64 dtype")
    const shape = broadcastShape(this.spec.shape, mask.spec.shape)
    if (shape.length !== this.spec.shape.length || shape.some((size, index) => size !== this.spec.shape[index])) throw new TypeError("masked_fill mask must broadcast to the input shape")
    if (!Number.isFinite(value)) throw new TypeError("masked_fill value must be finite")
    return this.owner[EMIT]("masked_fill", [this, mask], this.spec, { value })
  }
  #layer_norm(axis: number, epsilon = 1e-5): FormalTensor {
    const normalized = axis < 0 ? this.spec.shape.length + axis : axis
    if (!Number.isInteger(normalized) || normalized < 0 || normalized >= this.spec.shape.length || !Number.isFinite(epsilon) || epsilon <= 0) throw new TypeError("invalid layer_norm options")
    return this.owner[EMIT]("layer_norm", [this], this.spec, { axis: normalized, epsilon })
  }
  #embedding(indices: FormalTensor): FormalTensor {
    if (indices.spec.dtype !== "i64" || this.spec.shape.length !== 2) throw new TypeError("embedding requires an i64 index tensor and a rank-two table")
    return this.owner[EMIT]("embedding", [this, indices], freezeSpec(this.spec.dtype, [...indices.spec.shape, ...this.spec.shape.slice(1)]))
  }
  #gather(axis: number, indices: FormalTensor): FormalTensor {
    const normalized = axis < 0 ? this.spec.shape.length + axis : axis
    if (indices.spec.dtype !== "i64" || indices.spec.shape.length !== this.spec.shape.length || !Number.isInteger(normalized) || normalized < 0 || normalized >= this.spec.shape.length) throw new TypeError("invalid gather operands")
    for (let index = 0; index < this.spec.shape.length; index++) if (index !== normalized && this.spec.shape[index] !== indices.spec.shape[index]) throw new TypeError("gather index shape must match input outside the selected axis")
    return this.owner[EMIT]("gather", [this, indices], freezeSpec(this.spec.dtype, indices.spec.shape), { axis: normalized })
  }
  #oneHot(classes: number): FormalTensor {
    if (this.spec.dtype !== "i64" || !Number.isSafeInteger(classes) || classes <= 0) throw new TypeError("one_hot requires i64 indices and a positive class count")
    return this.owner[EMIT]("one_hot", [this], freezeSpec("f32", [...this.spec.shape, classes]), { num_classes: classes })
  }
  #where(onTrue: FormalTensor, onFalse: FormalTensor): FormalTensor {
    if (this.spec.dtype !== "i64") throw new TypeError("where condition must have i64 dtype")
    const result = arithmeticSpec(onTrue.spec, onFalse.spec)
    const shape = broadcastShape(this.spec.shape, result.shape)
    if (shape.length !== result.shape.length || shape.some((size, index) => size !== result.shape[index])) throw new TypeError("where condition must broadcast to its values")
    return this.owner[EMIT]("where", [this, onTrue, onFalse], result)
  }
  #stack(values: readonly FormalTensor[], axis: number): FormalTensor {
    if (values.some(value => !sameSpec(value.spec, this.spec))) throw new TypeError("stack requires identical tensor specs")
    const normalized = axis < 0 ? this.spec.shape.length + axis + 1 : axis
    if (!Number.isInteger(normalized) || normalized < 0 || normalized > this.spec.shape.length) throw new TypeError("stack axis is out of range")
    const shape = [...this.spec.shape]
    shape.splice(normalized, 0, values.length)
    return this.owner[EMIT]("stack", values, freezeSpec(this.spec.dtype, shape), { axis: normalized })
  }
  #cross_entropy(labels: FormalTensor): FormalTensor {
    if (this.spec.dtype !== "f32" && this.spec.dtype !== "f64") throw new TypeError("cross_entropy logits must have a floating-point dtype")
    if (labels.spec.dtype !== "i64") throw new TypeError("cross_entropy labels must have i64 dtype")
    if (this.spec.shape.length === 0) throw new TypeError("cross_entropy logits must include a class dimension")
    const labelShape = this.spec.shape.slice(0, -1)
    if (labels.spec.shape.length !== labelShape.length || labels.spec.shape.some((size, index) => size !== labelShape[index])) {
      throw new TypeError("cross_entropy labels must match the logits shape without its class dimension")
    }
    const result = scalarSpec(this.spec.dtype)
    if (this.spec.shape.length <= 2) return this.owner[EMIT]("cross_entropy", [this, labels], result, { reduction: "mean", class_axis: -1 })
    const classes = this.spec.shape.at(-1)!
    const rows = labelShape.reduce((product, value) => product * value, 1)
    return this.owner[EMIT]("cross_entropy", [this.#reshape([rows, classes]), labels.#reshape([rows])], result, { reduction: "mean", class_axis: 1 })
  }
  #index_select(axis: number, indices: FormalTensor): FormalTensor {
    const normalized = axis < 0 ? this.spec.shape.length + axis : axis
    if (indices.spec.dtype !== "i64" || indices.spec.shape.length !== 1 || !Number.isInteger(normalized) || normalized < 0 || normalized >= this.spec.shape.length) throw new TypeError("invalid index_select operands")
    const shape = [...this.spec.shape]
    shape[normalized] = indices.spec.shape[0]
    return this.owner[EMIT]("index_select", [this, indices], freezeSpec(this.spec.dtype, shape, this.spec.axes), { axis: normalized })
  }
}

/** Internal bridge used only by affon:ops while FormalTensor's implementation is encapsulated. */
export function $formalOperation(op: string, operands: readonly FormalTensor[], options: readonly unknown[] = []): FormalTensor {
  if (!operands.length || operands.some(value => !(value instanceof FormalTensor))) throw new TypeError(`${op} requires formal tensor operands`)
  return operands[0][FORMAL_OPERATION](op, operands, options)
}

/** Internal scalar hook used only by affon:ops composite operations. */
export function $formalScalarLike(reference: FormalTensor, value: number): FormalTensor {
  return (reference as any).owner[FORMAL_SCALAR](reference, value)
}

/** Internal parameter-declaration hook used by affon:nn factories. */
export function $formalParameter(reference: FormalTensor, name: string, spec: TensorSpec, options: { initializer?: Initializer } = {}): FormalTensor {
  if (!(reference instanceof FormalTensor)) throw new TypeError("parameterized layers require a formal tensor binding")
  return (reference as any).owner.parameter(name, spec, options)
}

function assertName(name: string, label: string): void {
  if (typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new TypeError(`${label} name must be a JavaScript identifier`)
}

function assertQualifiedName(name: string, label: string): void {
  if (typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*(\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*$/.test(name)) {
    throw new TypeError(`${label} name must be a dot-separated identifier`)
  }
}

function qualifiedName(...segments: string[]): string { return segments.join(".") }

function initializerSnapshot(value: Initializer, spec: TensorSpec): Initializer {
  if (!value || typeof value !== "object") throw new TypeError("initializer must be an Initializer descriptor")
  switch (value.kind) {
    case "zeros": case "ones":
      return Object.freeze({ kind: value.kind })
    case "xavier_uniform": case "xavier_normal": {
      const fanIn = spec.shape.at(-2) ?? spec.shape.at(-1) ?? 1
      const fanOut = spec.shape.at(-1) ?? 1
      if (fanIn + fanOut <= 0) throw new TypeError("xavier initializer requires a non-empty fan-in or fan-out")
      return Object.freeze({ kind: value.kind })
    }
    case "constant":
      if (!Number.isFinite(value.value)) throw new TypeError("constant initializer value must be finite")
      return Object.freeze({ kind: "constant", value: value.value })
    case "uniform":
      if (!Number.isFinite(value.min) || !Number.isFinite(value.max) || value.min > value.max) throw new TypeError("uniform initializer requires finite min <= max")
      return Object.freeze({ kind: "uniform", min: value.min, max: value.max })
    case "normal": {
      const mean = value.mean ?? 0
      const standard_deviation = value.standard_deviation ?? 1
      if (!Number.isFinite(mean) || !Number.isFinite(standard_deviation) || standard_deviation < 0) throw new TypeError("normal initializer requires a finite mean and non-negative standard_deviation")
      return Object.freeze({ kind: "normal", mean, standard_deviation })
    }
    default: throw new TypeError("unknown initializer kind")
  }
}

export class ProgramBuilder {
  readonly [BUILDER] = true
  private readonly programName: string
  private readonly nodeList: ProgramNode[] = []
  private readonly names = new Map<string, ProgramNode>()
  private finished = false

  constructor(name: string) { this.programName = name }

  private formal(role: "argument" | "parameter" | "state", name: string, spec: TensorSpec, path: ProgramPath = EMPTY_PROGRAM_PATH): FormalTensor {
    if (this.finished) throw new Error("ProgramBuilder may only be used inside program()")
    if (role === "argument") assertName(name, role)
    else assertQualifiedName(name, role)
    if (this.names.has(name)) throw new TypeError(`duplicate formal name: ${name}`)
    const node = Object.freeze({ id: this.nodeList.length, kind: role, path, role, name, provenance: `${this.programName}.${name}`, spec })
    this.nodeList.push(node)
    this.names.set(name, node)
    return new FormalTensor(this, node.id, spec, role, name)
  }

  argument(name: string, spec: TensorSpec): FormalTensor { return this.formal("argument", name, spec) }
  parameter(name: string, spec: TensorSpec, options: { initializer?: Initializer } = {}): FormalTensor {
    const initializer = options.initializer && initializerSnapshot(options.initializer, spec)
    const value = this.formal("parameter", name, spec)
    if (initializer) (this.nodeList[value.id] as any) = Object.freeze({ ...this.nodeList[value.id], options: Object.freeze({ initializer }) })
    return value
  }
  state(name: string, spec: TensorSpec, options: { initializer?: Initializer } = {}): FormalTensor {
    const initializer = options.initializer && initializerSnapshot(options.initializer, spec)
    const value = this.formal("state", name, spec)
    if (initializer) (this.nodeList[value.id] as any) = Object.freeze({ ...this.nodeList[value.id], options: Object.freeze({ initializer }) })
    return value
  }
  private constantAtPath(name: string, value: ProgramValue, spec: TensorSpec, path: ProgramPath): FormalTensor {
    assertQualifiedName(name, "constant")
    if (this.names.has(name)) throw new TypeError(`duplicate formal name: ${name}`)
    const snapshot = snapshotTensorData(value, { label: "constant", finite: true, integer: spec.dtype === "i64" })
    const scalarShorthand = typeof snapshot.data === "number" && elementCount(spec.shape) === 1
    if (!scalarShorthand && (snapshot.shape.length !== spec.shape.length || snapshot.shape.some((size, index) => size !== spec.shape[index]))) {
      throw new TypeError("constant value does not match its TensorSpec")
    }
    const node = Object.freeze({ id: this.nodeList.length, kind: "constant" as const, path, role: "constant" as const, name, provenance: `${this.programName}.${name}`, spec, value: snapshot.data })
    this.nodeList.push(node)
    this.names.set(name, node)
    return new FormalTensor(this, node.id, spec, "constant", name)
  }

  constant(name: string, value: ProgramValue, spec: TensorSpec): FormalTensor {
    return this.constantAtPath(name, value, spec, EMPTY_PROGRAM_PATH)
  }

  [EMIT](op: string, operands: readonly FormalTensor[], spec: TensorSpec, options?: Readonly<Record<string, unknown>>, path: ProgramPath = EMPTY_PROGRAM_PATH): FormalTensor {
    if (this.finished) throw new Error("FormalTensor values may only be used inside program()")
    if (operands.some(value => !(value instanceof FormalTensor) || (value as any).owner !== this)) throw new TypeError(`${op} operands must belong to this ProgramBuilder`)
    const node = Object.freeze({ id: this.nodeList.length, kind: "operation" as const, path, op, operands: Object.freeze(operands.map(value => value.id)), spec, ...(options ? { options: freezeMetadata({ ...options }) } : {}) })
    this.nodeList.push(node)
    return new FormalTensor(this, node.id, spec, "intermediate")
  }

  [FORMAL_SCALAR](reference: FormalTensor, value: number): FormalTensor {
    if (!Number.isFinite(value)) throw new TypeError("scalar operand must be finite")
    return this.constant(`scalar_${this.nodeList.length}`, value, Tensor.spec(reference.spec.dtype, [1]))
  }

  reduction(op: string, value: FormalTensor, axis?: number, keep_dims = false): FormalTensor {
    const dtype: ProgramDType = op === "argmin" || op === "argmax" ? "i64" : value.spec.dtype
    if (axis === undefined) return this[EMIT](op, [value], scalarSpec(dtype), { keep_dims })
    const normalized = axis < 0 ? value.spec.shape.length + axis : axis
    if (!Number.isInteger(normalized) || normalized < 0 || normalized >= value.spec.shape.length) throw new TypeError(`${op} axis is out of range`)
    const shape = keep_dims ? value.spec.shape.map((size, index) => index === normalized ? 1 : size) : value.spec.shape.filter((_, index) => index !== normalized)
    return this[EMIT](op, [value], freezeSpec(dtype, shape), { axis: normalized, keep_dims })
  }

  compose(child: Program, bindings: Record<string, FormalTensor>, instance = child.name): FormalTensor | readonly FormalTensor[] {
    const inspection = child.inspect()
    if (inspection.kind !== "authored") throw new TypeError("only authored Programs can be composed")
    const expectedNames = new Set(inspection.arguments.map(formal => formal.name))
    for (const name of Object.keys(bindings)) if (!expectedNames.has(name)) throw new TypeError(`unknown ${child.name} argument: ${name}`)
    const mapped = new Map<number, FormalTensor>()
    for (const formal of inspection.arguments) {
      const value = bindings[formal.name]
      if (!(value instanceof FormalTensor)) throw new TypeError(`${child.name} requires argument ${formal.name}`)
      if (!sameSpec(value.spec, formal.spec)) throw new TypeError(`${child.name}.${formal.name} does not match its TensorSpec`)
      mapped.set(inspection.nodes.find(node => node.name === formal.name && node.role === "argument")!.id, value)
    }
    for (const node of inspection.nodes) {
      if (mapped.has(node.id)) continue
      const path = composedPath(child.name, instance, node.path)
      if (node.role === "parameter" || node.role === "state") {
        const localName = qualifiedName(instance, node.name!)
        const formal = this.formal(node.role, localName, node.spec, path)
        if (node.options) (this.nodeList[formal.id] as any) = Object.freeze({ ...this.nodeList[formal.id], options: node.options })
        mapped.set(node.id, formal)
      } else if (node.role === "constant") {
        const localName = qualifiedName(instance, node.name!)
        mapped.set(node.id, this.constantAtPath(localName, node.value!, node.spec, path))
      } else if (node.operands) {
        mapped.set(node.id, this[EMIT](node.op ?? "composition", node.operands.map(id => mapped.get(id)!), node.spec, { ...(node.options ?? {}), from: child.name, instance }, path))
      }
    }
    const outputs = inspection.outputs.map(id => mapped.get(id)!)
    return outputs.length === 1 ? outputs[0] : Object.freeze(outputs)
  }

  finish(): readonly ProgramNode[] { this.finished = true; return Object.freeze([...this.nodeList]) }
}

export interface Callable<Bindings extends Record<string, FormalTensor>, Out> {
  (bindings: Bindings, instance?: string): Out
}
export interface Program<Bindings extends Record<string, FormalTensor> = Record<string, FormalTensor>, Out = FormalTensor | readonly FormalTensor[]> extends Callable<Bindings, Out> {
  readonly name: string
  readonly provenance: string
  inspect(): ProgramInspection
}
export type EvaluatedProgramOutput<Out> = Out extends FormalTensor
  ? Tensor
  : Out extends readonly FormalTensor[]
    ? readonly Tensor[]
    : never

type InternalProgram = Program & { readonly [PROGRAM]: true }
type LossKind = "cross_entropy" | "mean_squared_error" | "binary_cross_entropy" | "binary_cross_entropy_with_logits"
type LossCallable = Callable<{ input: FormalTensor; target: FormalTensor }, FormalTensor>
type LossCallableMetadata = Readonly<{ kind: LossKind; target: string }>
const LOSS_CALLABLE = Symbol.for("affon.nn.loss_callable")
let activeBuilder: ProgramBuilder | null = null

function isProgram(value: unknown): value is InternalProgram { return typeof value === "function" && (value as any)[PROGRAM] === true }

function formals(nodes: readonly ProgramNode[], role: ProgramFormal["role"]): readonly ProgramFormal[] {
  return Object.freeze(nodes.filter(node => node.role === role).map(node => Object.freeze({ name: node.name!, role, spec: node.spec, provenance: node.provenance! })))
}

function createProgram(name: string, kind: ProgramInspection["kind"], nodes: readonly ProgramNode[], outputs: readonly number[], transitions: ProgramInspection["transitions"] = []): Program {
  const provenance = `${name}:${JSON.stringify({
    kind,
    nodes: nodes.map(node => ({
      kind: node.kind,
      path: node.path,
      role: node.role,
      name: node.name,
      provenance: node.provenance,
      spec: node.spec,
      op: node.op,
      operands: node.operands,
      options: node.options,
      value: node.value,
    })),
    outputs,
    transitions,
  })}`
  const inspection: ProgramInspection = Object.freeze({
    name,
    provenance,
    kind,
    arguments: formals(nodes, "argument"),
    parameters: formals(nodes, "parameter"),
    state: formals(nodes, "state"),
    constants: formals(nodes, "constant"),
    nodes,
    outputs: Object.freeze([...outputs]),
    transitions,
  })
  const callable = ((...args: unknown[]) => {
    if (!activeBuilder) throw new TypeError("Programs are symbolic; call them only while authoring another program, then execute a compiled Executable")
    if (args.length < 1 || args.length > 2 || !args[0] || typeof args[0] !== "object" || Array.isArray(args[0]) || args[0] instanceof FormalTensor) {
      throw new TypeError(`${name} expects named bindings and an optional instance name`)
    }
    const instance = args[1] ?? name
    assertQualifiedName(instance as string, "composition instance")
    return activeBuilder.compose(callable as Program, args[0] as Record<string, FormalTensor>, instance as string)
  }) as InternalProgram
  Object.defineProperties(callable, {
    [PROGRAM]: { value: true },
    name: { value: name },
    provenance: { value: provenance, enumerable: true },
    inspect: { value: () => inspection },
  })
  return Object.freeze(callable)
}

export function program<Out extends FormalTensor | readonly FormalTensor[]>(name: string, author: (p: ProgramBuilder) => Out): Program<Record<string, FormalTensor>, Out> {
  assertName(name, "program")
  if (typeof author !== "function") throw new TypeError("program expects an authoring callback")
  const parentBuilder = activeBuilder
  const builder = new ProgramBuilder(name)
  activeBuilder = builder
  try {
    const result = author(builder)
    const outputs = Array.isArray(result) ? result : [result]
    if (outputs.length === 0 || outputs.some(value => !(value instanceof FormalTensor))) throw new TypeError("program callback must return a FormalTensor or a non-empty FormalTensor array")
    const nodes = builder.finish()
    return createProgram(name, "authored", nodes, outputs.map(value => value.id)) as Program<Record<string, FormalTensor>, Out>
  } finally {
    activeBuilder = parentBuilder
  }
}

export function gradient(loss: Program<Record<string, FormalTensor>, FormalTensor>, independent_variables: string): Program<Record<string, FormalTensor>, FormalTensor>
export function gradient(loss: Program<Record<string, FormalTensor>, FormalTensor>, independent_variables: readonly string[]): Program
export function gradient(loss: Program<Record<string, FormalTensor>, FormalTensor>, independent_variables: string | readonly string[]): Program {
  if (!isProgram(loss)) throw new TypeError("gradient expects a Program")
  const source = loss.inspect()
  const output = source.nodes[source.outputs[0]]
  if (source.outputs.length !== 1 || output.spec.shape.reduce((size, value) => size * value, 1) !== 1) throw new TypeError("gradient requires a scalar Program")
  const names = typeof independent_variables === "string" ? [independent_variables] : [...independent_variables]
  if (names.length === 0) throw new TypeError("gradient requires at least one independent variable")
  const nodes = [...source.nodes]
  const outputs = names.map(name => {
    const formal = source.parameters.find(parameter => parameter.name === name) ?? source.arguments.find(argument => argument.name === name)
    if (!formal) throw new TypeError(`unknown independent variable: ${name}`)
    const formalNode = source.nodes.find(node => node.provenance === formal.provenance)!
    const node = Object.freeze({ id: nodes.length, kind: "gradient" as const, path: formalNode.path, op: "gradient", operands: Object.freeze([source.outputs[0]]), spec: formal.spec, options: Object.freeze({ with_respect_to: formal.provenance }) })
    nodes.push(node)
    return node.id
  })
  return createProgram(`${loss.name}_gradient`, "gradient", Object.freeze(nodes), outputs)
}

function scheduleSnapshot(value: LRSchedule): LRSchedule {
  if (!value || typeof value !== "object") throw new TypeError("scheduled expects a learning-rate schedule")
  const nonNegative = (candidate: unknown, name: string): number => {
    if (typeof candidate !== "number" || !Number.isFinite(candidate) || candidate < 0) throw new TypeError(`${name} must be finite and non-negative`)
    return candidate
  }
  const positiveInteger = (candidate: unknown, name: string): number => {
    if (typeof candidate !== "number" || !Number.isSafeInteger(candidate) || candidate <= 0) throw new TypeError(`${name} must be a positive integer`)
    return candidate
  }
  if (value.kind === "constant") return Object.freeze({ kind: "constant", learning_rate: nonNegative(value.learning_rate, "learning_rate") })
  if (value.kind === "linear" || value.kind === "cosine") return Object.freeze({
    kind: value.kind,
    start: nonNegative(value.start, "start"),
    end: nonNegative(value.end, "end"),
    steps: positiveInteger(value.steps, "steps"),
  })
  if (value.kind === "step") return Object.freeze({
    kind: "step",
    base: nonNegative(value.base, "base"),
    gamma: nonNegative(value.gamma, "gamma"),
    every: positiveInteger(value.every, "every"),
    ...(value.steps === undefined ? {} : { steps: positiveInteger(value.steps, "steps") }),
  })
  if (value.kind === "warmup_cosine") {
    const warmup_steps = positiveInteger(value.warmup_steps, "warmup_steps")
    const total_steps = positiveInteger(value.total_steps, "total_steps")
    if (warmup_steps >= total_steps) throw new TypeError("warmup_steps must be less than total_steps")
    return Object.freeze({
      kind: "warmup_cosine",
      start: nonNegative(value.start, "start"),
      peak: nonNegative(value.peak, "peak"),
      end: nonNegative(value.end, "end"),
      warmup_steps,
      total_steps,
    })
  }
  if (value.kind === "sequence") {
    if (!Array.isArray(value.schedules) || value.schedules.length === 0) throw new TypeError("sequence requires at least one schedule")
    const schedules = value.schedules.map(scheduleSnapshot)
    for (let index = 0; index < schedules.length - 1; index++) {
      const schedule = schedules[index]
      if (schedule.kind === "constant" || (schedule.kind === "step" && schedule.steps === undefined) || schedule.kind === "sequence") {
        throw new TypeError("only the final sequence schedule may be unbounded or nested")
      }
    }
    return Object.freeze({ kind: "sequence", schedules: Object.freeze(schedules) })
  }
  throw new TypeError("unknown learning-rate schedule kind")
}

function scheduleDuration(value: LRSchedule): number | undefined {
  if (value.kind === "linear" || value.kind === "cosine") return value.steps
  if (value.kind === "warmup_cosine") return value.total_steps
  if (value.kind === "step") return value.steps
  if (value.kind === "sequence") {
    let total = 0
    for (const part of value.schedules) {
      const duration = scheduleDuration(part)
      if (duration === undefined) return undefined
      total += duration
    }
    return total
  }
  return undefined
}

function learningRateAt(value: LRSchedule, step: number): number {
  if (value.kind === "constant") return value.learning_rate
  if (value.kind === "linear") {
    const ratio = Math.min(step / value.steps, 1)
    return value.start + (value.end - value.start) * ratio
  }
  if (value.kind === "cosine") {
    const ratio = Math.min(step / value.steps, 1)
    return value.end + (value.start - value.end) * (1 + Math.cos(Math.PI * ratio)) / 2
  }
  if (value.kind === "step") {
    const bounded = value.steps === undefined ? step : Math.min(step, value.steps)
    return value.base * Math.pow(value.gamma, Math.floor(bounded / value.every))
  }
  if (value.kind === "warmup_cosine") {
    if (step <= value.warmup_steps) return value.start + (value.peak - value.start) * step / value.warmup_steps
    const ratio = Math.min((step - value.warmup_steps) / (value.total_steps - value.warmup_steps), 1)
    return value.end + (value.peak - value.end) * (1 + Math.cos(Math.PI * ratio)) / 2
  }
  if (value.kind !== "sequence") throw new TypeError("unknown learning-rate schedule")
  let offset = step
  for (const part of value.schedules) {
    const duration = scheduleDuration(part)
    if (duration === undefined || offset <= duration) return learningRateAt(part, offset)
    offset -= duration
  }
  const final = value.schedules.at(-1)!
  return learningRateAt(final, scheduleDuration(final) ?? offset)
}

function optimizerSnapshot(value: Optimizer): Optimizer {
  if (!value || typeof value !== "object") throw new TypeError("expected an Optimizer from affon:optim")
  if (value.kind === "accumulate") {
    if (!Number.isSafeInteger(value.steps) || value.steps < 2) throw new TypeError("accumulate steps must be an integer of at least 2")
    const optimizer = optimizerSnapshot(value.optimizer)
    if (optimizer.kind === "accumulate") throw new TypeError("accumulate cannot wrap another accumulating optimizer")
    return Object.freeze({ kind: "accumulate", optimizer, steps: value.steps })
  }
  if (value.kind === "scheduled") {
    const optimizer = optimizerSnapshot(value.optimizer)
    if (optimizer.kind === "accumulate" || optimizer.kind === "scheduled") throw new TypeError("scheduled must wrap a base optimizer")
    return Object.freeze({ kind: "scheduled", optimizer, schedule: scheduleSnapshot(value.schedule) })
  }
  if (value.kind !== "sgd" && value.kind !== "adam" && value.kind !== "adamw") throw new TypeError("unknown Optimizer kind")
  const finiteNonNegative = (candidate: unknown, name: string): number => {
    if (typeof candidate !== "number" || !Number.isFinite(candidate) || candidate < 0) throw new TypeError(`${name} must be finite and non-negative`)
    return candidate
  }
  const learning_rate = finiteNonNegative((value as any).learning_rate, "learning_rate")
  if (value.kind === "sgd") {
    const momentum = (value as any).momentum
    if (typeof momentum !== "number" || !Number.isFinite(momentum) || momentum < 0 || momentum >= 1) throw new TypeError("momentum must be in [0, 1)")
    return Object.freeze({ kind: "sgd", learning_rate, momentum })
  }
  const beta = (name: "beta1" | "beta2"): number => {
    const candidate = (value as any)[name]
    if (typeof candidate !== "number" || !Number.isFinite(candidate) || candidate < 0 || candidate >= 1) throw new TypeError(`${name} must be in [0, 1)`)
    return candidate
  }
  const epsilon = (value as any).epsilon
  if (typeof epsilon !== "number" || !Number.isFinite(epsilon) || epsilon <= 0) throw new TypeError("epsilon must be finite and positive")
  const common = { learning_rate, beta1: beta("beta1"), beta2: beta("beta2"), epsilon }
  if (value.kind === "adam") return Object.freeze({ kind: "adam", ...common })
  return Object.freeze({ kind: "adamw", ...common, weight_decay: finiteNonNegative((value as any).weight_decay, "weight_decay") })
}

function combineModelAndLoss(
  model: Program<Record<string, FormalTensor>, FormalTensor | readonly FormalTensor[]>,
  loss: Program<Record<string, FormalTensor>, FormalTensor>,
): Program<Record<string, FormalTensor>, FormalTensor> {
  if (!isProgram(model) || !isProgram(loss)) {
    throw new TypeError("optimize(model, loss, optimizer) expects model and loss Programs")
  }
  const modelInspection = model.inspect()
  const lossInspection = loss.inspect()
  if (modelInspection.kind !== "authored" || lossInspection.kind !== "authored") {
    throw new TypeError("optimize(model, loss, optimizer) requires authored Programs")
  }
  if (modelInspection.outputs.length > lossInspection.arguments.length) {
    throw new TypeError("loss must declare one leading argument for each model output")
  }

  const nodes: ProgramNode[] = [...modelInspection.nodes]
  const mapped = new Map<number, number>()
  const modelArgumentNames = new Set(modelInspection.arguments.map(formal => formal.name))
  const lossArgumentNodes = lossInspection.arguments.map(formal =>
    lossInspection.nodes.find(node => node.role === "argument" && node.provenance === formal.provenance)!,
  )

  for (let index = 0; index < modelInspection.outputs.length; index++) {
    const output = modelInspection.nodes[modelInspection.outputs[index]]
    const input = lossArgumentNodes[index]
    if (!sameSpec(output.spec, input.spec)) {
      throw new TypeError(`model output ${index} does not match loss argument ${input.name}`)
    }
    mapped.set(input.id, output.id)
  }

  for (const input of lossArgumentNodes.slice(modelInspection.outputs.length)) {
    if (modelArgumentNames.has(input.name!)) {
      throw new TypeError(`model and loss expose duplicate argument: ${input.name}`)
    }
  }

  for (const node of lossInspection.nodes) {
    if (mapped.has(node.id)) continue
    const operands = node.operands?.map(id => {
      const mappedId = mapped.get(id)
      if (mappedId === undefined) throw new TypeError("loss graph contains an unresolved operand")
      return mappedId
    })
    const clone = Object.freeze({
      ...node,
      id: nodes.length,
      ...(operands ? { operands: Object.freeze(operands) } : {}),
    })
    nodes.push(clone)
    mapped.set(node.id, clone.id)
  }

  return createProgram(
    `${model.name}_${loss.name}`,
    "authored",
    Object.freeze(nodes),
    lossInspection.outputs.map(id => mapped.get(id)!),
  ) as Program<Record<string, FormalTensor>, FormalTensor>
}

function lossProgramFromCallable(
  model: Program<Record<string, FormalTensor>, FormalTensor | readonly FormalTensor[]>,
  loss: LossCallable,
): Program<Record<string, FormalTensor>, FormalTensor> {
  const inspection = model.inspect()
  const metadata = (loss as any)[LOSS_CALLABLE] as LossCallableMetadata | undefined
  const kind = metadata?.kind ?? "custom"
  const targetName = metadata?.target ?? "target"
  if (inspection.outputs.length !== 1) throw new TypeError(`${kind} requires a model with one output`)
  const outputSpec = inspection.nodes[inspection.outputs[0]].spec
  if (outputSpec.dtype !== "f32" && outputSpec.dtype !== "f64") throw new TypeError(`${kind} requires a floating-point model output`)
  return program(`${model.name}_${kind}_loss`, p => {
    const input = p.argument("input", outputSpec)
    let targetSpec = outputSpec
    if (kind === "cross_entropy") {
      if (outputSpec.shape.length === 0) throw new TypeError("cross_entropy requires model logits with a class dimension")
      const labelAxes = outputSpec.axes?.slice(0, -1)
      targetSpec = Tensor.i64(outputSpec.shape.slice(0, -1), labelAxes ? { axes: labelAxes } : undefined)
    }
    const target = p.argument(targetName, targetSpec)
    return loss({ input, target }, "loss")
  })
}

export function optimize(
  model: Program<Record<string, FormalTensor>, FormalTensor | readonly FormalTensor[]>,
  loss: Program<Record<string, FormalTensor>, FormalTensor> | LossCallable,
  optimizer: Optimizer,
): Program<Record<string, FormalTensor>, FormalTensor>
export function optimize(
  model: Program<Record<string, FormalTensor>, FormalTensor | readonly FormalTensor[]>,
  loss: Program<Record<string, FormalTensor>, FormalTensor> | LossCallable,
  optimizer: Optimizer,
): Program<Record<string, FormalTensor>, FormalTensor> {
  if (!isProgram(model)) throw new TypeError("optimize expects a model Program")
  if (!isProgram(loss) && typeof loss !== "function") {
    throw new TypeError("optimize expects a loss Program or LossCallable")
  }
  const lossProgram = isProgram(loss) ? loss : lossProgramFromCallable(model, loss)
  const combined = combineModelAndLoss(model, lossProgram)
  const source = combined.inspect()
  const output = source.nodes[source.outputs[0]]
  if (source.outputs.length !== 1 || output.spec.shape.reduce((size, value) => size * value, 1) !== 1) throw new TypeError("optimize requires a scalar Program")
  if (source.parameters.length === 0) throw new TypeError("optimize requires at least one parameter")
  const derivatives = gradient(combined, source.parameters.map(parameter => parameter.name))
  return update_parameters(combined, derivatives, optimizer)
}

export function update_parameters(
  source: Program<Record<string, FormalTensor>, FormalTensor>,
  gradients: Program<Record<string, FormalTensor>, FormalTensor | readonly FormalTensor[]>,
  optimizer: Optimizer,
): Program<Record<string, FormalTensor>, FormalTensor> {
  if (!isProgram(source)) throw new TypeError("update_parameters expects a source Program")
  if (!isProgram(gradients) || gradients.inspect().kind !== "gradient") {
    throw new TypeError("update_parameters expects a gradient Program")
  }
  const sourceInspection = source.inspect()
  const gradientInspection = gradients.inspect()
  if (sourceInspection.kind !== "authored") throw new TypeError("update_parameters requires an authored source Program")
  if (sourceInspection.outputs.length !== 1) throw new TypeError("update_parameters requires a single-output source Program")
  const sameFormalSet = (left: readonly ProgramFormal[], right: readonly ProgramFormal[]): boolean =>
    left.length === right.length && left.every((formal, index) => {
      const other = right[index]
      return other?.provenance === formal.provenance && sameSpec(other.spec, formal.spec)
    })
  if (!sameFormalSet(sourceInspection.arguments, gradientInspection.arguments) ||
      !sameFormalSet(sourceInspection.parameters, gradientInspection.parameters) ||
      !sameFormalSet(sourceInspection.state, gradientInspection.state)) {
    throw new TypeError("gradient Program must be derived from the source Program")
  }
  const parametersByProvenance = new Map(sourceInspection.parameters.map(parameter => [parameter.provenance, parameter]))
  const selected = gradientInspection.outputs.map(id => {
    const node = gradientInspection.nodes.find(candidate => candidate.id === id)
    const provenance = node?.kind === "gradient" ? (node.options as any)?.with_respect_to : undefined
    const parameter = typeof provenance === "string" ? parametersByProvenance.get(provenance) : undefined
    if (!node || !parameter || !sameSpec(node.spec, parameter.spec)) {
      throw new TypeError("update_parameters gradients must correspond to source parameters")
    }
    return parameter.provenance
  })
  if (selected.length === 0 || new Set(selected).size !== selected.length) {
    throw new TypeError("update_parameters requires unique parameter gradients")
  }
  const snapshot = optimizerSnapshot(optimizer)
  const parameters = Object.freeze(selected)
  const transition = Object.freeze({ kind: "optimize" as const, optimizer: snapshot, parameters })
  const result = createProgram(`${source.name}_${snapshot.kind}_update`, "optimize", sourceInspection.nodes, sourceInspection.outputs, Object.freeze([...sourceInspection.transitions, transition])) as Program<Record<string, FormalTensor>, FormalTensor>
  parameterUpdateSources.set(result, { source, gradients, optimizer: snapshot, parameters })
  return result
}

function seededRandom(seed: number): () => number {
  let value = (Number.isSafeInteger(seed) ? seed : 0) >>> 0
  return () => {
    value = (value + 0x6d2b79f5) >>> 0
    let next = value
    next = Math.imul(next ^ (next >>> 15), next | 1)
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61)
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296
  }
}

function initializedValue(spec: TensorSpec, initializer: Initializer, random: () => number): ProgramValue {
  const fanIn = spec.shape.length >= 2 ? spec.shape[spec.shape.length - 2] : Math.max(spec.shape[0] ?? 1, 1)
  const fanOut = spec.shape.length >= 2 ? spec.shape[spec.shape.length - 1] : Math.max(spec.shape[0] ?? 1, 1)
  let spareNormal: number | undefined
  const normal = () => {
    if (spareNormal !== undefined) { const value = spareNormal; spareNormal = undefined; return value }
    const radius = Math.sqrt(-2 * Math.log(Math.max(random(), Number.EPSILON)))
    const angle = 2 * Math.PI * random()
    spareNormal = radius * Math.sin(angle)
    return radius * Math.cos(angle)
  }
  const scalar = (): number => {
    switch (initializer.kind) {
      case "zeros": return 0
      case "ones": return 1
      case "constant": return initializer.value
      case "uniform": return initializer.min + (initializer.max - initializer.min) * random()
      case "normal": return (initializer.mean ?? 0) + (initializer.standard_deviation ?? 1) * normal()
      case "xavier_uniform": { const bound = Math.sqrt(6 / (fanIn + fanOut)); return (2 * random() - 1) * bound }
      case "xavier_normal": return normal() * Math.sqrt(2 / (fanIn + fanOut))
    }
  }
  const build = (axis: number): ProgramValue => axis === spec.shape.length ? scalar() : Array.from({ length: spec.shape[axis] }, () => build(axis + 1))
  return build(0)
}

const tensorOwners = new WeakMap<object, Session>()
const disposedTensors = new WeakSet<object>()

/** Internal ownership hook used by affon:ops immediate dispatch. */
export function $sessionOf(value: Tensor): Session | undefined {
  const owner = tensorOwners.get(value as object)
  if (disposedTensors.has(value as object)) return undefined
  return owner
}

function metricValues(value: Tensor): number[] {
  const result: number[] = []
  const visit = (item: TensorData): void => {
    if (typeof item === "number") result.push(item)
    else for (const child of item) visit(child)
  }
  visit(value.to_array())
  return result
}

function metricPair(prediction: Tensor, target: Tensor): [number[], number[]] {
  const left = metricValues(prediction)
  const right = metricValues(target)
  if (left.length !== right.length) throw new TypeError("metric tensors must contain the same number of elements")
  return [left, right]
}

function binaryCounts(prediction: Tensor, target: Tensor, threshold: number): { tp: number; fp: number; fn: number; correct: number; total: number } {
  if (!Number.isFinite(threshold)) throw new TypeError("metric threshold must be finite")
  const [left, right] = metricPair(prediction, target)
  let tp = 0, fp = 0, fn = 0, correct = 0
  for (let index = 0; index < left.length; index++) {
    const predicted = left[index] > threshold
    const actual = right[index] !== 0
    if (predicted === actual) correct++
    if (predicted && actual) tp++
    else if (predicted) fp++
    else if (actual) fn++
  }
  return { tp, fp, fn, correct, total: left.length }
}

/** Reporting metrics for evaluated tensors. Metrics never mutate ExecutionState. */
export const metrics = Object.freeze({
  accuracy(prediction: Tensor, target: Tensor, options: { threshold?: number } = {}): number {
    if (prediction.shape.length === target.shape.length + 1 && prediction.shape.slice(0, -1).every((size, index) => size === target.shape[index])) {
      const classes = prediction.shape.at(-1)!
      const scores = metricValues(prediction)
      const labels = metricValues(target)
      let correct = 0
      for (let row = 0; row < labels.length; row++) {
        let selected = 0
        for (let index = 1; index < classes; index++) if (scores[row * classes + index] > scores[row * classes + selected]) selected = index
        if (selected === labels[row]) correct++
      }
      return labels.length === 0 ? 0 : correct / labels.length
    }
    const counts = binaryCounts(prediction, target, options.threshold ?? 0.5)
    return counts.total === 0 ? 0 : counts.correct / counts.total
  },
  precision(prediction: Tensor, target: Tensor, options: { threshold?: number } = {}): number {
    const { tp, fp } = binaryCounts(prediction, target, options.threshold ?? 0.5)
    return tp + fp === 0 ? 0 : tp / (tp + fp)
  },
  recall(prediction: Tensor, target: Tensor, options: { threshold?: number } = {}): number {
    const { tp, fn } = binaryCounts(prediction, target, options.threshold ?? 0.5)
    return tp + fn === 0 ? 0 : tp / (tp + fn)
  },
  f1(prediction: Tensor, target: Tensor, options: { threshold?: number } = {}): number {
    const precision = metrics.precision(prediction, target, options)
    const recall = metrics.recall(prediction, target, options)
    return precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall)
  },
  mean_squared_error(prediction: Tensor, target: Tensor): number {
    const [left, right] = metricPair(prediction, target)
    return left.length === 0 ? 0 : left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0) / left.length
  },
  mean_absolute_error(prediction: Tensor, target: Tensor): number {
    const [left, right] = metricPair(prediction, target)
    return left.length === 0 ? 0 : left.reduce((sum, value, index) => sum + Math.abs(value - right[index]), 0) / left.length
  },
  r2_score(prediction: Tensor, target: Tensor): number {
    const [left, right] = metricPair(prediction, target)
    if (right.length === 0) return 0
    const average = right.reduce((sum, value) => sum + value, 0) / right.length
    const total = right.reduce((sum, value) => sum + (value - average) ** 2, 0)
    if (total === 0) return 0
    return 1 - left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0) / total
  },
})

/** Internal symbolic hook used by affon:ops for variadic operations. */
export function $formalCat(values: readonly FormalTensor[], axis = 0): FormalTensor {
  if (!values.length) throw new TypeError("cat requires at least one tensor")
  const first = values[0].spec
  const normalized = axis < 0 ? first.shape.length + axis : axis
  if (!Number.isInteger(normalized) || normalized < 0 || normalized >= first.shape.length) throw new TypeError("cat axis is out of range")
  const shape = [...first.shape]
  for (const value of values.slice(1)) {
    if (value.spec.dtype !== first.dtype || value.spec.shape.length !== first.shape.length || value.spec.shape.some((size, index) => index !== normalized && size !== first.shape[index])) throw new TypeError("cat tensor specs are incompatible")
    if ((first.axes === undefined) !== (value.spec.axes === undefined) || (first.axes && first.axes.some((name, index) => name !== value.spec.axes![index]))) throw new TypeError("cat tensor axes are incompatible")
    shape[normalized] += value.spec.shape[normalized]
  }
  return (values[0] as any).owner[EMIT]("cat", values, freezeSpec(first.dtype, shape, first.axes), { axis: normalized })
}

export class ExecutionState {
  readonly parameters: Record<string, Tensor>
  readonly model_state: Record<string, Tensor>
  readonly optimizer_state: Record<string, Tensor | number>
  readonly rng_state: Record<string, number>
  readonly session: Session
  #disposed = false

  constructor(session: Session, values: { parameters?: Record<string, Tensor>; model_state?: Record<string, Tensor>; optimizer_state?: Record<string, Tensor | number>; rng_state?: Record<string, number> } = {}) {
    this.session = session
    this.parameters = { ...(values.parameters ?? {}) }
    this.model_state = { ...(values.model_state ?? {}) }
    this.optimizer_state = { ...(values.optimizer_state ?? {}) }
    this.rng_state = { ...(values.rng_state ?? {}) }
  }
  get disposed(): boolean { return this.#disposed }
  dispose(): void {
    if (this.#disposed) return
    for (const group of [this.parameters, this.model_state, this.optimizer_state]) {
      for (const name of Object.keys(group)) {
        const value = group[name] as any
        value?.dispose?.()
        delete group[name]
      }
    }
    for (const name of Object.keys(this.rng_state)) delete this.rng_state[name]
    this.#disposed = true
  }
  [Symbol.dispose](): void { this.dispose() }
}

export class Executable<Out extends FormalTensor | readonly FormalTensor[] = FormalTensor | readonly FormalTensor[]> {
  readonly program: Program<Record<string, FormalTensor>, Out>
  readonly session: Session
  readonly argument_names: readonly string[]
  readonly argument_specs: readonly ProgramFormal[]
  readonly parameter_names: readonly string[]
  readonly state_names: readonly string[]
  readonly native_input_order: readonly string[]
  private readonly nativeExecutable: any
  private readonly argumentFormals: readonly ProgramFormal[]
  private readonly parameterFormals: readonly ProgramFormal[]
  private readonly stateFormals: readonly ProgramFormal[]
  private readonly inputFormals: readonly ProgramFormal[]
  private readonly optimization?: { source: Executable; gradients: Executable; optimizer: Optimizer; parameters: readonly string[] }
  #disposed = false

  constructor(session: Session, source: Program<Record<string, FormalTensor>, Out>, nativeExecutable: any, optimization?: { source: Executable; gradients: Executable; optimizer: Optimizer; parameters: readonly string[] }) {
    const inspection = source.inspect()
    this.program = source
    this.session = session
    this.nativeExecutable = nativeExecutable
    this.optimization = optimization
    this.argumentFormals = inspection.arguments
    this.parameterFormals = inspection.parameters
    this.stateFormals = inspection.state
    const formalByProvenance = new Map([...inspection.arguments, ...inspection.parameters, ...inspection.state].map(value => [value.provenance, value]))
    this.inputFormals = Object.freeze(inspection.nodes.flatMap(node => node.provenance && formalByProvenance.has(node.provenance) ? [formalByProvenance.get(node.provenance)!] : []))
    this.argument_names = Object.freeze(inspection.arguments.map(value => value.name))
    this.argument_specs = inspection.arguments
    this.parameter_names = Object.freeze(inspection.parameters.map(value => value.name))
    this.state_names = Object.freeze(inspection.state.map(value => value.name))
    this.native_input_order = Object.freeze(this.inputFormals.map(value => value.name))
  }
  get disposed(): boolean { return this.#disposed }
  run(arguments_: ProgramArguments, state?: ExecutionState): EvaluatedProgramOutput<Out> {
    if (this.#disposed) throw new Error("Executable has been disposed")
    if (this.session.disposed) throw new Error("Session has been disposed")
    if (state && state.session !== this.session) throw new TypeError("ExecutionState belongs to a different Session")
    if (state?.disposed) throw new Error("ExecutionState has been disposed")
    if (this.optimization) {
      if (!state) throw new TypeError("optimize Executable.run requires an ExecutionState")
      const result = this.optimization.source.run(arguments_, state)
      let values: any[] = []
      try {
        const gradients = this.optimization.gradients.run(arguments_, state)
        values = Array.isArray(gradients) ? gradients : [gradients]
        this.session.applyOptimizer(this.program, state, values, this.optimization.optimizer, this.optimization.parameters)
        return result as EvaluatedProgramOutput<Out>
      } catch (error) {
        ;(result as any)?.dispose?.()
        throw error
      } finally {
        for (const value of values) value?.dispose?.()
      }
    }
    if (!arguments_ || typeof arguments_ !== "object" || Array.isArray(arguments_)) throw new TypeError("Executable.run expects named arguments")
    const expected = new Set(this.argument_names)
    for (const name of Object.keys(arguments_)) if (!expected.has(name)) throw new TypeError(`unknown Program argument: ${name}`)
    if ((this.parameterFormals.length > 0 || this.stateFormals.length > 0) && !state) throw new TypeError("Executable.run requires an ExecutionState")
    const inputs: unknown[] = []
    for (const formal of this.inputFormals) {
      let value: unknown
      if (formal.role === "argument") {
        if (!(formal.name in arguments_)) throw new TypeError(`missing Program argument: ${formal.name}`)
        value = arguments_[formal.name]
      } else if (formal.role === "parameter") {
        value = state!.parameters[formal.provenance]
        if (!value) throw new TypeError(`ExecutionState is missing parameter ${formal.provenance}`)
      } else {
        value = state!.model_state[formal.provenance]
        if (!value) throw new TypeError(`ExecutionState is missing model state ${formal.provenance}`)
      }
      if (!this.session.owns(value)) throw new TypeError(`${formal.name} is not a Tensor owned by this Session`)
      const tensor = value as Tensor
      if (tensor.dtype !== formal.spec.dtype || tensor.shape.length !== formal.spec.shape.length || tensor.shape.some((size, index) => size !== formal.spec.shape[index])) {
        throw new TypeError(`${formal.name} does not match its TensorSpec`)
      }
      inputs.push(value)
    }
    const outputs = native.runExecutable(this.nativeExecutable, inputs as any)
    for (const output of outputs) this.session.adopt(output)
    const visible = this.program.inspect().kind === "gradient" ? outputs.slice(1) : outputs
    if (this.program.inspect().kind === "gradient") outputs[0]?.dispose?.()
    const outputNodes = this.program.inspect().outputs.map(id => this.program.inspect().nodes.find(node => node.id === id)!)
    for (let index = 0; index < visible.length; index++) {
      const axes = outputNodes[index]?.spec.axes
      if (axes && visible[index] && !("axes" in visible[index])) {
        Object.defineProperty(visible[index], "axes", { value: Object.freeze([...axes]), enumerable: true })
      }
    }
    return (visible.length === 1 ? visible[0] : Object.freeze(visible)) as EvaluatedProgramOutput<Out>
  }
  dispose(): void {
    if (this.#disposed) return
    this.nativeExecutable?.dispose?.()
    this.#disposed = true
  }
  [Symbol.dispose](): void { this.dispose() }
}

export class Session {
  readonly device: Device
  readonly #cache = new Map<Program, Executable<any>>()
  readonly #native: any
  #disposed = false

  constructor(options: { device?: Device } = {}) {
    this.device = options.device ?? "cpu"
    this.#native = native.createSession(this.device as any)
  }
  get disposed(): boolean { return this.#disposed }
  tensor(values: TensorData, options: { dtype?: ProgramDType; axes?: readonly string[] } = {}): Tensor {
    return this[SESSION_TENSOR](values, options)
  }
  [SESSION_TENSOR](values: TensorData, options: TensorValueOptions = {}, requestedShape?: readonly number[]): Tensor {
    if (this.#disposed) throw new Error("Session has been disposed")
    const dtype = options.dtype ?? "f32"
    if (dtype !== "f32" && dtype !== "f64" && dtype !== "i64") throw new TypeError(`unsupported Tensor dtype: ${dtype}`)
    const snapshot = snapshotTensorData(values, { label: "Tensor", integer: dtype === "i64" })
    const shape = requestedShape ?? snapshot.shape
    if (options.axes && (options.axes.length !== shape.length || options.axes.some(axis => typeof axis !== "string" || axis.length === 0))) {
      throw new TypeError("Tensor axes must contain one non-empty name per dimension")
    }
    if (requestedShape && elementCount(requestedShape) !== elementCount(snapshot.shape)) throw new TypeError("Tensor shape does not match its values")
    const value = native.sessionTensor(this.#native, snapshot.data, dtype, shape)
    if (options.axes) {
      Object.defineProperty(value, "axes", { value: Object.freeze([...options.axes]), enumerable: true })
    }
    this.adopt(value)
    return value as Tensor
  }
  [SESSION_FULL](shape: readonly number[], fill: number, options: TensorValueOptions = {}): Tensor {
    if (this.#disposed) throw new Error("Session has been disposed")
    const dtype = options.dtype ?? "f32"
    if (dtype !== "f32" && dtype !== "f64" && dtype !== "i64") throw new TypeError(`unsupported Tensor dtype: ${dtype}`)
    if (options.axes && (options.axes.length !== shape.length || options.axes.some(axis => typeof axis !== "string" || axis.length === 0))) {
      throw new TypeError("Tensor axes must contain one non-empty name per dimension")
    }
    const value = native.sessionFull(this.#native, shape, fill, dtype)
    if (options.axes) Object.defineProperty(value, "axes", { value: Object.freeze([...options.axes]), enumerable: true })
    this.adopt(value)
    return value as Tensor
  }
  compile<Out extends FormalTensor | readonly FormalTensor[]>(source: Program<Record<string, FormalTensor>, Out>): Executable<Out> {
    if (this.#disposed) throw new Error("Session has been disposed")
    if (!isProgram(source)) throw new TypeError("Session.compile expects a Program")
    const cached = this.#cache.get(source)
    if (cached && !cached.disposed) return cached as Executable<Out>
    if (cached) this.#cache.delete(source)
    const optimization = parameterUpdateSources.get(source)
    let executable: Executable<Out>
    if (optimization) {
      executable = new Executable(this, source, null, {
        source: this.compile(optimization.source),
        gradients: this.compile(optimization.gradients),
        optimizer: optimization.optimizer,
        parameters: optimization.parameters,
      }) as Executable<Out>
    } else executable = new Executable(this, source, native.compileProgram(this.#native, JSON.stringify(source.inspect()))) as Executable<Out>
    this.#cache.set(source, executable)
    return executable
  }
  initialize(source: Program, options: { seed?: number; parameters?: Readonly<Record<string, TensorInitializerValue>>; model_state?: Readonly<Record<string, TensorInitializerValue>> } = {}): ExecutionState {
    if (this.#disposed) throw new Error("Session has been disposed")
    if (!isProgram(source)) throw new TypeError("Session.initialize expects a Program")
    const random = seededRandom(options.seed ?? 0)
    const inspection = source.inspect()
    const parameters: Record<string, Tensor> = {}
    const model_state: Record<string, Tensor> = {}
    try {
      for (const [group, supplied] of [[inspection.parameters, options.parameters], [inspection.state, options.model_state]] as const) {
        if (!supplied) continue
        const accepted = new Set(group.flatMap(formal => [formal.name, formal.provenance]))
        for (const name of Object.keys(supplied)) if (!accepted.has(name)) throw new TypeError(`unknown ExecutionState initializer: ${name}`)
      }
      for (const formal of [...inspection.parameters, ...inspection.state]) {
        const node = inspection.nodes.find(value => value.provenance === formal.provenance)!
        const initializer = (node.options as any)?.initializer as Initializer | undefined
        const overrides = formal.role === "parameter" ? options.parameters : options.model_state
        const supplied = overrides?.[formal.provenance] ?? overrides?.[formal.name]
        const raw = supplied === undefined ? initializedValue(formal.spec, initializer ?? { kind: "zeros" }, random) : supplied
        if (supplied !== undefined && raw && typeof raw === "object" && "shape" in raw && "dtype" in raw) {
          const tensor = raw as any
          if (tensor.dtype !== formal.spec.dtype || !Array.isArray(tensor.shape) || tensor.shape.length !== formal.spec.shape.length || tensor.shape.some((size: number, index: number) => size !== formal.spec.shape[index])) {
            throw new TypeError(`${formal.name} initializer does not match its TensorSpec`)
          }
        }
        const materialized = raw && typeof raw === "object" && "to_array" in raw && typeof (raw as any).to_array === "function" ? (raw as any).to_array() : raw
        const value = this.tensor(materialized as TensorData, { dtype: formal.spec.dtype, axes: formal.spec.axes })
        if (value.shape.length !== formal.spec.shape.length || value.shape.some((size, index) => size !== formal.spec.shape[index])) {
          value.dispose()
          throw new TypeError(`${formal.name} initializer does not match its TensorSpec`)
        }
        if (formal.role === "parameter") parameters[formal.provenance] = value
        else model_state[formal.provenance] = value
      }
    } catch (error) {
      for (const value of [...Object.values(parameters), ...Object.values(model_state)] as any[]) value?.dispose?.()
      throw error
    }
    return new ExecutionState(this, { parameters, model_state, optimizer_state: {}, rng_state: { seed: options.seed ?? 0, counter: 0 } })
  }
  adopt(value: unknown): void {
    if (value && (typeof value === "object" || typeof value === "function")) {
      const tensor = value as object & { dispose?: () => void; [Symbol.dispose]?: () => void }
      if (tensorOwners.has(tensor)) return
      const nativeDispose = tensor.dispose
      if (typeof nativeDispose !== "function") throw new TypeError("Session can only adopt a Tensor")
      const dispose = () => {
        if (disposedTensors.has(tensor)) return
        disposedTensors.add(tensor)
        tensorOwners.delete(tensor)
        nativeDispose.call(tensor)
      }
      Object.defineProperties(tensor, {
        disposed: { get: () => disposedTensors.has(tensor), enumerable: true },
        dispose: { value: dispose },
        [Symbol.dispose]: { value: dispose },
      })
      tensorOwners.set(tensor, this)
    }
  }
  applyOptimizer(source: Program, state: ExecutionState, gradients: readonly unknown[], optimizer: Optimizer, parameterProvenances?: readonly string[]): void {
    const available = source.inspect().parameters
    const parameters = parameterProvenances
      ? parameterProvenances.map(provenance => available.find(parameter => parameter.provenance === provenance)!)
      : available
    if (parameters.length !== gradients.length) throw new Error("gradient result does not match Program parameters")
    if (optimizer.kind === "accumulate") {
      const previousMicrostep = state.optimizer_state["$microstep"] ?? 0
      if (typeof previousMicrostep !== "number" || !Number.isSafeInteger(previousMicrostep) || previousMicrostep < 0 || previousMicrostep >= optimizer.steps) {
        throw new TypeError("ExecutionState optimizer microstep must be an integer within the accumulation window")
      }
      const boundary = previousMicrostep + 1 === optimizer.steps
      const accumulated: Array<{ key: string; old?: any; next: any }> = []
      const averaged: any[] = []
      let retained = false
      try {
        for (let index = 0; index < parameters.length; index++) {
          const formal = parameters[index]
          const gradient = gradients[index] as any
          if (!gradient) throw new TypeError(`missing optimizer value for ${formal.provenance}`)
          const key = `${formal.provenance}/accumulate/gradient`
          const old = state.optimizer_state[key] as any
          if (previousMicrostep > 0 && !old) throw new TypeError(`missing accumulated gradient for ${formal.provenance}`)
          let next: any
          if (old) {
            const sum = program(`accumulate_gradient_${index}`, p => $formalOperation("add", [
              p.argument("accumulated", formal.spec),
              p.argument("gradient", formal.spec),
            ]))
            next = this.compile(sum).run({ accumulated: old, gradient })
          } else {
            next = this.tensor(gradient.to_array(), { dtype: formal.spec.dtype, axes: formal.spec.axes })
          }
          accumulated.push({ key, old, next })
        }

        if (!boundary) {
          for (const change of accumulated) {
            state.optimizer_state[change.key] = change.next
            change.old?.dispose?.()
          }
          state.optimizer_state["$microstep"] = previousMicrostep + 1
          retained = true
          return
        }

        for (let index = 0; index < parameters.length; index++) {
          const formal = parameters[index]
          const average = program(`average_accumulated_gradient_${index}`, p => $formalOperation("div", [
            p.argument("gradient", formal.spec),
            p.constant("steps", optimizer.steps, scalarSpec(formal.spec.dtype)),
          ]))
          averaged.push(this.compile(average).run({ gradient: accumulated[index].next }))
        }
        this.applyOptimizer(source, state, averaged, optimizer.optimizer, parameterProvenances)
        for (const change of accumulated) {
          delete state.optimizer_state[change.key]
          change.old?.dispose?.()
        }
        state.optimizer_state["$microstep"] = 0
      } finally {
        if (!retained) for (const change of accumulated) change.next?.dispose?.()
        for (const value of averaged) value?.dispose?.()
      }
      return
    }
    if (optimizer.kind === "scheduled") {
      const previousStep = state.optimizer_state["$step"] ?? 0
      if (typeof previousStep !== "number" || !Number.isSafeInteger(previousStep) || previousStep < 0) throw new TypeError("ExecutionState optimizer step must be a non-negative integer")
      const learning_rate = learningRateAt(optimizer.schedule, previousStep)
      this.applyOptimizer(source, state, gradients, Object.freeze({ ...optimizer.optimizer, learning_rate }), parameterProvenances)
      return
    }
    const previousStep = state.optimizer_state["$step"] ?? 0
    if (typeof previousStep !== "number" || !Number.isSafeInteger(previousStep) || previousStep < 0) throw new TypeError("ExecutionState optimizer step must be a non-negative integer")
    const step = previousStep + 1
    const nextParameters: Array<{ key: string; old: any; next: any }> = []
    const nextOptimizer: Array<{ key: string; old?: any; next: any }> = []
    try {
      for (let index = 0; index < parameters.length; index++) {
        const formal = parameters[index]
        const parameter = state.parameters[formal.provenance] as any
        const grad = gradients[index] as any
        if (!parameter || !grad) throw new TypeError(`missing optimizer value for ${formal.provenance}`)
        const name = `optimizer_update_${index}`
        const update = program(name, p => {
          const current = p.argument("parameter", formal.spec)
          const derivative = p.argument("gradient", formal.spec)
          const scalar = (label: string, value: number) => p.constant(label, value, scalarSpec(formal.spec.dtype))
          const learningRate = scalar("learning_rate", optimizer.learning_rate)
          if (optimizer.kind === "sgd") {
            if ((optimizer.momentum ?? 0) === 0) return $formalOperation("sub", [current, $formalOperation("mul", [derivative, learningRate])])
            const velocity = p.argument("velocity", formal.spec)
            const nextVelocity = $formalOperation("add", [$formalOperation("mul", [velocity, scalar("momentum", optimizer.momentum ?? 0)]), derivative])
            return [$formalOperation("sub", [current, $formalOperation("mul", [nextVelocity, learningRate])]), nextVelocity]
          }
          const first = p.argument("first_moment", formal.spec)
          const second = p.argument("second_moment", formal.spec)
          const beta1 = optimizer.beta1 ?? 0.9
          const beta2 = optimizer.beta2 ?? 0.999
          const nextFirst = $formalOperation("add", [$formalOperation("mul", [first, scalar("beta1", beta1)]), $formalOperation("mul", [derivative, scalar("one_minus_beta1", 1 - beta1)])])
          const nextSecond = $formalOperation("add", [$formalOperation("mul", [second, scalar("beta2", beta2)]), $formalOperation("mul", [$formalOperation("mul", [derivative, derivative]), scalar("one_minus_beta2", 1 - beta2)])])
          const correctedFirst = $formalOperation("div", [nextFirst, p.argument("bias_correction1", scalarSpec(formal.spec.dtype))])
          const correctedSecond = $formalOperation("div", [nextSecond, p.argument("bias_correction2", scalarSpec(formal.spec.dtype))])
          let direction = $formalOperation("div", [correctedFirst, $formalOperation("add", [$formalOperation("sqrt", [correctedSecond]), scalar("epsilon", optimizer.epsilon ?? 1e-8)])])
          if (optimizer.kind === "adamw" && (optimizer.weight_decay ?? 0) !== 0) direction = $formalOperation("add", [direction, $formalOperation("mul", [current, scalar("weight_decay", optimizer.weight_decay ?? 0)])])
          return [$formalOperation("sub", [current, $formalOperation("mul", [direction, learningRate])]), nextFirst, nextSecond]
        })
        const args: Record<string, any> = { parameter, gradient: grad }
        const temporary: any[] = []
        const statePrefix = `${formal.provenance}/${optimizer.kind}`
        try {
          if (optimizer.kind === "sgd" && (optimizer.momentum ?? 0) !== 0) {
            const key = `${statePrefix}/velocity`
            const old = state.optimizer_state[key] as any
            args.velocity = old ?? this.tensor(initializedValue(formal.spec, { kind: "zeros" }, seededRandom(0)), { dtype: formal.spec.dtype })
            if (!old) temporary.push(args.velocity)
            const outputs = this.compile(update).run(args) as readonly any[]
            nextParameters.push({ key: formal.provenance, old: parameter, next: outputs[0] })
            nextOptimizer.push({ key, old, next: outputs[1] })
          } else if (optimizer.kind === "adam" || optimizer.kind === "adamw") {
            const firstKey = `${statePrefix}/first_moment`
            const secondKey = `${statePrefix}/second_moment`
            const oldFirst = state.optimizer_state[firstKey] as any
            const oldSecond = state.optimizer_state[secondKey] as any
            args.first_moment = oldFirst ?? this.tensor(initializedValue(formal.spec, { kind: "zeros" }, seededRandom(0)), { dtype: formal.spec.dtype })
            args.second_moment = oldSecond ?? this.tensor(initializedValue(formal.spec, { kind: "zeros" }, seededRandom(0)), { dtype: formal.spec.dtype })
            args.bias_correction1 = this.tensor([1 - Math.pow(optimizer.beta1 ?? 0.9, step)], { dtype: formal.spec.dtype })
            args.bias_correction2 = this.tensor([1 - Math.pow(optimizer.beta2 ?? 0.999, step)], { dtype: formal.spec.dtype })
            if (!oldFirst) temporary.push(args.first_moment)
            if (!oldSecond) temporary.push(args.second_moment)
            temporary.push(args.bias_correction1, args.bias_correction2)
            const outputs = this.compile(update).run(args) as readonly any[]
            nextParameters.push({ key: formal.provenance, old: parameter, next: outputs[0] })
            nextOptimizer.push({ key: firstKey, old: oldFirst, next: outputs[1] }, { key: secondKey, old: oldSecond, next: outputs[2] })
          } else {
            const next = this.compile(update).run(args)
            nextParameters.push({ key: formal.provenance, old: parameter, next })
          }
        } finally {
          for (const value of temporary) value.dispose?.()
        }
      }
      for (const change of nextParameters) { state.parameters[change.key] = change.next; change.old.dispose?.() }
      for (const change of nextOptimizer) { state.optimizer_state[change.key] = change.next; change.old?.dispose?.() }
      state.optimizer_state["$step"] = step
      state.rng_state.counter = Number(state.rng_state.counter ?? 0) + 1
    } catch (error) {
      for (const change of nextParameters) change.next?.dispose?.()
      for (const change of nextOptimizer) change.next?.dispose?.()
      throw error
    }
  }
  owns(value: unknown): boolean {
    return !this.#disposed
      && !!value
      && (typeof value === "object" || typeof value === "function")
      && !disposedTensors.has(value as object)
      && tensorOwners.get(value as object) === this
  }
  dispose(): void {
    if (this.#disposed) return
    for (const executable of this.#cache.values()) executable.dispose()
    this.#cache.clear()
    this.#native.dispose()
    this.#disposed = true
  }
  [Symbol.dispose](): void { this.dispose() }
}
