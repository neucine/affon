import native from "affon:compute/native"
import type { Optimizer } from "affon:optim"

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

type ProgramNode = Readonly<{
  id: number
  kind: NodeKind
  role?: Exclude<FormalRole, "intermediate">
  name?: string
  provenance?: string
  spec: TensorSpec
  op?: string
  operands?: readonly number[]
  options?: Readonly<Record<string, unknown>>
  value?: ProgramValue
}>

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
const optimizationSources = new WeakMap<Program, { loss: Program; optimizer: Optimizer }>()

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

export const Tensor = Object.freeze({
  spec(dtype: ProgramDType, shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec {
    return freezeSpec(dtype, shape, options.axes)
  },
  f32(shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec { return freezeSpec("f32", shape, options.axes) },
  f64(shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec { return freezeSpec("f64", shape, options.axes) },
  i64(shape: readonly number[], options: { axes?: readonly string[] } = {}): TensorSpec { return freezeSpec("i64", shape, options.axes) },
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
      case "add": case "sub": case "mul": case "div": case "matmul": case "dot": case "embedding": return this.dispatchBinary(op, operands[1])
      case "masked_fill": return this.#masked_fill(operands[1], options[0] as number)
      case "index_select": return this.#index_select(options[0] as number, operands[1])
      case "softmax": return this.#softmax(options[0] as number)
      case "cast": return this.#cast(options[0] as ProgramDType)
      case "sum": return this.#sum(options[0] as number | undefined, options[1] as boolean | undefined)
      case "mean": return this.#mean(options[0] as number | undefined, options[1] as boolean | undefined)
      case "reshape": return this.#reshape(options[0] as readonly number[], options[1] as readonly string[] | undefined)
      case "transpose": return this.#transpose(options[0] as readonly number[] | undefined)
      case "slice": return this.#slice(options[0] as readonly SliceRange[])
      case "squeeze": return this.#squeeze(options[0] as number | undefined)
      case "unsqueeze": return this.#unsqueeze(options[0] as number, options[1] as string | undefined)
      case "layer_norm": return this.#layer_norm(options[0] as number, options[1] as number | undefined)
      case "neg": return this.#neg(); case "abs": return this.#abs(); case "exp": return this.#exp(); case "log": return this.#log()
      case "sqrt": return this.#sqrt(); case "relu": return this.#relu(); case "sigmoid": return this.#sigmoid(); case "silu": return this.#silu()
      case "tanh": return this.#tanh(); case "erf": return this.#erf(); case "gelu": return this.#gelu(); case "contiguous": return this.#contiguous()
      default: throw new TypeError(`unsupported formal operation: ${op}`)
    }
  }

  private dispatchBinary(op: string, other: FormalTensor): FormalTensor {
    switch (op) {
      case "add": return this.#add(other); case "sub": return this.#sub(other); case "mul": return this.#mul(other); case "div": return this.#div(other)
      case "matmul": return this.#matmul(other); case "dot": return this.#dot(other); case "embedding": return this.#embedding(other)
      default: throw new TypeError(`unsupported binary formal operation: ${op}`)
    }
  }

  #add(other: FormalTensor): FormalTensor { return this.binary("add", other, arithmeticSpec(this.spec, other.spec)) }
  #sub(other: FormalTensor): FormalTensor { return this.binary("sub", other, arithmeticSpec(this.spec, other.spec)) }
  #mul(other: FormalTensor): FormalTensor { return this.binary("mul", other, arithmeticSpec(this.spec, other.spec)) }
  #div(other: FormalTensor): FormalTensor { return this.binary("div", other, arithmeticSpec(this.spec, other.spec)) }
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

function assertName(name: string, label: string): void {
  if (typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new TypeError(`${label} name must be a JavaScript identifier`)
}

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

export interface ProgramNN {
  linear(value: FormalTensor, options: { name: string; out_features: number; bias?: boolean }): FormalTensor
  embedding(indices: FormalTensor, options: { name: string; num_embeddings: number; embedding_dim: number; dtype?: ProgramDType }): FormalTensor
  layer_norm(value: FormalTensor, options: { name: string; normalized_shape?: number; epsilon?: number; affine?: boolean }): FormalTensor
  cross_entropy(logits: FormalTensor, labels: FormalTensor): FormalTensor
}

class BoundProgramNN implements ProgramNN {
  constructor(private readonly builder: ProgramBuilder) {}

  linear(value: FormalTensor, options: { name: string; out_features: number; bias?: boolean }): FormalTensor {
    assertName(options.name, "linear")
    const inFeatures = value.spec.shape.at(-1)
    if (!Number.isSafeInteger(inFeatures) || !Number.isSafeInteger(options.out_features) || options.out_features <= 0) throw new TypeError("linear requires known positive feature sizes")
    const weight = this.builder.parameter(`${options.name}_weight`, Tensor.spec(value.spec.dtype, [inFeatures!, options.out_features]), { initializer: { kind: "xavier_uniform" } })
    let result = $formalOperation("matmul", [value, weight])
    if (options.bias !== false) result = $formalOperation("add", [result, this.builder.parameter(`${options.name}_bias`, Tensor.spec(value.spec.dtype, [options.out_features]), { initializer: { kind: "zeros" } })])
    return result
  }

  embedding(indices: FormalTensor, options: { name: string; num_embeddings: number; embedding_dim: number; dtype?: ProgramDType }): FormalTensor {
    assertName(options.name, "embedding")
    if (!Number.isSafeInteger(options.num_embeddings) || options.num_embeddings <= 0) throw new TypeError("embedding num_embeddings must be a positive integer")
    if (!Number.isSafeInteger(options.embedding_dim) || options.embedding_dim <= 0) throw new TypeError("embedding embedding_dim must be a positive integer")
    const table = this.builder.parameter(`${options.name}_weight`, Tensor.spec(options.dtype ?? "f32", [options.num_embeddings, options.embedding_dim]), { initializer: { kind: "normal", mean: 0, standard_deviation: 0.02 } })
    return $formalOperation("embedding", [table, indices])
  }

  layer_norm(value: FormalTensor, options: { name: string; normalized_shape?: number; epsilon?: number; affine?: boolean }): FormalTensor {
    assertName(options.name, "layer_norm")
    const width = options.normalized_shape ?? value.spec.shape.at(-1) ?? 0
    if (!Number.isSafeInteger(width) || width <= 0) throw new TypeError("layer_norm normalized_shape must be a positive integer")
    if (value.spec.shape.at(-1) !== width) throw new TypeError("layer_norm normalized_shape must match the last input dimension")
    const epsilon = options.epsilon ?? 1e-5
    if (!Number.isFinite(epsilon) || epsilon <= 0) throw new TypeError("layer_norm epsilon must be positive and finite")
    let result = $formalOperation("layer_norm", [value], [value.spec.shape.length - 1, epsilon])
    if (options.affine !== false) {
      const weight = this.builder.parameter(`${options.name}_weight`, Tensor.spec(value.spec.dtype, [width]), { initializer: { kind: "ones" } })
      const bias = this.builder.parameter(`${options.name}_bias`, Tensor.spec(value.spec.dtype, [width]), { initializer: { kind: "zeros" } })
      result = $formalOperation("add", [$formalOperation("mul", [result, weight]), bias])
    }
    return result
  }

  cross_entropy(logits: FormalTensor, labels: FormalTensor): FormalTensor {
    if (logits.spec.dtype !== "f32" && logits.spec.dtype !== "f64") throw new TypeError("cross_entropy logits must have a floating-point dtype")
    if (labels.spec.dtype !== "i64") throw new TypeError("cross_entropy labels must have i64 dtype")
    if (logits.spec.shape.length === 0) throw new TypeError("cross_entropy logits must include a class dimension")
    const labelShape = logits.spec.shape.slice(0, -1)
    if (labels.spec.shape.length !== labelShape.length || labels.spec.shape.some((size, index) => size !== labelShape[index])) {
      throw new TypeError("cross_entropy labels must match the logits shape without its class dimension")
    }
    const result = scalarSpec(logits.spec.dtype)
    if (logits.spec.shape.length <= 2) return this.builder[EMIT]("cross_entropy", [logits, labels], result, { reduction: "mean", class_axis: -1 })
    const classes = logits.spec.shape.at(-1)!
    const rows = labelShape.reduce((product, value) => product * value, 1)
    return this.builder[EMIT]("cross_entropy", [$formalOperation("reshape", [logits], [[rows, classes]]), $formalOperation("reshape", [labels], [[rows]])], result, { reduction: "mean", class_axis: 1 })
  }
}

export class ProgramBuilder {
  readonly [BUILDER] = true
  readonly nn: ProgramNN
  private readonly programName: string
  private readonly nodeList: ProgramNode[] = []
  private readonly names = new Map<string, ProgramNode>()
  private finished = false

  constructor(name: string) { this.programName = name; this.nn = Object.freeze(new BoundProgramNN(this)) }

  private formal(role: "argument" | "parameter" | "state", name: string, spec: TensorSpec): FormalTensor {
    if (this.finished) throw new Error("ProgramBuilder may only be used inside program()")
    assertName(name, role)
    if (this.names.has(name)) throw new TypeError(`duplicate formal name: ${name}`)
    const node = Object.freeze({ id: this.nodeList.length, kind: role, role, name, provenance: `${this.programName}.${name}`, spec })
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
  constant(name: string, value: ProgramValue, spec: TensorSpec): FormalTensor {
    assertName(name, "constant")
    if (this.names.has(name)) throw new TypeError(`duplicate formal name: ${name}`)
    const snapshot = snapshotTensorData(value, { label: "constant", finite: true, integer: spec.dtype === "i64" })
    const scalarShorthand = typeof snapshot.data === "number" && elementCount(spec.shape) === 1
    if (!scalarShorthand && (snapshot.shape.length !== spec.shape.length || snapshot.shape.some((size, index) => size !== spec.shape[index]))) {
      throw new TypeError("constant value does not match its TensorSpec")
    }
    const node = Object.freeze({ id: this.nodeList.length, kind: "constant" as const, role: "constant" as const, name, provenance: `${this.programName}.${name}`, spec, value: snapshot.data })
    this.nodeList.push(node)
    this.names.set(name, node)
    return new FormalTensor(this, node.id, spec, "constant", name)
  }

  [EMIT](op: string, operands: readonly FormalTensor[], spec: TensorSpec, options?: Readonly<Record<string, unknown>>): FormalTensor {
    if (operands.some(value => !(value instanceof FormalTensor) || (value as any).owner !== this)) throw new TypeError(`${op} operands must belong to this ProgramBuilder`)
    const node = Object.freeze({ id: this.nodeList.length, kind: "operation" as const, op, operands: Object.freeze(operands.map(value => value.id)), spec, ...(options ? { options: freezeMetadata({ ...options }) } : {}) })
    this.nodeList.push(node)
    return new FormalTensor(this, node.id, spec, "intermediate")
  }

  reduction(op: string, value: FormalTensor, axis?: number, keep_dims = false): FormalTensor {
    if (axis === undefined) return this[EMIT](op, [value], scalarSpec(value.spec.dtype), { keep_dims })
    const normalized = axis < 0 ? value.spec.shape.length + axis : axis
    if (!Number.isInteger(normalized) || normalized < 0 || normalized >= value.spec.shape.length) throw new TypeError(`${op} axis is out of range`)
    const shape = keep_dims ? value.spec.shape.map((size, index) => index === normalized ? 1 : size) : value.spec.shape.filter((_, index) => index !== normalized)
    return this[EMIT](op, [value], freezeSpec(value.spec.dtype, shape), { axis: normalized, keep_dims })
  }

  use(child: Program, options: { as: string; [name: string]: unknown }): FormalTensor | readonly FormalTensor[] {
    if (!isProgram(child)) throw new TypeError("p.use expects a Program")
    const { as, ...bindings } = options
    assertName(as, "composition alias")
    return this.compose(child, bindings as Record<string, FormalTensor>, as)
  }

  compose(child: Program, bindingsOrArguments: readonly FormalTensor[] | Record<string, FormalTensor>, alias = child.name): FormalTensor | readonly FormalTensor[] {
    const inspection = child.inspect()
    if (inspection.kind !== "authored") throw new TypeError("only authored Programs can be composed")
    const expectedNames = new Set(inspection.arguments.map(formal => formal.name))
    if (Array.isArray(bindingsOrArguments)) {
      if (bindingsOrArguments.length !== inspection.arguments.length) throw new TypeError(`${child.name} expects ${inspection.arguments.length} arguments`)
    } else {
      for (const name of Object.keys(bindingsOrArguments)) if (!expectedNames.has(name)) throw new TypeError(`unknown ${child.name} argument: ${name}`)
    }
    const supplied: Record<string, FormalTensor | undefined> = Array.isArray(bindingsOrArguments)
      ? Object.fromEntries(inspection.arguments.map((formal, index) => [formal.name, bindingsOrArguments[index]]))
      : bindingsOrArguments as Record<string, FormalTensor>
    const mapped = new Map<number, FormalTensor>()
    for (const formal of inspection.arguments) {
      const value = supplied[formal.name]
      if (!(value instanceof FormalTensor)) throw new TypeError(`${child.name} requires argument ${formal.name}`)
      if (!sameSpec(value.spec, formal.spec)) throw new TypeError(`${child.name}.${formal.name} does not match its TensorSpec`)
      mapped.set(inspection.nodes.find(node => node.name === formal.name && node.role === "argument")!.id, value)
    }
    for (const node of inspection.nodes) {
      if (mapped.has(node.id)) continue
      if (node.role === "parameter" || node.role === "state") {
        const localName = `${alias}_${node.name}`
        const formal = this.formal(node.role, localName, node.spec)
        if (node.options) (this.nodeList[formal.id] as any) = Object.freeze({ ...this.nodeList[formal.id], options: node.options })
        mapped.set(node.id, formal)
      } else if (node.role === "constant") {
        const localName = `${alias}_${node.name}`
        mapped.set(node.id, this.constant(localName, node.value!, node.spec))
      } else if (node.operands) {
        mapped.set(node.id, this[EMIT](node.op ?? "composition", node.operands.map(id => mapped.get(id)!), node.spec, { ...(node.options ?? {}), from: child.name, as: alias }))
      }
    }
    const outputs = inspection.outputs.map(id => mapped.get(id)!)
    return outputs.length === 1 ? outputs[0] : Object.freeze(outputs)
  }

  finish(): readonly ProgramNode[] { this.finished = true; return Object.freeze([...this.nodeList]) }
}

export interface Program<Args extends readonly unknown[] = readonly FormalTensor[], Out = FormalTensor | readonly FormalTensor[]> {
  (...arguments_: Args): Out
  (arguments_: Record<string, FormalTensor>): Out
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
    const bindings = args.length === 1 && args[0] && typeof args[0] === "object" && !(args[0] instanceof FormalTensor) ? args[0] : args
    return activeBuilder.compose(callable as Program, bindings as any)
  }) as InternalProgram
  Object.defineProperties(callable, {
    [PROGRAM]: { value: true },
    name: { value: name },
    provenance: { value: provenance, enumerable: true },
    inspect: { value: () => inspection },
  })
  return Object.freeze(callable)
}

export function program<Out extends FormalTensor | readonly FormalTensor[]>(name: string, author: (p: ProgramBuilder) => Out): Program<readonly FormalTensor[], Out> {
  assertName(name, "program")
  if (typeof author !== "function") throw new TypeError("program expects an authoring callback")
  if (activeBuilder) throw new Error("program() cannot be nested; compose an existing Program by calling it")
  const builder = new ProgramBuilder(name)
  activeBuilder = builder
  try {
    const result = author(builder)
    const outputs = Array.isArray(result) ? result : [result]
    if (outputs.length === 0 || outputs.some(value => !(value instanceof FormalTensor))) throw new TypeError("program callback must return a FormalTensor or a non-empty FormalTensor array")
    const nodes = builder.finish()
    return createProgram(name, "authored", nodes, outputs.map(value => value.id)) as Program<readonly FormalTensor[], Out>
  } finally {
    activeBuilder = null
  }
}

export function gradient(loss: Program<readonly FormalTensor[], FormalTensor>, independent_variables: string): Program<readonly FormalTensor[], FormalTensor>
export function gradient(loss: Program<readonly FormalTensor[], FormalTensor>, independent_variables: readonly string[]): Program
export function gradient(loss: Program<readonly FormalTensor[], FormalTensor>, independent_variables: string | readonly string[]): Program {
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
    const node = Object.freeze({ id: nodes.length, kind: "gradient" as const, op: "gradient", operands: Object.freeze([source.outputs[0]]), spec: formal.spec, options: Object.freeze({ with_respect_to: formal.provenance }) })
    nodes.push(node)
    return node.id
  })
  return createProgram(`${loss.name}_gradient`, "gradient", Object.freeze(nodes), outputs)
}

function optimizerSnapshot(value: Optimizer): Optimizer {
  if (!value || typeof value !== "object") throw new TypeError("optimize expects an Optimizer from affon:optim")
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

export function optimize(loss: Program<readonly FormalTensor[], FormalTensor>, optimizer: Optimizer): Program<readonly FormalTensor[], FormalTensor> {
  if (!isProgram(loss)) throw new TypeError("optimize expects a Program")
  const source = loss.inspect()
  const output = source.nodes[source.outputs[0]]
  if (source.outputs.length !== 1 || output.spec.shape.reduce((size, value) => size * value, 1) !== 1) throw new TypeError("optimize requires a scalar Program")
  if (source.parameters.length === 0) throw new TypeError("optimize requires at least one parameter")
  const snapshot = optimizerSnapshot(optimizer)
  const transition = Object.freeze({ kind: "optimize" as const, optimizer: snapshot, parameters: Object.freeze(source.parameters.map(parameter => parameter.provenance)) })
  const result = createProgram(`${loss.name}_${snapshot.kind}`, "optimize", source.nodes, source.outputs, Object.freeze([...source.transitions, transition]))
  optimizationSources.set(result, { loss, optimizer: snapshot })
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
  readonly program: Program<readonly FormalTensor[], Out>
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
  private readonly optimization?: { loss: Executable; gradient: Executable; optimizer: Optimizer }
  #disposed = false

  constructor(session: Session, source: Program<readonly FormalTensor[], Out>, nativeExecutable: any, optimization?: { loss: Executable; gradient: Executable; optimizer: Optimizer }) {
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
      const result = this.optimization.loss.run(arguments_, state)
      let values: any[] = []
      try {
        const gradients = this.optimization.gradient.run(arguments_, state)
        values = Array.isArray(gradients) ? gradients : [gradients]
        this.session.applyOptimizer(this.program, state, values, this.optimization.optimizer)
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
    if (this.#disposed) throw new Error("Session has been disposed")
    const dtype = options.dtype ?? "f32"
    if (dtype !== "f32" && dtype !== "f64" && dtype !== "i64") throw new TypeError(`unsupported Tensor dtype: ${dtype}`)
    const snapshot = snapshotTensorData(values, { label: "Tensor", integer: dtype === "i64" })
    if (options.axes && (options.axes.length !== snapshot.shape.length || options.axes.some(axis => typeof axis !== "string" || axis.length === 0))) {
      throw new TypeError("Tensor axes must contain one non-empty name per dimension")
    }
    const value = native.sessionTensor(this.#native, snapshot.data, dtype)
    if (options.axes) {
      Object.defineProperty(value, "axes", { value: Object.freeze([...options.axes]), enumerable: true })
    }
    this.adopt(value)
    return value as Tensor
  }
  compile<Out extends FormalTensor | readonly FormalTensor[]>(source: Program<readonly FormalTensor[], Out>): Executable<Out> {
    if (this.#disposed) throw new Error("Session has been disposed")
    if (!isProgram(source)) throw new TypeError("Session.compile expects a Program")
    const cached = this.#cache.get(source)
    if (cached && !cached.disposed) return cached as Executable<Out>
    if (cached) this.#cache.delete(source)
    const optimization = optimizationSources.get(source)
    let executable: Executable<Out>
    if (optimization) {
      const names = optimization.loss.inspect().parameters.map(value => value.name)
      executable = new Executable(this, source, null, {
        loss: this.compile(optimization.loss),
        gradient: this.compile(gradient(optimization.loss, names)),
        optimizer: optimization.optimizer,
      })
    } else executable = new Executable(this, source, native.compileProgram(this.#native, JSON.stringify(source.inspect())))
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
  applyOptimizer(source: Program, state: ExecutionState, gradients: readonly unknown[], optimizer: Optimizer): void {
    const parameters = source.inspect().parameters
    if (parameters.length !== gradients.length) throw new Error("gradient result does not match Program parameters")
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
