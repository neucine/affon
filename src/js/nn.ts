import {
  $formalOperation,
  $formalParameter,
  $formalScalarLike,
  $formalCat,
  $formalZerosLike,
  Tensor,
  type Callable,
  type FormalTensor,
  type ProgramDType,
} from "affon:_internal/compute/program"

export interface LossCallable<Input extends FormalTensor = FormalTensor, Target extends FormalTensor = FormalTensor>
  extends Callable<{ input: Input; target: Target }, FormalTensor> {}

export type LinearOptions = Readonly<{
  out_features: number
  bias?: boolean
}>

export type Conv1DOptions = Readonly<{
  out_channels: number
  kernel_size: number
  stride?: number
  padding?: number
  dilation?: number
  groups?: number
  bias?: boolean
}>

export type ConvTranspose1DOptions = Conv1DOptions & Readonly<{ output_padding?: number }>

export type EmbeddingOptions = Readonly<{
  num_embeddings: number
  embedding_dim: number
  dtype?: ProgramDType
}>

export type LayerNormOptions = Readonly<{
  normalized_shape?: number
  epsilon?: number
  affine?: boolean
}>

export type LossOptions = Readonly<{
  target?: string
  reduction?: "mean"
}>

type LossKind = "cross_entropy" | "mean_squared_error" | "binary_cross_entropy" | "binary_cross_entropy_with_logits"
const LOSS_CALLABLE = Symbol.for("affon.nn.loss_callable")

function assertInstanceName(name: string, label: string): void {
  if (typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*(\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*$/.test(name)) {
    throw new TypeError(`${label} name must be a dot-separated identifier`)
  }
}

function binding<T extends FormalTensor>(bindings: unknown, key: string, label: string): T {
  if (!bindings || typeof bindings !== "object" || Array.isArray(bindings)) throw new TypeError(`${label} expects named bindings`)
  const names = Object.keys(bindings)
  if (names.length !== 1 || names[0] !== key) throw new TypeError(`${label} expects only the ${key} binding`)
  return (bindings as Record<string, T>)[key]
}

function positiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${label} must be a positive integer`)
}

function convolutionOptions(options: Conv1DOptions, label: string) {
  if (!options || typeof options !== "object") throw new TypeError(`${label} expects options`)
  const { out_channels, kernel_size } = options
  const stride = options.stride ?? 1
  const padding = options.padding ?? 0
  const dilation = options.dilation ?? 1
  const groups = options.groups ?? 1
  const bias = options.bias ?? true
  for (const [name, value] of Object.entries({ out_channels, kernel_size, stride, dilation, groups })) positiveInteger(value, `${label} ${name}`)
  if (!Number.isSafeInteger(padding) || padding < 0) throw new TypeError(`${label} padding must be a non-negative integer`)
  if (typeof bias !== "boolean") throw new TypeError(`${label} bias must be boolean`)
  if (out_channels % groups !== 0) throw new TypeError(`${label} out_channels must be divisible by groups`)
  return { out_channels, kernel_size, stride, padding, dilation, groups, bias }
}

function operation(op: string, values: FormalTensor[], options: unknown[] = []): FormalTensor {
  return $formalOperation(op, values, options)
}

function temporalSlice(x: FormalTensor, start: number, stop: number, step = 1): FormalTensor {
  return operation("slice", [x], [[
    { start: 0, stop: x.spec.shape[0] },
    { start: 0, stop: x.spec.shape[1] },
    { start, stop, step },
  ]])
}

function channelSlice(x: FormalTensor, start: number, stop: number): FormalTensor {
  return operation("slice", [x], [[
    { start: 0, stop: x.spec.shape[0] },
    { start, stop },
    { start: 0, stop: x.spec.shape[2] },
  ]])
}

function filterSlice(x: FormalTensor, start: number, stop: number): FormalTensor {
  return operation("slice", [x], [[
    { start, stop },
    { start: 0, stop: x.spec.shape[1] },
    { start: 0, stop: x.spec.shape[2] },
  ]])
}

function padded(x: FormalTensor, left: number, right: number): FormalTensor {
  const [batch, channels, length] = x.spec.shape
  if (![left, right, length + Math.max(0, left) + Math.max(0, right)].every(Number.isSafeInteger)) throw new TypeError("convolution padded length exceeds the safe integer range")
  const cropLeft = Math.max(0, -left)
  const cropRight = Math.max(0, -right)
  if (cropLeft + cropRight > length) throw new TypeError("convolution padding removes the entire input")
  const middle = cropLeft || cropRight ? temporalSlice(x, cropLeft, length - cropRight) : x
  const parts = [
    ...(left > 0 ? [$formalZerosLike(x, [batch, channels, left])] : []),
    middle,
    ...(right > 0 ? [$formalZerosLike(x, [batch, channels, right])] : []),
  ]
  return parts.length === 1 ? parts[0] : $formalCat(parts, 2)
}

function convolveValid(x: FormalTensor, weight: FormalTensor, stride: number, dilation: number, groups: number): FormalTensor {
  const length = x.spec.shape[2]
  const [outChannels, channelsPerGroup, kernel] = weight.spec.shape
  const effective = dilation * (kernel - 1) + 1
  if (!Number.isSafeInteger(effective)) throw new TypeError("convolution effective kernel exceeds the safe integer range")
  if (length < effective) throw new TypeError("convolution kernel exceeds padded input length")
  const outputLength = Math.floor((length - effective) / stride) + 1
  if (!Number.isSafeInteger(outputLength)) throw new TypeError("convolution output length exceeds the safe integer range")
  const outputs: FormalTensor[] = []
  for (let group = 0; group < groups; group++) {
    const input = channelSlice(x, group * channelsPerGroup, (group + 1) * channelsPerGroup)
    const groupOutputs = outChannels / groups
    const filter = filterSlice(weight, group * groupOutputs, (group + 1) * groupOutputs)
    const tapMajor = operation("transpose", [filter], [[0, 2, 1]])
    const matrix = operation("transpose", [operation("reshape", [operation("contiguous", [tapMajor])], [[groupOutputs, channelsPerGroup * kernel]])], [[1, 0]])
    const taps: FormalTensor[] = []
    for (let tap = 0; tap < kernel; tap++) {
      const offset = tap * dilation
      const selected = temporalSlice(input, offset, offset + (outputLength - 1) * stride + 1, stride)
      taps.push(operation("transpose", [selected], [[0, 2, 1]]))
    }
    const patches = taps.length === 1 ? taps[0] : $formalCat(taps, 2)
    // A shared filter does not need batched GEMM. Flatten sample/time rows so
    // the backend can select its tuned matrix implementation (e.g. MPS).
    const rows = operation("reshape", [patches], [[x.spec.shape[0] * outputLength, channelsPerGroup * kernel]])
    const projected = operation("matmul", [rows, matrix])
    const sequence = operation("reshape", [projected], [[x.spec.shape[0], outputLength, groupOutputs]])
    outputs.push(operation("transpose", [sequence], [[0, 2, 1]]))
  }
  return outputs.length === 1 ? outputs[0] : $formalCat(outputs, 1)
}

/** Learned one-dimensional convolution over NCL tensors. */
export function conv1d(options: Conv1DOptions): Callable<{ x: FormalTensor }, FormalTensor> {
  const config = convolutionOptions(options, "conv1d")
  return Object.freeze((bindings: { x: FormalTensor }, name = "conv1d") => {
    assertInstanceName(name, "conv1d")
    const x = binding<FormalTensor>(bindings, "x", "conv1d")
    if (x.spec.shape.length !== 3) throw new TypeError("conv1d input must have NCL rank three")
    const channels = x.spec.shape[1]
    positiveInteger(channels, "conv1d input channels")
    if (channels % config.groups !== 0) throw new TypeError("conv1d input channels must be divisible by groups")
    const weight = $formalParameter(x, `${name}.weight`, Tensor.spec(x.spec.dtype, [config.out_channels, channels / config.groups, config.kernel_size]), { initializer: { kind: "xavier_uniform" } })
    let result = convolveValid(padded(x, config.padding, config.padding), weight, config.stride, config.dilation, config.groups)
    if (config.bias) {
      const bias = $formalParameter(x, `${name}.bias`, Tensor.spec(x.spec.dtype, [config.out_channels]), { initializer: { kind: "zeros" } })
      result = operation("add", [result, operation("reshape", [bias], [[1, config.out_channels, 1]])])
    }
    return result
  })
}

/** Learned transposed one-dimensional convolution over NCL tensors. */
export function conv_transpose1d(options: ConvTranspose1DOptions): Callable<{ x: FormalTensor }, FormalTensor> {
  const config = convolutionOptions(options, "conv_transpose1d")
  const outputPadding = options.output_padding ?? 0
  if (!Number.isSafeInteger(outputPadding) || outputPadding < 0 || outputPadding >= config.stride) throw new TypeError("conv_transpose1d output_padding must be non-negative and smaller than stride")
  return Object.freeze((bindings: { x: FormalTensor }, name = "conv_transpose1d") => {
    assertInstanceName(name, "conv_transpose1d")
    const x = binding<FormalTensor>(bindings, "x", "conv_transpose1d")
    if (x.spec.shape.length !== 3) throw new TypeError("conv_transpose1d input must have NCL rank three")
    const [batch, channels, length] = x.spec.shape
    positiveInteger(channels, "conv_transpose1d input channels")
    positiveInteger(length, "conv_transpose1d input length")
    if (channels % config.groups !== 0) throw new TypeError("conv_transpose1d input channels must be divisible by groups")
    if (!Number.isSafeInteger(length * config.stride)) throw new TypeError("conv_transpose1d upsampled length exceeds the safe integer range")
    const weight = $formalParameter(x, `${name}.weight`, Tensor.spec(x.spec.dtype, [channels, config.out_channels / config.groups, config.kernel_size]), { initializer: { kind: "xavier_uniform" } })
    let upsampled = x
    if (config.stride > 1) {
      const samples = operation("reshape", [x], [[batch, channels, length, 1]])
      const gaps = $formalZerosLike(samples, [batch, channels, length, config.stride - 1])
      const interleaved = operation("reshape", [$formalCat([samples, gaps], 3)], [[batch, channels, length * config.stride]])
      upsampled = temporalSlice(interleaved, 0, (length - 1) * config.stride + 1)
    }
    const extent = config.dilation * (config.kernel_size - 1)
    const prepared = padded(upsampled, extent - config.padding, extent - config.padding + outputPadding)
    const grouped: FormalTensor[] = []
    const inputGroup = channels / config.groups
    const outputGroup = config.out_channels / config.groups
    for (let group = 0; group < config.groups; group++) {
      const input = channelSlice(prepared, group * inputGroup, (group + 1) * inputGroup)
      const filter = filterSlice(weight, group * inputGroup, (group + 1) * inputGroup)
      const reversed = $formalCat(Array.from({ length: config.kernel_size }, (_, tap) => temporalSlice(filter, config.kernel_size - 1 - tap, config.kernel_size - tap)), 2)
      const transposed = operation("transpose", [reversed], [[1, 0, 2]])
      grouped.push(convolveValid(input, transposed, 1, config.dilation, 1))
    }
    let result = grouped.length === 1 ? grouped[0] : $formalCat(grouped, 1)
    if (config.bias) {
      const bias = $formalParameter(x, `${name}.bias`, Tensor.spec(x.spec.dtype, [config.out_channels]), { initializer: { kind: "zeros" } })
      result = operation("add", [result, operation("reshape", [bias], [[1, config.out_channels, 1]])])
    }
    return result
  })
}

function lossBindings(bindings: unknown, label: string): { input: FormalTensor; target: FormalTensor } {
  if (!bindings || typeof bindings !== "object" || Array.isArray(bindings)) throw new TypeError(`${label} expects named bindings`)
  const names = Object.keys(bindings).sort()
  if (names.length !== 2 || names[0] !== "input" || names[1] !== "target") {
    throw new TypeError(`${label} expects only the input and target bindings`)
  }
  return bindings as { input: FormalTensor; target: FormalTensor }
}

function lossCallable(
  kind: LossKind,
  options: LossOptions,
  defaultTarget: string,
  author: (input: FormalTensor, target: FormalTensor) => FormalTensor,
): LossCallable {
  if (!options || typeof options !== "object") throw new TypeError(`${kind} expects options`)
  const target = options.target ?? defaultTarget
  if (typeof target !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(target)) throw new TypeError(`${kind} target must be a JavaScript identifier`)
  if (options.reduction !== undefined && options.reduction !== "mean") throw new TypeError(`${kind} currently supports only mean reduction`)
  const callable = ((bindings: { input: FormalTensor; target: FormalTensor }, name = kind) => {
    assertInstanceName(name, kind)
    const values = lossBindings(bindings, kind)
    return author(values.input, values.target)
  }) as LossCallable
  Object.defineProperty(callable, LOSS_CALLABLE, { value: Object.freeze({ kind, target }) })
  return Object.freeze(callable)
}

export function linear(options: LinearOptions): Callable<{ x: FormalTensor }, FormalTensor> {
  if (!options || typeof options !== "object") throw new TypeError("linear expects options")
  const out_features = options.out_features
  const bias = options.bias ?? true
  if (!Number.isSafeInteger(out_features) || out_features <= 0) throw new TypeError("linear out_features must be a positive integer")
  if (typeof bias !== "boolean") throw new TypeError("linear bias must be boolean")

  return Object.freeze((bindings: { x: FormalTensor }, name = "linear") => {
    assertInstanceName(name, "linear")
    const x = binding<FormalTensor>(bindings, "x", "linear")
    const in_features = x?.spec.shape.at(-1)
    if (!Number.isSafeInteger(in_features) || in_features! <= 0) throw new TypeError("linear requires a known positive input feature size")
    const weight = $formalParameter(x, `${name}.weight`, Tensor.spec(x.spec.dtype, [in_features!, out_features]), {
      initializer: { kind: "xavier_uniform" },
    })
    let result = $formalOperation("matmul", [x, weight])
    if (bias) {
      const offset = $formalParameter(x, `${name}.bias`, Tensor.spec(x.spec.dtype, [out_features]), {
        initializer: { kind: "zeros" },
      })
      result = $formalOperation("add", [result, offset])
    }
    return result
  })
}

export function embedding(options: EmbeddingOptions): Callable<{ indices: FormalTensor }, FormalTensor> {
  if (!options || typeof options !== "object") throw new TypeError("embedding expects options")
  const { num_embeddings, embedding_dim } = options
  const dtype = options.dtype ?? "f32"
  if (!Number.isSafeInteger(num_embeddings) || num_embeddings <= 0) throw new TypeError("embedding num_embeddings must be a positive integer")
  if (!Number.isSafeInteger(embedding_dim) || embedding_dim <= 0) throw new TypeError("embedding embedding_dim must be a positive integer")
  if (dtype !== "bf16" && dtype !== "f32" && dtype !== "f64" && dtype !== "u32" && dtype !== "i64") throw new TypeError("embedding dtype is unsupported")

  return Object.freeze((bindings: { indices: FormalTensor }, name = "embedding") => {
    assertInstanceName(name, "embedding")
    const indices = binding<FormalTensor>(bindings, "indices", "embedding")
    if (indices?.spec.dtype !== "i64") throw new TypeError("embedding indices must have i64 dtype")
    const weight = $formalParameter(indices, `${name}.weight`, Tensor.spec(dtype, [num_embeddings, embedding_dim]), {
      initializer: { kind: "normal", mean: 0, standard_deviation: 0.02 },
    })
    return $formalOperation("embedding", [weight, indices])
  })
}

export function layer_norm(options: LayerNormOptions = {}): Callable<{ x: FormalTensor }, FormalTensor> {
  if (!options || typeof options !== "object") throw new TypeError("layer_norm expects options")
  const normalized_shape = options.normalized_shape
  const epsilon = options.epsilon ?? 1e-5
  const affine = options.affine ?? true
  if (normalized_shape !== undefined && (!Number.isSafeInteger(normalized_shape) || normalized_shape <= 0)) {
    throw new TypeError("layer_norm normalized_shape must be a positive integer")
  }
  if (!Number.isFinite(epsilon) || epsilon <= 0) throw new TypeError("layer_norm epsilon must be positive and finite")
  if (typeof affine !== "boolean") throw new TypeError("layer_norm affine must be boolean")

  return Object.freeze((bindings: { x: FormalTensor }, name = "layer_norm") => {
    assertInstanceName(name, "layer_norm")
    const x = binding<FormalTensor>(bindings, "x", "layer_norm")
    const width = normalized_shape ?? x?.spec.shape.at(-1) ?? 0
    if (!Number.isSafeInteger(width) || width <= 0) throw new TypeError("layer_norm requires a known positive normalized shape")
    if (x.spec.shape.at(-1) !== width) throw new TypeError("layer_norm normalized_shape must match the last input dimension")
    let result = $formalOperation("layer_norm", [x], [x.spec.shape.length - 1, epsilon])
    if (affine) {
      const weight = $formalParameter(x, `${name}.weight`, Tensor.spec(x.spec.dtype, [width]), { initializer: { kind: "ones" } })
      const bias = $formalParameter(x, `${name}.bias`, Tensor.spec(x.spec.dtype, [width]), { initializer: { kind: "zeros" } })
      result = $formalOperation("add", [$formalOperation("mul", [result, weight]), bias])
    }
    return result
  })
}

/** Configure indexed multiclass cross entropy over the final input dimension. */
export function cross_entropy(options: LossOptions = {}): LossCallable {
  return lossCallable("cross_entropy", options, "labels", (input, target) => $formalOperation("cross_entropy", [input, target]))
}

/** Configure mean squared error for same-shaped input and target tensors. */
export function mean_squared_error(options: LossOptions = {}): LossCallable {
  return lossCallable("mean_squared_error", options, "target", (input, target) => {
    const difference = $formalOperation("sub", [input, target])
    return $formalOperation("mean", [$formalOperation("mul", [difference, difference])], [undefined, false])
  })
}

/** Configure binary cross entropy for probabilities and same-shaped targets. */
export function binary_cross_entropy(options: LossOptions = {}): LossCallable {
  return lossCallable("binary_cross_entropy", options, "target", (input, target) => {
    const one = $formalScalarLike(input, 1)
    const epsilon = $formalScalarLike(input, input.spec.dtype === "f32" ? 1e-7 : 1e-15)
    const positive = $formalOperation("mul", [target, $formalOperation("log", [$formalOperation("add", [input, epsilon])])])
    const negative = $formalOperation("mul", [
      $formalOperation("sub", [one, target]),
      $formalOperation("log", [$formalOperation("add", [$formalOperation("sub", [one, input]), epsilon])]),
    ])
    return $formalOperation("neg", [$formalOperation("mean", [$formalOperation("add", [positive, negative])], [undefined, false])])
  })
}

/** Configure numerically stable binary cross entropy for unnormalized logits. */
export function binary_cross_entropy_with_logits(options: LossOptions = {}): LossCallable {
  return lossCallable("binary_cross_entropy_with_logits", options, "target", (input, target) => {
    const one = $formalScalarLike(input, 1)
    const positive = $formalOperation("relu", [input])
    const linear = $formalOperation("mul", [input, target])
    const tail = $formalOperation("log", [$formalOperation("add", [one, $formalOperation("exp", [$formalOperation("neg", [$formalOperation("abs", [input])])])])])
    return $formalOperation("mean", [$formalOperation("add", [$formalOperation("sub", [positive, linear]), tail])], [undefined, false])
  })
}
