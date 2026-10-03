import {
  $formalOperation,
  $formalParameter,
  $formalScalarLike,
  Tensor,
  type Callable,
  type FormalTensor,
  type ProgramDType,
} from "affon:_internal/compute/program"

export interface Layer<Bindings extends Record<string, FormalTensor>, Out extends FormalTensor = FormalTensor> extends Callable<Bindings, Out> {}

export interface LossCallable<Input extends FormalTensor = FormalTensor, Target extends FormalTensor = FormalTensor>
  extends Callable<{ input: Input; target: Target }, FormalTensor> {}

export type LinearOptions = Readonly<{
  out_features: number
  bias?: boolean
}>

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

export function linear(options: LinearOptions): Layer<{ x: FormalTensor }> {
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

export function embedding(options: EmbeddingOptions): Layer<{ indices: FormalTensor }> {
  if (!options || typeof options !== "object") throw new TypeError("embedding expects options")
  const { num_embeddings, embedding_dim } = options
  const dtype = options.dtype ?? "f32"
  if (!Number.isSafeInteger(num_embeddings) || num_embeddings <= 0) throw new TypeError("embedding num_embeddings must be a positive integer")
  if (!Number.isSafeInteger(embedding_dim) || embedding_dim <= 0) throw new TypeError("embedding embedding_dim must be a positive integer")
  if (dtype !== "f32" && dtype !== "f64" && dtype !== "i64") throw new TypeError("embedding dtype is unsupported")

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

export function layer_norm(options: LayerNormOptions = {}): Layer<{ x: FormalTensor }> {
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
