declare module "affon:compute/graph.ts" {
type DType = "f32" | "f64" | "i64"
type Nested = number | Nested[]
type UnaryKind = "neg" | "relu" | "abs" | "exp" | "log" | "sqrt" | "sigmoid" | "silu" | "tanh" | "sign"
type BinaryKind = "add" | "sub" | "mul" | "div" | "gt"
type ReductionKind = "sum" | "mean" | "std" | "variance" | "min" | "max"
type IndexReductionKind = "argmin" | "argmax"
type MatmulExecutionHint = "projection" | "attention_scores" | "attention_values"
type MatmulExecutionSource = "higher_level_module"

export type NativeTensor = {
  shape: number[]
  dtype: DType
}
export type TensorMeta = {
  shape: number[]
  dtype: DType
}
export type MatmulExecutionOptions = {
  hint?: MatmulExecutionHint
  source?: MatmulExecutionSource
}

export function graph(fn: (...inputs: any[]) => any): any
export function graphWithArity(arity: number, fn: (...inputs: any[]) => any): any
export function graphWithInputMetas(arity: number, metas: TensorMeta[], fn: (...inputs: any[]) => any): any
export function graphTensor(data: number | Nested[], opts?: { dtype?: DType }): any | undefined
export function isCapturing(): boolean
export function withCaptureModulePath<T>(path: string, fn: () => T): T
export function captureBoundTensor(value: unknown): NativeTensor | any
export function captureUnary(kind: UnaryKind, opName: string, input: unknown): NativeTensor | any
export function captureBinary(kind: BinaryKind, opName: string, left: unknown, right: unknown): NativeTensor | any
export function captureClamp(input: unknown, min: unknown, max: unknown): NativeTensor | any
export function captureReshape(input: unknown, shape: unknown, opts?: unknown): NativeTensor | any
export function captureSlice(input: unknown, selectors: unknown[]): NativeTensor | any
export function captureSqueeze(input: unknown, axis?: unknown): NativeTensor | any
export function captureUnsqueeze(input: unknown, axis: unknown): NativeTensor | any
export function captureTranspose(input: unknown, dim1?: unknown, dim2?: unknown): NativeTensor | any
export function capturePermute(input: unknown, axes: unknown): NativeTensor | any
export function captureContiguous(input: unknown): NativeTensor | any
export function captureCast(input: unknown, dtype: DType): NativeTensor | any
export function captureGelu(input: unknown): NativeTensor | any
export function captureDot(left: unknown, right: unknown): NativeTensor | any
export function captureMatmul(left: unknown, right: unknown, execution?: MatmulExecutionOptions): NativeTensor | any
export function captureCat(inputs: unknown[], dim?: unknown): NativeTensor | any
export function captureStack(inputs: unknown[], dim?: unknown): NativeTensor | any
export function captureWhere(cond: unknown, onTrue: unknown, onFalse: unknown): NativeTensor | any
export function captureMaskedFill(input: unknown, mask: unknown, value: unknown): NativeTensor | any
export function captureSoftmax(input: unknown, dim: unknown): NativeTensor | any
export function captureCrossEntropyIndexed(logits: unknown, targets: unknown, axis: unknown): NativeTensor | any
export function captureOneHot(input: unknown, numClasses: unknown): NativeTensor | any
export function captureTopK(input: unknown, k: unknown, dim?: unknown): { values: NativeTensor | any; indices: NativeTensor | any }
export function captureIndexSelect(input: unknown, dim: unknown, index: unknown): NativeTensor | any
export function captureGather(input: unknown, dim: unknown, index: unknown): NativeTensor | any
export function captureReduction(kind: ReductionKind, input: unknown, axis?: unknown, keepdim?: unknown): NativeTensor | any
export function captureIndexReduction(kind: IndexReductionKind, input: unknown, axis?: unknown, keepdim?: unknown): NativeTensor | any
export function captureRandom(kind: "rand" | "randn", shape: unknown, opts?: { dtype?: DType }): NativeTensor | any
export function rejectIfGraphArgs(opName: string, values: unknown[]): void

declare const graphSupport: {
  graph: typeof graph
  graphWithArity: typeof graphWithArity
  graphWithInputMetas: typeof graphWithInputMetas
  graphTensor: typeof graphTensor
  isCapturing: typeof isCapturing
  withCaptureModulePath: typeof withCaptureModulePath
  captureBoundTensor: typeof captureBoundTensor
  captureUnary: typeof captureUnary
  captureBinary: typeof captureBinary
  captureClamp: typeof captureClamp
  captureReshape: typeof captureReshape
  captureSlice: typeof captureSlice
  captureSqueeze: typeof captureSqueeze
  captureUnsqueeze: typeof captureUnsqueeze
  captureTranspose: typeof captureTranspose
  capturePermute: typeof capturePermute
  captureContiguous: typeof captureContiguous
  captureCast: typeof captureCast
  captureGelu: typeof captureGelu
  captureDot: typeof captureDot
  captureMatmul: typeof captureMatmul
  captureCat: typeof captureCat
  captureStack: typeof captureStack
  captureWhere: typeof captureWhere
  captureMaskedFill: typeof captureMaskedFill
  captureSoftmax: typeof captureSoftmax
  captureCrossEntropyIndexed: typeof captureCrossEntropyIndexed
  captureOneHot: typeof captureOneHot
  captureTopK: typeof captureTopK
  captureIndexSelect: typeof captureIndexSelect
  captureGather: typeof captureGather
  captureReduction: typeof captureReduction
  captureIndexReduction: typeof captureIndexReduction
  captureRandom: typeof captureRandom
  rejectIfGraphArgs: typeof rejectIfGraphArgs
}
export default graphSupport
}
