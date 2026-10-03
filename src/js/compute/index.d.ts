declare module "affon:compute/legacy" {
  /** Inference-only CPU centered reflect STFT. Signal f32, window f64; output f64 [bins, frames]. */
  export function stft_power(signal: Tensor, window: Tensor, hop: number, paddedLength?: number, frames?: number): Tensor
  export function filterbank(spectrum: Tensor, filters: Tensor): Tensor

type Device = "cpu" | "metal" | "cuda" | `cuda:${number}`
type DType = "f32" | "f64" | "i64"
type TensorInput = number | readonly number[] | readonly TensorInput[]
type TensorOptions = { dtype?: DType; device?: Device; axes?: readonly string[] }
type ParameterOptions = { dtype?: "f32" | "f64"; device?: Device; axes?: readonly string[] }

export interface Tensor {
  readonly shape: readonly number[]
  readonly ndim: number
  readonly dtype: DType
  readonly device: Device
  readonly grad?: Tensor
  readonly grad_device?: Device
  item(): number
  to_array(): unknown
  toString(): string
  repr(): string
  to(device: Device): Tensor
  backward(): void
}

export type DurationUnit = "step" | "epoch"
export type DurationValue = { unit: DurationUnit; value: number }
export type TrainContext = { readonly epoch: number; readonly step: number }
export type LRSchedule = ((context: TrainContext) => number) & {
  __duration?: DurationValue | null
  __unit?: DurationUnit | null
}
export type ComputeStep = ((parameters: readonly any[]) => void) & {
  lr: number
  state(): any
  restore(state: any): void
}
export type ScheduledStep = ComputeStep & {
  readonly context: TrainContext
  epoch(value: number): void
}
export type ComputeState = Record<string, any> | any[] | Tensor | number | string | boolean | null
export type Module<Args extends readonly any[] = readonly any[], Out = any> = ((...args: Args) => Out) & {
  readonly __affon_compute_module: true
  readonly parameters: readonly Tensor[] & {
    named(): readonly (readonly [string, Tensor])[]
  }
  readonly training: boolean
  readonly module_path?: string
  state(): ComputeState
  restore(state: ComputeState): void
  mode(value?: "train" | "eval"): Module<Args, Out>
  train(): Module<Args, Out>
  eval(): Module<Args, Out>
  save(path: string): void
  load(path: string): Module<Args, Out>
  metadata(): { parameters: ReturnType<Module<Args, Out>["parameters"]["named"]>; state: ComputeState; training: boolean; module_path?: string }
}

export const all: Readonly<{ kind: "all" }>
export const axes: Readonly<{
  batch: "batch"
  token: "token"
  sequence: "sequence"
  feature: "feature"
  hidden: "hidden"
  channel: "channel"
  height: "height"
  width: "width"
  head: "head"
  vocab: "vocab"
}>
export function range(start: number, end: number, step?: number): Readonly<{ kind: "range"; start: number; end: number; step: number }>

export const Duration: Readonly<{
  steps(value: number): DurationValue
  epochs(value: number): DurationValue
}>
export const schedules: Readonly<{
  constant(lr: number): LRSchedule
  linear(options: { start: number; end: number; duration: DurationValue }): LRSchedule
  cosine(options: { start: number; end: number; duration: DurationValue }): LRSchedule
  step(options: { base: number; gamma: number; every: DurationValue; duration?: DurationValue }): LRSchedule
  sequence(...parts: LRSchedule[]): LRSchedule
}>

export function compile<F extends (...args: any[]) => any>(fn: F): F
export function module<Args extends readonly any[] = readonly any[], Out = any>(
  state: ComputeState,
  apply: (state: any, ...args: Args) => Out,
  evalApply?: (state: any, ...args: Args) => Out,
): Module<Args, Out>
export function scheduled(baseStep: ComputeStep, lrSchedule: LRSchedule): ScheduledStep
export function sgd(options?: { lr?: number; momentum?: number }): ComputeStep
export function adam(options?: { lr?: number; beta1?: number; beta2?: number; eps?: number }): ComputeStep
export function adamw(options?: { lr?: number; beta1?: number; beta2?: number; eps?: number; weight_decay?: number }): ComputeStep

export const tensor: (values: TensorInput, options?: TensorOptions) => Tensor
export const empty: (shape: readonly number[], options?: TensorOptions) => Tensor
export const zeros: (shape: readonly number[], options?: TensorOptions) => Tensor
export const ones: (shape: readonly number[], options?: TensorOptions) => Tensor
export const full: (shape: readonly number[], fill: number, options?: TensorOptions) => Tensor
export function parameter(shape: readonly number[], options?: ParameterOptions): Tensor
export const setDevice: (device: Device) => void
export function copy<T extends Tensor>(target: T, source: Tensor): T
export const clip_grad_norm: (parameters: readonly Tensor[], max_norm: number, eps?: number) => number
export function clear_grad(parameters: readonly Tensor[]): void
export function grad(loss: Tensor, parameters: readonly Tensor[], opts?: { assign?: "replace" | "accumulate" }): void
export function rand(shape: readonly number[], opts?: TensorOptions): Tensor
export function randn(shape: readonly number[], opts?: TensorOptions): Tensor
export const seed: (value: number) => void
export function arange(start: number, end?: number, step?: number, opts?: TensorOptions): Tensor
export function linspace(start: number, end: number, steps?: number, opts?: TensorOptions): Tensor

export function add(lhs: any, rhs: any): Tensor
export function sub(lhs: any, rhs: any): Tensor
export function mul(lhs: any, rhs: any): Tensor
export function div(lhs: any, rhs: any): Tensor
export const matmul: (lhs: any, rhs: any, execution?: any) => Tensor
export const dot: (lhs: any, rhs: any) => Tensor
export function square(value: any): Tensor
export const gt_scalar: (value: any, threshold: number) => Tensor
export const cast: (value: any, dtype: DType) => Tensor
export const abs: (value: any) => Tensor
export const exp: (value: any) => Tensor
export const log: (value: any) => Tensor
export const neg: (value: any) => Tensor
export const sqrt: (value: any) => Tensor
export const sign: (value: any) => Tensor
export const relu: (value: any) => Tensor
export const sigmoid: (value: any) => Tensor
export const silu: (value: any) => Tensor
export const swish: typeof silu
export const erf: (value: any) => Tensor
export const tanh: (value: any) => Tensor
export const gelu: (value: any) => Tensor
export const clamp: (value: any, min: number, max: number) => Tensor
export const layer_norm: (value: Tensor, axis: number, eps?: number) => Tensor
export const softmax: (value: any, dim: number) => Tensor

export function sum(value: any, axis?: number, keepdim?: boolean): Tensor
export function mean(value: any, axis?: number, keepdim?: boolean): Tensor
export function min(value: any, axis?: number, keepdim?: boolean): Tensor
export function max(value: any, axis?: number, keepdim?: boolean): Tensor
export function variance(value: any, axis?: number, keepdim?: boolean): Tensor
export function std(value: any, axis?: number, keepdim?: boolean): Tensor
export function argmin(value: any, axis?: number, keepdim?: boolean): Tensor
export function argmax(value: any, axis?: number, keepdim?: boolean): Tensor

export const reshape: (value: any, shape: number[], opts?: any) => Tensor
export function slice(value: any, ...selectors: any[]): Tensor
export function at(value: any, ...selectors: number[]): Tensor
export const contiguous: (value: any) => Tensor
export const permute: (value: any, axes: number[]) => Tensor
export function transpose(value: any, dim1: number, dim2: number): Tensor
export const squeeze: (value: any, axis?: number) => Tensor
export const unsqueeze: (value: any, axis: number) => Tensor
export const cat: (inputs: any[], dim?: number) => Tensor
export const stack: (inputs: any[], dim?: number) => Tensor
export const one_hot: (input: any, numClasses: number) => Tensor
export const gather: (input: any, dim: number, index: any) => Tensor
export const index_select: (input: any, dim: number, index: any) => Tensor
export const topk: (input: any, k: number, dim?: number) => { values: Tensor; indices: Tensor }
export const where: (cond: any, onTrue: any, onFalse: any) => Tensor
export const masked_fill: (input: any, mask: any, value: number) => Tensor
export const cross_entropy_indexed: (logits: any, targets: any, axis?: number) => Tensor
export function move(value: any, device: Device): Tensor
export function no_grad<T>(fn: () => T): T
export function exportBundleFile(..._args: any[]): never

export function finite_summary(value: any): { finite: boolean; nan: number; posinf: number; neginf: number; total: number }
export function finite_abs_max(value: any): number | null
export function accuracy(pred: any, target: any, options?: { threshold?: number }): number
export function mse(pred: any, target: any): number
export function mae(pred: any, target: any): number
export function precision(pred: any, target: any, options?: { threshold?: number }): number
export function recall(pred: any, target: any, options?: { threshold?: number }): number
export function f1(pred: any, target: any, options?: { threshold?: number }): number
export function r2(pred: any, target: any): number

declare const compute: {
  stft_power: typeof stft_power;
  filterbank: typeof filterbank;
  axes: typeof axes
  all: typeof all
  range: typeof range
  Duration: typeof Duration
  schedules: typeof schedules
  scheduled: typeof scheduled
  compile: typeof compile
  module: typeof module
  sgd: typeof sgd
  adam: typeof adam
  adamw: typeof adamw
  tensor: typeof tensor
  empty: typeof empty
  zeros: typeof zeros
  ones: typeof ones
  full: typeof full
  parameter: typeof parameter
  setDevice: typeof setDevice
  copy: typeof copy
  grad: typeof grad
  clip_grad_norm: typeof clip_grad_norm
  clear_grad: typeof clear_grad
  rand: typeof rand
  randn: typeof randn
  seed: typeof seed
  arange: typeof arange
  linspace: typeof linspace
  add: typeof add
  sub: typeof sub
  mul: typeof mul
  div: typeof div
  matmul: typeof matmul
  dot: typeof dot
  square: typeof square
  gt_scalar: typeof gt_scalar
  cast: typeof cast
  abs: typeof abs
  exp: typeof exp
  log: typeof log
  neg: typeof neg
  sqrt: typeof sqrt
  sign: typeof sign
  relu: typeof relu
  sigmoid: typeof sigmoid
  silu: typeof silu
  swish: typeof swish
  tanh: typeof tanh
  erf: typeof erf
  gelu: typeof gelu
  clamp: typeof clamp
  layer_norm: typeof layer_norm
  softmax: typeof softmax
  sum: typeof sum
  mean: typeof mean
  min: typeof min
  max: typeof max
  variance: typeof variance
  std: typeof std
  argmin: typeof argmin
  argmax: typeof argmax
  reshape: typeof reshape
  slice: typeof slice
  at: typeof at
  contiguous: typeof contiguous
  permute: typeof permute
  transpose: typeof transpose
  squeeze: typeof squeeze
  unsqueeze: typeof unsqueeze
  cat: typeof cat
  stack: typeof stack
  one_hot: typeof one_hot
  gather: typeof gather
  index_select: typeof index_select
  topk: typeof topk
  where: typeof where
  masked_fill: typeof masked_fill
  cross_entropy_indexed: typeof cross_entropy_indexed
  move: typeof move
  no_grad: typeof no_grad
  finite_summary: typeof finite_summary
  finite_abs_max: typeof finite_abs_max
  accuracy: typeof accuracy
  precision: typeof precision
  recall: typeof recall
  f1: typeof f1
  mse: typeof mse
  mae: typeof mae
  r2: typeof r2
}
export default compute
}
