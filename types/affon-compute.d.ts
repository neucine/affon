declare module "affon:compute" {
  export type Shape = readonly number[]
  export type AxisName = string
  export type DType = "f32" | "f64" | "i64"
  export type Device = "cpu" | "metal"
  export type Selector = number | { kind: "range"; start: number; end: number; step?: number } | { kind: "all" }
  export type DurationUnit = "step" | "epoch"
  export type DurationValue = { unit: DurationUnit; value: number }
  export type TrainContext = { readonly epoch: number; readonly step: number }
  export type LRSchedule = ((context: TrainContext) => number) & { __duration?: DurationValue | null; __unit?: DurationUnit | null }
  export type ComputeStep = ((parameters: readonly Parameter[]) => void) & { lr: number; state(): any; restore(state: any): void }
  export type ScheduledStep = ComputeStep & { readonly context: TrainContext; epoch(value: number): void }
  export const Duration: { steps(value: number): DurationValue & { unit: "step" }; epochs(value: number): DurationValue & { unit: "epoch" } }
  export const schedules: {
    constant(lr: number): LRSchedule
    linear(options: { start: number; end: number; duration: DurationValue }): LRSchedule
    cosine(options: { start: number; end: number; duration: DurationValue }): LRSchedule
    step(options: { base: number; gamma: number; every: DurationValue; duration?: DurationValue }): LRSchedule
    sequence(...parts: LRSchedule[]): LRSchedule
  }
  export function scheduled(step: ComputeStep, schedule: LRSchedule): ScheduledStep
  export type Compiled<F extends (...args: any[]) => any> = F & {
    run: F
    summary(...args: any[]): any
    graph(...args: any[]): any
    plan(...args: any[]): any
    capturedProgram(...args: any[]): any
    captureSummary(...args: any[]): any
    exportBundle(...args: any[]): any
    exportReport(...args: any[]): any
  }
  export function compile<F extends (...args: any[]) => any>(fn: F): Compiled<F>
  export interface Module<Args extends readonly any[] = readonly any[], Out = any> {
    (...args: Args): Out
    readonly training: boolean
    readonly parameters: readonly Parameter[]
    state(): any
    restore(state: any): void
    mode(): "train" | "eval"
    mode(value: "train" | "eval"): this
    train(): void
    eval(): void
    readonly module_path: string | null
    metadata(path: string): this
  }
  export function module<Args extends readonly any[] = readonly any[], Out = any>(state: any, apply: (state: any, ...args: Args) => Out, evalApply?: (state: any, ...args: Args) => Out): Module<Args, Out>
  export function sgd(options?: { lr?: number; momentum?: number }): ComputeStep
  export function adam(options?: { lr?: number; beta1?: number; beta2?: number; eps?: number }): ComputeStep
  export function adamw(options?: { lr?: number; beta1?: number; beta2?: number; eps?: number; weight_decay?: number }): ComputeStep
  export const axes: Readonly<Record<"batch" | "token" | "sequence" | "feature" | "hidden" | "channel" | "height" | "width" | "head" | "vocab", AxisName>>
  export const all: { readonly kind: "all" }
  export function range(start: number, end: number, step?: number): Readonly<{ kind: "range"; start: number; end: number; step: number }>
  export interface TensorOptions<D extends Device = Device> { dtype?: DType; device?: D; axes?: readonly AxisName[] }

  export interface Tensor<S extends Shape = Shape, D extends Device = Device> {
    readonly shape: S
    readonly ndim: number
    readonly dtype: DType
    readonly device: D
    readonly axes?: readonly AxisName[]
    readonly grad: Tensor | null
    readonly grad_device: Device | undefined
    item(): number
    to_array(): unknown
    toString(): string
    repr(): string
    to<T extends Device>(device: T): Tensor<S, T>
    backward(): void
  }

  export function tensor<D extends Device = Device>(values: number | readonly number[] | readonly (number | readonly number[])[], options?: TensorOptions<D>): Tensor<Shape, D>
  export function empty(shape: readonly number[]): Tensor
  export function zeros(shape: readonly number[]): Tensor
  export function ones(shape: readonly number[]): Tensor
  export function full(shape: readonly number[], fill: number): Tensor
  export interface Parameter extends Tensor {
    zeros(): this
    ones(): this
    full(value: number): this
    rand(): this
    randn(): this
    xavier_uniform(): this
    xavier_normal(): this
    kaiming_uniform(): this
    kaiming_normal(): this
  }
  export function parameter<D extends Device = Device>(shape: readonly number[], options?: TensorOptions<D>): Parameter & Tensor<Shape, D>
  export function setDevice(device: Device): void
  export function copy<T extends Tensor>(target: T, source: Tensor): T
  export function grad(loss: Tensor, parameters: readonly Parameter[]): void
  export function clip_grad_norm(parameters: readonly Parameter[], max_norm: number, eps?: number): number
  export function clear_grad(parameters: readonly Parameter[]): void
  export function rand<D extends Device = Device>(shape: readonly number[], options?: TensorOptions<D>): Tensor<Shape, D>
  export function randn<D extends Device = Device>(shape: readonly number[], options?: TensorOptions<D>): Tensor<Shape, D>
  export function seed(value: number): void
  export function arange<D extends Device = Device>(start: number, end?: number, step?: number, options?: TensorOptions<D>): Tensor<Shape, D>
  export function linspace<D extends Device = Device>(start: number, end: number, steps?: number, options?: TensorOptions<D>): Tensor<Shape, D>
  export function add(lhs: Tensor, rhs: Tensor): Tensor
  export function sub(lhs: Tensor, rhs: Tensor): Tensor
  export function mul(lhs: Tensor, rhs: Tensor): Tensor
  export function div(lhs: Tensor, rhs: Tensor): Tensor
  export type MatmulExecutionHint = "projection" | "attention_scores" | "attention_values"
  export type MatmulExecutionOptions = { hint?: MatmulExecutionHint; source?: "higher_level_module" }
  export function matmul(lhs: Tensor, rhs: Tensor, execution?: MatmulExecutionOptions): Tensor
  export function dot(lhs: Tensor, rhs: Tensor): Tensor
  export function square(value: Tensor): Tensor
  export function gt_scalar(value: Tensor, threshold: number): Tensor
  export function cast(value: Tensor, dtype: DType): Tensor
  export function abs(value: Tensor): Tensor
  export function exp(value: Tensor): Tensor
  export function log(value: Tensor): Tensor
  export function neg(value: Tensor): Tensor
  export function sqrt(value: Tensor): Tensor
  export function sign(value: Tensor): Tensor
  export function relu(value: Tensor): Tensor
  export function sigmoid(value: Tensor): Tensor
  export function silu(value: Tensor): Tensor
  export function tanh(value: Tensor): Tensor
  export function gelu(value: Tensor): Tensor
  export function clamp(value: Tensor, min: number, max: number): Tensor
  export function softmax(value: Tensor, axis: number): Tensor
  export function sum(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function mean(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function min(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function max(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function variance(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function std(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function argmin(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function argmax(value: Tensor, axis?: number, keepdim?: boolean): Tensor
  export function reshape(value: Tensor, shape: readonly number[]): Tensor
  export function slice(value: Tensor, ...selectors: Selector[]): Tensor
  export function at(value: Tensor, ...selectors: number[]): Tensor
  export function contiguous(value: Tensor): Tensor
  export function permute(value: Tensor, axes: readonly number[]): Tensor
  export function transpose(value: Tensor, axisA: number, axisB: number): Tensor
  export function squeeze(value: Tensor, axis?: number): Tensor
  export function unsqueeze(value: Tensor, axis: number): Tensor
  export function cat(values: readonly Tensor[], axis?: number): Tensor
  export function stack(values: readonly Tensor[], axis?: number): Tensor
  export function one_hot(indices: Tensor, classes: number): Tensor
  export function gather(value: Tensor, axis: number, index: Tensor): Tensor
  export function index_select(value: Tensor, axis: number, index: Tensor): Tensor
  export function topk(value: Tensor, k: number, axis?: number): { values: Tensor; indices: Tensor }
  export function where(condition: Tensor, onTrue: Tensor, onFalse: Tensor): Tensor
  export function masked_fill(value: Tensor, mask: Tensor, fill: number): Tensor
  export function cross_entropy_indexed(logits: Tensor, targets: Tensor, axis?: number): Tensor
  export const swish: typeof silu
  export function move(value: Tensor, device: Device): Tensor
  export function no_grad<T>(fn: () => T): T
  export function finite_summary(value: Tensor): { ok: boolean; first_bad_flat_index: number; first_bad_value: number }
  export function finite_abs_max(value: Tensor): number | null
  export function accuracy(pred: Tensor, target: Tensor, options?: { threshold?: number }): number
  export function mse(pred: Tensor, target: Tensor): number
  export function mae(pred: Tensor, target: Tensor): number
  export function precision(pred: Tensor, target: Tensor, options?: { threshold?: number }): number
  export function recall(pred: Tensor, target: Tensor, options?: { threshold?: number }): number
  export function f1(pred: Tensor, target: Tensor, options?: { threshold?: number }): number
  export function r2(pred: Tensor, target: Tensor): number

  const compute: {
    compile: typeof compile
    module: typeof module
    tensor: typeof tensor
    empty: typeof empty
    zeros: typeof zeros
    ones: typeof ones
    full: typeof full
    parameter: typeof parameter
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
    tanh: typeof tanh
    gelu: typeof gelu
    clamp: typeof clamp
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
    all: typeof all
    range: typeof range
    swish: typeof swish
    move: typeof move
    no_grad: typeof no_grad
    finite_summary: typeof finite_summary
    finite_abs_max: typeof finite_abs_max
    accuracy: typeof accuracy
    mse: typeof mse
    mae: typeof mae
    precision: typeof precision
    recall: typeof recall
    f1: typeof f1
    r2: typeof r2
    axes: typeof axes
    Duration: typeof Duration
    schedules: typeof schedules
    scheduled: typeof scheduled
    sgd: typeof sgd
    adam: typeof adam
    adamw: typeof adamw
  }

  export default compute
}
