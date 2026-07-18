import native from "affon:compute/native"
import graphSupport from "affon:compute/graph.ts"

export const all = Object.freeze({ kind: "all" as const })
export const axes = Object.freeze({
  batch: "batch",
  token: "token",
  sequence: "sequence",
  feature: "feature",
  hidden: "hidden",
  channel: "channel",
  height: "height",
  width: "width",
  head: "head",
  vocab: "vocab",
} as const)
export function range(start: number, end: number, step = 1) {
  if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(step) || step === 0) throw new Error("invalid range")
  return Object.freeze({ kind: "range" as const, start, end, step })
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

function duration(unit: DurationUnit, value: number): DurationValue {
  if (!Number.isFinite(value) || Math.floor(value) !== value || value <= 0) throw new RangeError("duration must be a positive integer")
  return { unit, value }
}

function schedule(fn: (context: TrainContext) => number, value: DurationValue | null): LRSchedule {
  const result = fn as LRSchedule
  result.__duration = value
  result.__unit = value?.unit ?? null
  return result
}

export const Duration = Object.freeze({
  steps: (value: number) => duration("step", value),
  epochs: (value: number) => duration("epoch", value),
})

export const schedules = Object.freeze({
  constant(lr: number): LRSchedule {
    return schedule(() => lr, null)
  },
  linear(options: { start: number; end: number; duration: DurationValue }): LRSchedule {
    const value = options.duration
    return schedule(context => {
      const progress = value.unit === "step" ? context.step : context.epoch
      const ratio = Math.min(Math.max(progress / value.value, 0), 1)
      return options.start + (options.end - options.start) * ratio
    }, value)
  },
  cosine(options: { start: number; end: number; duration: DurationValue }): LRSchedule {
    const value = options.duration
    return schedule(context => {
      const progress = value.unit === "step" ? context.step : context.epoch
      const ratio = Math.min(Math.max(progress / value.value, 0), 1)
      const weight = (1 + Math.cos(Math.PI * ratio)) / 2
      return options.end + (options.start - options.end) * weight
    }, value)
  },
  step(options: { base: number; gamma: number; every: DurationValue; duration?: DurationValue }): LRSchedule {
    const every = options.every
    const limit = options.duration
    if (limit && limit.unit !== every.unit) throw new TypeError("step schedule duration must use the same unit as every")
    const result = schedule(context => {
      const progress = every.unit === "step" ? context.step : context.epoch
      const bounded = limit ? Math.min(progress, limit.value) : progress
      return options.base * Math.pow(options.gamma, Math.floor(bounded / every.value))
    }, limit ?? null)
    result.__unit = every.unit
    return result
  },
  sequence(...parts: LRSchedule[]): LRSchedule {
    if (parts.length === 0) throw new TypeError("sequence requires at least one schedule")
    let unit: DurationUnit | null = null
    const durations: Array<DurationValue | null> = []
    for (let index = 0; index < parts.length; index++) {
      const value = parts[index].__duration
      const partUnit = parts[index].__unit
      if (value === undefined) throw new TypeError("sequence only supports schedules created by schedules")
      durations.push(value)
      if (partUnit != null) {
        if (unit == null) unit = partUnit
        else if (unit !== partUnit) throw new TypeError("all schedules in a sequence must use the same duration unit")
      }
      if (value === null && index !== parts.length - 1) throw new TypeError("only the final schedule may be unbounded")
      if (value !== null && value.unit !== unit) throw new TypeError("all schedules in a sequence must use the same duration unit")
    }
    const sequenceUnit = unit ?? "step"
    const result = schedule(context => {
      const progress = sequenceUnit === "step" ? context.step : context.epoch
      let offset = 0
      for (let index = 0; index < parts.length; index++) {
        const value = durations[index]
        if (value === null || progress <= offset + value.value) {
          const innerProgress = progress - offset
          return parts[index](sequenceUnit === "step"
            ? { epoch: context.epoch, step: innerProgress }
            : { epoch: innerProgress, step: context.step })
        }
        offset += value.value
      }
      return parts[parts.length - 1](context)
    }, null)
    result.__unit = sequenceUnit
    return result
  },
})

export function scheduled(baseStep: ComputeStep, lrSchedule: LRSchedule): ScheduledStep {
  let epochValue = 0
  let stepValue = 0
  const context = {} as TrainContext
  Object.defineProperties(context, {
    epoch: { enumerable: true, get: () => epochValue },
    step: { enumerable: true, get: () => stepValue },
  })
  const wrapped = ((parameters: readonly any[]) => {
    baseStep.lr = lrSchedule(context)
    baseStep(parameters)
    stepValue += 1
  }) as ScheduledStep
  Object.defineProperty(wrapped, "lr", {
    enumerable: true,
    get: () => baseStep.lr,
    set: (value: number) => { baseStep.lr = value },
  })
  Object.defineProperty(wrapped, "context", { enumerable: true, get: () => context })
  wrapped.epoch = (value: number) => {
    if (!Number.isFinite(value) || Math.floor(value) !== value || value < 0) throw new RangeError("epoch must be a non-negative integer")
    epochValue = value
  }
  wrapped.state = () => baseStep.state()
  wrapped.restore = state => baseStep.restore(state)
  return wrapped
}

type OptimizerState = { kind: string; scalars: Record<string, number>; tensors: Record<string, any> }

function scalarLike(value: any, number: number): any {
  return native.tensor(number, { dtype: value.dtype }).to(value.device)
}

function zerosLike(value: any): any {
  return native.mul(value, scalarLike(value, 0))
}

function cloneTensor(value: any): any {
  return native.mul(value, scalarLike(value, 1))
}

function validateOptimizerParams(parameters: readonly any[]): void {
  if (!Array.isArray(parameters)) throw new TypeError("optimizer step expects a parameter array")
  for (const parameter of parameters) {
    if (parameter.device === "metal" && parameter.dtype !== "f32") throw new TypeError("metal optimizers require f32 parameters")
  }
}

function optimizerScalar(value: unknown, fallback: number, name: string): number {
  if (value === undefined) return fallback
  if (typeof value !== "number" || !Number.isFinite(value)) throw new TypeError(`${name} must be finite`)
  return value
}

function optimizerStateTensor(state: OptimizerState, key: string): any {
  const value = state.tensors[key]
  if (!value || typeof value !== "object" || !Array.isArray(value.shape)) throw new TypeError(`optimizer state is missing ${key}`)
  return value
}

function makeOptimizerStep(
  kind: string,
  defaults: { lr: number; beta1?: number; beta2?: number; eps?: number; momentum?: number; weight_decay?: number },
): ComputeStep {
  let bound: readonly any[] | null = null
  let pendingState: OptimizerState | null = null
  let stepCount = 0
  let first = defaults.momentum === undefined ? [] as any[] : null
  let second: any[] = []
  const lrState = { value: defaults.lr }
  const beta1 = defaults.beta1
  const beta2 = defaults.beta2
  const eps = defaults.eps
  const momentum = defaults.momentum
  const weightDecay = defaults.weight_decay

  const step = ((parameters: readonly any[]) => {
    validateOptimizerParams(parameters)
    if (bound === null) {
      bound = parameters
      if (momentum !== undefined && momentum > 0) first = parameters.map(zerosLike)
      if (beta1 !== undefined) {
        first = parameters.map(zerosLike)
        second = parameters.map(zerosLike)
      }
      if (pendingState) {
        step.restore(pendingState)
        pendingState = null
      }
    } else if (parameters !== bound) {
      throw new TypeError("optimizer step is bound to one parameter list")
    }

    native.no_grad(() => {
      if (beta1 === undefined && momentum !== undefined && momentum > 0) {
        for (let index = 0; index < parameters.length; index++) {
          const gradient = parameters[index].grad
          if (gradient == null) continue
          const velocity = first![index]
          const nextVelocity = native.add(native.mul(velocity, scalarLike(velocity, momentum)), gradient.to(parameters[index].device))
          copy(velocity, nextVelocity)
          copy(parameters[index], native.add(parameters[index], native.mul(velocity, scalarLike(velocity, -lrState.value))))
        }
        return
      }

      if (beta1 === undefined) {
        for (let index = 0; index < parameters.length; index++) {
          const gradient = parameters[index].grad
          if (gradient == null) continue
          copy(parameters[index], native.add(parameters[index], native.mul(gradient.to(parameters[index].device), scalarLike(parameters[index], -lrState.value))))
        }
        return
      }

      stepCount += 1
      for (let index = 0; index < parameters.length; index++) {
        const parameter = parameters[index]
        const gradient = parameter.grad
        if (gradient == null) {
          if (weightDecay !== undefined && weightDecay !== 0) copy(parameter, native.sub(parameter, native.mul(parameter, scalarLike(parameter, lrState.value * weightDecay))))
          continue
        }
        const updateGradient = gradient.to(parameter.device)
        const m = first![index]
        const v = second[index]
        const nextM = native.add(native.mul(m, scalarLike(m, beta1!)), native.mul(updateGradient, scalarLike(m, 1 - beta1!)))
        const nextV = native.add(native.mul(v, scalarLike(v, beta2!)), native.mul(native.mul(updateGradient, updateGradient), scalarLike(v, 1 - beta2!)))
        copy(m, nextM)
        copy(v, nextV)
        const correctedM = native.mul(m, scalarLike(m, 1 / (1 - Math.pow(beta1!, stepCount))))
        const correctedV = native.mul(v, scalarLike(v, 1 / (1 - Math.pow(beta2!, stepCount))))
        const denominator = native.add(native.sqrt(correctedV), scalarLike(v, eps!))
        let update = native.div(correctedM, denominator)
        if (weightDecay !== undefined && weightDecay !== 0) update = native.add(update, native.mul(parameter, scalarLike(parameter, weightDecay)))
        copy(parameter, native.sub(parameter, native.mul(update, scalarLike(parameter, lrState.value))))
      }
    })
  }) as ComputeStep

  Object.defineProperty(step, "lr", {
    enumerable: true,
    get: () => lrState.value,
    set: value => {
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new TypeError("optimizer lr must be non-negative and finite")
      lrState.value = value
    },
  })
  step.state = () => {
    const tensors: Record<string, any> = {}
    if (first) first.forEach((value, index) => { tensors[`first_${index}`] = cloneTensor(value) })
    second.forEach((value, index) => { tensors[`second_${index}`] = cloneTensor(value) })
    return { kind, scalars: { lr: lrState.value, step: stepCount }, tensors }
  }
  step.restore = (state: OptimizerState) => {
    if (!state || state.kind !== kind) throw new TypeError(`optimizer.restore expected ${kind} state`)
    lrState.value = state.scalars.lr
    stepCount = state.scalars.step ?? 0
    if (bound === null) {
      pendingState = state
      return
    }
    if (first) first.forEach((value, index) => copy(value, optimizerStateTensor(state, `first_${index}`)))
    second.forEach((value, index) => copy(value, optimizerStateTensor(state, `second_${index}`)))
  }
  return step
}

export function sgd(options: { lr?: number; momentum?: number } = {}): ComputeStep {
  const lr = optimizerScalar(options.lr, 0.01, "sgd lr")
  const momentum = optimizerScalar(options.momentum, 0, "sgd momentum")
  if (momentum < 0) throw new RangeError("sgd momentum must be non-negative")
  return makeOptimizerStep("SGD", { lr, momentum })
}

export function adam(options: { lr?: number; beta1?: number; beta2?: number; eps?: number } = {}): ComputeStep {
  return makeOptimizerStep("Adam", {
    lr: optimizerScalar(options.lr, 0.001, "adam lr"),
    beta1: optimizerScalar(options.beta1, 0.9, "adam beta1"),
    beta2: optimizerScalar(options.beta2, 0.999, "adam beta2"),
    eps: optimizerScalar(options.eps, 1e-8, "adam eps"),
  })
}

export function adamw(options: { lr?: number; beta1?: number; beta2?: number; eps?: number; weight_decay?: number } = {}): ComputeStep {
  return makeOptimizerStep("AdamW", {
    lr: optimizerScalar(options.lr, 0.001, "adamw lr"),
    beta1: optimizerScalar(options.beta1, 0.9, "adamw beta1"),
    beta2: optimizerScalar(options.beta2, 0.999, "adamw beta2"),
    eps: optimizerScalar(options.eps, 1e-8, "adamw eps"),
    weight_decay: optimizerScalar(options.weight_decay, 0.01, "adamw weight_decay"),
  })
}

export const tensor = native.tensor
export const empty = native.empty
export const zeros = native.zeros
export const ones = native.ones
export const full = native.full
export const parameter = native.parameter
export const copy = native.copy
export const grad = native.grad
export const clip_grad_norm = native.clip_grad_norm
export const clear_grad = native.clear_grad
export const rand = (shape: readonly number[]) => graphSupport.captureRandom("rand", shape, "f32")
export const randn = (shape: readonly number[]) => graphSupport.captureRandom("randn", shape, "f32")
export const seed = native.seed
export const arange = native.arange
export const linspace = native.linspace
export function add(lhs: Tensor, rhs: Tensor): Tensor {
  return graphSupport.captureBinary("add", "add", lhs, rhs)
}

export function sub(lhs: Tensor, rhs: Tensor): Tensor {
  return graphSupport.captureBinary("sub", "sub", lhs, rhs)
}

export function mul(lhs: Tensor, rhs: Tensor): Tensor {
  return graphSupport.captureBinary("mul", "mul", lhs, rhs)
}

export function div(lhs: Tensor, rhs: Tensor): Tensor {
  return graphSupport.captureBinary("div", "div", lhs, rhs)
}
export const matmul = (lhs: any, rhs: any, execution?: any) => graphSupport.captureMatmul(lhs, rhs, execution)
export const dot = (lhs: any, rhs: any) => graphSupport.captureDot(lhs, rhs)
export function square(value: any) { return mul(value, value) }
export const gt_scalar = (value: any, threshold: number) => graphSupport.captureBinary("gt", "gt_scalar", value, threshold)
export const cast = (value: any, dtype: "f32" | "f64") => graphSupport.captureCast(value, dtype)
export const abs = (value: any) => graphSupport.captureUnary("abs", "abs", value)
export const exp = (value: any) => graphSupport.captureUnary("exp", "exp", value)
export const log = (value: any) => graphSupport.captureUnary("log", "log", value)
export const neg = (value: any) => graphSupport.captureUnary("neg", "neg", value)
export const sqrt = (value: any) => graphSupport.captureUnary("sqrt", "sqrt", value)
export const sign = (value: any) => graphSupport.captureUnary("sign", "sign", value)
export const relu = (value: any) => graphSupport.captureUnary("relu", "relu", value)
export const sigmoid = (value: any) => graphSupport.captureUnary("sigmoid", "sigmoid", value)
export const silu = (value: any) => graphSupport.captureUnary("silu", "silu", value)
export const swish = silu
export const tanh = (value: any) => graphSupport.captureUnary("tanh", "tanh", value)
export const gelu = (value: any) => graphSupport.captureUnary("gelu", "gelu", value)
export const clamp = (value: any, min: number, max: number) => graphSupport.captureClamp(value, min, max)
export const softmax = (value: any, dim: number) => graphSupport.captureSoftmax(value, dim)
function reduce(value: any, kind: any, axis?: number, keepdim?: boolean) {
  if (kind === "sum" || kind === "mean" || kind === "std" || kind === "variance" || kind === "min" || kind === "max") {
    return graphSupport.captureReduction(kind, value, axis, keepdim)
  }
  if (kind === "argmin" || kind === "argmax") {
    return graphSupport.captureIndexReduction(kind, value, axis, keepdim)
  }
  if (axis === undefined) return native[kind](value)
  if (keepdim === undefined) return native[kind](value, axis)
  return native[kind](value, axis, keepdim)
}
export function sum(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "sum", axis, keepdim) }
export function mean(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "mean", axis, keepdim) }
export function min(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "min", axis, keepdim) }
export function max(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "max", axis, keepdim) }
export function variance(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "variance", axis, keepdim) }
export function std(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "std", axis, keepdim) }
export function argmin(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "argmin", axis, keepdim) }
export function argmax(value: any, axis?: number, keepdim?: boolean) { return reduce(value, "argmax", axis, keepdim) }
export const reshape = (value: any, shape: number[], opts?: any) => graphSupport.captureReshape(value, shape, opts)
export function slice(value: any, ...selectors: any[]) {
  const ranges = selectors.map(selector => {
    if (typeof selector === "number") return selector
    if (selector?.kind === "all") return ":"
    if (selector?.kind === "range") {
      return selector.step === undefined
        ? `${selector.start}:${selector.end}`
        : `${selector.start}:${selector.end}:${selector.step}`
    }
    throw new TypeError("invalid slice selector")
  })
  return native.slice(value, ranges)
}
export function at(value: any, ...selectors: number[]) {
  return slice(value, ...selectors)
}
export const contiguous = (value: any) => graphSupport.captureContiguous(value)
export const permute = (value: any, axes: number[]) => graphSupport.capturePermute(value, axes)
export const transpose = native.transpose
export const squeeze = (value: any, axis?: number) => graphSupport.captureSqueeze(value, axis)
export const unsqueeze = (value: any, axis: number) => graphSupport.captureUnsqueeze(value, axis)
export const cat = (inputs: any[], dim?: number) => graphSupport.captureCat(inputs, dim)
export const stack = (inputs: any[], dim?: number) => graphSupport.captureStack(inputs, dim)
export const one_hot = (input: any, numClasses: number) => graphSupport.captureOneHot(input, numClasses)
export const gather = (input: any, dim: number, index: any) => graphSupport.captureGather(input, dim, index)
export const index_select = (input: any, dim: number, index: any) => graphSupport.captureIndexSelect(input, dim, index)
export const topk = native.topk
export const where = (cond: any, onTrue: any, onFalse: any) => graphSupport.captureWhere(cond, onTrue, onFalse)
export const masked_fill = (input: any, mask: any, value: number) => graphSupport.captureMaskedFill(input, mask, value)
export const cross_entropy_indexed = (logits: any, targets: any, axis = 1) => graphSupport.captureCrossEntropyIndexed(logits, targets, axis)
export function move(value: any, device: "cpu") {
  return value.to(device)
}
export function no_grad<T>(fn: () => T): T {
  return native.no_grad(fn) as T
}

function flatValues(value: any): number[] {
  const out: number[] = []
  const visit = (item: any) => Array.isArray(item) ? item.forEach(visit) : out.push(Number(item))
  visit(value.to_array())
  return out
}

export function finite_summary(value: any) {
  const values = flatValues(value)
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) return { ok: false, first_bad_flat_index: index, first_bad_value: values[index] }
  }
  return { ok: true, first_bad_flat_index: -1, first_bad_value: 0 }
}

export function finite_abs_max(value: any): number | null {
  const summary = finite_summary(value)
  if (!summary.ok) return null
  return Math.max(...flatValues(value).map(Math.abs))
}

function pairedValues(left: any, right: any): [number[], number[]] {
  const lhs = flatValues(left)
  const rhs = flatValues(right)
  if (lhs.length !== rhs.length) throw new Error("tensor sizes must match")
  return [lhs, rhs]
}

export function accuracy(pred: any, target: any, options?: { threshold?: number }): number {
  const [lhs, rhs] = pairedValues(pred, target)
  const threshold = options?.threshold ?? 0.5
  return lhs.length === 0 ? 0 : lhs.reduce((sum, value, index) => sum + ((value >= threshold) === (rhs[index] >= threshold) ? 1 : 0), 0) / lhs.length
}

export function mse(pred: any, target: any): number {
  const [lhs, rhs] = pairedValues(pred, target)
  return lhs.length === 0 ? 0 : lhs.reduce((sum, value, index) => sum + (value - rhs[index]) ** 2, 0) / lhs.length
}

export function mae(pred: any, target: any): number {
  const [lhs, rhs] = pairedValues(pred, target)
  return lhs.length === 0 ? 0 : lhs.reduce((sum, value, index) => sum + Math.abs(value - rhs[index]), 0) / lhs.length
}

export function precision(pred: any, target: any, options?: { threshold?: number }): number {
  const [lhs, rhs] = pairedValues(pred, target)
  const threshold = options?.threshold ?? 0.5
  let truePositive = 0
  let falsePositive = 0
  lhs.forEach((value, index) => {
    const predicted = value >= threshold
    const actual = rhs[index] >= threshold
    if (predicted && actual) truePositive++
    else if (predicted) falsePositive++
  })
  return truePositive + falsePositive === 0 ? 0 : truePositive / (truePositive + falsePositive)
}

export function recall(pred: any, target: any, options?: { threshold?: number }): number {
  const [lhs, rhs] = pairedValues(pred, target)
  const threshold = options?.threshold ?? 0.5
  let truePositive = 0
  let falseNegative = 0
  lhs.forEach((value, index) => {
    const predicted = value >= threshold
    const actual = rhs[index] >= threshold
    if (predicted && actual) truePositive++
    else if (!predicted && actual) falseNegative++
  })
  return truePositive + falseNegative === 0 ? 0 : truePositive / (truePositive + falseNegative)
}

export function f1(pred: any, target: any, options?: { threshold?: number }): number {
  const p = precision(pred, target, options)
  const r = recall(pred, target, options)
  return p + r === 0 ? 0 : (2 * p * r) / (p + r)
}

export function r2(pred: any, target: any): number {
  const [lhs, rhs] = pairedValues(pred, target)
  if (rhs.length === 0) return 0
  const mean = rhs.reduce((sum, value) => sum + value, 0) / rhs.length
  const total = rhs.reduce((sum, value) => sum + (value - mean) ** 2, 0)
  if (total === 0) return lhs.every((value, index) => value === rhs[index]) ? 1 : 0
  return 1 - lhs.reduce((sum, value, index) => sum + (value - rhs[index]) ** 2, 0) / total
}

export default { axes, all, range, Duration, schedules, scheduled, sgd, adam, adamw, tensor, empty, zeros, ones, full, parameter, copy, grad, clip_grad_norm, clear_grad, rand, randn, seed, arange, linspace, add, sub, mul, div, matmul, dot, square, gt_scalar, cast, abs, exp, log, neg, sqrt, sign, relu, sigmoid, silu, swish, tanh, gelu, clamp, softmax, sum, mean, min, max, variance, std, argmin, argmax, reshape, slice, at, contiguous, permute, transpose, squeeze, unsqueeze, cat, stack, one_hot, gather, index_select, topk, where, masked_fill, cross_entropy_indexed, move, no_grad, finite_summary, finite_abs_max, accuracy, precision, recall, f1, mse, mae, r2 }
