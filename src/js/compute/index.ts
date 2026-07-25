import { AffonError } from "affon:errors"
import native from "affon:compute/native"
import graphSupport from "affon:compute/graph.ts"
import { compileWithHelpers } from "affon:compute/compile.ts"
import { loadStateTree, restorePersistedState, saveStateTree } from "affon:compute/persistence.ts"

type Device = "cpu" | "metal"
type TensorOptions = { dtype?: "f32" | "f64" | "i64"; device?: Device; axes?: readonly string[] }
type Tensor = any

function sameDevice<T extends Tensor>(value: T, device?: Device): T {
  if (!device || value.device === device) return value
  return value.to(device) as T
}

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

function isTensorLike(value: unknown): value is { shape: number[]; dtype: "f32" | "f64" | "i64" } {
  return !!value && typeof value === "object" && Array.isArray((value as any).shape) && typeof (value as any).dtype === "string"
}

function isCapturedTensor(value: unknown): boolean {
  return !!value && typeof value === "object" && (value as any).$graph === true
}

function requireTensorArgs(args: readonly unknown[], name: string): any[] {
  const tensors: any[] = []
  for (const arg of args) {
    if (graphSupport.isCapturing() && isCapturedTensor(arg)) {
      tensors.push(arg)
      continue
    }
    if (!isTensorLike(arg) || typeof (arg as any).item !== "function") {
      throw new AffonError("invalid_arg", `${name} expects compute values/parameters`)
    }
    tensors.push(arg)
  }
  return tensors
}

function collectProgramStateParts(value: any, seen: Set<any>, out: string[]): void {
  if (!value || (typeof value !== "object" && typeof value !== "function") || seen.has(value)) return
  seen.add(value)
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      const nested: string[] = []
      collectProgramStateParts(item, seen, nested)
      if (nested.length > 0) out.push(`${index}(${nested.join("|")})`)
    })
    return
  }
  if (typeof value.training === "boolean") out.push(value.training ? "train" : "eval")
  if (value.__affon_compute_module === true) {
    collectProgramStateParts(value.__affon_compute_state, seen, out)
    return
  }
  for (const key of Object.keys(value)) {
    const nested: string[] = []
    collectProgramStateParts(value[key], seen, nested)
    if (nested.length > 0) out.push(`${key}(${nested.join("|")})`)
  }
}

function programStateKey(source: any): string {
  const parts: string[] = []
  collectProgramStateParts(source, new Set<any>(), parts)
  return parts.length > 0 ? parts.join("|") : "default"
}

export function compile<F extends (...args: any[]) => any>(fn: F): F {
  return compileWithHelpers(fn, {
    graphSupport,
    isTensorLike,
    programStateKey,
    programArity: (source, fallback) => source?.__affon_compute_arity ?? fallback.length,
    replayCapturedProgram: (source, inputs) => source(...inputs),
    isProgramSource: source => source?.__affon_compute_module === true,
    isReplayableProgramSource: () => false,
    finalizeExecutable: (executable, original) => {
      if (original?.__affon_compute_module !== true) return executable
      Object.defineProperties(executable, {
        __affon_compute_module: { value: true },
        parameters: { enumerable: true, get: () => original.parameters },
        training: { enumerable: true, get: () => original.training },
      })
      executable.state = () => original.state()
      executable.restore = (state: any) => original.restore(state)
      executable.mode = (value?: "train" | "eval") => value === undefined ? original.mode() : (original.mode(value), executable)
      executable.train = () => original.train()
      executable.eval = () => original.eval()
      return executable
    },
  }) as F
}

type ModuleState = any
type ModuleTensor = { shape: readonly number[]; dtype: "f32" | "f64" | "i64"; device: "cpu" | "metal"; [key: string]: any }

function isParameter(value: unknown): value is ModuleTensor {
  return !!value && typeof value === "object" && (value as any).$compute?.role === "parameter"
}

function markParameter<T extends ModuleTensor>(param: T): T {
  const metadata = (param as any).$compute
  if (metadata && typeof metadata === "object") return param
  Object.defineProperty(param, "$compute", { configurable: true, value: { role: "parameter" } })
  return param
}

function cloneModuleState(value: any): any {
  if (value && (typeof value === "object" || typeof value === "function") && value.__affon_compute_module === true) {
    return cloneModuleState(value.__affon_compute_state)
  }
  if (isParameter(value)) {
    const result = native.parameter([...value.shape], { dtype: value.dtype, device: value.device })
    copy(result, value)
    return markParameter(result as ModuleTensor)
  }
  if (isTensorLike(value)) return native.mul(value, scalarLike(value, 1))
  if (Array.isArray(value)) return value.map(cloneModuleState)
  if (value && typeof value === "object") {
    const result: Record<string, any> = {}
    for (const key of Object.keys(value)) result[key] = cloneModuleState(value[key])
    return result
  }
  return value
}

function restoreModuleState(target: any, source: any): void {
  if (target && (typeof target === "object" || typeof target === "function") && target.__affon_compute_module === true) {
    restoreModuleState(target.__affon_compute_state, source)
    return
  }
  if (isTensorLike(target)) {
    if (!isTensorLike(source)) throw new TypeError("module.restore expects matching tensor state")
    copy(target, source)
    return
  }
  if (Array.isArray(target)) {
    if (!Array.isArray(source)) throw new TypeError("module.restore expects matching array state")
    target.length = source.length
    for (let index = 0; index < source.length; index++) {
      if (index < target.length && (isTensorLike(target[index]) || Array.isArray(target[index]) || (target[index] && (typeof target[index] === "object" || typeof target[index] === "function")))) restoreModuleState(target[index], source[index])
      else target[index] = cloneModuleState(source[index])
    }
    return
  }
  if (target && typeof target === "object" && source && typeof source === "object") {
    for (const key of Object.keys(source)) {
      if (key in target && (isTensorLike(target[key]) || Array.isArray(target[key]) || (target[key] && (typeof target[key] === "object" || typeof target[key] === "function")))) restoreModuleState(target[key], source[key])
      else target[key] = cloneModuleState(source[key])
    }
    return
  }
  throw new TypeError("module.restore expects matching state")
}

function collectModuleParameters(value: any, out: ModuleTensor[]): void {
  if (!value || (typeof value !== "object" && typeof value !== "function")) return
  if (isParameter(value)) {
    if (!out.includes(value)) out.push(value)
    return
  }
  if (value.__affon_compute_module === true) {
    collectModuleParameters(value.__affon_compute_state, out)
    return
  }
  for (const key of Object.keys(value)) {
    const item = value[key]
    if (isParameter(item)) {
      if (!out.includes(item)) out.push(item)
    } else if (!isTensorLike(item) && item && (typeof item === "object" || typeof item === "function")) {
      collectModuleParameters(item, out)
    }
  }
}

const reservedModuleKeys = new Set(["state", "restore", "mode", "train", "eval", "save", "load", "metadata", "parameters", "training", "module_path"])

function exposeModuleStateProperties(target: any, state: any): void {
  if (!state || typeof state !== "object" || Array.isArray(state)) return
  for (const key of Object.keys(state)) {
    if (reservedModuleKeys.has(key)) continue
    Object.defineProperty(target, key, {
      get: () => state[key],
      set: (value: any) => { state[key] = value },
      enumerable: true,
      configurable: true,
    })
  }
}

function collectNamedModuleParameters(value: any, prefix: string, out: Array<[string, ModuleTensor]>, seen: Set<any>, seenParameters: Set<any>): void {
  if (!value || (typeof value !== "object" && typeof value !== "function")) return
  if (isParameter(value)) {
    if (!seenParameters.has(value)) {
      seenParameters.add(value)
      out.push([prefix, value])
    }
    return
  }
  if (value.__affon_compute_module === true) {
    collectNamedModuleParameters(value.__affon_compute_state, prefix, out, seen, seenParameters)
    return
  }
  if (seen.has(value)) return
  seen.add(value)
  if (Array.isArray(value)) {
    value.forEach((item, index) => collectNamedModuleParameters(item, prefix ? `${prefix}.${index}` : String(index), out, seen, seenParameters))
    return
  }
  for (const key of Object.keys(value)) collectNamedModuleParameters(value[key], prefix ? `${prefix}.${key}` : key, out, seen, seenParameters)
}

function moduleParameterCollection(state: any): any[] {
  const parameters: ModuleTensor[] = []
  collectModuleParameters(state, parameters)
  const named: Array<[string, ModuleTensor]> = []
  collectNamedModuleParameters(state, "", named, new Set<any>(), new Set<any>())
  const collection = parameters.slice() as any
  Object.defineProperty(collection, "named", { value: () => Object.freeze(named.map(([name, parameter]) => Object.freeze([name, parameter] as const))) })
  return Object.freeze(collection)
}

function visitNestedModules(value: any, visit: (moduleValue: any) => void, seen: Set<any>): void {
  if (!value || (typeof value !== "object" && typeof value !== "function") || seen.has(value)) return
  seen.add(value)
  if (value.__affon_compute_module === true) {
    visit(value)
    visitNestedModules(value.__affon_compute_state, visit, seen)
    return
  }
  if (Array.isArray(value)) {
    for (const item of value) visitNestedModules(item, visit, seen)
    return
  }
  for (const key of Object.keys(value)) visitNestedModules(value[key], visit, seen)
}

function assignNestedModulePaths(value: any, prefix: string, seen: Set<any>): void {
  if (!value || (typeof value !== "object" && typeof value !== "function") || seen.has(value)) return
  seen.add(value)
  if (value.__affon_compute_module === true) {
    if (value.module_path == null) value.module_path = prefix
    assignNestedModulePaths(value.__affon_compute_state, value.module_path, seen)
    return
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => assignNestedModulePaths(item, `${prefix}.${index}`, seen))
    return
  }
  for (const key of Object.keys(value)) assignNestedModulePaths(value[key], `${prefix}.${key}`, seen)
}

export function module<Args extends readonly any[] = readonly any[], Out = any>(
  state: ModuleState,
  apply: (state: any, ...args: Args) => Out,
  evalApply?: (state: any, ...args: Args) => Out,
): any {
  if (typeof apply !== "function" || (evalApply !== undefined && typeof evalApply !== "function")) throw new TypeError("module expects apply and optional eval functions")
  let mode: "train" | "eval" = "train"
  const callable: any = (...args: Args) => {
    requireTensorArgs(args, "module call")
    const fn = mode === "eval" && evalApply ? evalApply : apply
    return graphSupport.withCaptureModulePath(callable.module_path ?? undefined, () => fn(state, ...args))
  }
  Object.defineProperties(callable, {
    __affon_compute_module: { value: true },
    __affon_compute_state: { value: state },
    __affon_compute_arity: { value: Math.max(0, apply.length - 1, (evalApply?.length ?? 1) - 1) },
    module_path: { enumerable: true, writable: true, value: null },
    training: { enumerable: true, get: () => mode === "train" },
    parameters: { enumerable: true, get: () => moduleParameterCollection(state) },
  })
  exposeModuleStateProperties(callable, state)
  callable.state = () => cloneModuleState(state)
  callable.restore = (nextState: any) => restoreModuleState(state, nextState)
  callable.mode = (nextMode?: "train" | "eval") => {
    if (nextMode === undefined) return mode
    if (nextMode !== "train" && nextMode !== "eval") throw new TypeError("module.mode expects train or eval")
    mode = nextMode
    visitNestedModules(state, nested => {
      if (nested !== callable) nested.mode(nextMode)
    }, new Set([callable]))
    return callable
  }
  callable.train = () => { callable.mode("train") }
  callable.eval = () => { callable.mode("eval") }
  callable.save = (path: string) => { saveStateTree(state, path) }
  callable.load = (path: string) => { restorePersistedState(state, loadStateTree(path)) }
  callable.metadata = (path: string) => {
    if (typeof path !== "string" || path.length === 0) throw new TypeError("module.metadata expects a non-empty path")
    callable.module_path = path
    assignNestedModulePaths(state, path, new Set([callable]))
    return callable
  }
  return callable
}

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
          native.$muladd_(velocity, momentum, gradient.to(parameters[index].device))
          native.$axpy_(parameters[index], -lrState.value, velocity)
        }
        return
      }

      if (beta1 === undefined) {
        for (let index = 0; index < parameters.length; index++) {
          const gradient = parameters[index].grad
          if (gradient == null) continue
          native.$axpy_(parameters[index], -lrState.value, gradient.to(parameters[index].device))
        }
        return
      }

      stepCount += 1
      const updateParameters: any[] = []
      const updateMoments: any[] = []
      const updateSecondMoments: any[] = []
      const updateGradients: any[] = []
      for (let index = 0; index < parameters.length; index++) {
        const parameter = parameters[index]
        const gradient = parameter.grad
        if (gradient == null) {
          if (weightDecay !== undefined && weightDecay !== 0) copy(parameter, native.sub(parameter, native.mul(parameter, scalarLike(parameter, lrState.value * weightDecay))))
          continue
        }
        updateParameters.push(parameter)
        updateMoments.push(first![index])
        updateSecondMoments.push(second[index])
        updateGradients.push(gradient.to(parameter.device))
      }
      if (updateParameters.length > 0 && updateParameters.every(parameter => parameter.dtype === "f32" || parameter.dtype === "f64")) {
        native.$adam_step_many_(
          updateParameters,
          updateMoments,
          updateSecondMoments,
          updateGradients,
          beta1!,
          beta2!,
          1 - Math.pow(beta1!, stepCount),
          1 - Math.pow(beta2!, stepCount),
          eps!,
          lrState.value,
          weightDecay ?? 0,
        )
        return
      }
      for (let index = 0; index < updateParameters.length; index++) {
        const parameter = updateParameters[index]
        const updateGradient = updateGradients[index]
        const m = updateMoments[index]
        const v = updateSecondMoments[index]
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
    if (beta1 === undefined) {
      if (first) first.forEach((value, index) => { tensors[`velocity_${index}`] = cloneTensor(value) })
      return { kind, scalars: { lr: lrState.value, momentum: momentum ?? 0 }, tensors }
    }
    first?.forEach((value, index) => { tensors[`m_${index}`] = cloneTensor(value) })
    second.forEach((value, index) => { tensors[`v_${index}`] = cloneTensor(value) })
    const scalars: Record<string, number> = {
      lr: lrState.value,
      beta1,
      beta2: beta2!,
      eps: eps!,
      t: stepCount,
    }
    if (weightDecay !== undefined) scalars.weight_decay = weightDecay
    return { kind, scalars, tensors }
  }
  step.restore = (state: OptimizerState) => {
    if (!state || state.kind !== kind) throw new TypeError(`optimizer.restore expected ${kind} state`)
    lrState.value = state.scalars.lr
    stepCount = state.scalars.t ?? state.scalars.step ?? 0
    if (bound === null) {
      pendingState = state
      return
    }
    if (beta1 === undefined) {
      if (first) first.forEach((value, index) => copy(value, optimizerStateTensor(state, `velocity_${index}`)))
      return
    }
    first?.forEach((value, index) => copy(value, optimizerStateTensor(state, `m_${index}`)))
    second.forEach((value, index) => copy(value, optimizerStateTensor(state, `v_${index}`)))
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
type ParameterTensor = {
  shape: readonly number[]
  ndim: number
  dtype: "f32" | "f64" | "i64"
  device: "cpu" | "metal"
  grad: any
  [key: string]: any
}

function fanInOut(param: ParameterTensor): { fan_in: number; fan_out: number } {
  if (param.ndim < 2) throw new Error("parameter init expects at least 2 dimensions for Xavier/Kaiming initialization")
  if (param.ndim === 2) return { fan_in: param.shape[0], fan_out: param.shape[1] }
  let receptive = 1
  for (let index = 2; index < param.shape.length; index++) receptive *= param.shape[index]
  return { fan_in: param.shape[1] * receptive, fan_out: param.shape[0] * receptive }
}

function fillScalar_(param: ParameterTensor, scalar: number): ParameterTensor {
  const tmp = native.mul(native.ones([...param.shape], { dtype: param.dtype }), native.tensor(scalar, { dtype: param.dtype })).to(param.device)
  native.$muladd_(param, 0.0, tmp)
  return param
}

function fillUniform_(param: ParameterTensor, lower: number, upper: number): ParameterTensor {
  const span = upper - lower
  const tmp = native.add(
    native.mul(native.rand([...param.shape], { dtype: param.dtype }).to(param.device), native.tensor(span, { dtype: param.dtype })),
    native.tensor(lower, { dtype: param.dtype }),
  )
  native.$muladd_(param, 0.0, tmp.to(param.device))
  return param
}

function fillNormal_(param: ParameterTensor, standardDeviation: number): ParameterTensor {
  const tmp = native.mul(native.randn([...param.shape], { dtype: param.dtype }).to(param.device), native.tensor(standardDeviation, { dtype: param.dtype }))
  native.$muladd_(param, 0.0, tmp.to(param.device))
  return param
}

function attachParameterInitMethods(param: ParameterTensor): ParameterTensor {
  if (typeof param.zeros === "function") return param
  Object.defineProperties(param, {
    zeros: { value: () => fillScalar_(param, 0.0) },
    ones: { value: () => fillScalar_(param, 1.0) },
    full: { value: (value: number) => fillScalar_(param, value) },
    rand: { value: () => { const tmp = native.rand([...param.shape], { dtype: param.dtype }).to(param.device); native.$muladd_(param, 0.0, tmp); return param } },
    randn: { value: () => { const tmp = native.randn([...param.shape], { dtype: param.dtype }).to(param.device); native.$muladd_(param, 0.0, tmp); return param } },
    xavier_uniform: { value: () => { const { fan_in, fan_out } = fanInOut(param); return fillUniform_(param, -Math.sqrt(6.0 / (fan_in + fan_out)), Math.sqrt(6.0 / (fan_in + fan_out))) } },
    xavier_normal: { value: () => { const { fan_in, fan_out } = fanInOut(param); return fillNormal_(param, Math.sqrt(2.0 / (fan_in + fan_out))) } },
    kaiming_uniform: { value: () => { const { fan_in } = fanInOut(param); return fillUniform_(param, -Math.sqrt(6.0 / fan_in), Math.sqrt(6.0 / fan_in)) } },
    kaiming_normal: { value: () => { const { fan_in } = fanInOut(param); return fillNormal_(param, Math.sqrt(2.0 / fan_in)) } },
  })
  return param
}

export function parameter(shape: readonly number[], options?: { dtype?: "f32" | "f64"; device?: "cpu" | "metal"; axes?: readonly string[] }) {
  const param = native.parameter([...shape], { dtype: options?.dtype ?? "f32", device: options?.device, axes: options?.axes })
  return attachParameterInitMethods(markParameter(param as ParameterTensor))
}
export const setDevice = native.setDevice
;(globalThis as any).setDevice = setDevice
function copyTensorValue(target: any, source: any): void {
  const adapted = source.dtype === target.dtype
    ? (source.device === target.device ? source : source.to(target.device))
    : native.cast(source, target.dtype).to(target.device)
  native.$muladd_(target, 0.0, adapted)
}
export function copy<T extends Tensor>(target: T, source: Tensor): T {
  copyTensorValue(target, source)
  return target
}
export const clip_grad_norm = native.clip_grad_norm
export function clear_grad(parameters: readonly Tensor[]): void {
  for (let index = 0; index < parameters.length; index++) native.$zero_grad_(parameters[index])
}

function internalBackward(loss: Tensor): void {
  native.$backward_(loss)
}

export function grad(loss: Tensor, parameters: readonly Tensor[], opts?: { assign?: 'replace' | 'accumulate' }): void {
  if (!loss || typeof loss !== 'object' || typeof (loss as any).shape === 'undefined') {
    throw new AffonError('invalid_arg', 'grad(loss, params, opts?) expects a differentiable compute loss value')
  }
  if (opts?.assign !== 'accumulate') clear_grad(parameters)
  internalBackward(loss)
}
export function rand(shape: readonly number[], opts?: TensorOptions): Tensor {
  const captured = graphSupport.captureRandom("rand", shape, opts?.dtype ?? "f32")
  if (captured) return captured as Tensor
  return sameDevice(native.rand(shape.slice() as number[], { dtype: opts?.dtype ?? "f32", axes: opts?.axes }), opts?.device)
}
export function randn(shape: readonly number[], opts?: TensorOptions): Tensor {
  const captured = graphSupport.captureRandom("randn", shape, opts?.dtype ?? "f32")
  if (captured) return captured as Tensor
  return sameDevice(native.randn(shape.slice() as number[], { dtype: opts?.dtype ?? "f32", axes: opts?.axes }), opts?.device)
}
export const seed = native.seed
export function arange(start: number, end?: number, step = 1, opts?: TensorOptions): Tensor {
  const actualStart = end === undefined ? 0 : start
  const actualEnd = end === undefined ? start : end
  if (!Number.isFinite(actualStart) || !Number.isFinite(actualEnd) || !Number.isFinite(step) || step === 0) {
    throw new Error("arange(start, end?, step?, opts?) expects finite numeric bounds and a non-zero step")
  }
  const values: number[] = []
  if (step > 0) {
    for (let value = actualStart; value < actualEnd; value += step) values.push(value)
  } else {
    for (let value = actualStart; value > actualEnd; value += step) values.push(value)
  }
  return tensor(values, opts)
}
export function linspace(start: number, end: number, steps = 100, opts?: TensorOptions): Tensor {
  if (!Number.isInteger(steps) || steps < 0) {
    throw new Error("linspace(start, end, steps?, opts?) expects steps to be a non-negative integer")
  }
  if (steps === 0) return tensor([], opts)
  const output = native.linspace(start, end, steps, { dtype: opts?.dtype ?? "f32", axes: opts?.axes })
  return sameDevice(output, opts?.device)
}
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
export const cast = (value: any, dtype: "f32" | "f64" | "i64") => graphSupport.captureCast(value, dtype)
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
export function transpose(value: any, dim1: number, dim2: number) {
  const dims = Array.from({ length: value.ndim }, (_, index) => index)
  const first = dims[dim1]
  dims[dim1] = dims[dim2]
  dims[dim2] = first
  return graphSupport.capturePermute(value, dims)
}
export const squeeze = (value: any, axis?: number) => graphSupport.captureSqueeze(value, axis)
export const unsqueeze = (value: any, axis: number) => graphSupport.captureUnsqueeze(value, axis)
export const cat = (inputs: any[], dim?: number) => graphSupport.captureCat(inputs, dim)
export const stack = (inputs: any[], dim?: number) => graphSupport.captureStack(inputs, dim)
export const one_hot = (input: any, numClasses: number) => graphSupport.captureOneHot(input, numClasses)
export const gather = (input: any, dim: number, index: any) => graphSupport.captureGather(input, dim, index)
export const index_select = (input: any, dim: number, index: any) => graphSupport.captureIndexSelect(input, dim, index)
export const topk = (input: any, k: number, dim?: number) => graphSupport.captureTopK(input, k, dim)

export const where = (cond: any, onTrue: any, onFalse: any) => graphSupport.captureWhere(cond, onTrue, onFalse)
export const masked_fill = (input: any, mask: any, value: number) => graphSupport.captureMaskedFill(input, mask, value)
export const cross_entropy_indexed = (logits: any, targets: any, axis = 1) => graphSupport.captureCrossEntropyIndexed(logits, targets, axis)
export function move(value: any, device: Device) {
  return value.to(device)
}
export function no_grad<T>(fn: () => T): T {
  return native.no_grad(fn) as T
}

// Export/report plumbing is intentionally deferred, but keep the declared
// surface loadable for workflows that only train and checkpoint.
export function exportBundleFile(..._args: any[]): never {
  throw new AffonError('not_implemented', 'compute bundle export is not implemented yet')
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
  return lhs.length === 0 ? 0 : lhs.reduce((sum, value, index) => sum + ((value > threshold) === (rhs[index] !== 0) ? 1 : 0), 0) / lhs.length
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
    const predicted = value > threshold
    const actual = rhs[index] !== 0
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
    const predicted = value > threshold
    const actual = rhs[index] !== 0
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
  if (total === 0) return 0
  return 1 - lhs.reduce((sum, value, index) => sum + (value - rhs[index]) ** 2, 0) / total
}

export default { axes, all, range, Duration, schedules, scheduled, compile, module, sgd, adam, adamw, tensor, empty, zeros, ones, full, parameter, setDevice, copy, grad, clip_grad_norm, clear_grad, rand, randn, seed, arange, linspace, add, sub, mul, div, matmul, dot, square, gt_scalar, cast, abs, exp, log, neg, sqrt, sign, relu, sigmoid, silu, swish, tanh, gelu, clamp, softmax, sum, mean, min, max, variance, std, argmin, argmax, reshape, slice, at, contiguous, permute, transpose, squeeze, unsqueeze, cat, stack, one_hot, gather, index_select, topk, where, masked_fill, cross_entropy_indexed, move, no_grad, finite_summary, finite_abs_max, accuracy, precision, recall, f1, mse, mae, r2 }
