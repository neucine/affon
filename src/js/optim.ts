export type SGD = Readonly<{
  kind: 'sgd'
  learning_rate: number
  momentum: number
}>
export type Adam = Readonly<{
  kind: 'adam'
  learning_rate: number
  beta1: number
  beta2: number
  epsilon: number
}>
export type AdamW = Readonly<{
  kind: 'adamw'
  learning_rate: number
  beta1: number
  beta2: number
  epsilon: number
  weight_decay: number
}>
export type BaseOptimizer = SGD | Adam | AdamW
export type LRSchedule =
  | Readonly<{ kind: 'constant'; learning_rate: number }>
  | Readonly<{ kind: 'linear' | 'cosine'; start: number; end: number; steps: number }>
  | Readonly<{ kind: 'step'; base: number; gamma: number; every: number; steps?: number }>
  | Readonly<{ kind: 'warmup_cosine'; start: number; peak: number; end: number; warmup_steps: number; total_steps: number }>
  | Readonly<{ kind: 'sequence'; schedules: readonly LRSchedule[] }>
export type ScheduledOptimizer = Readonly<{
  kind: 'scheduled'
  optimizer: BaseOptimizer
  schedule: LRSchedule
}>
export type UpdatingOptimizer = BaseOptimizer | ScheduledOptimizer
export type AccumulatingOptimizer = Readonly<{
  kind: 'accumulate'
  optimizer: UpdatingOptimizer
  steps: number
}>
export type Optimizer = UpdatingOptimizer | AccumulatingOptimizer

function finiteNonNegative(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new TypeError(`${name} must be finite and non-negative`)
  return value
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${name} must be a positive integer`)
  return value
}

function boundedSchedule(kind: 'linear' | 'cosine', options: { start: number; end: number; steps: number }): LRSchedule {
  return Object.freeze({
    kind,
    start: finiteNonNegative(options.start, 'start'),
    end: finiteNonNegative(options.end, 'end'),
    steps: positiveInteger(options.steps, 'steps'),
  })
}

export const schedules = Object.freeze({
  constant(learning_rate: number): LRSchedule {
    return Object.freeze({ kind: 'constant', learning_rate: finiteNonNegative(learning_rate, 'learning_rate') })
  },
  linear(options: { start: number; end: number; steps: number }): LRSchedule {
    return boundedSchedule('linear', options)
  },
  cosine(options: { start: number; end: number; steps: number }): LRSchedule {
    return boundedSchedule('cosine', options)
  },
  step(options: { base: number; gamma: number; every: number; steps?: number }): LRSchedule {
    const gamma = finiteNonNegative(options.gamma, 'gamma')
    return Object.freeze({
      kind: 'step',
      base: finiteNonNegative(options.base, 'base'),
      gamma,
      every: positiveInteger(options.every, 'every'),
      ...(options.steps === undefined ? {} : { steps: positiveInteger(options.steps, 'steps') }),
    })
  },
  warmupCosine(options: { start: number; peak: number; end: number; warmup_steps: number; total_steps: number }): LRSchedule {
    const warmup_steps = positiveInteger(options.warmup_steps, 'warmup_steps')
    const total_steps = positiveInteger(options.total_steps, 'total_steps')
    if (warmup_steps >= total_steps) throw new TypeError('warmup_steps must be less than total_steps')
    return Object.freeze({
      kind: 'warmup_cosine',
      start: finiteNonNegative(options.start, 'start'),
      peak: finiteNonNegative(options.peak, 'peak'),
      end: finiteNonNegative(options.end, 'end'),
      warmup_steps,
      total_steps,
    })
  },
  sequence(...parts: readonly LRSchedule[]): LRSchedule {
    if (parts.length === 0) throw new TypeError('sequence requires at least one schedule')
    for (let index = 0; index < parts.length - 1; index++) {
      const part = parts[index]
      if (part.kind === 'constant' || (part.kind === 'step' && part.steps === undefined) || part.kind === 'sequence') {
        throw new TypeError('only the final sequence schedule may be unbounded or nested')
      }
    }
    return Object.freeze({ kind: 'sequence', schedules: Object.freeze([...parts]) })
  },
})

export function sgd(options: { learning_rate?: number; momentum?: number } = {}): SGD {
  const momentum = options.momentum ?? 0
  if (!Number.isFinite(momentum) || momentum < 0 || momentum >= 1) throw new TypeError('momentum must be in [0, 1)')
  return Object.freeze({ kind: 'sgd', learning_rate: finiteNonNegative(options.learning_rate ?? 0.01, 'learning_rate'), momentum })
}

export function adam(options: { learning_rate?: number; beta1?: number; beta2?: number; epsilon?: number } = {}): Adam {
  const beta1 = options.beta1 ?? 0.9
  const beta2 = options.beta2 ?? 0.999
  if (!Number.isFinite(beta1) || beta1 < 0 || beta1 >= 1) throw new TypeError('beta1 must be in [0, 1)')
  if (!Number.isFinite(beta2) || beta2 < 0 || beta2 >= 1) throw new TypeError('beta2 must be in [0, 1)')
  const epsilon = options.epsilon ?? 1e-8
  if (!Number.isFinite(epsilon) || epsilon <= 0) throw new TypeError('epsilon must be finite and positive')
  return Object.freeze({ kind: 'adam', learning_rate: finiteNonNegative(options.learning_rate ?? 0.001, 'learning_rate'), beta1, beta2, epsilon })
}

export function adamw(options: { learning_rate?: number; beta1?: number; beta2?: number; epsilon?: number; weight_decay?: number } = {}): AdamW {
  return Object.freeze({ ...adam(options), kind: 'adamw', weight_decay: finiteNonNegative(options.weight_decay ?? 0.01, 'weight_decay') })
}

export function scheduled(optimizer: BaseOptimizer, schedule: LRSchedule): ScheduledOptimizer {
  if (!optimizer || typeof optimizer !== 'object' || !['sgd', 'adam', 'adamw'].includes(optimizer.kind)) throw new TypeError('scheduled expects a base Optimizer')
  if (!schedule || typeof schedule !== 'object' || typeof schedule.kind !== 'string') throw new TypeError('scheduled expects a learning-rate schedule')
  return Object.freeze({ kind: 'scheduled', optimizer, schedule })
}

export function accumulate(optimizer: UpdatingOptimizer, options: { steps: number }): AccumulatingOptimizer {
  if (!optimizer || typeof optimizer !== 'object' || !['sgd', 'adam', 'adamw', 'scheduled'].includes(optimizer.kind)) throw new TypeError('accumulate expects an updating Optimizer')
  if (!options || !Number.isSafeInteger(options.steps) || options.steps < 2) throw new TypeError('accumulate steps must be an integer of at least 2')
  return Object.freeze({ kind: 'accumulate', optimizer, steps: options.steps })
}
