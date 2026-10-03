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
export type Optimizer = SGD | Adam | AdamW

function finiteNonNegative(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new TypeError(`${name} must be finite and non-negative`)
  return value
}

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
