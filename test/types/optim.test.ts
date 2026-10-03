import { Duration, adam, clip_grad_norm, parameter, scheduled, schedules, sgd } from 'affon:compute/legacy'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const params = [parameter([1])]
const optimizer = sgd()
const adamStep = adam()
const duration = Duration.steps(10)
const schedule = schedules.linear({
  start: 0,
  end: 1e-3,
  duration,
})
const sequence = schedules.sequence(
  schedule,
  schedules.constant(1e-4)
)
const scheduledStep = scheduled(optimizer, sequence)
const lr = sequence({ epoch: 0, step: 5 })
const clippedNorm = clip_grad_norm(params, 1.0)

assertType<IsExact<typeof optimizer.lr, number>>()
assertType<IsExact<typeof adamStep.lr, number>>()
assertType<IsExact<typeof duration.unit, 'step'>>()
assertType<IsExact<typeof duration.value, number>>()
assertType<IsExact<typeof lr, number>>()
assertType<IsExact<typeof clippedNorm, number>>()
assertType<IsExact<typeof scheduledStep.context.epoch, number>>()
assertType<IsExact<typeof scheduledStep.context.step, number>>()
