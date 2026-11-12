// Run with: affon examples/first-program.ts
// CPU-only, deterministic, and independent of downloaded models or datasets.
import { Session, Tensor, optimize, program } from 'affon:core'
import { mean_squared_error, mul } from 'affon:ops'
import { sgd } from 'affon:optim'

const model = program('line', p => mul(
  p.argument('x', Tensor.f32([3], { axes: ['sample'] })),
  p.parameter('weight', Tensor.f32([1]), { initializer: { kind: 'zeros' } }),
))
const loss = program('squared_error', p => mean_squared_error(
  p.argument('prediction', Tensor.f32([3])),
  p.argument('target', Tensor.f32([3])),
))
const training = optimize(model, loss, sgd({ learning_rate: 0.05 }))
const inspection = training.inspect()
console.log('contract:', inspection.arguments.map(value => `${value.name}: ${value.spec.dtype}[${value.spec.shape}]`).join(', '))
console.log('graph:', inspection.nodes.length, 'nodes;', inspection.transitions.length, 'optimizer transition')

const session = new Session({ device: 'cpu' })
const state = session.initialize(training)
const x = session.tensor([1, 2, 3])
const target = session.tensor([2, 4, 6])
const step = session.compile(training)
let firstLoss = 0
let finalLoss = 0
for (let i = 0; i < 12; i++) {
  const result = step.run({ x, target }, state)
  finalLoss = result.item()
  if (i === 0) firstLoss = finalLoss
  result.dispose()
}
const prediction = session.compile(model).run({ x }, state)
console.log('loss:', firstLoss.toFixed(6), '->', finalLoss.toFixed(6))
console.log('prediction:', prediction.to_array())
if (!(finalLoss < firstLoss / 100)) throw new Error('training did not converge')
prediction.dispose()
x.dispose()
target.dispose()
state.dispose()
session.dispose()
