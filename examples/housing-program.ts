// Run from the repository root: affon examples/housing-program.ts
// Uses the bundled California housing CSV; no model download or GPU needed.
import { read } from 'affon:dataset'
import { Session, Tensor, metrics, optimize, program } from 'affon:core'
import { linear } from 'affon:nn'
import { div, gelu, mean, mean_squared_error, mul, reshape, std, sub } from 'affon:ops'
import { adam } from 'affon:optim'

const batchSize = 512
const features = ['median_income', 'latitude', 'longitude', 'housing_median_age']
const session = new Session({ device: 'cpu' })
const [trainingRows, heldOutRows] = read('examples/california_housing.csv')
  .shuffle()
  .sample(batchSize * 2)
  .split(0.5)
const train = trainingRows.features(...features).target('median_house_value').toTensors({ session })
const heldOut = heldOutRows.features(...features).target('median_house_value').toTensors({ session })

// Fit scaling on training features and apply it to both sets.
const center = mean(train.X, 0)
const spread = std(train.X, 0)
const x = div(sub(train.X, center), spread)
const testX = div(sub(heldOut.X, center), spread)
const priceScale = session.tensor([100_000])
const target = div(train.y, priceScale)

const hidden = linear({ out_features: 16 })
const output = linear({ out_features: 1 })
const model = program('house_price', p => {
  const input = p.argument('x', Tensor.f32([batchSize, features.length]))
  return reshape(output({ x: gelu(hidden({ x: input }, 'hidden')) }, 'output'), [batchSize])
})
const training = optimize(model, program('price_error', p => mean_squared_error(
  p.argument('prediction', Tensor.f32([batchSize])),
  p.argument('target', Tensor.f32([batchSize])),
)), adam({ learning_rate: 0.01 }))
const inspection = training.inspect()
console.log('Program:', inspection.arguments.map(value => value.name).join(', '), '|', inspection.parameters.length, 'parameters')

const state = session.initialize(training, { seed: 7 })
const step = session.compile(training)
let firstLoss = 0
let finalLoss = 0
for (let i = 0; i < 100; i++) {
  const result = step.run({ x, target }, state)
  finalLoss = result.item()
  if (i === 0) firstLoss = finalLoss
}

const predictions = session.compile(model).run({ x: testX }, state)
const prices = mul(predictions, priceScale)
const mae = metrics.mean_absolute_error(prices, heldOut.y)
const r2 = metrics.r2_score(prices, heldOut.y)
console.log('training loss:', firstLoss.toFixed(3), '->', finalLoss.toFixed(3))
console.log('held-out MAE:', Math.round(mae), 'USD; R²:', r2.toFixed(3))
if (!Number.isFinite(mae) || finalLoss >= firstLoss) throw new Error('training did not improve')
