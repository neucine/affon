// Run from the repository root: affon examples/housing-program.ts
// Uses the bundled California housing CSV; no model download or GPU needed.
import fs from 'std:fs'
import { Session, Tensor, optimize, program } from 'affon:core'
import { linear } from 'affon:nn'
import { gelu, mean_squared_error, reshape } from 'affon:ops'
import { adam } from 'affon:optim'

const batchSize = 512
const featureNames = ['median_income', 'latitude', 'longitude', 'housing_median_age']
const session = new Session({ device: 'cpu' })
const lines = fs.readFileSync('examples/california_housing.csv').trim().split('\n')
const columns = lines.shift()!.split(',')
const columnIndices = [...featureNames, 'median_house_value'].map(name => columns.indexOf(name))
const rows = lines.map(line => {
  const cells = line.split(',')
  return columnIndices.map(index => Number(cells[index]))
})

// Shuffle deterministically, then fit scaling on training rows only.
let seed = 7
const indices = rows.map((_, i) => i)
for (let i = indices.length - 1; i > 0; i--) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
  const j = seed % (i + 1)
  const current = indices[i]
  indices[i] = indices[j]
  indices[j] = current
}
const selected = indices.slice(0, batchSize * 2).map(index => rows[index])
const trainRows = selected.slice(0, batchSize)
const testRows = selected.slice(batchSize)
const means = featureNames.map((_, j) => trainRows.reduce((sum, row) => sum + row[j], 0) / batchSize)
const scales = featureNames.map((_, j) => Math.sqrt(trainRows.reduce((sum, row) => sum + (row[j] - means[j]) ** 2, 0) / batchSize))
const features = (data: number[][]) => data.map(row => featureNames.map((_, j) => (row[j] - means[j]) / scales[j]))
const targets = (data: number[][]) => data.map(row => row[4] / 100_000)

const hidden = linear({ out_features: 16 })
const output = linear({ out_features: 1 })
const model = program('house_price', p => {
  const x = p.argument('x', Tensor.f32([batchSize, featureNames.length]))
  return reshape(output({ x: gelu(hidden({ x }, 'hidden')) }, 'output'), [batchSize])
})
const loss = program('price_error', p => mean_squared_error(
  p.argument('prediction', Tensor.f32([batchSize])),
  p.argument('target', Tensor.f32([batchSize])),
))
const training = optimize(model, loss, adam({ learning_rate: 0.01 }))
const inspection = training.inspect()
console.log('Program:', inspection.arguments.map(value => value.name).join(', '), '|', inspection.parameters.length, 'parameters')

const state = session.initialize(training, { seed: 7 })
const x = session.tensor(features(trainRows))
const target = session.tensor(targets(trainRows))
const testX = session.tensor(features(testRows))
const testY = targets(testRows)
const step = session.compile(training)
let firstLoss = 0
let finalLoss = 0
for (let i = 0; i < 100; i++) {
  const result = step.run({ x, target }, state)
  finalLoss = result.item()
  if (i === 0) firstLoss = finalLoss
  result.dispose()
}
const predictions = session.compile(model).run({ x: testX }, state)
const values = predictions.to_array() as number[]
const mae = values.reduce((sum, value, i) => sum + Math.abs(value - testY[i]), 0) / batchSize * 100_000
const meanTrainPrice = trainRows.reduce((sum, row) => sum + row[4], 0) / batchSize / 100_000
const baseline = testY.reduce((sum, value) => sum + Math.abs(value - meanTrainPrice), 0) / batchSize * 100_000
console.log('training loss:', firstLoss.toFixed(3), '->', finalLoss.toFixed(3))
console.log('held-out MAE:', Math.round(mae), 'USD; mean-price baseline:', Math.round(baseline), 'USD')
console.log('sample predicted / actual:', Math.round(values[0] * 100_000), '/', Math.round(testY[0] * 100_000), 'USD')
if (!Number.isFinite(mae) || finalLoss >= firstLoss || mae >= baseline) throw new Error('training did not beat the baseline')

predictions.dispose()
x.dispose()
target.dispose()
testX.dispose()
state.dispose()
session.dispose()
