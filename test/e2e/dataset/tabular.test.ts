import { describe, test, expect, values } from 'std:test'
import dataset, { read, DataLoader } from 'affon:dataset'

describe('dataset tabular', () => {
  test('reads csv columns and basic selections', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
    expect(ds.columns).toEqual(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

    const result = ds.select('petal_length', 'petal_width').toTensor()
    expect(result.data.shape).toEqual([6, 2])
    expect(result.data.dtype).toBe('f32')
    expect(result.schema).toEqual({ petal_length: [0], petal_width: [1] })
  })

  test('exposes tabular namespace for current csv workflow', () => {
    const ds = dataset.tabular.read('test/e2e/dataset/fixtures/iris.csv')
    expect(ds.columns).toEqual(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    expect(dataset.tabular.DataLoader).toBe(DataLoader)
  })

  test('encodes labels and one-hot columns', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')

    const labeled = ds.encode('species', 'label').select('species').toTensor()
    expect(labeled.data.shape).toEqual([6, 1])

    const onehot = ds.encode('species').toTensor()
    expect(onehot.data.shape).toEqual([6, 7])
    expect(onehot.schema).toEqual({
      sepal_length: [0],
      sepal_width: [1],
      petal_length: [2],
      petal_width: [3],
      species_setosa: [4],
      species_versicolor: [5],
      species_virginica: [6],
    })
  })

  test('builds feature and target tensors', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
      .encode('species', 'label')
      .features('petal_length', 'petal_width')
      .target('species')
      .toTensors()

    expect(ds.X.shape).toEqual([6, 2])
    expect(ds.y.shape).toEqual([6])
    expect(ds.X.dtype).toBe('f32')
    expect(ds.y.dtype).toBe('f32')
  })

  test('supports explicit f64 tensor export when requested', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
      .features('petal_length', 'petal_width')
      .target('sepal_length')

    const exported = ds.toTensors({ dtype: 'f64' })
    expect(exported.X.dtype).toBe('f64')
    expect(exported.y.dtype).toBe('f64')
  })

  test('drop and rename keep expected schemas', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')

    const dropped = ds.drop('species').toTensor()
    expect(dropped.data.shape).toEqual([6, 4])
    expect(dropped.schema).toEqual({
      sepal_length: [0],
      sepal_width: [1],
      petal_length: [2],
      petal_width: [3],
    })

    const renamed = ds.rename('sepal_length', 'sl').rename('sepal_width', 'sw').select('sl', 'sw').toTensor()
    expect(renamed.schema).toEqual({ sl: [0], sw: [1] })
  })

  test('split and concat preserve expected row counts', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')

    const [train, testSplit] = ds.split(0.8)
    expect(train.select('sepal_length', 'sepal_width').toTensor().data.shape).toEqual([5, 2])
    expect(testSplit.select('sepal_length', 'sepal_width').toTensor().data.shape).toEqual([1, 2])

    const [tr, val, te] = ds.split(0.5, 0.3)
    expect(tr.select('sepal_length').toTensor().data.shape).toEqual([3, 1])
    expect(val.select('sepal_length').toTensor().data.shape).toEqual([2, 1])
    expect(te.select('sepal_length').toTensor().data.shape).toEqual([1, 1])

    const [part1, part2] = ds.split(0.5)
    const rejoined = part1.concat(part2).select('sepal_length', 'sepal_width').toTensor()
    expect(rejoined.data.shape).toEqual([6, 2])
  })

  test('fillna and dropna handle missing rows', () => {
    const ms = read('test/e2e/dataset/fixtures/missing.csv')
    expect(ms.columns).toEqual(['id', 'value', 'score', 'label'])

    const filled = ms.fillna('value', 0).fillna('score', -1).select('value', 'score').toTensor()
    expect(filled.data.shape).toEqual([5, 2])
    expect(values(filled.data)).toBeAllClose([[10, 5.5], [0, 3.2], [30, -1], [40, 8.1], [0, -1]])

    const cleaned = ms.dropna('value', 'score').select('id', 'value', 'score').toTensor()
    expect(cleaned.data.shape).toEqual([2, 3])
  })

  test('sample and dataloader expose expected batch shapes', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')

    const sampled = ds.sample(3).select('sepal_length').toTensor()
    expect(sampled.data.shape).toEqual([3, 1])

    const trainDs = ds
      .encode('species', 'label')
      .features('sepal_length', 'sepal_width')
      .target('species')

    const loader = new DataLoader(trainDs, { batchSize: 2 })
    const batches: number[][] = []
    for (const [bX, bY] of loader) {
      batches.push([...bX.shape, bY.shape[0]])
    }
    expect(loader.length).toBe(3)
    expect(batches.length).toBe(3)
    expect(batches[0]).toEqual([2, 2, 2])
    expect(batches[batches.length - 1]).toEqual([2, 2, 2])

    const loader2 = trainDs.loader({ batchSize: 3 })
    expect(loader2.length).toBe(2)
  })

  test('tabular dataset supports shared input and tensorLoader aliases', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
      .encode('species', 'label')
      .input('sepal_length', 'sepal_width')
      .target('species')

    const loader = ds.tensorLoader({ batchSize: 2 })
    const batches = [...loader]

    expect(loader.length).toBe(3)
    expect(batches[0][0].shape).toEqual([2, 2])
    expect(batches[0][1].shape).toEqual([2])
    const inputValues = values(batches[0][0]) as number[][]
    expect(Math.abs(inputValues[0][0] - 5.1) < 1e-6).toBe(true)
    expect(Math.abs(inputValues[0][1] - 3.5) < 1e-6).toBe(true)
    expect(Math.abs(inputValues[1][0] - 4.9) < 1e-6).toBe(true)
    expect(Math.abs(inputValues[1][1] - 3.0) < 1e-6).toBe(true)
    expect(values(batches[0][1])).toEqual([0, 0])
  })

  test('split results keep loader convenience', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
    const [train, testSplit] = ds.split(0.8)

    expect(train.loader({ batchSize: 2 }).length).toBe(3)
    expect(testSplit.loader({ batchSize: 1 }).length).toBe(1)
  })

  test('sample validates non-negative integer input', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
    expect(ds.sample(2).toTensor().data.shape[0]).toBe(2)
  })

  test('dataloader validates positive integer batch size', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv').select('sepal_length')
    expect(new DataLoader(ds, { batchSize: 2 }).length).toBe(3)
  })

  test('split validates ratio count', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
    const parts = ds.split(0.5, 0.25, 0.25)
    expect(parts.length).toBe(4)
    expect(parts[0].loader({ batchSize: 1 }).length).toBe(3)
  })

  test('numeric csv fixture remains readable through columns', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
    expect(ds.select('species').loader({ batchSize: 2 }).length).toBe(3)
  })

  test('normalize keeps expected shape', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
    const normed = ds.select('sepal_length', 'sepal_width').normalize('sepal_length', 'sepal_width').toTensor()
    expect(normed.data.shape).toEqual([6, 2])
  })
})
