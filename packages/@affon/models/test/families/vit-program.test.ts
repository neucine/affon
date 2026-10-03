import { expect, test } from 'std:test'
import { Session } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { create_vit } from '../../src/vit/model.ts'

test('runs ViT patch extraction and encoder through one Program', () => {
  const source = new Session({ device: 'cpu' })
  const value = (shape: number[], fill = 0.1): Tensor => {
    const data = (dimensions: number[]): any => dimensions.length
      ? Array.from({ length: dimensions[0] }, () => data(dimensions.slice(1)))
      : fill
    return source.tensor(data(shape)) as Tensor
  }
  const affine = (input: number, output: number) => ({ weight: value([input, output]), bias: value([output], 0) })
  const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
  const model = create_vit({ width: 4, innerWidth: 8, heads: 2, layers: 1, imageSize: 2, patchSize: 1, epsilon: 1e-5 }, {
    classToken: value([1, 1, 4]),
    positionEmbedding: value([1, 5, 4]),
    patchProjection: { weight: value([4, 3, 1, 1]), bias: value([4], 0) },
    finalNorm: norm(),
    classifier: affine(4, 3),
    blocks: [{ query: affine(4, 4), key: affine(4, 4), value: affine(4, 4), attentionOutput: affine(4, 4), attentionNorm: norm(), feedForwardNorm: norm(), expand: affine(4, 8), contract: affine(8, 4) }],
  })
  const runtime = new Session({ device: 'cpu' })
  const state = runtime.initialize(model.forward, { parameters: model.parameters })
  const pixels = runtime.tensor(value([1, 3, 2, 2], 0.5).to_array())
  const [output, ...hiddenStates] = runtime.compile(model.forward).run({ pixels }, state) as Tensor[]
  expect(output.shape).toEqual([1, 3])
  expect(hiddenStates.map(hidden => hidden.shape)).toEqual([[1, 5, 4], [1, 5, 4]])
  expect((output.to_array() as number[][]).flat().every(Number.isFinite)).toBe(true)

  output.dispose()
  for (const hidden of hiddenStates) hidden.dispose()
  pixels.dispose()
  state.dispose()
  runtime.dispose()
  source.dispose()
})
