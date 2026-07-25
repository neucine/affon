import { describe, test, expect, values } from 'std:test'
import { grad, tensor } from 'affon:compute'
import nn from 'affon:nn'

function roundNested(value: any): any {
  if (Array.isArray(value)) return value.map(roundNested)
  return Math.round(value * 1e6) / 1e6
}

describe('nn embedding', () => {
  test('looks up token rows for sequence and batch inputs', () => {
    const embedding = nn.Embedding(5, 3)
    embedding.restore({
      weight: tensor([
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
        [4, 40, 400],
        [5, 50, 500],
      ])
    })

    const seq = embedding(tensor([0, 2, 4], { dtype: 'i64' }))
    expect(seq.shape).toEqual([3, 3])
    expect(values(seq)).toEqual([
      [1, 10, 100],
      [3, 30, 300],
      [5, 50, 500],
    ])

    const batch = embedding(tensor([
      [1, 0],
      [4, 3],
    ], { dtype: 'i64' }))
    expect(batch.shape).toEqual([2, 2, 3])
    expect(values(batch)).toEqual([
      [[2, 20, 200], [1, 10, 100]],
      [[5, 50, 500], [4, 40, 400]],
    ])
  })

  test('accumulates gradients into the referenced embedding rows', () => {
    const embedding = nn.Embedding(4, 2)
    embedding.restore({
      weight: tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ])
    })

    const out = embedding(tensor([1, 1, 3], { dtype: 'i64' }))
    const loss = out.sum()
    grad(loss, embedding.parameters)

    expect(roundNested(values(embedding.weight.grad))).toEqual([
      [0, 0],
      [2, 2],
      [0, 0],
      [1, 1],
    ])
  })
})
