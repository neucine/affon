import { describe, expect, test, values } from 'std:test'
import { read } from 'affon:dataset'
import { Session } from 'affon:compute'
import { captureError } from '../../support/errors.ts'

describe('tabular batch contents and recovery', () => {
  test('iterates exact rows, including a final short batch, on repeated passes', () => {
    const session = new Session({ device: 'cpu' })
    try {
      const data = read('test/e2e/dataset/fixtures/iris.csv')
        .features('sepal_length', 'sepal_width').target('petal_length')
      const loader = data.loader({ session, batchSize: 4 })
      const expectedX = [
        [[5.1, 3.5], [4.9, 3], [7, 3.2], [6.4, 3.2]],
        [[6.3, 3.3], [5.8, 2.7]],
      ]
      const expectedY = [[1.4, 1.4, 4.7, 4.5], [6, 5.1]]
      expect(loader.length).toBe(2)
      for (let pass = 0; pass < 2; pass++) {
        const batches = [...loader]
        expect(batches.length).toBe(2)
        for (let i = 0; i < batches.length; i++) {
          expect(values(batches[i][0])).toBeAllClose(expectedX[i])
          expect(values(batches[i][1])).toBeAllClose(expectedY[i])
          batches[i][0].dispose(); batches[i][1].dispose()
        }
      }
    } finally { session.dispose() }
  })

  test('a bad numeric row does not prevent a later valid loader', () => {
    const session = new Session({ device: 'cpu' })
    try {
      const error = captureError(() => [...read('test/e2e/dataset/fixtures/invalid-numeric.csv')
        .select('a').loader({ session, batchSize: 2 })])
      expect(error.message).toContain('InvalidNumericValue')
      const valid = read('test/e2e/dataset/fixtures/iris.csv')
        .features('sepal_length').target('petal_length')
      const batches = [...valid.loader({ session, batchSize: 5 })]
      expect(batches.map(([x]) => x.shape[0])).toEqual([5, 1])
      expect(values(batches[1][0])).toBeAllClose([[5.8]])
      expect(values(batches[1][1])).toBeAllClose([5.1])
      for (const batch of batches) for (const tensor of batch) tensor.dispose()
    } finally { session.dispose() }
  })
})
