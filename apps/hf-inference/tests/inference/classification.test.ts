import { expect, test } from 'std:test'
import { rank_classes } from '../../src/inference/classification.ts'

test('classification normalizes over every class before selecting top five', () => {
  const results = rank_classes([1000, 1000, 1000, 1000, 1000, 1000], {
    '0': 'cat',
  })
  expect(results.length).toBe(5)
  expect(results[0].label).toBe('cat')
  expect(results.map((item) => item.id)).toEqual([0, 1, 2, 3, 4])
  expect(results[0].score).toBe(1 / 6)
  expect(() => rank_classes([NaN], {})).toThrow()
})
