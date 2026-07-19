import { describe, test, expect, values } from 'std:test'
import { linspace } from 'affon:compute'

describe('compute linspace', () => {
  test('creates common linspace variants', () => {
    const defaultValues = values(linspace(0, 1, 100)) as number[]
    expect(defaultValues).toHaveLength(100)
    expect(defaultValues[0]).toBe(0)
    expect(defaultValues[99]).toBe(1)
    expect(values(linspace(0, 1, 5))).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(values(linspace(2, 9, 1))).toEqual([2])
    expect(values(linspace(2, 9, 0))).toEqual([])
    expect(values(linspace(3, -1, 5))).toEqual([3, 2, 1, 0, -1])
  })
})
