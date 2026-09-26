import { describe, test, expect } from 'std:test'
import { compare_values } from '../audit/compare.ts'

describe('HF parity comparison', () => {
  test('uses elementwise absolute and relative tolerance', () => {
    expect(compare_values([0.00005, 100.005], [0, 100]).passed).toBe(true)
    expect(compare_values([0.001, 100], [0, 100]).mismatches).toBe(1)
  })
  test('nonfinite values never pass parity', () => {
    expect(compare_values([NaN, Infinity], [0, Infinity]).mismatches).toBe(2)
  })
  test('empty or mismatched references cannot pass', () => {
    expect(() => compare_values([], [])).toThrow()
    expect(() => compare_values([1], [1, 2])).toThrow()
  })
})
