import { Session, Tensor, program } from 'affon:compute'
import { add, argmax, argmin, max, mean, min, std, sum, variance } from 'affon:ops'
import { describe, expect, test, values } from 'std:test'

const data = [[1, 5, 5], [3, -1, 7]]
const cases = [
  { name: 'sum', run: sum, expected: [11, 9] },
  { name: 'mean', run: mean, expected: [11 / 3, 3] },
  { name: 'min', run: min, expected: [1, -1] },
  { name: 'max', run: max, expected: [5, 7] },
  { name: 'variance', run: variance, expected: [32 / 9, 32 / 3] },
  { name: 'std', run: std, expected: [Math.sqrt(32 / 9), Math.sqrt(32 / 3)] },
  { name: 'argmin', run: argmin, expected: [0, 1] },
  { name: 'argmax', run: argmax, expected: [1, 2] },
] as const

describe('Reduction axes, values, and repeatability', () => {
  test('Metal keeps a parallel axis reduction between GPU producer and consumer', () => {
    const source = program('gpu_axis_sum_resident', p => {
      const input = p.argument('input', Tensor.f32([129, 257]))
      const reduced = sum(add(input, input), 1)
      return add(reduced, reduced)
    })
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source, { explanationLevel: 'detailed' })
    const input = session.tensor(Array.from({ length: 129 }, (_, row) => Array(257).fill(row)))
    const output = executable.run({ input })
    expect(output.to_array()).toEqual(Array.from({ length: 129 }, (_, row) => row * 4 * 257))
    expect(JSON.stringify(executable.explain())).toContain('metal.reduction.parallel_generic')
    output.dispose(); input.dispose(); executable.dispose(); session.dispose()
  })

  test('Metal keeps parallel axis variance between GPU producer and consumer', () => {
    const source = program('gpu_axis_variance_resident', p => {
      const input = p.argument('input', Tensor.f32([129, 257]))
      const zeros = p.argument('zeros', Tensor.f32([129, 257]))
      const reduced = variance(add(input, zeros), 1)
      return add(reduced, reduced)
    })
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source, { explanationLevel: 'detailed' })
    const rows = Array.from({ length: 129 }, () => Array.from({ length: 257 }, (_, column) => column - 128))
    const input = session.tensor(rows)
    const zeros = session.tensor(Array.from({ length: 129 }, () => Array(257).fill(0)))
    const output = executable.run({ input, zeros })
    const expected = 2 * (257 * 257 - 1) / 12
    expect(output.to_array().every((value: number) => Math.abs(value - expected) < 0.01)).toBe(true)
    expect(JSON.stringify(executable.explain())).toContain('metal.reduction.parallel_generic')
    output.dispose(); zeros.dispose(); input.dispose(); executable.dispose(); session.dispose()
  })

  test('Metal parallel argmax keeps first ties and first NaN indices', () => {
    const source = program('gpu_argmax_axis_ties', p => {
      const input = p.argument('input', Tensor.f32([2, 128]))
      const zeros = p.argument('zeros', Tensor.f32([2, 128]))
      const selected = argmax(add(input, zeros), 1)
      return add(selected, selected)
    })
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source)
    const rows = [Array(128).fill(0), Array(128).fill(0)]
    rows[0][1] = 5; rows[0][2] = 5; rows[1][0] = NaN; rows[1][7] = 9
    const input = session.tensor(rows)
    const zeros = session.tensor([Array(128).fill(0), Array(128).fill(0)])
    const output = executable.run({ input, zeros })
    expect(output.to_array()).toEqual([2, 0])
    output.dispose(); zeros.dispose(); input.dispose(); executable.dispose(); session.dispose()
  })

  for (const device of ['cpu', 'metal'] as const) {
    for (const entry of cases) {
      test(`${entry.name} on ${device} handles negative axis and keep_dims`, () => {
        const source = program(`reduction_${entry.name}_${device}`, p => entry.run(
          p.argument('input', Tensor.f32([2, 3])), -1, true,
        ))
        const session = new Session({ device })
        const input = session.tensor(data)
        const direct = entry.run(input, -1, true)
        const executable = session.compile(source)
        const compiled = executable.run({ input })
        const expected = entry.expected.map(value => [value])
        expect(direct.shape).toEqual([2, 1])
        expect(compiled.shape).toEqual([2, 1])
        expect(values(direct)).toBeAllClose(expected, { atol: 1e-4, rtol: 1e-4 })
        expect(values(compiled)).toBeAllClose(expected, { atol: 1e-4, rtol: 1e-4 })
        const again = executable.run({ input })
        expect(values(again)).toBeAllClose(expected, { atol: 1e-4, rtol: 1e-4 })
        direct.dispose(); compiled.dispose(); again.dispose(); input.dispose(); session.dispose()
      })
    }
  }

  test('column reductions use independent columns with integer argmax output', () => {
    const session = new Session({ device: 'cpu' })
    const input = session.tensor(data, { axes: ['batch', 'feature'] })
    const total = sum(input, 0)
    const largest = argmax(input, 0)
    expect(total.to_array()).toEqual([4, 4, 12])
    expect(largest.to_array()).toEqual([1, 0, 1])
    expect(total.dtype).toBe('f32')
    expect(largest.dtype).toBe('i64')
    total.dispose(); largest.dispose(); input.dispose(); session.dispose()
  })

  test('invalid axis leaves the tensor usable for a valid reduction', () => {
    const session = new Session({ device: 'cpu' })
    const input = session.tensor(data)
    expect(() => sum(input, 2)).toThrow()
    expect(() => argmin(input, -3)).toThrow()
    const result = sum(input, 1)
    expect(result.to_array()).toEqual([11, 9])
    result.dispose(); input.dispose(); session.dispose()
  })
})
