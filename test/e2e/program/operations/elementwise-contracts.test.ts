import { testDevices } from '../../devices.ts'
import { Session, Tensor, program } from 'affon:compute'
import { abs, add, clamp, div, eq, exp, gt, gt_scalar, log, lt, mul, neg, relu, sign, sqrt, sub, where } from 'affon:ops'
import { describe, expect, test, values } from 'std:test'

describe('Elementwise operation contracts', () => {
  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`broadcasts a row and column through evaluated and compiled arithmetic on ${device}`, () => {
    const source = program('elementwise_broadcast', p => {
      const left = p.argument('left', Tensor.f32([2, 1]))
      const right = p.argument('right', Tensor.f32([1, 3]))
      return [add(left, right), sub(left, right), mul(left, right), div(right, left)]
    })
    const session = new Session({ device })
    const left = session.tensor([[2], [4]])
    const right = session.tensor([[1, 2, 4]])
    const expected = [
      [[3, 4, 6], [5, 6, 8]],
      [[1, 0, -2], [3, 2, 0]],
      [[2, 4, 8], [4, 8, 16]],
      [[0.5, 1, 2], [0.25, 0.5, 1]],
    ]
    const direct = [add(left, right), sub(left, right), mul(left, right), div(right, left)]
    const executable = session.compile(source)
    const compiled = executable.run({ left, right }) as Tensor[]
    for (let index = 0; index < expected.length; index++) {
      expect(direct[index].to_array()).toEqual(expected[index])
      expect(compiled[index].to_array()).toEqual(expected[index])
    }
    const changed = session.tensor([[8], [16]])
    const repeated = executable.run({ left: changed, right }) as Tensor[]
    expect(repeated[0].to_array()).toEqual([[9, 10, 12], [17, 18, 20]])
    expect(repeated[2].to_array()).toEqual([[8, 16, 32], [16, 32, 64]])
    for (const tensor of [...direct, ...compiled, ...repeated, left, right, changed]) tensor.dispose()
    session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`comparisons, thresholds, and where use element values after broadcasting on ${device}`, () => {
    const source = program('elementwise_decisions', p => {
      const data = p.argument('data', Tensor.f32([2, 3]))
      const threshold = p.argument('threshold', Tensor.f32([1, 3]))
      return [eq(data, threshold), lt(data, threshold), gt(data, threshold),
        gt_scalar(data, 0), where(gt(data, threshold), data, threshold)]
    })
    const session = new Session({ device })
    const data = session.tensor([[-2, 0, 3], [2, 0, -3]])
    const threshold = session.tensor([[0, 0, 1]])
    const expected = [
      [[0, 1, 0], [0, 1, 0]],
      [[1, 0, 0], [0, 0, 1]],
      [[0, 0, 1], [1, 0, 0]],
      [[0, 0, 1], [1, 0, 0]],
      [[0, 0, 3], [2, 0, 1]],
    ]
    const comparison = gt(data, threshold)
    const direct = [eq(data, threshold), lt(data, threshold), comparison,
      gt_scalar(data, 0), where(comparison, data, threshold)]
    const compiled = session.compile(source).run({ data, threshold }) as Tensor[]
    for (let index = 0; index < expected.length; index++) {
      expect(direct[index].to_array()).toEqual(expected[index])
      expect(compiled[index].to_array()).toEqual(expected[index])
    }
    for (const tensor of [...direct, ...compiled, data, threshold]) tensor.dispose()
    session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`unary boundaries and compositions retain signs and clamped values on ${device}`, () => {
    const session = new Session({ device })
    const input = session.tensor([-4, -1, 0, 1, 4])
    const cases = [
      [() => neg(input), [4, 1, -0, -1, -4]],
      [() => abs(input), [4, 1, 0, 1, 4]],
      [() => sign(input), [-1, -1, 0, 1, 1]],
      [() => relu(input), [0, 0, 0, 1, 4]],
      [() => clamp(input, -1, 1), [-1, -1, 0, 1, 1]],
    ] as const
    for (const [run, expected] of cases) {
      const result = run()
      expect(result.to_array()).toEqual(expected)
      result.dispose()
    }
    const positive = session.tensor([0.25, 1, 4])
    const rooted = sqrt(positive)
    expect(rooted.to_array()).toEqual([0.5, 1, 2])
    const exponentiated = exp(positive)
    const restored = log(exponentiated)
    expect(values(restored)).toBeAllClose([0.25, 1, 4])
    rooted.dispose(); restored.dispose(); exponentiated.dispose(); positive.dispose(); input.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`bad broadcast shapes fail without invalidating operands or a compiled Program on ${device}`, () => {
    const session = new Session({ device })
    const a = session.tensor([[1, 2], [3, 4]])
    const invalid = session.tensor([1, 2, 3])
    const valid = session.tensor([10, 20])
    expect(() => add(a, invalid)).toThrow('cannot broadcast')
    const source = program('broadcast_recovery', p => add(
      p.argument('a', Tensor.f32([2, 2])), p.argument('valid', Tensor.f32([2])),
    ))
    const executable = session.compile(source)
    const first = executable.run({ a, valid })
    const second = executable.run({ a, valid })
    expect(first.to_array()).toEqual([[11, 22], [13, 24]])
    expect(second.to_array()).toEqual(first.to_array())
    first.dispose(); second.dispose(); a.dispose(); invalid.dispose(); valid.dispose(); session.dispose()
  })
})
