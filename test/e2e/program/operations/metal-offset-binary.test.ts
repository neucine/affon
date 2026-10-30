import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { add, div, eq, gt, mul, slice, sub } from 'affon:ops'

describe('Metal f32 offset binary inputs', () => {
  test('uses both view offsets in arithmetic and comparison', () => {
    const session = new Session({ device: 'metal' })
    const x = session.tensor([[1, 2, 3], [4, 5, 6]])
    const y = session.tensor([[10, 20, 30], [40, 50, 60]])
    const source = program('offset_binary_contract', p => {
      const a = slice(p.argument('x', Tensor.f32([2, 3])), [{ start: 1, stop: 2 }, { start: 1, stop: 3 }])
      const b = slice(p.argument('y', Tensor.f32([2, 3])), [{ start: 1, stop: 2 }, { start: 1, stop: 3 }])
      return [add(a, b), sub(b, a), mul(a, b), div(b, a), eq(a, b), gt(b, a)]
    })
    const executable = session.compile(source)
    const outputs = executable.run({ x, y })
    expect(outputs.map(output => output.to_array())).toEqual([
      [[55, 66]], [[45, 54]], [[250, 360]], [[10, 10]], [[0, 0]], [[1, 1]],
    ])
    outputs.forEach(output => output.dispose())
    executable.dispose(); x.dispose(); y.dispose(); session.dispose()
  })
})
