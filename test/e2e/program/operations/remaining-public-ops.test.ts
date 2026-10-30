import { Session, Tensor, program } from 'affon:compute'
import { erf, gelu, log_softmax, permute, rms_norm, silu, square } from 'affon:ops'
import { describe, expect, test, values } from 'std:test'

describe('remaining public operation behavior', () => {
  test('runs activation and normalization calls in evaluated and compiled forms', () => {
    const source = program('activation_and_norm', p => {
      const input = p.argument('input', Tensor.f32([3]))
      return [square(input), silu(input), erf(input), gelu(input), log_softmax(input, 0), rms_norm(input, 0)]
    })
    expect(source.inspect().nodes.filter(node => node.kind === 'operation').map(node => node.op)).toEqual([
      'mul', 'silu', 'erf', 'gelu', 'log_softmax', 'rms_norm',
    ])
    const session = new Session({ device: 'cpu' })
    const input = session.tensor([-1, 0, 1])
    const direct = [square(input), silu(input), erf(input), gelu(input), log_softmax(input, 0), rms_norm(input, 0)]
    const compiled = session.compile(source).run({ input }) as Tensor[]
    const expected = [
      [1, 0, 1],
      [-0.2689414, 0, 0.7310586],
      [-0.8427008, 0, 0.8427008],
      [-0.158808, 0, 0.841192],
      [-2.407606, -1.407606, -0.407606],
      [-1.2247357, 0, 1.2247357],
    ]
    for (let index = 0; index < direct.length; index++) {
      expect(values(direct[index])).toBeAllClose(expected[index], { atol: 1e-4, rtol: 1e-4 })
      expect(values(compiled[index])).toBeAllClose(expected[index], { atol: 1e-4, rtol: 1e-4 })
    }
    for (const tensor of [...direct, ...compiled, input]) tensor.dispose()
    session.dispose()
  })

  test('permutes three axes in evaluated and compiled forms', () => {
    const source = program('permute_three_axes', p => permute(
      p.argument('input', Tensor.f32([2, 1, 3])), [2, 0, 1],
    ))
    const session = new Session({ device: 'cpu' })
    const input = session.tensor([[[1, 2, 3]], [[4, 5, 6]]])
    const expected = [[[1], [4]], [[2], [5]], [[3], [6]]]
    const direct = permute(input, [2, 0, 1])
    const compiled = session.compile(source).run({ input }) as Tensor
    expect(direct.to_array()).toEqual(expected)
    expect(compiled.to_array()).toEqual(expected)
    direct.dispose(); compiled.dispose(); input.dispose(); session.dispose()
  })
})
