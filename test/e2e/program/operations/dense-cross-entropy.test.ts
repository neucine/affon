import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { dense_cross_entropy, log_softmax_nll } from 'affon:ops'

for (const device of ['cpu', 'metal'] as const) describe(`dense_cross_entropy ${device}`, () => {
  test('evaluates soft targets through immediate and Program calls', () => {
    const session = new Session({ device })
    const logits = session.tensor([[0, 0, 0], [0, 0, 0]])
    const targets = session.tensor([[0.5, 0.5, 0], [0, 0.5, 0.5]])
    const immediate = dense_cross_entropy(logits, targets)
    expect(immediate.shape).toEqual([1])
    expect(immediate.dtype).toBe('f32')
    if (Math.abs((immediate.to_array() as number[])[0] - Math.log(3)) > 1e-6) throw Error('incorrect soft-target loss')
    const source = program('dense_loss_contract', p => dense_cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])), p.argument('targets', Tensor.f32([2, 3]))))
    const executable = session.compile(source, { explanationLevel: 'detailed' })
    const result = executable.run({ logits, targets })
    if (Math.abs((result.to_array() as number[])[0] - Math.log(3)) > 1e-6) throw Error('incorrect Program soft-target loss')
    result.dispose(); executable.dispose(); immediate.dispose(); targets.dispose(); logits.dispose(); session.dispose()
  })

  test('rejects targets with a different shape', () => {
    const source = () => program('invalid_dense_loss', p => dense_cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])), p.argument('targets', Tensor.f32([2, 2]))))
    expect(source).toThrow()
  })

  test('log_softmax_nll accepts logits and soft targets', () => {
    const session = new Session({ device })
    const logits = session.tensor([[0, 0, 0]])
    const targets = session.tensor([[0.5, 0.5, 0]])
    const immediate = log_softmax_nll(logits, targets)
    if (Math.abs((immediate.to_array() as number[])[0] - Math.log(3)) > 1e-6) throw Error('incorrect immediate NLL')
    const source = program('nll_contract', p => log_softmax_nll(
      p.argument('logits', Tensor.f32([1, 3])), p.argument('targets', Tensor.f32([1, 3]))))
    const executable = session.compile(source)
    const result = executable.run({ logits, targets })
    if (Math.abs((result.to_array() as number[])[0] - Math.log(3)) > 1e-6) throw Error('incorrect Program NLL')
    result.dispose(); executable.dispose(); immediate.dispose(); targets.dispose(); logits.dispose(); session.dispose()
  })

  test('reduces rank-three soft targets across batch and spatial positions', () => {
    const session = new Session({ device })
    const logits = session.tensor([[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]])
    const targets = session.tensor([[[0.5, 0], [0.5, 0.5], [0, 0.5]],
      [[0, 0.5], [0.5, 0.5], [0.5, 0]]])
    for (const loss of [dense_cross_entropy, log_softmax_nll]) {
      const source = program('batched_dense_loss_contract', p => loss(
        p.argument('logits', Tensor.f32([2, 3, 2])),
        p.argument('targets', Tensor.f32([2, 3, 2]))))
      const executable = session.compile(source)
      const result = executable.run({ logits, targets })
      expect(result.shape).toEqual([1])
      if (Math.abs((result.to_array() as number[])[0] - Math.log(3)) > 1e-6) throw Error('incorrect batched soft-target loss')
      result.dispose(); executable.dispose()
    }
    targets.dispose(); logits.dispose(); session.dispose()
  })
})
