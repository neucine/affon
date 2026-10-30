import { Session, Tensor, optimize, program } from 'affon:compute'
import { cross_entropy, index_select, mean, mul } from 'affon:ops'
import { sgd } from 'affon:optim'
import { describe, expect, test } from 'std:test'
import { captureError } from '../../../support/errors.ts'

describe('Executable recovery after rejected runs', () => {
  test('backend index error leaves the compiled executable reusable', () => {
    const source = program('index_recovery', p => index_select(
      p.argument('table', Tensor.f32([3, 2])), 0,
      p.argument('indices', Tensor.i64([2])),
    ))
    const session = new Session({ device: 'cpu' })
    const executable = session.compile(source)
    const table = session.tensor([[1, 2], [3, 4], [5, 6]])
    const invalid = session.tensor([0, 3], { dtype: 'i64' })
    const valid = session.tensor([2, 0], { dtype: 'i64' })
    const error = captureError(() => executable.run({ table, indices: invalid }))
    expect(error.message).toContain('Program execution failed: InvalidCommand')
    const output = executable.run({ table, indices: valid })
    expect(output.to_array()).toEqual([[5, 6], [1, 2]])
    output.dispose(); valid.dispose(); invalid.dispose(); table.dispose(); session.dispose()
  })

  test('failed optimizer forward leaves parameters and step unchanged', () => {
    const model = program('optimizer_label_recovery', p => mul(
      p.argument('input', Tensor.f32([1, 2])),
      p.parameter('weight', Tensor.f32([1, 2]), { initializer: { kind: 'ones' } }),
    ))
    const loss = program('optimizer_label_loss', p => cross_entropy(
      p.argument('prediction', Tensor.f32([1, 2])),
      p.argument('labels', Tensor.i64([1])),
    ))
    const train = optimize(model, loss, sgd({ learning_rate: 0.1 }))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(train)
    const executable = session.compile(train)
    const key = train.inspect().parameters[0].provenance
    const input = session.tensor([[1, 2]])
    const valid = session.tensor([1], { dtype: 'i64' })
    const first = executable.run({ input, labels: valid }, state)
    expect(Number.isFinite(first.item())).toBe(true)
    first.dispose()
    const before = state.parameters[key].to_array()
    expect(state.optimizer_state.$step).toBe(1)
    const invalid = session.tensor([2], { dtype: 'i64' })
    expect(captureError(() => executable.run({ input, labels: invalid }, state)).message)
      .toContain('Program execution failed: InvalidCommand')
    expect(state.parameters[key].to_array()).toEqual(before)
    expect(state.optimizer_state.$step).toBe(1)
    const output = executable.run({ input, labels: valid }, state)
    expect(Number.isFinite(output.item())).toBe(true)
    expect(state.optimizer_state.$step).toBe(2)
    output.dispose(); valid.dispose(); invalid.dispose(); input.dispose(); state.dispose(); session.dispose()
  })

  test('invalid arguments leave training state and the executable usable', () => {
    const model = program('recovery_model', p => mul(
      p.argument('value', Tensor.f32([1])),
      p.parameter('weight', Tensor.f32([1]), { initializer: { kind: 'ones' } }),
    ))
    const loss = program('recovery_loss', p => mean(p.argument('prediction', Tensor.f32([1]))))
    const train = optimize(model, loss, sgd({ learning_rate: 0.1 }))
    const session = new Session({ device: 'cpu' })
    const foreignSession = new Session({ device: 'cpu' })
    const state = session.initialize(train)
    const executable = session.compile(train)
    const valid = session.tensor([2])
    const first = executable.run({ value: valid }, state)
    expect(first.item()).toBe(2)
    first.dispose()
    const key = train.inspect().parameters[0].provenance
    const snapshot = state.parameters[key].to_array()
    const step = state.optimizer_state.$step

    const wrongShape = session.tensor([2, 3])
    const foreign = foreignSession.tensor([2])
    const disposed = session.tensor([2])
    disposed.dispose()
    expect(() => executable.run({} as any, state)).toThrow('missing Program argument')
    expect(() => executable.run({ value: wrongShape }, state)).toThrow('TensorSpec')
    expect(() => executable.run({ value: foreign }, state)).toThrow('not a Tensor owned')
    expect(() => executable.run({ value: disposed }, state)).toThrow('not a Tensor owned')
    expect(state.parameters[key].to_array()).toEqual(snapshot)
    expect(state.optimizer_state.$step).toBe(step)

    const second = executable.run({ value: valid }, state)
    expect(Math.abs(second.item() - 1.6) < 1e-6).toBe(true)
    expect(state.optimizer_state.$step).toBe(2)
    second.dispose(); valid.dispose(); wrongShape.dispose(); foreign.dispose()
    state.dispose(); session.dispose(); foreignSession.dispose()
  })
})
