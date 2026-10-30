import checkpoint from 'affon:checkpoint'
import { read } from 'affon:dataset'
import { Session, Tensor, optimize, program } from 'affon:compute'
import { matmul, mean_squared_error, mul, reshape } from 'affon:ops'
import { adam, sgd } from 'affon:optim'
import { describe, expect, test } from 'std:test'

describe('Program training workflow', () => {
  test('restores optimizer tensors and step for the same next update', () => {
    const model = program('resumable_model', p => mul(
      p.argument('x', Tensor.f32([2])),
      p.parameter('weight', Tensor.f32([2]), { initializer: { kind: 'ones' } }),
    ))
    const loss = program('resumable_loss', p => mean_squared_error(
      p.argument('prediction', Tensor.f32([2])),
      p.argument('target', Tensor.f32([2])),
    ))
    const train = optimize(model, loss, adam({ learning_rate: 0.05 }))
    const firstSession = new Session({ device: 'cpu' })
    const firstState = firstSession.initialize(train)
    const firstX = firstSession.tensor([1, 2])
    const firstTarget = firstSession.tensor([2, 4])
    const firstStep = firstSession.compile(train)
    for (let index = 0; index < 3; index++) firstStep.run({ x: firstX, target: firstTarget }, firstState).dispose()

    const optimizerTensors: Record<string, Tensor> = {}
    const optimizerScalars: Record<string, number> = {}
    for (const [name, value] of Object.entries(firstState.optimizer_state)) {
      if (typeof value === 'number') optimizerScalars[name] = value
      else optimizerTensors[name] = value
    }
    const prefix = `/tmp/affon-resume-workflow-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    checkpoint.saveBundle(prefix, {
      state: firstState.parameters,
      tensorGroups: { optimizer: optimizerTensors },
      manifest: { optimizerScalars },
    })
    const uninterrupted = firstStep.run({ x: firstX, target: firstTarget }, firstState)

    const loaded = checkpoint.loadBundle(prefix)
    const secondSession = new Session({ device: 'cpu' })
    const parameters = Object.fromEntries(Object.entries(loaded.state).map(([name, value]) => [name, value.to_array()]))
    const restoredState = secondSession.initialize(train, { parameters })
    for (const [name, value] of Object.entries(loaded.tensorGroups.optimizer)) {
      restoredState.optimizer_state[name] = secondSession.tensor(value.to_array(), { dtype: value.dtype })
    }
    Object.assign(restoredState.optimizer_state, loaded.manifest.optimizerScalars)
    const secondX = secondSession.tensor([1, 2])
    const secondTarget = secondSession.tensor([2, 4])
    const resumed = secondSession.compile(train).run({ x: secondX, target: secondTarget }, restoredState)

    expect(Math.abs(uninterrupted.item() - resumed.item()) < 1e-6).toBe(true)
    expect(restoredState.optimizer_state.$step).toBe(firstState.optimizer_state.$step)
    for (const [name, value] of Object.entries(firstState.parameters)) {
      const restored = restoredState.parameters[name].to_array() as number[]
      const expected = value.to_array() as number[]
      expect(restored.every((item, index) => Math.abs(item - expected[index]) < 1e-6)).toBe(true)
    }
    for (const [name, value] of Object.entries(firstState.optimizer_state)) {
      if (typeof value === 'number') expect(restoredState.optimizer_state[name]).toBe(value)
      else {
        const actual = (restoredState.optimizer_state[name] as Tensor).to_array() as number[]
        const expected = value.to_array() as number[]
        expect(actual.every((item, index) => Math.abs(item - expected[index]) < 1e-6)).toBe(true)
      }
    }

    for (const value of Object.values(loaded.state)) value.dispose()
    for (const value of Object.values(loaded.tensorGroups.optimizer)) value.dispose()
    uninterrupted.dispose(); resumed.dispose()
    firstX.dispose(); firstTarget.dispose(); secondX.dispose(); secondTarget.dispose()
    firstState.dispose(); restoredState.dispose(); firstSession.dispose(); secondSession.dispose()
  })

  test('trains, checkpoints, restores, and resumes with consistent state', () => {
    const model = program('linear_workflow', p => mul(
      p.argument('x', Tensor.f32([3])),
      p.parameter('weight', Tensor.f32([1]), { initializer: { kind: 'zeros' } }),
    ))
    const loss = program('regression_workflow_loss', p => mean_squared_error(
      p.argument('prediction', Tensor.f32([3])),
      p.argument('target', Tensor.f32([3])),
    ))
    const train = optimize(model, loss, sgd({ learning_rate: 0.05 }))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(train)
    const x = session.tensor([1, 2, 3])
    const target = session.tensor([2, 4, 6])
    const step = session.compile(train)
    const parameterName = train.inspect().parameters[0].provenance
    let firstLoss = 0
    let lastLoss = 0
    for (let index = 0; index < 12; index++) {
      const result = step.run({ x, target }, state)
      const lossValue = result.item()
      if (index === 0) firstLoss = lossValue
      if (index > 0) expect(lossValue < lastLoss).toBe(true)
      lastLoss = lossValue
      result.dispose()
    }
    expect(lastLoss < firstLoss / 100).toBe(true)
    expect(state.optimizer_state.$step).toBe(12)

    const path = `/tmp/affon-program-workflow-${Date.now()}-${Math.floor(Math.random() * 1e6)}.safetensors`
    checkpoint.save({ weight: state.parameters[parameterName] }, path)
    const saved = checkpoint.load(path) as { weight: Tensor }
    const restored = session.initialize(model, { parameters: { [parameterName]: saved.weight.to_array() } })
    const before = session.compile(model).run({ x }, state)
    const after = session.compile(model).run({ x }, restored)
    expect(after.to_array()).toEqual(before.to_array())
    expect((after.to_array() as number[])[0] > 1.9).toBe(true)

    const resumed = session.compile(train).run({ x, target }, restored)
    expect(resumed.item() < lastLoss).toBe(true)
    expect(restored.optimizer_state.$step).toBe(1)
    resumed.dispose(); before.dispose(); after.dispose(); saved.weight.dispose()
    restored.dispose(); x.dispose(); target.dispose(); state.dispose(); session.dispose()
  })

  test('feeds dataset tensors into a compiled training Program', () => {
    const session = new Session({ device: 'cpu' })
    const dataset = read('test/e2e/dataset/fixtures/iris.csv')
      .features('sepal_width', 'petal_length')
      .target('sepal_length')
      .toTensors({ session })
    expect(dataset.X.shape).toEqual([6, 2])
    expect(dataset.y.shape).toEqual([6])

    const model = program('tabular_workflow', p => reshape(matmul(
      p.argument('X', Tensor.f32([6, 2])),
      p.parameter('weight', Tensor.f32([2, 1]), { initializer: { kind: 'zeros' } }),
    ), [6]))
    const loss = program('tabular_workflow_loss', p => mean_squared_error(
      p.argument('prediction', Tensor.f32([6])),
      p.argument('target', Tensor.f32([6])),
    ))
    const train = optimize(model, loss, sgd({ learning_rate: 0.001 }))
    const state = session.initialize(train)
    const executable = session.compile(train)
    const losses: number[] = []
    for (let index = 0; index < 8; index++) {
      const result = executable.run({ X: dataset.X, target: dataset.y }, state)
      losses.push(result.item())
      result.dispose()
    }
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(losses.at(-1)! < losses[0]).toBe(true)
    expect(state.optimizer_state.$step).toBe(8)
    dataset.X.dispose(); dataset.y.dispose(); state.dispose(); session.dispose()
  })
})
