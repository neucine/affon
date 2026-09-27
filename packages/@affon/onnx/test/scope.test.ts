import { test, expect } from 'std:test'
import fs from 'std:fs'
import { getEnv } from 'std:process'
import { tensor, add, neg, no_grad } from 'affon:compute'
import native from 'affon:compute/native'
import { load_graph } from '../src/index.ts'
import { load_graph_for_scope_validation } from '../src/runtime.ts'

const device = getEnv('AFFON_DEVICE') === 'metal' ? 'metal' : 'cpu'
const directory = 'packages/@affon/onnx/test/fixtures'
const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))
const inputs = () => ({ x: tensor(reference.input, { dtype: 'f32', device }) })

for (const [label, load] of [['default', load_graph], ['diagnostic', load_graph_for_scope_validation]] as const) {
  test(`${label} ONNX execution matches references and keeps outputs across calls`, () => {
    const model = load(directory, device)
    const first = model.forward(inputs())
    const second = model.forward(inputs())
    for (const outputs of [first, second]) {
      for (const [name, value] of Object.entries(reference.expected)) {
        const expected = (value as number[]).flat(Infinity) as number[]
        const actual = (outputs[name].to_array() as number[]).flat(Infinity) as number[]
        expect(actual.length).toBe(expected.length)
        expect(actual.every((x, i) => Number.isFinite(x) &&
          Math.abs(x - expected[i]) <= 1e-5 + 1e-5 * Math.abs(expected[i]))).toBe(true)
      }
    }
    if (!('scope_stats' in model)) return
    const stats = model.scope_stats()
    if (device === 'metal') {
      expect(stats!.encoded > 0).toBe(true)
      expect(stats!.submitted <= stats!.encoded).toBe(true)
    } else expect(stats).toBe(null)
  })

  test(`${label} profiling bypasses the scope and callback failures recover`, () => {
    const model = load(directory, device)
    let visits = 0
    model.forward(inputs(), () => {
      visits++
      // A new native scope succeeds only if profiling is outside the graph scope.
      const value = device === 'metal'
        ? no_grad(() => native.$with_graph_execution(() => neg(tensor([2], { device })))).value
        : neg(tensor([2], { device }))
      expect(value.to_array()).toEqual([-2])
    })
    expect(visits > 0).toBe(true)
    if ('scope_stats' in model) expect(model.scope_stats()).toBe(null)
    const failure = Error('profile callback failure')
    try {
      model.forward(inputs(), () => { throw failure })
      throw Error('callback did not throw')
    } catch (error) { expect(error).toBe(failure) }
    if ('scope_stats' in model) expect(model.scope_stats()).toBe(null)
    expect(model.forward(inputs()).result.shape).toEqual([1, 2])
  })
}

if (device === 'metal') {
  test('private native scope batches scalar broadcast chains across chunk limits', () => {
    const initial = tensor([2, 3], { dtype: 'f32', device })
    const increment = tensor(1, { dtype: 'f32', device })
    const result = no_grad(() => native.$with_graph_execution(() => {
      let value = initial
      for (let i = 0; i < 70; i++) value = add(value, increment)
      return value
    }))
    expect(result.value.to_array()).toEqual([72, 73])
    expect(result.encoded).toBe(70)
    expect(result.submitted).toBe(3)
  })

  test('private native scope preserves exceptions, rejects nesting and thenables, and recovers', () => {
    const failure = Error('graph failure')
    no_grad(() => {
      try {
        native.$with_graph_execution(() => {
          neg(tensor([2], { device }))
          throw failure
        })
        throw Error('graph did not throw')
      } catch (error) { expect(error).toBe(failure) }
      expect(() => native.$with_graph_execution(() =>
        native.$with_graph_execution(() => 1))).toThrow()
      expect(() => native.$with_graph_execution(() => ({ then() {} }))).toThrow()
      const result = native.$with_graph_execution(() => neg(tensor([2], { device })))
      expect(result.value.to_array()).toEqual([-2])
      expect(result.encoded).toBe(1)
      expect(result.submitted).toBe(1)
    })
  })
}
