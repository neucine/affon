import { test, expect, afterAll } from 'std:test'
import fs from 'std:fs'
import { getEnv } from 'std:process'
import { Session } from 'affon:compute'
import type { Device } from 'affon:compute'
import { load_graph } from '../src/index.ts'
import { load_graph_for_scope_validation } from '../src/runtime.ts'

const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const session = new Session({ device })
const directory = 'packages/@affon/onnx/test/fixtures'
const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))
const inputs = () => ({ x: session.tensor(reference.input) as any })
afterAll(() => session.dispose())

for (const [label, load] of [['default', load_graph], ['diagnostic', load_graph_for_scope_validation]] as const) {
  test(`${label} ONNX execution keeps canonical outputs across calls`, () => {
    const model = load(directory, device)
    const firstInput = inputs(), secondInput = inputs()
    const first = model.forward(firstInput), second = model.forward(secondInput)
    for (const outputs of [first, second]) for (const [name, value] of Object.entries(reference.expected)) {
      const expected = (value as number[]).flat(Infinity) as number[]
      const actual = (outputs[name].to_array() as number[]).flat(Infinity) as number[]
      expect(actual.length).toBe(expected.length)
      expect(actual.every((entry, index) => Number.isFinite(entry) && Math.abs(entry - expected[index]) <= 1e-5 + 1e-5 * Math.abs(expected[index]))).toBe(true)
    }
    firstInput.x.dispose(); secondInput.x.dispose()
    Object.values(first).forEach(value => value.dispose()); Object.values(second).forEach(value => value.dispose())
    model.dispose()
  })

  test(`${label} profiling reports every evaluated graph node and propagates callback failures`, () => {
    const model = load(directory, device)
    const input = inputs(); let visits = 0
    const result = model.forward(input, () => { visits++ })
    expect(visits).toBe(model.graph.nodes.length)
    Object.values(result).forEach(value => value.dispose())
    const failure = Error('profile callback failure')
    try { model.forward(input, () => { throw failure }); throw Error('callback did not throw') }
    catch (error) { expect(error).toBe(failure) }
    expect(model.forward(input).result.shape).toEqual([1, 2])
    input.x.dispose(); model.dispose()
  })
}
