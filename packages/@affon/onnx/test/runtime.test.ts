import { test, expect, beforeAll, afterAll } from 'std:test'
import fs from 'std:fs'
import { getEnv, run } from 'std:process'
import { Session, type Device, type Tensor } from 'affon:compute'
import { load_graph, semantic_loss_report } from '../src/index.ts'

const directory = 'packages/@affon/onnx/test/fixtures'
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
let temporary = ''
beforeAll(async () => { temporary = (await run({ cmd: 'mktemp', args: ['-d', '/tmp/affon-onnx-test.XXXXXX'] })).stdout.trim() })
afterAll(async () => { if (temporary) await run({ cmd: 'rm', args: ['-rf', temporary] }) })

function runtime(model: ReturnType<typeof load_graph>) {
  const session = new Session({ device })
  const state = session.initialize(model.forward, { parameters: model.parameters })
  const executable = session.compile(model.forward)
  return {
    forward(values: Record<string, unknown>, dtype: 'f32' | 'f64' = 'f32') {
      const inputs = Object.fromEntries(Object.entries(values).map(([name, value]) => [name, session.tensor(value as any, { dtype })])) as Record<string, Tensor>
      try {
        const result = executable.run(inputs, state) as Tensor | Tensor[]
        const outputs = Array.isArray(result) ? result : [result]
        return Object.fromEntries(model.output_names.map((name, index) => [name, outputs[index]])) as Record<string, Tensor>
      } finally { for (const input of Object.values(inputs)) input.dispose() }
    },
    dispose() { state.dispose(); session.dispose() },
  }
}

function compare(actual: Tensor, expected: unknown) {
  const values = (actual.to_array() as number[]).flat(Infinity) as number[]
  const references = (expected as number[]).flat(Infinity) as number[]
  expect(values.length).toBe(references.length)
  expect(values.every((value, index) => Number.isFinite(value) && Math.abs(value - references[index]) <= 1e-5 + 1e-5 * Math.abs(references[index]))).toBe(true)
}

test('import reports preserved, inferred, decomposed, unsupported and source-export-lost meaning', () => {
  const report = semantic_loss_report(directory)
  expect(report.format).toBe('affon-program-import-semantics/v1')
  expect(report.unsupported.length).toBe(0)
  expect(report.source_export_lost.length > 0).toBe(true)
})

test('imports the generic graph as one authored Program with explicit parameters', () => {
  const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))
  const model = load_graph(directory)
  expect(model.forward.inspect().kind).toBe('authored')
  expect(model.forward.inspect().arguments.map(value => value.name)).toEqual(Object.keys(model.graph.inputs))
  expect(model.forward.inspect().parameters.length).toBe(Object.keys(model.graph.constants).length)
  const execution = runtime(model)
  for (let repeat = 0; repeat < 2; repeat++) {
    const outputs = execution.forward({ x: reference.input })
    for (const [name, expected] of Object.entries(reference.expected)) { compare(outputs[name], expected); outputs[name].dispose() }
  }
  execution.dispose()
})

test('Program input contracts reject wrong shape, dtype and names', () => {
  const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))
  const model = load_graph(directory), execution = runtime(model)
  expect(() => execution.forward({ x: [1, 2] })).toThrow()
  expect(() => execution.forward({ x: reference.input }, 'f64')).toThrow()
  expect(() => execution.forward({})).toThrow()
  expect(() => execution.forward({ other: reference.input })).toThrow()
  execution.dispose()
})

test('unsupported operations and missing dependencies fail before loading weights', () => {
  const manifest = JSON.parse(fs.readFileSync(`${directory}/graph.json`))
  manifest.nodes[0].op = 'Unsupported'
  fs.writeFileSync(`${temporary}/graph.json`, JSON.stringify(manifest))
  expect(() => load_graph(temporary)).toThrow()
  manifest.nodes[0].op = 'Conv'; manifest.nodes[0].inputs[0] = 'missing'
  fs.writeFileSync(`${temporary}/graph.json`, JSON.stringify(manifest))
  expect(() => load_graph(temporary)).toThrow()
})

for (const fixture of ['spatial', 'conv1d']) test(`canonical Program lowering matches ONNX Runtime for ${fixture}`, () => {
  const root = `${directory}/${fixture}`
  const reference = JSON.parse(fs.readFileSync(`${root}/reference.json`))
  const model = load_graph(root), execution = runtime(model)
  const outputs = execution.forward({ x: reference.input })
  for (const [name, expected] of Object.entries(reference.expected)) { compare(outputs[name], expected); outputs[name].dispose() }
  execution.dispose()
})
