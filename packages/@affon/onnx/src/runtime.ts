// Experimental executor for convert.py's bounded static-f32 ONNX manifest.
// It uses canonical session tensors at the boundary and evaluates the imported
// static graph deterministically on host values.
import { capabilities } from './capabilities.ts'
import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { Session } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { convHost, hostTensor, nested, padHost, reshapeHost } from './spatial.ts'
import type { HostTensor } from './spatial.ts'

type Node = { op: string; name: string; inputs: string[]; output: string; shape: number[]; attrs: Record<string, any> }
type Manifest = { format: string; opset: number; inputs: Record<string, number[]>; outputs: string[]; constants: Record<string, number[]>; nodes: Node[] }
export type SemanticLossEntry = { node: string; op: string; reason: string }
export type SemanticLossReport = {
  format: 'affon-program-import-semantics/v1'
  preserved: SemanticLossEntry[]
  inferred: SemanticLossEntry[]
  decomposed: SemanticLossEntry[]
  unsupported: SemanticLossEntry[]
  source_export_lost: SemanticLossEntry[]
}

const supported = new Set<string>(capabilities.operators)
const same = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((value, index) => value === b[index])
const size = (shape: readonly number[]) => shape.reduce((product, value) => product * value, 1)
const strides = (shape: readonly number[]) => shape.map((_, index) => size(shape.slice(index + 1)))

function manifest(directory: string): Manifest {
  const graph = JSON.parse(fs.readFileSync(`${directory}/graph.json`)) as Manifest
  if (graph.format !== 'affon-onnx-static/v1' || graph.opset !== 17 || !Array.isArray(graph.nodes)) throw Error('Invalid graph manifest')
  return graph
}

export function semantic_loss_report(directory: string): SemanticLossReport { return classify_import_semantics(manifest(directory)) }

function classify_import_semantics(graph: Manifest): SemanticLossReport {
  const report: SemanticLossReport = { format: 'affon-program-import-semantics/v1', preserved: [], inferred: [], decomposed: [], unsupported: [], source_export_lost: [] }
  const preserved = new Set(['Add', 'Mul', 'Div', 'MatMul', 'Concat', 'Reshape', 'Transpose', 'Softmax', 'Erf', 'Clip'])
  const inferred = new Set(['Identity', 'Cast', 'Flatten'])
  const decomposed = new Set(['Gemm', 'LayerNormalization', 'Gather', 'Slice', 'Conv', 'Pad', 'GlobalAveragePool'])
  for (const node of graph.nodes) {
    const entry = { node: node.name, op: node.op, reason: '' }
    if (preserved.has(node.op)) { entry.reason = 'operation and attributes retain their imported mathematical meaning'; report.preserved.push(entry) }
    else if (inferred.has(node.op)) { entry.reason = 'result dtype or dimensions are re-established by static evaluation'; report.inferred.push(entry) }
    else if (decomposed.has(node.op)) { entry.reason = 'importer lowers the source operation to canonical tensor semantics'; report.decomposed.push(entry) }
    else { entry.reason = 'no candidate lowering is declared'; report.unsupported.push(entry) }
  }
  report.source_export_lost.push({ node: '<source-export>', op: 'ConstantSubgraph', reason: 'offline conversion folds source constant subgraphs, so their original node topology is unavailable to the runtime importer' })
  return report
}

function coordinates(linear: number, shape: readonly number[]): number[] {
  const result: number[] = []
  for (const stride of strides(shape)) { result.push(Math.floor(linear / stride)); linear %= stride }
  return result
}

function offset(indices: readonly number[], shape: readonly number[]): number {
  const steps = strides(shape)
  return indices.reduce((value, coordinate, axis) => value + coordinate * steps[axis], 0)
}

function broadcastShape(a: readonly number[], b: readonly number[]): number[] {
  const rank = Math.max(a.length, b.length), result = Array(rank).fill(1)
  for (let index = 0; index < rank; index++) {
    const av = a[a.length - rank + index] ?? 1, bv = b[b.length - rank + index] ?? 1
    if (av !== bv && av !== 1 && bv !== 1) throw Error('Invalid broadcast dimensions')
    result[index] = Math.max(av, bv)
  }
  return result
}

function project(indices: readonly number[], sourceShape: readonly number[]): number {
  const start = indices.length - sourceShape.length
  return offset(sourceShape.map((dimension, axis) => dimension === 1 ? 0 : indices[start + axis]), sourceShape)
}

function binary(a: HostTensor, b: HostTensor, operation: (left: number, right: number) => number): HostTensor {
  const shape = broadcastShape(a.shape, b.shape)
  const values = Array.from({ length: size(shape) }, (_, linear) => {
    const index = coordinates(linear, shape)
    return operation(a.values[project(index, a.shape)], b.values[project(index, b.shape)])
  })
  return { shape, values }
}

function transpose(input: HostTensor, permutation: number[]): HostTensor {
  const shape = permutation.map(axis => input.shape[axis])
  return { shape, values: Array.from({ length: input.values.length }, (_, linear) => {
    const target = coordinates(linear, shape), source = Array(input.shape.length)
    permutation.forEach((axis, index) => { source[axis] = target[index] })
    return input.values[offset(source, input.shape)]
  }) }
}

function matmul(a: HostTensor, b: HostTensor): HostTensor {
  if (a.shape.length < 2 || b.shape.length < 2) throw Error('MatMul requires rank at least two')
  const m = a.shape.at(-2)!, k = a.shape.at(-1)!, bk = b.shape.at(-2)!, n = b.shape.at(-1)!
  if (k !== bk) throw Error('MatMul contraction mismatch')
  const batchShape = broadcastShape(a.shape.slice(0, -2), b.shape.slice(0, -2))
  const shape = [...batchShape, m, n]
  const values = Array(size(shape)).fill(0)
  for (let linear = 0; linear < values.length; linear++) {
    const target = coordinates(linear, shape), batch = target.slice(0, -2), row = target.at(-2)!, column = target.at(-1)!
    let value = 0
    for (let inner = 0; inner < k; inner++) {
      const ai = [...batch.slice(batch.length - (a.shape.length - 2)), row, inner]
      const bi = [...batch.slice(batch.length - (b.shape.length - 2)), inner, column]
      value += a.values[project(ai, a.shape)] * b.values[project(bi, b.shape)]
    }
    values[linear] = value
  }
  return { shape, values }
}

function reduce(input: HostTensor, axis: number, keep: boolean, average: boolean): HostTensor {
  if (axis < 0) axis += input.shape.length
  const shape = keep ? input.shape.map((value, index) => index === axis ? 1 : value) : input.shape.filter((_, index) => index !== axis)
  const values = Array(size(shape)).fill(0)
  for (let linear = 0; linear < input.values.length; linear++) {
    const index = coordinates(linear, input.shape), target = keep ? index.map((value, i) => i === axis ? 0 : value) : index.filter((_, i) => i !== axis)
    values[offset(target, shape)] += input.values[linear]
  }
  if (average) for (let index = 0; index < values.length; index++) values[index] /= input.shape[axis]
  return { shape, values }
}

function softmax(input: HostTensor, axis: number): HostTensor {
  if (axis < 0) axis += input.shape.length
  const outer = size(input.shape.slice(0, axis)), width = input.shape[axis], inner = size(input.shape.slice(axis + 1)), values = Array(input.values.length)
  for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
    let maximum = -Infinity
    for (let x = 0; x < width; x++) maximum = Math.max(maximum, input.values[(o * width + x) * inner + i])
    let total = 0
    for (let x = 0; x < width; x++) total += Math.exp(input.values[(o * width + x) * inner + i] - maximum)
    for (let x = 0; x < width; x++) values[(o * width + x) * inner + i] = Math.exp(input.values[(o * width + x) * inner + i] - maximum) / total
  }
  return { shape: [...input.shape], values }
}

function erf(value: number) {
  const sign = value < 0 ? -1 : 1, x = Math.abs(value), t = 1 / (1 + 0.3275911 * x)
  const polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
  return sign * (1 - polynomial * Math.exp(-x * x))
}

function layerNorm(input: HostTensor, axis: number, epsilon: number, scale: HostTensor, bias: HostTensor): HostTensor {
  if (axis < 0) axis += input.shape.length
  const width = size(input.shape.slice(axis)), groups = input.values.length / width, values = Array(input.values.length)
  for (let group = 0; group < groups; group++) {
    const start = group * width
    let mean = 0
    for (let index = 0; index < width; index++) mean += input.values[start + index]
    mean /= width
    let variance = 0
    for (let index = 0; index < width; index++) variance += (input.values[start + index] - mean) ** 2
    variance /= width
    for (let index = 0; index < width; index++) values[start + index] = (input.values[start + index] - mean) / Math.sqrt(variance + epsilon) * scale.values[index % scale.values.length] + bias.values[index % bias.values.length]
  }
  return { shape: [...input.shape], values }
}

function concat(inputs: HostTensor[], axis: number): HostTensor {
  if (axis < 0) axis += inputs[0].shape.length
  const shape = [...inputs[0].shape]; shape[axis] = inputs.reduce((total, input) => total + input.shape[axis], 0)
  const values = Array(size(shape)), targetStrides = strides(shape)
  let axisOffset = 0
  for (const input of inputs) {
    for (let linear = 0; linear < input.values.length; linear++) {
      const index = coordinates(linear, input.shape); index[axis] += axisOffset
      values[index.reduce((total, value, i) => total + value * targetStrides[i], 0)] = input.values[linear]
    }
    axisOffset += input.shape[axis]
  }
  return { shape, values }
}

function sliced(input: HostTensor, selectors: Array<string | number>): HostTensor {
  const choices = selectors.map((selector, axis) => {
    if (typeof selector === 'number') return [selector < 0 ? input.shape[axis] + selector : selector]
    if (selector === ':') return Array.from({ length: input.shape[axis] }, (_, index) => index)
    const [rawStart, rawStop, rawStep] = selector.split(':')
    const step = rawStep ? Number(rawStep) : 1
    let start = rawStart ? Number(rawStart) : 0, stop = rawStop ? Number(rawStop) : input.shape[axis]
    if (start < 0) start += input.shape[axis]; if (stop < 0) stop += input.shape[axis]
    const values: number[] = []; for (let value = start; value < stop; value += step) values.push(value)
    return values
  })
  const shape = choices.map((values, axis) => typeof selectors[axis] === 'number' ? 0 : values.length).filter(Boolean)
  const values: number[] = []
  const visit = (axis: number, indices: number[]) => {
    if (axis === choices.length) { values.push(input.values[offset(indices, input.shape)]); return }
    for (const value of choices[axis]) visit(axis + 1, [...indices, value])
  }
  visit(0, [])
  return { shape, values }
}

function validateGraph(graph: Manifest) {
  if (!graph.nodes.length) throw Error('Invalid graph manifest')
  const known = new Set([...Object.keys(graph.inputs), ...Object.keys(graph.constants)])
  for (const node of graph.nodes) {
    if (!supported.has(node.op)) throw Error(`Unsupported operation: ${node.op}`)
    if (known.has(node.output) || node.inputs.some(input => !known.has(input))) throw Error(`Invalid graph ordering: ${node.name}`)
    known.add(node.output)
  }
  if (graph.outputs.some(output => !known.has(output))) throw Error('Missing graph output')
}

function create_graph(directory: string, device: Device, diagnostic: boolean) {
  const graph = manifest(directory)
  validateGraph(graph)
  const loaded = checkpoint.load(`${directory}/weights.safetensors`)
  const constants = new Map<string, HostTensor>()
  try {
    for (const [name, shape] of Object.entries(graph.constants)) {
      const value = loaded[name]
      if (!value || value.dtype !== 'f32' || !same(value.shape, shape)) throw Error(`Invalid constant: ${name}`)
      constants.set(name, hostTensor(value)); delete loaded[name]; value.dispose()
    }
    if (Object.keys(loaded).length) throw Error('Unexpected graph weights')
  } finally { for (const value of Object.values(loaded)) value.dispose() }
  const session = new Session({ device })

  function evaluate(node: Node, args: HostTensor[]): HostTensor {
    const [x, y, z] = args, a = node.attrs
    switch (node.op) {
      case 'Identity': case 'Cast': return { shape: [...x.shape], values: [...x.values] }
      case 'Add': return binary(x, y, (left, right) => left + right)
      case 'Mul': return binary(x, y, (left, right) => left * right)
      case 'Div': return binary(x, y, (left, right) => left / right)
      case 'MatMul': return matmul(x, y)
      case 'Concat': return concat(args, a.axis)
      case 'Flatten': case 'Reshape': return reshapeHost(x, a.shape)
      case 'Transpose': return transpose(x, a.perm ?? [...x.shape.keys()].reverse())
      case 'Softmax': return softmax(x, a.axis)
      case 'Erf': return { shape: [...x.shape], values: x.values.map(erf) }
      case 'LayerNormalization': return layerNorm(x, a.axis, a.epsilon ?? 1e-5, y, z)
      case 'Gather': {
        const ranges = x.shape.map((dimension, axis) => axis === a.axis ? a.index : ':')
        return reshapeHost(sliced(x, ranges), node.shape)
      }
      case 'Gemm': {
        const left = a.transA ? transpose(x, [1, 0]) : x, right = a.transB ? transpose(y, [1, 0]) : y
        let result = matmul(left, right)
        if ((a.alpha ?? 1) !== 1) result = { shape: result.shape, values: result.values.map(value => value * a.alpha) }
        if (z) result = binary(result, { shape: z.shape, values: z.values.map(value => value * (a.beta ?? 1)) }, (leftValue, rightValue) => leftValue + rightValue)
        return result
      }
      case 'Slice': return reshapeHost(sliced(x, a.selectors), node.shape)
      case 'Pad': return padHost(x, a.pads)
      case 'Conv': {
        if (x.shape.length === 4) return convHost(x, y, z, a as any)
        const expanded = reshapeHost(x, [x.shape[0], x.shape[1], 1, x.shape[2]])
        const weights = reshapeHost(y, [y.shape[0], y.shape[1], 1, y.shape[2]])
        return reshapeHost(convHost(expanded, weights, z, { kernel: [1, a.kernel[0]], strides: [1, a.strides[0]], dilations: [1, a.dilations[0]], pads: [0, a.pads[0], 0, a.pads[1]], group: a.group }), node.shape)
      }
      case 'Clip': return { shape: [...x.shape], values: x.values.map(value => Math.min(Math.max(value, a.min), a.max)) }
      case 'GlobalAveragePool': return reshapeHost(reduce(reduce(x, 3, true, true), 2, true, true), node.shape)
      default: throw Error(`Unsupported operation: ${node.op}`)
    }
  }

  function forward(inputs: Record<string, Tensor>, profile?: (event: { name: string; op: string; shape: number[]; elapsed_ms: number }) => void) {
    if (Object.keys(inputs).length !== Object.keys(graph.inputs).length) throw Error('Incorrect graph inputs')
    const values = new Map(constants)
    for (const [name, shape] of Object.entries(graph.inputs)) {
      const value = inputs[name]
      if (!value || value.dtype !== 'f32' || !same(value.shape, shape)) throw Error(`Expected f32 input ${name} ${shape}`)
      values.set(name, hostTensor(value))
    }
    for (const node of graph.nodes) {
      const started = profile ? Date.now() : 0
      const result = evaluate(node, node.inputs.map(name => values.get(name)!))
      if (!same(result.shape, node.shape)) throw Error(`Output shape mismatch: ${node.name}: ${result.shape} != ${node.shape}`)
      values.set(node.output, result)
      profile?.({ name: node.name, op: node.op, shape: node.shape, elapsed_ms: Date.now() - started })
    }
    return Object.fromEntries(graph.outputs.map(name => {
      const value = values.get(name)!
      return [name, session.tensor(nested(value.values, value.shape)) as Tensor]
    }))
  }
  function dispose() { session.dispose() }
  return { graph, forward, dispose, ...(diagnostic ? { scope_stats: () => null } : {}) }
}

export function load_graph(directory: string, device: Device = 'cpu') {
  const model = create_graph(directory, device, false)
  return { ...model, semanticLoss: classify_import_semantics(model.graph) }
}

export function load_graph_for_scope_validation(directory: string, device: Device = 'cpu', _scoped = true) {
  return create_graph(directory, device, true)
}
