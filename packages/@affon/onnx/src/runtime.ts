import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { Tensor, program, type FormalTensor } from 'affon:compute'
import { add, cat, contiguous, div, erf, layer_norm, matmul, mean, mul, relu, reshape, slice, softmax, sub, transpose } from 'affon:ops'
import { capabilities } from './capabilities.ts'

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
const product = (shape: readonly number[]) => shape.reduce((value, dimension) => value * dimension, 1)
const zeros = (shape: readonly number[]): any => shape.length ? Array.from({ length: shape[0] }, () => zeros(shape.slice(1))) : 0

function read_manifest(directory: string): Manifest {
  const graph = JSON.parse(fs.readFileSync(`${directory}/graph.json`)) as Manifest
  if (graph.format !== 'affon-onnx-static/v1' || graph.opset !== 17 || !Array.isArray(graph.nodes)) throw Error('Invalid graph manifest')
  return graph
}

function validate_graph(graph: Manifest) {
  if (!graph.nodes.length) throw Error('Invalid graph manifest')
  const known = new Set([...Object.keys(graph.inputs), ...Object.keys(graph.constants)])
  for (const node of graph.nodes) {
    if (!supported.has(node.op)) throw Error(`Unsupported operation: ${node.op}`)
    if (known.has(node.output) || node.inputs.some(input => !known.has(input))) throw Error(`Invalid graph ordering: ${node.name}`)
    known.add(node.output)
  }
  if (graph.outputs.some(output => !known.has(output))) throw Error('Missing graph output')
}

function classify_import_semantics(graph: Manifest): SemanticLossReport {
  const report: SemanticLossReport = { format: 'affon-program-import-semantics/v1', preserved: [], inferred: [], decomposed: [], unsupported: [], source_export_lost: [] }
  const preserved = new Set(['Add', 'Mul', 'Div', 'MatMul', 'Concat', 'Reshape', 'Transpose', 'Softmax', 'Erf', 'Clip'])
  const inferred = new Set(['Identity', 'Cast', 'Flatten'])
  const decomposed = new Set(['Gemm', 'LayerNormalization', 'Gather', 'Slice', 'Conv', 'Pad', 'GlobalAveragePool'])
  for (const node of graph.nodes) {
    const entry = { node: node.name, op: node.op, reason: '' }
    if (preserved.has(node.op)) { entry.reason = 'operation and attributes retain their imported mathematical meaning'; report.preserved.push(entry) }
    else if (inferred.has(node.op)) { entry.reason = 'result dtype or dimensions are re-established by the authored Program'; report.inferred.push(entry) }
    else if (decomposed.has(node.op)) { entry.reason = 'importer lowers the source operation to canonical Program operations'; report.decomposed.push(entry) }
    else { entry.reason = 'no candidate lowering is declared'; report.unsupported.push(entry) }
  }
  report.source_export_lost.push({ node: '<source-export>', op: 'ConstantSubgraph', reason: 'offline conversion folds source constant subgraphs, so their original node topology is unavailable to the runtime importer' })
  return report
}

export function semantic_loss_report(directory: string): SemanticLossReport { return classify_import_semantics(read_manifest(directory)) }

function selectors(value: FormalTensor, node: Node) {
  const ranges = (node.attrs.selectors as Array<string | number>).map((selector, axis) => {
    if (typeof selector === 'number') {
      const index = selector < 0 ? value.spec.shape[axis] + selector : selector
      return { start: index, stop: index + 1 }
    }
    const [rawStart, rawStop, rawStep] = selector.split(':')
    let start = rawStart ? Number(rawStart) : 0
    let stop = rawStop ? Number(rawStop) : value.spec.shape[axis]
    if (start < 0) start += value.spec.shape[axis]
    if (stop < 0) stop += value.spec.shape[axis]
    return { start, stop, step: rawStep ? Number(rawStep) : 1 }
  })
  return reshape(contiguous(slice(value, ranges)), node.shape)
}

function zero_pad(p: any, node: Node, input: FormalTensor, pads: readonly number[]) {
  if (input.spec.shape.length !== 4 || pads.length !== 4) throw Error('Pad requires NCHW input and four spatial pads')
  const [batch, channels, height, width] = input.spec.shape
  const [top, left, bottom, right] = pads
  let result = input
  if (left || right) {
    const parts: FormalTensor[] = []
    if (left) {
      const shape = [batch, channels, height, left]
      parts.push(p.constant(`${node.output}_pad_left`, zeros(shape), Tensor.f32(shape)))
    }
    parts.push(result)
    if (right) {
      const shape = [batch, channels, height, right]
      parts.push(p.constant(`${node.output}_pad_right`, zeros(shape), Tensor.f32(shape)))
    }
    result = cat(parts, 3)
  }
  if (top || bottom) {
    const paddedWidth = width + left + right
    const parts: FormalTensor[] = []
    if (top) {
      const shape = [batch, channels, top, paddedWidth]
      parts.push(p.constant(`${node.output}_pad_top`, zeros(shape), Tensor.f32(shape)))
    }
    parts.push(result)
    if (bottom) {
      const shape = [batch, channels, bottom, paddedWidth]
      parts.push(p.constant(`${node.output}_pad_bottom`, zeros(shape), Tensor.f32(shape)))
    }
    result = cat(parts, 2)
  }
  return result
}

function convolution(p: any, node: Node, input: FormalTensor, weight: FormalTensor, bias?: FormalTensor) {
  const oneDimensional = input.spec.shape.length === 3
  let x = oneDimensional ? reshape(input, [input.spec.shape[0], input.spec.shape[1], 1, input.spec.shape[2]]) : input
  let w = oneDimensional ? reshape(weight, [weight.spec.shape[0], weight.spec.shape[1], 1, weight.spec.shape[2]]) : weight
  const attrs = node.attrs
  const kernel = oneDimensional ? [1, attrs.kernel[0]] : attrs.kernel
  const strides = oneDimensional ? [1, attrs.strides[0]] : attrs.strides
  const dilations = oneDimensional ? [1, attrs.dilations[0]] : attrs.dilations
  const pads = oneDimensional ? [0, attrs.pads[0], 0, attrs.pads[1]] : attrs.pads
  x = zero_pad(p, node, x, pads)
  const [batch, channels, height, width] = x.spec.shape
  const [outputs, channelsPerGroup, kh, kw] = w.spec.shape
  const groups = attrs.group ?? 1
  const outputsPerGroup = outputs / groups
  const oh = Math.floor((height - dilations[0] * (kh - 1) - 1) / strides[0]) + 1
  const ow = Math.floor((width - dilations[1] * (kw - 1) - 1) / strides[1]) + 1
  const groupOutputs: FormalTensor[] = []
  for (let group = 0; group < groups; group++) {
    let accumulated: FormalTensor | undefined
    for (let ky = 0; ky < kh; ky++) for (let kx = 0; kx < kw; kx++) {
      const patch = slice(x, [
        { start: 0, stop: batch },
        { start: group * channelsPerGroup, stop: (group + 1) * channelsPerGroup },
        { start: ky * dilations[0], stop: ky * dilations[0] + (oh - 1) * strides[0] + 1, step: strides[0] },
        { start: kx * dilations[1], stop: kx * dilations[1] + (ow - 1) * strides[1] + 1, step: strides[1] },
      ])
      const matrix = reshape(contiguous(transpose(patch, [0, 2, 3, 1])), [batch * oh * ow, channelsPerGroup])
      const kernelMatrix = reshape(contiguous(slice(w, [
        { start: group * outputsPerGroup, stop: (group + 1) * outputsPerGroup },
        { start: 0, stop: channelsPerGroup }, { start: ky, stop: ky + 1 }, { start: kx, stop: kx + 1 },
      ])), [outputsPerGroup, channelsPerGroup])
      const contribution = matmul(matrix, transpose(kernelMatrix, [1, 0]))
      accumulated = accumulated ? add(accumulated, contribution) : contribution
    }
    groupOutputs.push(accumulated!)
  }
  let result = transpose(reshape(groupOutputs.length === 1 ? groupOutputs[0] : cat(groupOutputs, 1), [batch, oh, ow, outputs]), [0, 3, 1, 2])
  if (bias) result = add(result, reshape(bias, [1, outputs, 1, 1]))
  return oneDimensional ? reshape(result, node.shape) : result
}

function lower(p: any, node: Node, args: FormalTensor[]) {
  const [x, y, z] = args, a = node.attrs
  const scalar = (suffix: string, value: number) => p.constant(`${node.output}_${suffix}`, value, Tensor.f32([1])) as FormalTensor
  switch (node.op) {
    case 'Identity': case 'Cast': return x
    case 'Add': return add(x, y)
    case 'Mul': return mul(x, y)
    case 'Div': return div(x, y)
    case 'MatMul': return matmul(x, y)
    case 'Concat': return cat(args, a.axis)
    case 'Flatten': case 'Reshape': return reshape(x, a.shape)
    case 'Transpose': return transpose(x, a.perm)
    case 'Softmax': return softmax(x, a.axis)
    case 'Erf': return erf(x)
    case 'LayerNormalization': return add(mul(layer_norm(x, a.axis, a.epsilon ?? 1e-5), y), z)
    case 'Gather': {
      const index = a.index < 0 ? x.spec.shape[a.axis] + a.index : a.index
      const ranges = x.spec.shape.map((dimension, axis) => ({ start: axis === a.axis ? index : 0, stop: axis === a.axis ? index + 1 : dimension }))
      return reshape(contiguous(slice(x, ranges)), node.shape)
    }
    case 'Gemm': {
      const left = a.transA ? transpose(x, [1, 0]) : x
      const right = a.transB ? transpose(y, [1, 0]) : y
      let result = matmul(left, right)
      if ((a.alpha ?? 1) !== 1) result = mul(result, scalar('alpha', a.alpha))
      if (z) result = add(result, (a.beta ?? 1) === 1 ? z : mul(z, scalar('beta', a.beta)))
      return result
    }
    case 'Slice': return selectors(x, node)
    case 'Pad': return zero_pad(p, node, x, a.pads)
    case 'Conv': return convolution(p, node, x, y, z)
    case 'Clip': {
      const minimum = scalar('clip_min', a.min), maximum = scalar('clip_max', a.max)
      return sub(maximum, relu(sub(maximum, add(relu(sub(x, minimum)), minimum))))
    }
    case 'GlobalAveragePool': return mean(mean(x, 3, true), 2, true)
    default: throw Error(`Unsupported operation: ${node.op}`)
  }
}

/** Import a prepared static ONNX manifest as one canonical authored Program. */
export function load_graph(directory: string) {
  const graph = read_manifest(directory)
  validate_graph(graph)
  const loaded = checkpoint.load(`${directory}/weights.safetensors`) as Record<string, import('affon:compute').Tensor>
  for (const [name, shape] of Object.entries(graph.constants)) {
    const value = loaded[name]
    if (!value || value.dtype !== 'f32' || !same(value.shape, shape)) throw Error(`Invalid constant: ${name}`)
  }
  if (Object.keys(loaded).some(name => !Object.hasOwn(graph.constants, name))) throw Error('Unexpected graph weights')
  const forward = program('onnx_import', p => {
    const values = new Map<string, FormalTensor>()
    for (const [name, shape] of Object.entries(graph.inputs)) values.set(name, p.argument(name, Tensor.f32(shape)))
    for (const [name, shape] of Object.entries(graph.constants)) values.set(name, p.parameter(name, Tensor.f32(shape)))
    for (const node of graph.nodes) values.set(node.output, lower(p, node, node.inputs.map(name => values.get(name)!)))
    return graph.outputs.map(name => values.get(name)!)
  })
  return {
    graph,
    forward,
    parameters: loaded,
    output_names: [...graph.outputs],
    semanticLoss: classify_import_semantics(graph),
  }
}
