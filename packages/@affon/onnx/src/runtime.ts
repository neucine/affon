// Experimental executor for convert.py's bounded static-f32 ONNX manifest.
// No HF configuration, architecture dispatch, or model-specific weight names.
import { capabilities } from './capabilities.ts'
import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import * as compute from 'affon:compute'
import native from 'affon:compute/native'
import {
  abs,
  add,
  cat,
  clamp,
  compile,
  mean,
  contiguous,
  div,
  exp,
  layer_norm,
  matmul,
  mul,
  neg,
  no_grad,
  permute,
  reshape,
  sign,
  softmax,
  sub,
  tensor,
  transpose,
} from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'

import { prepare_conv, prepare_pad } from './spatial.ts'

type Node = {
  op: string
  name: string
  inputs: string[]
  output: string
  shape: number[]
  attrs: Record<string, any>
}
type Manifest = {
  format: string
  opset: number
  inputs: Record<string, number[]>
  outputs: string[]
  constants: Record<string, number[]>
  nodes: Node[]
}
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
const same = (a: number[], b: number[]) =>
  JSON.stringify(a) === JSON.stringify(b)

export function load_graph(directory: string, device: Device = 'cpu') {
  const { graph, forward } = create_graph(directory, device, true)
  return { graph, forward, semanticLoss: classify_import_semantics(graph) }
}

export function semantic_loss_report(directory: string): SemanticLossReport {
  const graph = JSON.parse(fs.readFileSync(`${directory}/graph.json`)) as Manifest
  if (graph.format !== 'affon-onnx-static/v1' || graph.opset !== 17 || !Array.isArray(graph.nodes))
    throw Error('Invalid graph manifest')
  return classify_import_semantics(graph)
}

function classify_import_semantics(graph: Manifest): SemanticLossReport {
  const report: SemanticLossReport = {
    format: 'affon-program-import-semantics/v1',
    preserved: [], inferred: [], decomposed: [], unsupported: [], source_export_lost: [],
  }
  const preserved = new Set(['Add', 'Mul', 'Div', 'MatMul', 'Concat', 'Reshape', 'Transpose', 'Softmax', 'Erf', 'Clip'])
  const inferred = new Set(['Identity', 'Cast', 'Flatten'])
  const decomposed = new Set(['Gemm', 'LayerNormalization', 'Gather', 'Slice', 'Conv', 'Pad', 'GlobalAveragePool'])
  for (const node of graph.nodes) {
    const entry = { node: node.name, op: node.op, reason: '' }
    if (preserved.has(node.op)) {
      entry.reason = 'operation and attributes retain their imported mathematical meaning'
      report.preserved.push(entry)
    } else if (inferred.has(node.op)) {
      entry.reason = 'result dtype or dimensions are re-established by Program abstract evaluation'
      report.inferred.push(entry)
    } else if (decomposed.has(node.op)) {
      entry.reason = 'importer lowers the source operation to ordinary Program operations'
      report.decomposed.push(entry)
    } else {
      entry.reason = 'no candidate Program lowering is declared'
      report.unsupported.push(entry)
    }
  }
  report.source_export_lost.push({ node: '<source-export>', op: 'ConstantSubgraph', reason: 'offline conversion folds source constant subgraphs, so their original node topology is unavailable to the runtime importer' })
  return report
}

// Internal validation entry; deliberately not exported by the package index.
// The switch preserves an ordinary-execution oracle without a public policy knob.
export function load_graph_for_scope_validation(
  directory: string,
  device: Device = 'cpu',
  scoped = true,
) {
  return create_graph(directory, device, scoped)
}

function create_graph(directory: string, device: Device, scoped: boolean) {
  let scope_stats: Omit<
    ReturnType<typeof native.$with_graph_execution>,
    'value'
  > | null = null
  const graph = JSON.parse(
    fs.readFileSync(`${directory}/graph.json`),
  ) as Manifest
  if (
    graph.format !== 'affon-onnx-static/v1' ||
    graph.opset !== 17 ||
    !graph.nodes?.length
  )
    throw Error('Invalid graph manifest')
  const known = new Set([
    ...Object.keys(graph.inputs),
    ...Object.keys(graph.constants),
  ])
  const uses = new Map<string, number>()
  for (const node of graph.nodes) {
    if (!supported.has(node.op))
      throw Error(`Unsupported operation: ${node.op}`)
    if (known.has(node.output) || node.inputs.some((x) => !known.has(x)))
      throw Error(`Invalid graph ordering: ${node.name}`)
    node.inputs.forEach((x) => uses.set(x, (uses.get(x) ?? 0) + 1))
    known.add(node.output)
  }
  if (graph.outputs.some((x) => !known.has(x)))
    throw Error('Missing graph output')
  graph.outputs.forEach((x) => uses.set(x, (uses.get(x) ?? 0) + 1))
  const loaded = checkpoint.load(`${directory}/weights.safetensors`) as Record<
    string,
    Tensor
  >
  const constants = new Map<string, Tensor>()
  for (const [name, shape] of Object.entries(graph.constants)) {
    const value = loaded[name]
    if (!value || value.dtype !== 'f32' || !same(value.shape, shape))
      throw Error(`Invalid constant: ${name}`)
    constants.set(name, value.device === device ? value : value.to(device))
    delete loaded[name]
  }
  if (Object.keys(loaded).length) throw Error('Unexpected graph weights')
  const spatial = new Map<string, (x: Tensor) => Tensor>()
  const shapes = new Map(
    Object.entries({ ...graph.inputs, ...graph.constants }),
  )
  // Fuse single-consumer constant spatial padding into CPU depthwise kernels.
  // The serialized graph stays intact; only the prepared execution schedule changes.
  const padByConv = new Map<string, { input: string; pads: number[] }>(),
    skippedPads = new Set<string>()
  if (device === 'cpu')
    graph.nodes.forEach((pad, index) => {
      const next = graph.nodes[index + 1]
      if (
        pad.op !== 'Pad' ||
        !next ||
        next.op !== 'Conv' ||
        next.inputs[0] !== pad.output ||
        uses.get(pad.output) !== 1 ||
        pad.shape.length !== 4
      )
        return
      const weight = constants.get(next.inputs[1])!
      if (
        weight.shape.length !== 4 ||
        weight.shape[1] !== 1 ||
        next.attrs.group !== weight.shape[0]
      )
        return
      const pads = pad.attrs.pads.map(
        (n: number, i: number) => n + (next.attrs.pads?.[i] ?? 0),
      )
      padByConv.set(next.output, { input: pad.inputs[0], pads })
      skippedPads.add(pad.output)
    })
  const clipByConv = new Map<string, number[]>(),
    fusedClips = new Set<string>()
  if (device === 'cpu')
    graph.nodes.forEach((node, index) => {
      const next = graph.nodes[index + 1],
        a = node.attrs
      if (
        node.op !== 'Conv' ||
        !next ||
        next.op !== 'Clip' ||
        next.inputs[0] !== node.output ||
        uses.get(node.output) !== 1 ||
        !same(node.shape, next.shape)
      )
        return
      const weight = constants.get(node.inputs[1])!
      if (weight.shape.length !== 4) return
      const pointwise =
        (a.group ?? 1) === 1 &&
        a.kernel.every((n: number) => n === 1) &&
        (a.strides ?? a.kernel).every((n: number) => n === 1) &&
        (a.pads ?? [0, 0, 0, 0]).every((n: number) => n === 0)
      const depthwise = weight.shape[1] === 1 && a.group === weight.shape[0]
      if (!pointwise && !depthwise && (a.group ?? 1) !== 1) return
      clipByConv.set(node.output, [next.attrs.min, next.attrs.max])
      fusedClips.add(next.output)
    })
  for (const node of graph.nodes) {
    if (node.op === 'Conv' && shapes.get(node.inputs[0])!.length === 3) {
      const shape = shapes.get(node.inputs[0])!,
        w = constants.get(node.inputs[1])!,
        a = node.attrs
      const run = prepare_conv(
        [shape[0], shape[1], 1, shape[2]],
        reshape(w, [w.shape[0], w.shape[1], 1, w.shape[2]]),
        constants.get(node.inputs[2]),
        {
          kernel: [1, a.kernel[0]],
          strides: [1, a.strides[0]],
          dilations: [1, a.dilations[0]],
          pads: [0, a.pads[0], 0, a.pads[1]],
          group: a.group,
        },
        device,
      )
      spatial.set(node.output, (x) =>
        reshape(run(reshape(x, [shape[0], shape[1], 1, shape[2]])), node.shape),
      )
    } else if (node.op === 'Conv')
      spatial.set(
        node.output,
        prepare_conv(
          shapes.get(padByConv.get(node.output)?.input ?? node.inputs[0])!,
          constants.get(node.inputs[1])!,
          constants.get(node.inputs[2]),
          (padByConv.has(node.output)
            ? { ...node.attrs, pads: padByConv.get(node.output)!.pads }
            : node.attrs) as any,
          device,
          clipByConv.get(node.output),
        ),
      )
    if (node.op === 'Pad' && !skippedPads.has(node.output))
      spatial.set(
        node.output,
        prepare_pad(shapes.get(node.inputs[0])!, node.attrs.pads, device),
      )
    shapes.set(node.output, node.shape)
  }
  // A&S fallback for CUDA and older runtimes without native erf.
  function erfFallback(x: Tensor) {
    const scalar = (n: number) => tensor(n, { dtype: 'f32', device })
    const t = div(scalar(1), add(scalar(1), mul(scalar(0.3275911), abs(x))))
    let p = add(mul(scalar(1.061405429), t), scalar(-1.453152027))
    for (const c of [1.421413741, -0.284496736, 0.254829592])
      p = add(mul(p, t), scalar(c))
    return mul(sign(x), sub(scalar(1), mul(mul(p, t), exp(neg(mul(x, x))))))
  }
  const evaluateErf =
    device !== 'cuda' && typeof compute.erf === 'function'
      ? compute.erf
      : erfFallback
  // Use the measured Metal affine path automatically. The compute graph owns
  // eligibility and falls back when layout or broadcast constraints do not match.
  const affine = compile((value: Tensor, scale: Tensor, bias: Tensor) =>
    add(mul(value, scale), bias),
  )
  const useAffine = device === 'metal'
  function forward(
    inputs: Record<string, Tensor>,
    profile?: (event: {
      name: string
      op: string
      shape: number[]
      elapsed_ms: number
    }) => void,
  ) {
    scope_stats = null
    if (Object.keys(inputs).length !== Object.keys(graph.inputs).length)
      throw Error('Incorrect graph inputs')
    const execute = () => {
      const values = new Map(constants),
        remaining = new Map(uses)
      for (const [name, shape] of Object.entries(graph.inputs)) {
        const x = inputs[name]
        if (!x || x.dtype !== 'f32' || !same(x.shape, shape))
          throw Error(`Expected f32 input ${name} ${shape}`)
        values.set(name, x.device === device ? x : x.to(device))
      }
      for (const node of graph.nodes) {
        if (skippedPads.has(node.output)) continue
        const started = profile ? Date.now() : 0
        const nodeInputs = padByConv.has(node.output)
          ? [padByConv.get(node.output)!.input, ...node.inputs.slice(1)]
          : node.inputs
        const args = nodeInputs.map((name) => values.get(name)!)
        const [x, y, z] = args,
          a = node.attrs
        let result: Tensor
        switch (node.op) {
          case 'Identity':
            result = x
            break
          case 'Cast':
            if (a.to !== 1) throw Error('Only f32 Cast supported')
            result = x
            break
          case 'Add':
            result = add(x, y)
            break
          case 'Mul':
            result = mul(x, y)
            break
          case 'Div':
            result = div(x, y)
            break
          case 'MatMul':
            result = matmul(x, y)
            break
          case 'Concat':
            result = cat(args, a.axis)
            break
          case 'Flatten':
          case 'Reshape':
            result = reshape(contiguous(x), a.shape)
            break
          case 'Transpose':
            result = permute(
              x,
              a.perm ??
                Array.from({ length: x.ndim }, (_, i) => x.ndim - 1 - i),
            )
            break
          case 'Softmax':
            result = softmax(x, a.axis)
            break
          case 'Erf':
            result = evaluateErf(x)
            break
          case 'LayerNormalization':
            if (
              a.axis !== x.ndim - 1 ||
              (a.stash_type ?? 1) !== 1 ||
              args.length !== 3
            )
              throw Error('Unsupported normalization contract')
            const normalized = layer_norm(x, a.axis, a.epsilon ?? 1e-5)
            result = useAffine
              ? affine(normalized, y, z)
              : add(mul(normalized, y), z)
            break
          case 'Gather': {
            const ranges: (string | number)[] = Array.from(
              { length: x.ndim },
              () => ':',
            )
            ranges[a.axis] = a.index
            result = reshape(contiguous(x.slice(ranges)), node.shape)
            break
          }
          case 'Gemm': {
            const product = matmul(
              a.transA ? transpose(x, 0, 1) : x,
              a.transB ? transpose(y, 0, 1) : y,
            )
            const scale = (v: Tensor, n: number) =>
              n === 1 ? v : mul(v, tensor(n, { dtype: 'f32', device }))
            result = scale(product, a.alpha ?? 1)
            if (z) result = add(result, scale(z, a.beta ?? 1))
            break
          }
          case 'Slice':
            result = contiguous(x.slice(a.selectors))
            break
          case 'Conv':
          case 'Pad':
            result = spatial.get(node.output)!(x)
            break
          case 'Clip':
            result = fusedClips.has(node.output) ? x : clamp(x, a.min, a.max)
            break
          case 'GlobalAveragePool':
            result = reshape(
              mean(
                reshape(contiguous(x), [
                  x.shape[0],
                  x.shape[1],
                  x.shape[2] * x.shape[3],
                ]),
                2,
                true,
              ),
              node.shape,
            )
            break
          default:
            throw Error(`Unsupported operation: ${node.op}`)
        }
        profile?.({
          name: node.name,
          op: node.op,
          shape: node.shape,
          elapsed_ms: Date.now() - started,
        })
        if (!same(result.shape, node.shape))
          throw Error(
            `Output shape mismatch: ${node.name}: ${result.shape} != ${node.shape}`,
          )
        values.set(node.output, result)
        for (const name of nodeInputs) {
          const count = remaining.get(name)! - 1
          remaining.set(name, count)
          if (count === 0) values.delete(name)
        }
      }
      return Object.fromEntries(
        graph.outputs.map((name) => [name, values.get(name)!]),
      )
    }
    return no_grad(() => {
      // A profiling callback is arbitrary user code and retains synchronous
      // elapsed-time semantics; it must never run inside the native scope.
      if (!scoped || device !== 'metal' || profile) return execute()
      const { value, ...stats } = native.$with_graph_execution(execute)
      scope_stats = stats
      return value
    })
  }
  return { graph, forward, scope_stats: () => scope_stats }
}
