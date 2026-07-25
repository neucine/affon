import { describe, expect, test } from 'std:test'
import nn from 'affon:nn'
import {
  abs,
  add,
  argmax,
  argmin,
  arange,
  clamp,
  clear_grad,
  clip_grad_norm,
  compile,
  cat,
  contiguous,
  copy,
  cross_entropy_indexed,
  dot,
  empty,
  finite_abs_max,
  finite_summary,
  gelu,
  gather,
  gt_scalar,
  max,
  masked_fill,
  min,
  index_select,
  no_grad,
  one_hot,
  permute,
  randn,
  seed,
  softmax,
  sign,
  squeeze,
  stack,
  std,
  sum,
  topk,
  unsqueeze,
  grad,
  matmul,
  mean,
  module,
  mul,
  parameter,
  sgd,
  square,
  sub,
  variance,
  tensor,
  where,
} from 'affon:compute'

describe('compute', () => {
  test('supports value creation, callable modules, grad, step, and compile', () => {
    setDevice('cpu')

    const linear = module(
      {
        w: parameter([1, 1]).ones(),
        b: parameter([1]).zeros(),
        config: { scale: 2 },
      },
      ({ w, b, config }, x) => mul(add(matmul(x, w), b), tensor(config.scale)),
    )

    const params = linear.parameters
    expect(params.length).toBe(2)

    const x = tensor([[1], [2], [3]])
    const target = tensor([[2], [4], [6]])

    clear_grad(params)
    const loss = mean(square(sub(linear(x), target)))
    grad(loss, params)
    expect(params[0].grad !== null).toBe(true)

    const step = sgd({ lr: 0.01 })
    expect(step.lr).toBe(0.01)
    step(params)

    const compiled = compile(linear)
    const compiledOut = compiled(x)
    expect(compiledOut.shape).toEqual([3, 1])
    expect(compiled.parameters.length).toBe(2)
    const captured = (compiled as any).capturedProgram?.()
    expect(typeof captured).toBe('object')
    expect(captured.inputArity).toBe(1)
    expect(captured.boundInputCount >= 0).toBe(true)
    expect(Array.isArray(captured.nodes)).toBe(true)

    const state = linear.state() as any
    expect(Array.isArray(state.w.shape)).toBe(true)
    expect(state.config.scale).toBe(2)

    const seq = arange(0, 4)
    expect(seq.to_array()).toEqual([0, 1, 2, 3])
  })

  test('module state root can be a parameter', () => {
    setDevice('cpu')

    const scale = module(
      parameter([1]).ones(),
      (w, x) => mul(x, w),
    )

    const x = tensor([2, 3, 4])
    const y = scale(x)

    expect(y.to_array()).toEqual([2, 3, 4])
    expect(scale.parameters.length).toBe(1)
    expect(Array.isArray((scale.state() as any).shape)).toBe(true)
  })

  test('promotes wave-1 math ops onto the compute surface', () => {
    setDevice('cpu')

    const x = tensor([-2, -1, 3, 4])
    expect(abs(x).to_array()).toEqual([2, 1, 3, 4])
    expect(sign(x).to_array()).toEqual([-1, -1, 1, 1])
    expect(clamp(x, -1, 3).to_array()).toEqual([-1, -1, 3, 3])

    const a = tensor([1, 2, 3])
    const b = tensor([4, 5, 6])
    expect(dot(a, b).item()).toBe(32)

    const grid = tensor([[1, 2], [3, 4]])
    expect(variance(grid).item()).toBe(1.25)
    expect(Math.abs(std(grid).item() - Math.sqrt(1.25)) < 1e-6).toBe(true)
    expect(min(grid, 1).to_array()).toEqual([1, 3])
    expect(max(grid, 0).to_array()).toEqual([3, 4])
    expect(argmin(grid, 1).to_array()).toEqual([0, 0])
    expect(argmax(grid, 0).to_array()).toEqual([1, 1])
  })

  test('grad handles scalar-like losses', () => {
    setDevice('cpu')

    const x = parameter([3]).ones()
    const y = tensor([4, 5, 6])
    const loss = sum(mul(x, y))

    expect(loss.shape).toEqual([1])

    grad(loss, [x])

    expect(x.grad?.shape).toEqual([3])
    expect(x.grad?.to_array()).toEqual([4, 5, 6])
  })

  test('promotes wave-2 shape and selection ops onto the compute surface', () => {
    setDevice('cpu')

    const rows = tensor([[1, 2], [3, 4]])
    const other = tensor([[5, 6]])
    expect(cat([rows, other], 0).to_array()).toEqual([[1, 2], [3, 4], [5, 6]])

    const v1 = tensor([1, 2])
    const v2 = tensor([3, 4])
    expect(stack([v1, v2], 0).to_array()).toEqual([[1, 2], [3, 4]])

    const expanded = unsqueeze(v1, 0)
    expect(expanded.shape).toEqual([1, 2])
    expect(squeeze(expanded).to_array()).toEqual([1, 2])

    const cond = tensor([[1, 0], [0, 1]])
    expect(where(cond, rows, tensor([[9, 9], [9, 9]])).to_array()).toEqual([[1, 9], [9, 4]])
    expect(masked_fill(rows, cond, -1).to_array()).toEqual([[-1, 2], [3, -1]])

    const probs = softmax(tensor([[1, 2, 3]]), 1).to_array() as number[][]
    expect(probs.length).toBe(1)
    expect(probs[0].length).toBe(3)
    const total = probs[0][0] + probs[0][1] + probs[0][2]
    expect(Math.abs(total - 1) < 1e-6).toBe(true)
  })

  test('promotes wave-3 indexed ops onto the compute surface', () => {
    setDevice('cpu')

    expect(one_hot(tensor([0, 2, 1]), 3).to_array()).toEqual([
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ])

    const source = tensor([[10, 20], [30, 40]])
    const gatherIndex = tensor([[1, 0], [0, 1]])
    expect(gather(source, 1, gatherIndex).to_array()).toEqual([
      [20, 10],
      [30, 40],
    ])

    const selectIndex = tensor([1, 0])
    expect(index_select(source, 0, selectIndex).to_array()).toEqual([
      [30, 40],
      [10, 20],
    ])

    const ranked = topk(tensor([[1, 4, 2], [3, 0, 5]]), 2, 1)
    expect(ranked.values.to_array()).toEqual([
      [4, 2],
      [5, 3],
    ])
    expect(ranked.indices.to_array()).toEqual([
      [1, 2],
      [2, 0],
    ])
  })

  test('promotes runtime and layout helpers onto the compute surface', () => {
    setDevice('cpu')

    seed(123)
    const a = randn([2, 2]).to_array()
    seed(123)
    const b = randn([2, 2]).to_array()
    expect(a).toEqual(b)

    const w = parameter([1]).ones()
    const tracked = mul(w, w)
    const untracked = no_grad(() => mul(w, w))
    clear_grad([w])
    grad(sum(tracked), [w])
    expect(w.grad !== null).toBe(true)
    clear_grad([w])
    expect(() => grad(sum(untracked), [w])).toThrow()

    const view = permute(tensor([[[1, 2], [3, 4], [5, 6]]]), [0, 2, 1])
    expect(contiguous(view).to_array()).toEqual([[[1, 3, 5], [2, 4, 6]]])

    const summary = finite_summary(tensor([1, Number.POSITIVE_INFINITY, 3]))
    expect(summary.ok).toBe(false)
    expect(summary.first_bad_flat_index).toBe(1)
    expect(summary.first_bad_value).toBe(Number.POSITIVE_INFINITY)
    expect(finite_abs_max(tensor([-2, 5, -3], { dtype: 'f32' }))).toBe(5)
    expect(finite_abs_max(tensor([1, Number.NaN, 3], { dtype: 'f32' }))).toBe(null)

    const scratch = empty([2, 2], { dtype: 'f32' })
    expect(scratch.shape).toEqual([2, 2])
    copy(scratch, tensor([[1, 2], [3, 4]], { dtype: 'f32' }))
    expect(scratch.to_array()).toEqual([[1, 2], [3, 4]])

    const indexValues = tensor([0, 2, 1], { dtype: 'i64' })
    expect(indexValues.dtype).toBe('i64')
    expect(gt_scalar(indexValues, 0).to_array()).toEqual([0, 1, 1])
  })

  test('supports clip_grad_norm on compute parameters', () => {
    setDevice('cpu')

    const w = parameter([2], { dtype: 'f32' }).ones()
    const params = [w]
    clear_grad(params)
    const loss = sum(mul(w, tensor([10, 0], { dtype: 'f32' })))
    grad(loss, params)

    const totalNorm = clip_grad_norm(params, 5)
    expect(totalNorm > 5).toBe(true)
    const clipped = w.grad?.to_array() as number[]
    expect(Math.abs(clipped[0] - 5) < 1e-6).toBe(true)
    expect(Math.abs(clipped[1]) < 1e-6).toBe(true)
  })

  test('routes compiled matmul-to-indexed-loss through native graph lowering', () => {
    setDevice('cpu')

    const compiled = compile((x, w, targets) => cross_entropy_indexed(matmul(x, w), targets))
    const x = tensor([[2, -1], [0.5, 3]], { dtype: 'f32' })
    const w = tensor([[0.2, 1.1, -0.4], [0.7, -0.3, 0.9]], { dtype: 'f32' })
    const targets = tensor([1, 2], { dtype: 'i64' })

    const eager = cross_entropy_indexed(matmul(x, w), targets)
    const callable = compiled(x, w, targets)
    const lowered = compiled.run(x, w, targets)

    expect(Math.abs(callable.item() - eager.item()) < 1e-6).toBe(true)
    expect(Math.abs(lowered.item() - eager.item()) < 1e-6).toBe(true)
    expect(Math.abs(callable.item() - lowered.item()) < 1e-6).toBe(true)

    const graphProgram = (compiled as any).graph?.()
    expect(graphProgram.executionRuntime()).toBe('native-graph')

    const summary = graphProgram.summary(x, w, targets) as any
    expect(summary.runtime).toBe('native-graph')
    expect(summary.summaryKind).toBe('execution')
    expect(summary.regions).toEqual([])
    expect(summary.templates).toEqual([])
    expect(summary.staticMetadata).toEqual({
      kind: 'provisional_capture_metadata',
      source: 'compute_core_with_ts_shadow',
      fields: ['nodes.outputShape'],
    })
    expect(summary.inferenceConflicts).toEqual([])
    expect(summary.loweringAnalysis).toEqual({ lowerable: true })

    const compiledSummary = (compiled as any).summary()
    expect(compiledSummary.graphRuntime).toBe('native-graph')
    expect(compiledSummary.graphLoweringAnalysis).toEqual({ lowerable: true })
  })

  test('routes compiled where through native graph lowering', () => {
    setDevice('cpu')

    const compiled = compile((cond, x, y) => where(cond, x, y))
    const cond = tensor([[1, 0, 1]], { dtype: 'i64' })
    const x = tensor([[1, 2, 3]], { dtype: 'f32' })
    const y = tensor([[9, 9, 9]], { dtype: 'f32' })

    expect((compiled(cond, x, y) as any).to_array()).toEqual([[1, 9, 3]])
    expect((compiled.run(cond, x, y) as any).to_array()).toEqual([[1, 9, 3]])

    const graphProgram = (compiled as any).graph?.()
    expect(graphProgram.executionRuntime()).toBe('native-graph')

    const summary = graphProgram.summary(cond, x, y) as any
    expect(summary.runtime).toBe('native-graph')
    expect(summary.summaryKind).toBe('execution')
    expect(summary.regions).toEqual([])
    expect(summary.templates).toEqual([])
    expect(summary.loweringAnalysis).toEqual({ lowerable: true })

    const compiledSummary = (compiled as any).summary()
    expect(compiledSummary.graphRuntime).toBe('native-graph')
    expect(compiledSummary.graphLoweringAnalysis).toEqual({ lowerable: true })
  })

  test('routes compiled negative slice selectors through native graph lowering', () => {
    setDevice('cpu')

    const compiled = compile((x) => x.slice(['-1:', ':']))
    const x = tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' })

    expect((compiled(x) as any).to_array()).toEqual([[4, 5, 6]])
    expect((compiled.run(x) as any).to_array()).toEqual([[4, 5, 6]])

    const graphProgram = (compiled as any).graph?.()
    const summary = graphProgram.summary(x) as any
    expect(summary.runtime).toBe('native-graph')
    expect(summary.inferenceConflicts).toEqual([])
  })

  test('routes compiled softmax through native graph lowering', () => {
    setDevice('cpu')

    const compiled = compile((x) => softmax(x, 1))
    const x = tensor([[1, 2, 3]], { dtype: 'f32' })

    const callable = compiled(x)
    const runnable = compiled.run(x)
    const eager = softmax(x, 1)

    expect(callable.to_array()).toEqual(eager.to_array())
    expect(runnable.to_array()).toEqual(eager.to_array())

    const graphProgram = (compiled as any).graph?.()
    expect(graphProgram.executionRuntime()).toBe('native-graph')

    const summary = graphProgram.summary(x) as any
    expect(summary.runtime).toBe('native-graph')
    expect(summary.summaryKind).toBe('execution')
    expect(summary.loweringAnalysis).toEqual({ lowerable: true })
  })

  test.skip(() => true)('summarizes native plan regions for compiled matmul and gelu chains', () => {
    setDevice('cpu')

    const compiled = compile((x, w1, b1, w2, b2) =>
      add(matmul(gelu(add(matmul(x, w1), b1)), w2), b2))

    const x = tensor([[2, -1], [0.5, 3]], { dtype: 'f32' })
    const w1 = tensor([[0.2, 1.1, -0.4], [0.7, -0.3, 0.9]], { dtype: 'f32' })
    const b1 = tensor([[0.1, -0.2, 0.3]], { dtype: 'f32' })
    const w2 = tensor([[0.5, -0.1], [0.2, 0.7], [-0.4, 0.9]], { dtype: 'f32' })
    const b2 = tensor([[0.05, -0.15]], { dtype: 'f32' })

    const eager = add(matmul(gelu(add(matmul(x, w1), b1)), w2), b2)
    const callable = compiled(x, w1, b1, w2, b2)
    const runnable = compiled.run(x, w1, b1, w2, b2)

    expect(callable.to_array()).toEqual(eager.to_array())
    expect(runnable.to_array()).toEqual(eager.to_array())

    const compiledSummary = (compiled as any).summary()
    expect(compiledSummary.mode).toBe('graph')
    expect(compiledSummary.graphRuntime).toBe('native-graph')
    expect(compiledSummary.graphLoweringAnalysis).toEqual({ lowerable: true })

    const plannedSummary = (compiled as any).summary(x, w1, b1, w2, b2)
    expect(plannedSummary.planAvailable).toBe(true)

    const report = JSON.parse((compiled as any).exportReport(x, w1, b1, w2, b2, { format: 'json' }))
    expect(report.kind).toBe('report')
    expect(report.graph_plan.kind).toBe('graph_plan')
    expect(report.graph_plan.steps.length).toBe(5)
    expect(report.graph_plan.regions.length).toBe(2)
    expect(report.graph_plan.regions.map((region: any) => region.kind)).toEqual(['matmul_epilogue', 'matmul_epilogue'])
    expect(report.graph_plan.regions.map((region: any) => region.matmul_epilogue_activation)).toEqual(['gelu', 'none'])
    expect(report.graph_plan.steps.filter((step: any) => step.matmul?.family === 'gemm_2d').length).toBe(2)

    const captureSummary = (compiled as any).captureSummary?.()
    expect(captureSummary.nodeKinds).toContain('gelu')
  })

  test('reports graph capture errors when compile specialization cannot build a graph', () => {
    setDevice('cpu')

    const compiled = compile((x) => {
      finite_summary(x)
      return add(x, x)
    })
    const x = tensor([[1, 2, 3]], { dtype: 'f32' })

    expect(compiled(x).to_array()).toEqual([[2, 4, 6]])

    const compiledSummary = (compiled as any).summary()
    expect(compiledSummary.mode).toBe('eager-forward')
    expect(compiledSummary.graphRuntime).toBe(null)
    expect(compiledSummary.graphLoweringAnalysis).toBe(null)
    expect(typeof compiledSummary.graphCaptureError).toBe('string')
    expect(compiledSummary.graphCaptureError.length > 0).toBe(true)
    expect(compiledSummary.graphCaptureFailure.stage).toBe('capture_adapter')
    expect(compiledSummary.graphCaptureFailure.reason).toBe(compiledSummary.graphCaptureError)
    expect(compiledSummary.eagerFallbackCount > 0).toBe(true)
    expect(compiledSummary.lastExecutionMode).toBe('eager-forward')
    expect(compiledSummary.lastFallbackKind).toBe('graph_capture_failed')
    expect(typeof compiledSummary.lastFallbackReason).toBe('string')
  })

  test('classifies malformed graph capture arguments as syntax failures', () => {
    setDevice('cpu')

    const compiled = compile((x) => softmax(x, 1.5))
    const x = tensor([[1, 2, 3]], { dtype: 'f32' })

    expect(() => compiled(x)).toThrow('integer dim')

    const compiledSummary = (compiled as any).summary()
    expect(compiledSummary.graphCaptureFailure.stage).toBe('syntax')
    expect(compiledSummary.graphCaptureFailure.reason).toContain('integer dim')
    expect(compiledSummary.lastFallbackKind).toBe('graph_capture_failed')
  })

  test('reports specialization reuse and signature-change fallback in compiled summary', () => {
    setDevice('cpu')

    const compiled = compile((x) => add(x, x))
    const first = tensor([[1, 2, 3]], { dtype: 'f32' })
    const second = tensor([[4, 5, 6]], { dtype: 'f32' })
    const differentShape = tensor([[1, 2], [3, 4]], { dtype: 'f32' })

    expect(compiled(first).to_array()).toEqual([[2, 4, 6]])
    expect(compiled(second).to_array()).toEqual([[8, 10, 12]])

    const reusedSummary = (compiled as any).summary(second)
    expect(reusedSummary.specializationActive).toBe(true)
    expect(reusedSummary.specializationCaptureCount).toBe(1)
    expect(reusedSummary.specializationReuseCount > 0).toBe(true)
    expect(reusedSummary.activeTensorSignature).toEqual([{ dtype: 'f32', shape: [1, 3] }])
    expect(reusedSummary.lastExecutionMode).toBe('graph')
    expect(reusedSummary.lastFallbackReason).toBe(null)
    expect(reusedSummary.nativeState.mode).toBe('graph')
    expect(reusedSummary.nativeState.planOutcome).toBe('native-graph')
    expect(reusedSummary.nativeState.planOutcomeReason).toBe(null)
    expect(reusedSummary.nativeState.specializationCaptureCount).toBe(1)
    expect(reusedSummary.nativeState.specializationReuseCount > 0).toBe(true)

    expect(compiled(differentShape).to_array()).toEqual([[2, 4], [6, 8]])

    const fallbackSummary = (compiled as any).summary(differentShape)
    expect(fallbackSummary.mode).toBe('eager-forward')
    expect(fallbackSummary.specializationActive).toBe(true)
    expect(fallbackSummary.eagerFallbackCount > 0).toBe(true)
    expect(fallbackSummary.lastExecutionMode).toBe('eager-forward')
    expect(fallbackSummary.lastFallbackKind).toBe('tensor_signature_changed')
    expect(fallbackSummary.lastFallbackReason).toBe('tensor signature changed')
    expect(fallbackSummary.nativeState.mode).toBe('eager-forward')
    expect(fallbackSummary.nativeState.planOutcome).toBe('eager-forward')
    expect(fallbackSummary.nativeState.planOutcomeReason).toBe('tensor signature changed')
    expect(fallbackSummary.nativeState.lastFallbackKind).toBe('tensor_signature_changed')
    expect(fallbackSummary.nativeState.lastFallbackReason).toBe('tensor signature changed')
  })

  test('reports program-state fallback when a compiled module changes mode', () => {
    setDevice('cpu')

    const layer = module(
      {
        w: parameter([1], { dtype: 'f32' }).ones(),
      },
      ({ w }, x) => mul(x, w),
    )
    const compiled = compile(layer)
    const x = tensor([1, 2, 3], { dtype: 'f32' })

    expect(compiled(x).to_array()).toEqual([1, 2, 3])

    compiled.mode('eval')
    expect(compiled(x).to_array()).toEqual([1, 2, 3])

    const summary = (compiled as any).summary(x)
    expect(summary.mode).toBe('eager-forward')
    expect(summary.specializationActive).toBe(true)
    expect(summary.eagerFallbackCount > 0).toBe(true)
    expect(summary.lastExecutionMode).toBe('eager-forward')
    expect(summary.lastFallbackKind).toBe('program_state_changed')
    expect(summary.lastFallbackReason).toBe('program state changed')
  })

  test('preserves shape metadata through gelu capture', () => {
    setDevice('cpu')

    const compiled = compile((x) => {
      const y = gelu(x)
      if (y.ndim !== 3) {
        throw new Error('expected rank-3 output')
      }
      if (y.shape[2] !== 4) {
        throw new Error('expected hidden dimension to stay intact')
      }
      return y
    })
    const x = tensor([[[1, -2, 3, -4], [5, -6, 7, -8]]], { dtype: 'f32' })

    expect(compiled(x).to_array()).toEqual(gelu(x).to_array())

    const compiledSummary = (compiled as any).summary()
    expect(compiledSummary.mode).toBe('graph')
    expect(compiledSummary.graphRuntime).toBe('native-graph')
    expect(compiledSummary.graphCaptureError).toBe(null)
    expect(compiledSummary.graphLoweringAnalysis).toEqual({ lowerable: true })
  })

  test('routes compiled index_select through native graph lowering and preserves metadata', () => {
    setDevice('cpu')

    const compiled = compile((table, ids) => {
      const y = index_select(table, 0, ids)
      if (y.ndim !== 2) {
        throw new Error('expected rank-2 output')
      }
      if (y.shape[1] !== 2) {
        throw new Error('expected embedding width to stay intact')
      }
      return y
    })
    const table = tensor([[10, 20], [30, 40], [50, 60]], { dtype: 'f32' })
    const ids = tensor([2, 0], { dtype: 'f32' })

    expect(compiled(table, ids).to_array()).toEqual([[50, 60], [10, 20]])

    const compiledSummary = (compiled as any).summary()
    expect(compiledSummary.mode).toBe('graph')
    expect(compiledSummary.graphRuntime).toBe('native-graph')
    expect(compiledSummary.graphCaptureError).toBe(null)
    expect(compiledSummary.graphLoweringAnalysis).toEqual({ lowerable: true })
  })

  test('preserves autograd links through compiled native graph forwards', () => {
    setDevice('cpu')

    const layer = module(
      {
        w: parameter([2, 3], { dtype: 'f32' }).ones(),
        b: parameter([1, 3], { dtype: 'f32' }).zeros(),
      },
      ({ w, b }, x) => mean(gelu(add(matmul(x, w), b))),
    )
    const compiled = compile(layer)
    const x = tensor([[1, -2], [3, 4]], { dtype: 'f32' })

    clear_grad(compiled.parameters)
    const loss = compiled(x)
    grad(loss, compiled.parameters)

    expect((compiled as any).summary().graphRuntime).toBe('native-graph')
    expect(compiled.parameters[0].grad?.shape).toEqual([2, 3])
    expect(compiled.parameters[1].grad?.shape).toEqual([1, 3])
  })

  test.skip(() => true)('carries module metadata paths into compiled graph export', () => {
    setDevice('cpu')

    const block = module(
      {
        w: parameter([2, 2], { dtype: 'f32' }).ones(),
      },
      ({ w }, x) => matmul(x, w),
    ).metadata('toy.block')

    const compiled = compile(block)
    const x = tensor([[1, 2]], { dtype: 'f32' })

    compiled(x)
    const text = (compiled as any).exportReport(x, {
      mode: 'annotated',
      boundary: 'step',
      phase: 'forward',
      runId: 'test',
      graphId: 'toy',
      step: 1,
    })

    expect(typeof text).toBe('string')
    expect(text).toContain('module_path=toy.block')
  })

  test.skip(() => true)('exports canonical report json for offline tooling', () => {
    setDevice('cpu')

    const block = module(
      {
        w: parameter([2, 2], { dtype: 'f32' }).ones(),
      },
      ({ w }, x) => matmul(x, w),
    ).metadata('toy.block')

    const compiled = compile(block)
    const x = tensor([[1, 2]], { dtype: 'f32' })

    compiled(x)
    const json = (compiled as any).exportReport(x, {
      format: 'json',
      boundary: 'step',
      phase: 'forward',
      runId: 'test',
      graphId: 'toy',
      step: 1,
    })

    expect(typeof json).toBe('string')

    const payload = JSON.parse(json)
    expect(payload.kind).toBe('report')
    expect(payload.context.boundary).toBe('step')
    expect(payload.context.phase).toBe('forward')
    expect(payload.graph?.kind).toBe('graph')
    expect(payload.graph_plan?.kind).toBe('graph_plan')
    expect(payload.graph_plan?.steps[0]?.kind).toBe('single_op')
    expect(payload.graph_plan?.steps[0]?.execution_kind).toBe('reduction')
    expect(payload.graph?.nodes.some((node: any) => node.module_path === 'toy.block')).toBe(true)
    expect(Array.isArray(payload.graph?.edges)).toBe(true)
    expect(payload.graph.edges.length > 0).toBe(true)
    expect(typeof payload.graph.edges[0].edge_id).toBe('number')
    expect(typeof payload.graph.edges[0].value_id).toBe('number')
    expect(typeof payload.graph.edges[0].producer_node_id).toBe('number')
    expect(typeof payload.graph.edges[0].consumer_node_id).toBe('number')
    expect(Array.isArray(payload.graph?.module_nodes)).toBe(true)
    const moduleNode = payload.graph.module_nodes.find((node: any) => node.path === 'toy.block')
    expect(typeof moduleNode?.module_id).toBe('number')
    expect(moduleNode?.path).toBe('toy.block')
    expect(Array.isArray(moduleNode?.node_ids)).toBe(true)
    const summary = (compiled as any).summary(x)
    expect(summary.planAvailable).toBe(true)
    expect(summary.planStepCount).toBe(payload.graph_plan.steps.length)
    expect(summary.planRegionCount).toBe(payload.graph_plan.regions.length)
    expect(summary.planOutputCount).toBe(payload.graph_plan.outputs.length)
  })
})
