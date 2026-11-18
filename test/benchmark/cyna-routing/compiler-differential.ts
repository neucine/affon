// Run with a ReleaseFast Cyna-enabled Affon build and AFFON_TEST_CYNA_SESSION=1.
import { add, contiguous, embedding, gelu, matmul, mean, mul, slice, softmax, sub, topk, transpose, where } from 'affon:ops'
import { program, Session, Tensor } from 'affon:core'
import { representativePrograms } from './route-representative.ts'

const operations = { add, mul, sub }
const goals = ['balanced', 'latency', 'throughput', 'memory'] as const
const cynaElementwiseNames = new Set([
  'cpu_mul', 'cpu_sub', 'cpu_div', 'cpu_abs', 'cpu_neg', 'cpu_exp',
  'cpu_log', 'cpu_sqrt', 'cpu_sign', 'cpu_relu', 'cpu_sigmoid',
  'cpu_silu', 'cpu_tanh', 'cpu_sin', 'cpu_cos', 'cpu_erf', 'cpu_gelu',
])
const mismatches: unknown[] = []
let compared = 0

function compare(
  name: string,
  source: ReturnType<typeof program>,
  searchLimits?: Readonly<Record<string, number>>,
): void {
  for (const goal of goals) {
    const cyna = new Session({ device: 'cpu' })
    const zig = new Session({ device: 'cpu', cpuExecution: 'zig' })
    const options = { optimizationGoal: goal, searchLimits }
    const selected = cyna.compile(source, options)
    const reference = zig.compile(source, options)
    const actual = selected.explain() as any
    const expected = reference.explain() as any
    if (actual.execution_backend !== 'cyna') throw Error(`${name}/${goal} did not route to Cyna`)
    const cynaNodes = actual.cyna_selected_physical_nodes
      .map((node: any) => ({ node: node.node_id, sources: node.program_origins, implementation: node.implementation, view: node.is_view }))
    const zigNodes = expected.executable.physical_nodes
      .map((node: any) => ({ node: node.id, sources: node.origins, implementation: node.implementation_id, view: node.action === 'view' }))
    const canonical = (nodes: any[]) => nodes.filter(node => node.sources.length > 0).map(node => ({
      sources: node.sources,
      view: node.view,
      implementation: node.view ? 'view' : cynaElementwiseNames.has(node.implementation)
        ? 'cpu_elementwise' : node.implementation,
    })).sort((a, b) => a.sources[0] - b.sources[0])
    const conversions = (nodes: any[]) => nodes.filter(node => node.sources.length === 0)
      .map(node => node.implementation).sort()
    const scheduledSources = (nodes: any[], trace: any) => {
      if (trace.steps.length !== nodes.length) throw Error(`${name}/${goal} has incomplete schedule rows`)
      const scheduled = trace.steps.map((step: any) => {
        const node = nodes.find(candidate => candidate.node === step.node)
        if (!node) throw Error(`${name}/${goal} has an unknown scheduled node`)
        return node.sources
      })
      return scheduled.filter((sources: number[]) => sources.length > 0)
    }
    const cynaPeak = actual.selected_physical_liveness.peak_active_bytes
    const zigPeak = expected.selected_physical_liveness.peak_active_bytes
    const cynaOrder = scheduledSources(cynaNodes, actual.selected_physical_liveness)
    const zigOrder = scheduledSources(zigNodes, expected.selected_physical_liveness)
    if (JSON.stringify(canonical(cynaNodes)) !== JSON.stringify(canonical(zigNodes)) ||
        JSON.stringify(conversions(cynaNodes)) !== JSON.stringify(conversions(zigNodes)) ||
        JSON.stringify(cynaOrder) !== JSON.stringify(zigOrder) || cynaPeak !== zigPeak)
      mismatches.push({ name, goal, cyna: cynaNodes, zig: zigNodes, cynaPeak, zigPeak, cynaOrder, zigOrder })
    compared++
    selected.dispose(); reference.dispose(); cyna.dispose(); zig.dispose()
  }
}

const boundedSearchProgram = program('differential_bounded_search', p => add(
  add(p.argument('a', Tensor.f32([2])), p.argument('b', Tensor.f32([2]))),
  p.argument('c', Tensor.f32([2])),
))
for (const [name, searchLimits] of [
  ['structural_states', { structuralStates: 1 }],
  ['structural_expansions', { structuralExpansions: 1 }],
  ['complete_evaluations', { completeEvaluations: 1 }],
  ['physical_work', { physicalWork: 1 }],
  ['ready_work', { readyWork: 1 }],
] as const)
  compare(`bounded_search_${name}`, boundedSearchProgram, searchLimits)

for (const [name, source] of [
  ['add_chain', boundedSearchProgram],
  ['matmul_bias', program('differential_physical_work_matmul_bias', p => add(
    matmul(p.argument('left', Tensor.f32([3, 4])), p.argument('right', Tensor.f32([4, 5]))),
    p.argument('bias', Tensor.f32([5])),
  ))],
  ['embedding_topk', program('differential_physical_work_embedding_topk', p => topk(
    embedding(p.argument('table', Tensor.f32([8, 4])), p.argument('indices', Tensor.i64([3]))),
    2,
    1,
  ))],
  ['transpose', program('differential_physical_work_transpose', p =>
    transpose(p.argument('x', Tensor.f32([2, 3, 4])), [2, 0, 1]))],
  ['softmax', program('differential_physical_work_softmax', p =>
    softmax(p.argument('x', Tensor.f32([3, 4])), -1))],
  ['optimizer_update', program('differential_physical_work_optimizer_update', p => sub(
    p.argument('parameter', Tensor.f32([4])),
    mul(p.argument('gradient', Tensor.f32([4])), p.argument('rate', Tensor.f32([4]))),
  ))],
] as const)
  compare(`physical_work_single_offer_${name}`, source, { physicalWork: 1 })

for (const [firstName, first] of Object.entries(operations))
  for (const [secondName, second] of Object.entries(operations))
    for (const [lastName, last] of Object.entries(operations)) {
      const name = `${firstName}_${secondName}_${lastName}`
      compare(name, program(`differential_${name}`, p => {
        const a = p.argument('a', Tensor.f32([8]))
        const b = p.argument('b', Tensor.f32([8]))
        const c = p.argument('c', Tensor.f32([8]))
        const d = p.argument('d', Tensor.f32([8]))
        const shared = add(a, b)
        const left = first(shared, c)
        const right = second(shared, d)
        return gelu(last(left, right))
      }))
    }

for (const edge of ['reused_product', 'same_factor', 'parameter_factor', 'product_left', 'broadcast'] as const) {
  compare(edge, program(`differential_${edge}`, p => {
    if (edge === 'broadcast') {
      const a = p.argument('a', Tensor.f32([2, 2]))
      const b = p.argument('b', Tensor.f32([2, 1]))
      const c = p.argument('c', Tensor.f32([1, 2]))
      return sub(a, mul(b, c))
    }
    const a = p.argument('a', Tensor.f32([8]))
    const b = p.argument('b', Tensor.f32([8]))
    const c = p.argument('c', Tensor.f32([8]))
    if (edge === 'same_factor') return sub(a, mul(b, b))
    if (edge === 'parameter_factor') return sub(a, mul(a, c))
    const product = mul(b, c)
    if (edge === 'product_left') return sub(product, a)
    return add(sub(a, product), product)
  }))
}

for (const dtype of ['f32', 'f64', 'i64'] as const) {
  const spec = dtype === 'f32' ? Tensor.f32([2]) : dtype === 'f64' ? Tensor.f64([2]) : Tensor.i64([2])
  compare(`optimizer_${dtype}`, program(`differential_optimizer_${dtype}`, p => sub(
    p.argument('parameter', spec),
    mul(p.argument('gradient', spec), p.argument('rate', spec)),
  )))
}

for (const dtype of ['f32', 'f64'] as const)
  for (const size of [2, 4, 8, 16, 32, 64, 128])
    for (const activation of [false, true]) {
      if (dtype === 'f64' && activation) continue
      const name = `matmul_bias_${dtype}_${size}${activation ? '_gelu' : ''}`
      const matrix = dtype === 'f32' ? Tensor.f32([size, size]) : Tensor.f64([size, size])
      const bias = dtype === 'f32' ? Tensor.f32([size]) : Tensor.f64([size])
      compare(name, program(`differential_${name}`, p => {
        const sum = add(matmul(p.argument('lhs', matrix), p.argument('rhs', matrix)), p.argument('bias', bias))
        return activation ? gelu(sum) : sum
      }))
    }

for (const [rows, inner, columns] of [[3, 5, 4], [7, 9, 6], [17, 13, 11]] as const)
  for (const dtype of ['f32', 'f64'] as const)
    for (const activation of [false, true]) {
      if (dtype === 'f64' && activation) continue
      const name = `matmul_bias_${dtype}_${rows}x${inner}x${columns}${activation ? '_gelu' : ''}`
      const left = dtype === 'f32' ? Tensor.f32([rows, inner]) : Tensor.f64([rows, inner])
      const right = dtype === 'f32' ? Tensor.f32([inner, columns]) : Tensor.f64([inner, columns])
      const bias = dtype === 'f32' ? Tensor.f32([columns]) : Tensor.f64([columns])
      compare(name, program(`differential_${name}`, p => {
        const sum = add(matmul(p.argument('lhs', left), p.argument('rhs', right)), p.argument('bias', bias))
        return activation ? gelu(sum) : sum
      }))
    }

for (const dtype of ['f32', 'f64'] as const)
  for (const activation of [false, true]) {
    if (dtype === 'f64' && activation) continue
    const name = `batched_matmul_bias_${dtype}${activation ? '_gelu' : ''}`
    const left = dtype === 'f32' ? Tensor.f32([2, 3, 4]) : Tensor.f64([2, 3, 4])
    const right = dtype === 'f32' ? Tensor.f32([2, 4, 5]) : Tensor.f64([2, 4, 5])
    const bias = dtype === 'f32' ? Tensor.f32([5]) : Tensor.f64([5])
    compare(name, program(`differential_${name}`, p => {
      const sum = add(
        matmul(p.argument('lhs', left), p.argument('rhs', right)),
        p.argument('bias', bias),
      )
      return activation ? gelu(sum) : sum
    }))
  }

compare('matmul_bias_shared_left_operand', program('differential_matmul_bias_shared_left_operand', p => {
  const left = p.argument('left', Tensor.f32([3, 3]))
  return add(matmul(left, left), p.argument('bias', Tensor.f32([3])))
}))

compare('matmul_bias_reuses_left_matrix', program('differential_matmul_bias_reuses_left_matrix', p => {
  const left = p.argument('left', Tensor.f32([3, 3]))
  return add(matmul(left, p.argument('right', Tensor.f32([3, 3]))), left)
}))

compare('matmul_bias_reuses_right_matrix', program('differential_matmul_bias_reuses_right_matrix', p => {
  const right = p.argument('right', Tensor.f32([3, 3]))
  return add(p.argument('bias', Tensor.f32([3])), matmul(p.argument('left', Tensor.f32([3, 3])), right))
}))

compare('matmul_bias_broadcast_batch', program('differential_matmul_bias_broadcast_batch', p => add(
  matmul(p.argument('left', Tensor.f32([2, 3, 4])), p.argument('right', Tensor.f32([4, 5]))),
  p.argument('bias', Tensor.f32([5])),
)))

compare('matmul_bias_rank_four_broadcast_batches', program('differential_matmul_bias_rank_four_broadcast_batches', p => add(
  matmul(p.argument('left', Tensor.f32([2, 1, 3, 4])), p.argument('right', Tensor.f32([1, 2, 4, 5]))),
  p.argument('bias', Tensor.f32([5])),
)))

compare('batched_matmul_bias_transposed_right', program('differential_batched_matmul_bias_transposed_right', p => add(
  matmul(
    p.argument('left', Tensor.f32([2, 3, 5])),
    transpose(p.argument('right', Tensor.f32([2, 4, 5])), [0, 2, 1]),
  ),
  p.argument('bias', Tensor.f32([4])),
)))

compare('batched_matmul_bias_transposed_right_f64', program('differential_batched_matmul_bias_transposed_right_f64', p => add(
  matmul(
    p.argument('left', Tensor.f64([2, 3, 5])),
    transpose(p.argument('right', Tensor.f64([2, 4, 5])), [0, 2, 1]),
  ),
  p.argument('bias', Tensor.f64([4])),
)))

compare('batched_matmul_bias_batch_bias', program('differential_batched_matmul_bias_batch_bias', p => add(
  matmul(p.argument('left', Tensor.f32([2, 3, 4])), p.argument('right', Tensor.f32([2, 4, 5]))),
  p.argument('bias', Tensor.f32([2, 1, 5])),
)))

compare('matmul_bias_broadcast_right_with_batch_bias', program('differential_matmul_bias_broadcast_right_with_batch_bias', p => add(
  matmul(p.argument('left', Tensor.f32([2, 3, 4])), p.argument('right', Tensor.f32([4, 5]))),
  p.argument('bias', Tensor.f32([2, 1, 5])),
)))

compare('add_gelu_repeated_operand', program('differential_add_gelu_repeated_operand', p => {
  const value = p.argument('value', Tensor.f32([4]))
  return gelu(add(value, value))
}))

compare('add_gelu_broadcast_operand', program('differential_add_gelu_broadcast_operand', p => gelu(add(
  p.argument('left', Tensor.f32([2, 4])),
  p.argument('right', Tensor.f32([4])),
))))

compare('sliced_matmul_add', program('differential_sliced_matmul_add', p => add(
  matmul(
    contiguous(slice(p.argument('left', Tensor.f32([3, 5])), [{ start: 1, stop: 3 }, { start: 1, stop: 5 }])),
    p.argument('right', Tensor.f32([4, 2])),
  ),
  p.argument('bias', Tensor.f32([2])),
)))

compare('transposed_slice_binary', program('differential_transposed_slice_binary', p => add(
  slice(transpose(p.argument('matrix', Tensor.f32([3, 4])), [1, 0]), [
    { start: 0, stop: 3 },
    { start: 1, stop: 3 },
  ]),
  p.argument('bias', Tensor.f32([2])),
)))

compare('strided_slice_gelu_add', program('differential_strided_slice_gelu_add', p => add(
  gelu(slice(p.argument('matrix', Tensor.f32([3, 5])), [
    { start: 0, stop: 3 },
    { start: 1, stop: 5, step: 2 },
  ])),
  p.argument('bias', Tensor.f32([2])),
)))

compare('singleton_broadcast_add', program('differential_singleton_broadcast_add', p => add(
  p.argument('left', Tensor.f32([2, 1])),
  p.argument('right', Tensor.f32([1, 4])),
)))

for (const dtype of ['f32', 'f64'] as const)
  for (const size of [2, 8, 16]) {
    const name = `transposed_matmul_bias_${dtype}_${size}`
    const matrix = dtype === 'f32' ? Tensor.f32([size, size]) : Tensor.f64([size, size])
    const bias = dtype === 'f32' ? Tensor.f32([size]) : Tensor.f64([size])
    compare(name, program(`differential_${name}`, p => add(
      matmul(p.argument('lhs', matrix), transpose(p.argument('rhs', matrix), [1, 0])),
      p.argument('bias', bias),
    )))
  }

for (const size of [4, 16, 64]) {
  const matrix = Tensor.f32([size, size])
  compare(`softmax_add_${size}`, program(`differential_softmax_add_${size}`, p => add(
    softmax(p.argument('x', matrix), -1), p.argument('bias', Tensor.f32([size])),
  )))
  compare(`mean_keepdim_add_${size}`, program(`differential_mean_keepdim_add_${size}`, p => add(
    mean(p.argument('x', matrix), -1, true), p.argument('bias', Tensor.f32([size])),
  )))
  compare(`transposed_contiguous_add_${size}`, program(`differential_transposed_contiguous_add_${size}`, p => add(
    contiguous(transpose(p.argument('x', matrix), [1, 0])), p.argument('bias', matrix),
  )))
  compare(`where_broadcast_${size}`, program(`differential_where_broadcast_${size}`, p => where(
    p.argument('mask', Tensor.i64([size, 1])),
    add(p.argument('x', matrix), p.argument('bias', Tensor.f32([size]))),
    p.argument('fallback', Tensor.f32([1, size])),
  )))
  compare(`embedding_topk_${size}`, program(`differential_embedding_topk_${size}`, p => topk(
    embedding(p.argument('table', matrix), p.argument('indices', Tensor.i64([size]))),
    Math.min(size, 8), 1,
  )))
}

for (const entry of representativePrograms)
  compare(`representative_${entry.name}`, entry.source)

console.log(JSON.stringify({ compared, mismatchCount: mismatches.length }))
for (const item of mismatches as any[]) console.log(JSON.stringify({
  name: item.name, goal: item.goal,
  cyna: item.cyna.map((node: any) => `${node.sources.join(',')}:${node.implementation}`),
  zig: item.zig.map((node: any) => `${node.sources.join(',')}:${node.implementation}`),
  cynaPeak: item.cynaPeak, zigPeak: item.zigPeak,
  cynaOrder: item.cynaOrder, zigOrder: item.zigOrder,
}))
if (mismatches.length !== 0) throw Error(`${mismatches.length} selected plans differ`)
