import { Session, Tensor, program, type Program } from 'affon:core'
import {
  abs, add, argmax, argmin, binary_cross_entropy,
  binary_cross_entropy_with_logits, cast, cat, clamp, contiguous, cos,
  cross_entropy, dense_cross_entropy, div, dot, embedding, eq, erf, exp, gather,
  gelu, gt, gt_scalar, index_select, layer_norm, log, log_softmax,
  log_softmax_nll, lt, masked_fill, matmul, max, mean, mean_absolute_error,
  mean_squared_error, min, mul, neg, one_hot, permute, quantized_matmul, reduce_to_shape,
  relu, reshape, rms_norm, scatter_add, sigmoid, sign, silu, sin, slice, softmax,
  sqrt, square, squeeze, stack, std, sub, sum, tanh, topk, transpose, unsqueeze,
  variance, where,
} from 'affon:ops'
import { importedAndModelPrograms } from './route-model-corpus.ts'
import { buildRoutingReport, type Routing, type RoutingCorpus, type RoutingEntry } from './route-report.ts'

const routingSymbol = Symbol.for('affon.compute.executable_routing')

export type RegistryEntry = Readonly<{
  name: string
  corpus: RoutingCorpus
  source: Program
  options?: Readonly<{ telemetry?: { captureRecords?: number; computeTiming?: 'none' | 'invocation' | 'kernel' } }>
}>

const vector = (name: string, shape: readonly number[] = [4]) => (p: any) => p.argument(name, Tensor.f32(shape))
const bf16Vector = (name: string, shape: readonly number[] = [4]) => (p: any) => p.argument(name, Tensor.bf16(shape))
const f64Vector = (name: string, shape: readonly number[] = [4]) => (p: any) => p.argument(name, Tensor.f64(shape))
const define = (name: string, author: (p: any) => Program | any, options?: RegistryEntry['options']): RegistryEntry => ({
  name,
  corpus: 'authored',
  source: program(`routing_${name}`, author),
  options,
})

export const representativePrograms: readonly RegistryEntry[] = Object.freeze([
  define('identity', p => p.argument('x', Tensor.f32([2, 3]))),
  define('binary_add', p => add(vector('a')(p), vector('b')(p))),
  define('binary_sub', p => sub(vector('a')(p), vector('b')(p))),
  define('binary_mul', p => mul(vector('a')(p), vector('b')(p))),
  define('binary_div', p => div(vector('a')(p), vector('b')(p))),
  define('compare_eq', p => eq(vector('a')(p), vector('b')(p))),
  define('compare_lt', p => lt(vector('a')(p), vector('b')(p))),
  define('compare_gt', p => gt(vector('a')(p), vector('b')(p))),
  define('compare_scalar_gt', p => gt_scalar(vector('x')(p), 0.5)),
  define('matmul', p => matmul(p.argument('a', Tensor.f32([2, 3])), p.argument('b', Tensor.f32([3, 4])))),
  define('matmul_batched', p => matmul(p.argument('a', Tensor.f32([2, 3, 4])), p.argument('b', Tensor.f32([2, 4, 5])))),
  define('dot', p => dot(vector('a')(p), vector('b')(p))),
  define('quantized_matmul', p => quantized_matmul(
    p.argument('x', Tensor.f32([2, 64])),
    p.argument('weights', Tensor.u32([16, 16])),
    p.argument('scales', Tensor.bf16([16, 1])),
    p.argument('biases', Tensor.bf16([16, 1])),
    { bits: 8, group_size: 64 },
  )),
  define('unary_abs', p => abs(vector('x')(p))),
  define('unary_neg', p => neg(vector('x')(p))),
  define('unary_exp', p => exp(vector('x')(p))),
  define('unary_log', p => log(vector('x')(p))),
  define('unary_sqrt', p => sqrt(vector('x')(p))),
  define('unary_square', p => square(vector('x')(p))),
  define('unary_sign', p => sign(vector('x')(p))),
  define('activation_relu', p => relu(vector('x')(p))),
  define('activation_sigmoid', p => sigmoid(vector('x')(p))),
  define('activation_silu', p => silu(vector('x')(p))),
  define('activation_tanh', p => tanh(vector('x')(p))),
  define('activation_sin', p => sin(vector('x')(p))),
  define('activation_cos', p => cos(vector('x')(p))),
  define('activation_erf', p => erf(vector('x')(p))),
  define('activation_gelu', p => gelu(vector('x')(p))),
  define('clamp', p => clamp(vector('x')(p), -1, 1)),
  define('softmax', p => softmax(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('softmax_negative_axis', p => softmax(p.argument('x', Tensor.f32([2, 4])), -1)),
  define('log_softmax', p => log_softmax(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('log_softmax_negative_axis', p => log_softmax(p.argument('x', Tensor.f32([2, 4])), -1)),
  define('reduce_sum_all', p => sum(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_sum_axis', p => sum(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_sum_keepdim', p => sum(p.argument('x', Tensor.f32([2, 4])), 1, true)),
  define('reduce_sum_negative_axis_keepdim', p => sum(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_mean_all', p => mean(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_mean', p => mean(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_mean_negative_axis_keepdim', p => mean(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_min_all', p => min(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_min', p => min(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_min_negative_axis_keepdim', p => min(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_max_all', p => max(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_max', p => max(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_max_negative_axis_keepdim', p => max(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_variance_all', p => variance(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_variance', p => variance(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_variance_negative_axis_keepdim', p => variance(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_std_all', p => std(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_std', p => std(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_std_negative_axis_keepdim', p => std(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_argmin_all', p => argmin(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_argmin', p => argmin(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_argmin_negative_axis_keepdim', p => argmin(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_argmax_all', p => argmax(p.argument('x', Tensor.f32([2, 4])))),
  define('reduce_argmax', p => argmax(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('reduce_argmax_negative_axis_keepdim', p => argmax(p.argument('x', Tensor.f32([2, 4])), -1, true)),
  define('reduce_large_axis', p => sum(p.argument('x', Tensor.f32([8, 257])), 1)),
  define('reduce_to_shape', p => reduce_to_shape(p.argument('x', Tensor.f32([2, 3, 4])), [2, 1, 1])),
  define('reshape', p => reshape(p.argument('x', Tensor.f32([2, 3, 4])), [6, 4])),
  define('contiguous', p => contiguous(transpose(p.argument('x', Tensor.f32([2, 3])), [1, 0]))),
  define('transpose', p => transpose(p.argument('x', Tensor.f32([2, 3, 4])), [2, 0, 1])),
  define('permute', p => permute(p.argument('x', Tensor.f32([2, 3, 4])), [2, 0, 1])),
  define('broadcast_add', p => add(p.argument('a', Tensor.f32([3, 4])), p.argument('b', Tensor.f32([4])))),
  define('singleton_add', p => add(p.argument('a', Tensor.f32([1, 4])), p.argument('b', Tensor.f32([1, 4])))),
  define('rank_zero_identity', p => p.argument('x', Tensor.f32([]))),
  define('rank_zero_add', p => add(p.argument('a', Tensor.f32([])), p.argument('b', Tensor.f32([])))),
  define('slice', p => slice(p.argument('x', Tensor.f32([4, 6])), [{ start: 1, stop: 4, step: 2 }, { start: 1, stop: 4 }])),
  define('squeeze', p => squeeze(p.argument('x', Tensor.f32([1, 4])), 0)),
  define('unsqueeze', p => unsqueeze(vector('x')(p), 1)),
  define('cast_f64_to_f32', p => cast(p.argument('x', Tensor.f64([4])), 'f32')),
  define('dtype_f64_unary', p => exp(f64Vector('x')(p))),
  define('dtype_f64_reduce', p => sum(f64Vector('x')(p), 0)),
  define('dtype_f64_matmul', p => matmul(p.argument('a', Tensor.f64([2, 3])), p.argument('b', Tensor.f64([3, 2])))),
  define('dtype_bf16_unary', p => exp(bf16Vector('x')(p))),
  define('dtype_bf16_reduce', p => sum(bf16Vector('x')(p), 0)),
  define('dtype_bf16_matmul', p => matmul(p.argument('a', Tensor.bf16([2, 3])), p.argument('b', Tensor.bf16([3, 2])))),
  define('masked_fill', p => masked_fill(p.argument('x', Tensor.f32([2, 3])), p.argument('mask', Tensor.i64([2, 3])), 0)),
  define('cat', p => cat([p.argument('a', Tensor.f32([2, 3])), p.argument('b', Tensor.f32([2, 3]))], 0)),
  define('stack', p => stack([vector('a')(p), vector('b')(p)], 0)),
  define('where', p => where(p.argument('mask', Tensor.i64([2, 3])), p.argument('a', Tensor.f32([2, 3])), p.argument('b', Tensor.f32([2, 3])))),
  define('layer_norm', p => layer_norm(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('layer_norm_negative_axis', p => layer_norm(p.argument('x', Tensor.f32([2, 4])), -1)),
  define('rms_norm', p => rms_norm(p.argument('x', Tensor.f32([2, 4])), 1)),
  define('rms_norm_negative_axis', p => rms_norm(p.argument('x', Tensor.f32([2, 4])), -1)),
  define('embedding', p => embedding(p.argument('table', Tensor.f32([8, 4])), p.argument('indices', Tensor.i64([2, 3])))),
  define('index_select', p => index_select(p.argument('x', Tensor.f32([4, 3])), 0, p.argument('indices', Tensor.i64([2])))),
  define('gather', p => gather(p.argument('x', Tensor.f32([2, 4])), 1, p.argument('indices', Tensor.i64([2, 2])))),
  define('scatter_add', p => scatter_add(p.argument('x', Tensor.f32([3, 4])), p.argument('indices', Tensor.i64([2, 4])), p.argument('updates', Tensor.f32([2, 4])), 0)),
  define('topk', p => topk(p.argument('x', Tensor.f32([2, 6])), 2, 1)),
  define('one_hot', p => one_hot(p.argument('indices', Tensor.i64([2, 3])), 5)),
  define('cross_entropy', p => cross_entropy(p.argument('logits', Tensor.f32([2, 5])), p.argument('labels', Tensor.i64([2])))),
  define('dense_cross_entropy', p => dense_cross_entropy(p.argument('logits', Tensor.f32([2, 5])), p.argument('targets', Tensor.f32([2, 5])))),
  define('log_softmax_nll', p => log_softmax_nll(p.argument('logits', Tensor.f32([2, 5])), p.argument('targets', Tensor.f32([2, 5])))),
  define('mse', p => mean_squared_error(p.argument('prediction', Tensor.f32([2, 4])), p.argument('target', Tensor.f32([2, 4])))),
  define('mae', p => mean_absolute_error(p.argument('prediction', Tensor.f32([2, 4])), p.argument('target', Tensor.f32([2, 4])))),
  define('bce', p => binary_cross_entropy(p.argument('prediction', Tensor.f32([2, 4])), p.argument('target', Tensor.f32([2, 4])))),
  define('bce_with_logits', p => binary_cross_entropy_with_logits(p.argument('logits', Tensor.f32([2, 4])), p.argument('target', Tensor.f32([2, 4])))),
  define('linear_gelu_residual', p => add(
    gelu(matmul(p.argument('x', Tensor.f32([2, 8])), p.argument('weights', Tensor.f32([8, 8])))),
    p.argument('residual', Tensor.f32([2, 8])),
  )),
  define('attention_scores_softmax', p => {
    const query = p.argument('query', Tensor.f32([2, 4, 8]))
    const key = p.argument('key', Tensor.f32([2, 4, 8]))
    const scores = matmul(query, transpose(key, [0, 2, 1]))
    return matmul(softmax(scores, 2), p.argument('value', Tensor.f32([2, 4, 8])))
  }),
  define('transformer_block', p => {
    const input = p.argument('x', Tensor.f32([2, 4, 8]))
    const normalized = layer_norm(input, -1)
    const query = matmul(normalized, p.argument('query_weight', Tensor.f32([8, 8])))
    const key = matmul(normalized, p.argument('key_weight', Tensor.f32([8, 8])))
    const value = matmul(normalized, p.argument('value_weight', Tensor.f32([8, 8])))
    const scores = matmul(query, transpose(key, [0, 2, 1]))
    const attended = matmul(softmax(scores, -1), value)
    const projected = matmul(attended, p.argument('output_weight', Tensor.f32([8, 8])))
    return rms_norm(add(input, projected), -1)
  }),
  define('observability_evidence_capture', p => add(vector('a')(p), vector('b')(p)), { telemetry: { captureRecords: 8 } }),
  define('observability_compute_timing', p => add(vector('a')(p), vector('b')(p)), { telemetry: { computeTiming: 'invocation' } }),
])

export const routingPrograms: readonly RegistryEntry[] = Object.freeze([
  ...representativePrograms,
  ...importedAndModelPrograms,
])

export function measureRepresentativeRouting(entries: readonly RegistryEntry[] = routingPrograms): RoutingEntry[] {
  return entries.map(entry => {
    const session = new Session({ device: 'cpu', telemetry: entry.options?.telemetry })
    try {
      const executable = session.compile(entry.source)
      try {
        return { name: entry.name, corpus: entry.corpus, routing: Object.freeze((executable as any)[routingSymbol]() as Routing) }
      } finally {
        executable.dispose()
      }
    } finally {
      session.dispose()
    }
  })
}

const report = buildRoutingReport(measureRepresentativeRouting())
console.log(JSON.stringify(report, null, 2))
