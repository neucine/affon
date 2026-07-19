import {
  abs,
  add,
  axes as computeAxes,
  cast,
  clamp,
  contiguous,
  copy,
  div,
  empty,
  exp,
  finite_summary,
  gt_scalar,
  gelu,
  index_select,
  log,
  matmul,
  max,
  mean,
  masked_fill,
  module as computeModule,
  move,
  mul,
  neg,
  no_grad,
  ones,
  parameter as computeParameter,
  rand,
  randn,
  relu,
  reshape,
  sigmoid,
  softmax,
  squeeze,
  sqrt,
  stack,
  sub,
  sum,
  tanh,
  unsqueeze,
  tensor,
  variance,
  zeros,
} from 'affon:compute'
import graphSupport from 'affon:compute/graph.ts'
import { allBinary, allInRange, rowsAreOneHot } from 'affon:compute/native'

interface ComputeStyleModule {
  readonly __affon_compute_module: true
  readonly parameters: ParameterCollection
  state(): any
  restore(state: any): void
  mode(): 'train' | 'eval'
  mode(mode: 'train' | 'eval'): ComputeStyleModule
  train(): void
  eval(): void
  save(path: string): void
  load(path: string): void
}
type Module = ComputeStyleModule & ((x: Tensor, ...args: any[]) => Tensor)
interface LinearModule extends Module { weight: Tensor; bias: Tensor }
interface EmbeddingModule extends Module { weight: Tensor }
interface SequentialModule extends Module {}
interface ParameterCollection extends ReadonlyArray<Tensor> { named(): ReadonlyArray<ParamEntry> }
interface ModuleList { length: number; [index: number]: Module; readonly parameters: ParameterCollection; mode?(mode?: 'train' | 'eval'): ModuleList | 'train' | 'eval'; train?(): void; eval?(): void }
interface SimpleRNNModule extends Module { layers: ModuleList; hidden_state(): Tensor | null }
interface RNNModule extends Module { layers: ModuleList; hidden_state(): Tensor | null }
interface LSTMModule extends Module { layers: ModuleList; hidden_state(): Tensor | null; cell_state(): Tensor | null }
interface DropoutModule extends Module {}
interface BatchNormModule extends Module { gamma: Tensor; beta: Tensor }
interface LayerNormModule extends Module { gamma: Tensor; beta: Tensor }
type ParamEntry = readonly [string, Tensor]
type TensorDType = "f32" | "f64"
type AxisName = string
type CrossEntropyTargetMode = 'auto' | 'index' | 'probability' | 'one_hot'
type CrossEntropyReduction = 'mean'
type DiagnosticMode = 'off' | 'error' | 'warn'
type DiagnosticKind = 'finite'
interface DiagnosticConfig {
  mode?: DiagnosticMode
  include?: string[]
  exclude?: string[]
}
interface DiagnosticAssertOptions {
  path: string
  value?: Tensor
}

function graphCaptureAdapterError(code: string, message: string): AffonError {
  const error = new AffonError(code as any, message)
  ;(error as any).graphCaptureStage = 'capture_adapter'
  return error
}

interface SinusoidalEncodingOptions {
  dtype?: TensorDType
  device?: 'cpu' | 'metal'
}
interface EmbeddingOptions {
  dtype?: TensorDType
  axes?: readonly AxisName[]
}
interface CrossEntropyLossOptions {
  reduction?: CrossEntropyReduction
  target?: CrossEntropyTargetMode
  axis?: number
}

  function isTensor(val: any): val is Tensor {
    return !!(val && val.shape && typeof val.ndim === 'number')
  }

  function isCapturedGraphTensor(val: any): boolean {
    return !!(val && (val as any).$graph === true)
  }

  function describeDiagnosticValue(value: number): string {
    if (Number.isNaN(value)) return 'NaN'
    if (value === Number.POSITIVE_INFINITY) return 'Infinity'
    if (value === Number.NEGATIVE_INFINITY) return '-Infinity'
    return String(value)
  }

  function escapeRegExp(value: string): string {
    return value.replace(/[\\^$+?.()|[\]{}]/g, '\\$&')
  }

  function diagnosticPatternMatches(pattern: string, key: string): boolean {
    if (pattern === '*' || pattern === key) return true
    const source = `^${pattern.split('*').map(escapeRegExp).join('.*')}$`
    return new RegExp(source).test(key)
  }

  function diagnosticFilterMatches(patterns: readonly string[] | undefined, key: string): boolean {
    if (!patterns || patterns.length === 0) return false
    for (let i = 0; i < patterns.length; i++) {
      if (diagnosticPatternMatches(patterns[i], key)) return true
    }
    return false
  }

  function diagnosticEnabled(config: Required<DiagnosticConfig>, kind: DiagnosticKind, path: string): boolean {
    if (config.mode === 'off') return false
    const key = `${kind}:${path}`
    if (diagnosticFilterMatches(config.exclude, key)) return false
    if (config.include.length === 0) return true
    return diagnosticFilterMatches(config.include, key)
  }


function isComputationModule(value: any): value is Module | ComputeStyleModule {
  return typeof value === 'function'
    && value.__affon_compute_module === true
}

function assertModuleListEntry(entry: any, source: string): void {
  if (!isComputationModule(entry)) {
    throw new AffonError('invalid_arg', `${source} expects compute modules or built-in nn layers`)
  }
}

  function prod(shape: readonly number[], start: number): number {
    let value = 1
    for (let i = start; i < shape.length; i++) {
      value *= shape[i]
    }
    return value
  }

  function fanInOut(tensor: Tensor): { fan_in: number; fan_out: number } {
    if (tensor.ndim < 2) {
      throw new AffonError('invalid_shape', 'weight init expects tensor with at least 2 dimensions')
    }
    if (tensor.ndim === 2) {
      return { fan_in: tensor.shape[0], fan_out: tensor.shape[1] }
    }
    const receptive = prod(tensor.shape, 2)
    return {
      fan_in: tensor.shape[1] * receptive,
      fan_out: tensor.shape[0] * receptive,
    }
  }

  function dtypeOf(tensor: Tensor): TensorDType {
    return tensor.dtype as TensorDType
  }

  function defaultModuleDType(): TensorDType {
    return "f32"
  }

  const axis = Object.freeze({
    batch: computeAxes.batch,
    token: computeAxes.token,
    feature: computeAxes.feature,
    hidden: computeAxes.hidden,
    vocab: computeAxes.vocab,
    layer: 'layer',
    gate: 'gate',
  } as const)

  function tensorOpts(dtype: TensorDType, axes?: readonly AxisName[], device?: 'cpu' | 'metal') {
    return device ? { dtype, axes, device } : { dtype, axes }
  }

  function scalarLike(scalar: number, like: Tensor): Tensor {
    const build = (shape: number[]): any => {
      if (shape.length === 0) return scalar
      const [head, ...tail] = shape
      const out = new Array(head)
      for (let i = 0; i < head; i++) out[i] = build(tail)
      return out
    }
    const out = tensor(build(like.shape as number[]), tensorOpts(dtypeOf(like), like.axes))
    return isCapturedGraphTensor(like) ? out as Tensor : move(out, like.device) as Tensor
  }

  function probabilityClampEpsilon(like: Tensor): number {
    return dtypeOf(like) === 'f32' ? 1e-7 : 1e-15
  }

  function randLike(shape: number[], like: Tensor, axes?: readonly AxisName[]): Tensor {
    const out = rand(shape, tensorOpts(dtypeOf(like), axes))
    return isCapturedGraphTensor(like) ? out as Tensor : move(out, like.device) as Tensor
  }

  function randnLike(shape: number[], like: Tensor, axes?: readonly AxisName[]): Tensor {
    const out = randn(shape, tensorOpts(dtypeOf(like), axes))
    return isCapturedGraphTensor(like) ? out as Tensor : move(out, like.device) as Tensor
  }

  function onesLike(shape: number[], like: Tensor, axes?: readonly AxisName[]): Tensor {
    const out = ones(shape, tensorOpts(dtypeOf(like), axes))
    return isCapturedGraphTensor(like) ? out as Tensor : move(out, like.device) as Tensor
  }

  function zerosLike(shape: number[], like: Tensor, axes?: readonly AxisName[]): Tensor {
    const out = zeros(shape, tensorOpts(dtypeOf(like), axes))
    return isCapturedGraphTensor(like) ? out as Tensor : move(out, like.device) as Tensor
  }

  function axis0Index(i: number): Tensor {
    return tensor([i], { dtype: 'i64', axes: [axis.token] }) as Tensor
  }

  const causal_mask_cache = new Map<string, Tensor>()

  function causal_mask(
    length: number,
    opts?: { dtype?: TensorDType; device?: 'cpu' | 'metal' },
  ): Tensor {
    if (!Number.isInteger(length) || length <= 0) {
      throw new AffonError('invalid_arg', 'causal_mask length must be a positive integer')
    }
    const dtype = opts?.dtype ?? 'f32'
    const device = opts?.device ?? 'cpu'
    const cacheKey = `${length}:${dtype}:${device}`
    const cached = causal_mask_cache.get(cacheKey)
    if (cached) return cached

    const rows: number[][] = []
    for (let row = 0; row < length; row++) {
      const cols: number[] = []
      for (let col = 0; col < length; col++) cols.push(col > row ? 1 : 0)
      rows.push(cols)
    }
    const mask = tensor(rows, { dtype, device, axes: [axis.token, axis.token] }) as Tensor
    if (!isCapturedGraphTensor(mask)) {
      causal_mask_cache.set(cacheKey, mask)
    }
    return mask
  }

  function apply_causal_mask(scores: Tensor, value: number = -1e9): Tensor {
    if (scores.ndim < 2) {
      throw new AffonError('invalid_shape', 'apply_causal_mask expects a tensor with at least 2 dimensions')
    }
    const seqLen = scores.shape[scores.ndim - 1]
    return masked_fill(scores, causal_mask(seqLen as number, {
      dtype: dtypeOf(scores),
      device: scores.device,
    }), value)
  }

  function sinusoidal_encoding(
    length: number,
    dim: number,
    opts?: SinusoidalEncodingOptions,
  ): Tensor {
    if (!Number.isInteger(length) || length <= 0) {
      throw new AffonError('invalid_arg', 'sinusoidal_encoding length must be a positive integer')
    }
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new AffonError('invalid_arg', 'sinusoidal_encoding dim must be a positive integer')
    }
    const rows: number[][] = []
    for (let pos = 0; pos < length; pos++) {
      const row: number[] = []
      for (let i = 0; i < dim; i++) {
        const exponent = (2 * Math.floor(i / 2)) / dim
        const angle = pos / Math.pow(10000, exponent)
        row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle))
      }
      rows.push(row)
    }
    return tensor(rows, { dtype: opts?.dtype ?? 'f32', device: opts?.device, axes: [axis.token, axis.feature] }) as Tensor
  }

  function position_ids(length: number): Tensor {
    if (!Number.isInteger(length) || length <= 0) {
      throw new AffonError('invalid_arg', 'position_ids length must be a positive integer')
    }
    const ids: number[] = []
    for (let i = 0; i < length; i++) ids.push(i)
    return tensor(ids, { dtype: 'f32', axes: [axis.token] }) as Tensor
  }

  function sameShape(a: Tensor, b: Tensor): boolean {
    if (a.ndim !== b.ndim) return false
    for (let i = 0; i < a.ndim; i++) {
      if (a.shape[i] !== b.shape[i]) return false
    }
    return true
  }

  function assertLossShape(name: string, prediction: Tensor, target: Tensor): void {
    if (!sameShape(prediction, target)) {
      throw new AffonError('shape_mismatch', name + ' expects prediction and target to have the same shape after alignment')
    }
  }

  function assertBinaryTargets(name: string, target: Tensor): void {
    if (!allBinary(target)) {
      throw new AffonError('invalid_arg', name + ' expects binary targets containing only 0 or 1')
    }
  }

  function assertProbabilityInputs(name: string, prediction: Tensor): void {
    if (!allInRange(prediction, 0, 1)) {
      throw new AffonError('invalid_arg', name + ' expects probability inputs in the range [0, 1]')
    }
  }

  function assertOneHotTargets(name: string, logits: Tensor, targets: Tensor): void {
    if (logits.ndim !== 2 || targets.ndim !== 2) {
      throw new AffonError('invalid_shape', name + ' expects logits and targets shaped [batch, classes]')
    }
    if (!sameShape(logits, targets)) {
      throw new AffonError('shape_mismatch', name + ' expects logits and one-hot targets to have the same shape')
    }
    const eps = 1e-6
    if (!allInRange(targets, -eps, 1 + eps)) {
      throw new AffonError('invalid_arg', name + ' expects one-hot targets containing only 0 or 1')
    }
    if (!rowsAreOneHot(targets, eps)) {
      throw new AffonError('invalid_arg', name + ' expects each one-hot target row to contain exactly one 1')
    }
  }

  function assertCrossEntropyIndexedShape(name: string, logits: Tensor, targets: Tensor, axis: number): void {
    if (axis !== 1) {
      throw new AffonError('invalid_arg', name + ' currently supports indexed targets with axis=1')
    }
    if (logits.ndim !== 2 || targets.ndim !== 1) {
      throw new AffonError('invalid_shape', name + ' indexed targets expect logits [batch, classes] and targets [batch]')
    }
    if (logits.shape[0] !== targets.shape[0]) {
      throw new AffonError('shape_mismatch', name + ' indexed target batch dimension must match logits')
    }
  }

  function looksLikeIndexedCrossEntropyTargets(logits: Tensor, targets: Tensor, axis: number): boolean {
    return axis === 1 && logits.ndim === 2 && targets.ndim === 1 && logits.shape[0] === targets.shape[0]
  }

  function denseCrossEntropyLoss(logits: Tensor, targets: Tensor): Tensor {
    assertOneHotTargets('CrossEntropyLoss', logits, targets)
    const max_val = max(logits, 1, true)
    const shifted = sub(logits, max_val)
    const exp_shifted = exp(shifted)
    const sum_exp = sum(exp_shifted, 1, true)
    const log_sum_exp = log(sum_exp)
    const log_softmax = sub(shifted, log_sum_exp)
    const nll = mul(targets, log_softmax)
    const per_example = sum(nll, 1, false)
    return mean(neg(per_example))
  }

  function fillUniform_(tensor: Tensor, lower: number, upper: number): Tensor {
    const tmp = add(
      mul(randLike(tensor.shape, tensor, tensor.axes), scalarLike(upper - lower, tensor)),
      scalarLike(lower, tensor),
    )
    copy(tensor, tmp)
    return tensor
  }

  function fillNormal_(tensor: Tensor, std: number): Tensor {
    const tmp = mul(randnLike(tensor.shape, tensor, tensor.axes), scalarLike(std, tensor))
    copy(tensor, tmp)
    return tensor
  }

  function fillScalar_(tensor: Tensor, value: number): Tensor {
    const tmp = mul(onesLike(tensor.shape, tensor, tensor.axes), scalarLike(value, tensor))
    copy(tensor, tmp)
    return tensor
  }

  function trainableZeros(shape: number[], dtype: TensorDType, axes?: readonly AxisName[]): Tensor {
    return computeParameter(shape, { dtype, axes }).zeros()
  }

  function trainableOnes(shape: number[], dtype: TensorDType, axes?: readonly AxisName[]): Tensor {
    return computeParameter(shape, { dtype, axes }).ones()
  }

  const nn = {} as {
    module_list: {
      (modules: Module[]): ModuleList
      (length: number, make: (index: number) => Module): ModuleList
    }
    parameter: {
      <S extends number[] = number[], D extends 'f32' | 'f64' = 'f64'>(tensor: Tensor): Tensor
      <S extends number[] = number[], D extends 'f32' | 'f64' = 'f64'>(name: string, tensor: Tensor): Tensor
    }
    init: {
      xavier_uniform: (tensor: Tensor) => Tensor
      xavier_normal: (tensor: Tensor) => Tensor
      kaiming_uniform: (tensor: Tensor) => Tensor
      kaiming_normal: (tensor: Tensor) => Tensor
      zeros: (tensor: Tensor) => Tensor
      ones: (tensor: Tensor) => Tensor
    }
    Linear: (in_features: number, out_features: number) => LinearModule
    Embedding: (num_embeddings: number, embedding_dim: number, opts?: EmbeddingOptions) => EmbeddingModule
    SimpleRNN: (input_size: number, hidden_size: number, opts?: { num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu' }) => SimpleRNNModule
    RNN: (input_size: number, hidden_size: number, opts?: { num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu'; batch_first?: boolean }) => RNNModule
    LSTM: (input_size: number, hidden_size: number, opts?: { num_layers?: number; bias?: boolean; batch_first?: boolean }) => LSTMModule
    relu: (x: Tensor) => Tensor
    gelu: (x: Tensor) => Tensor
    sigmoid: (x: Tensor) => Tensor
    Sequential: (...layers: (Module | ((x: Tensor) => Tensor))[]) => SequentialModule
    MSELoss: (opts?: { reduction?: string }) => (prediction: Tensor, target: Tensor) => Tensor
    BCELoss: (opts?: { reduction?: string }) => (prediction: Tensor, target: Tensor) => Tensor
    BCEWithLogitsLoss: (opts?: { reduction?: string }) => (logits: Tensor, targets: Tensor) => Tensor
    CrossEntropyLoss: (opts?: CrossEntropyLossOptions) => (logits: Tensor, targets: Tensor) => Tensor
    diagnostics: {
      configure: (config: DiagnosticConfig) => void
      get_config: () => Required<DiagnosticConfig>
      assert: (kind: DiagnosticKind, opts: DiagnosticAssertOptions) => void
    }
    causal_mask: (length: number, opts?: { dtype?: TensorDType; device?: 'cpu' | 'metal' }) => Tensor
    apply_causal_mask: (scores: Tensor, value?: number) => Tensor
    sinusoidal_encoding: (length: number, dim: number, opts?: SinusoidalEncodingOptions) => Tensor
    position_ids: (length: number) => Tensor
    Dropout: (p?: number) => DropoutModule
    BatchNorm: (num_features: number, opts?: { momentum?: number; eps?: number }) => BatchNormModule
    LayerNorm: (normalized_shape: number, opts?: { eps?: number }) => LayerNormModule
  }

  const GELU_COEFF = Math.sqrt(2.0 / Math.PI)
  const GELU_CUBIC = 0.044715
  const diagnosticConfig: Required<DiagnosticConfig> = {
    mode: 'off',
    include: [],
    exclude: [],
  }

  nn.diagnostics = {
    configure(config: DiagnosticConfig): void {
      if (config.mode !== undefined) {
        if (config.mode !== 'off' && config.mode !== 'error' && config.mode !== 'warn') {
          throw new AffonError('invalid_arg', 'nn.diagnostics.configure mode must be off, error, or warn')
        }
        diagnosticConfig.mode = config.mode
      }
      if (config.include !== undefined) diagnosticConfig.include = config.include.slice()
      if (config.exclude !== undefined) diagnosticConfig.exclude = config.exclude.slice()
    },
    get_config(): Required<DiagnosticConfig> {
      return {
        mode: diagnosticConfig.mode,
        include: diagnosticConfig.include.slice(),
        exclude: diagnosticConfig.exclude.slice(),
      }
    },
    assert(kind: DiagnosticKind, opts: DiagnosticAssertOptions): void {
      if (kind !== 'finite') {
        throw new AffonError('invalid_arg', `nn.diagnostics.assert unsupported kind: ${kind}`)
      }
      if (!diagnosticEnabled(diagnosticConfig, kind, opts.path)) return
      if (!opts.value) {
        throw new AffonError('invalid_arg', 'nn.diagnostics.assert finite requires value')
      }
      if (isCapturedGraphTensor(opts.value)) return
      const summary = finite_summary(opts.value)
      if (summary.ok) return
      const message = `${opts.path} produced a non-finite tensor: shape=${JSON.stringify(opts.value.shape)} first_bad_flat_index=${summary.first_bad_flat_index} first_bad_value=${describeDiagnosticValue(summary.first_bad_value)}`
      if (diagnosticConfig.mode === 'warn') {
        console.warn(`Affon diagnostics warning: ${message}`)
        return
      }
      throw new AffonError('grad_error', message)
    },
  }

  nn.module_list = function(modulesOrLength: Module[] | number, make?: (index: number) => Module): ModuleList {
    let modules: Module[]
    if (typeof modulesOrLength === 'number') {
      if (!make) throw new AffonError('invalid_arg', 'nn.module_list(length, make) requires a factory function')
      modules = []
      for (let i = 0; i < modulesOrLength; i++) {
        const module = make(i)
        assertModuleListEntry(module, 'nn.module_list factory')
        modules.push(module)
      }
    } else {
      modules = modulesOrLength.slice()
      for (let i = 0; i < modules.length; i++) {
        assertModuleListEntry(modules[i], 'nn.module_list')
      }
    }

    const list = modules as unknown as ModuleList
    const owner = computeModule(modules as unknown as any, (_state, x: Tensor): Tensor => x)
    const collectListParameters = (): ParameterCollection => {
      const params: Tensor[] = []
      const named: Array<[string, Tensor]> = []
      for (let i = 0; i < modules.length; i++) {
        const collection = (modules[i] as any).parameters as ParameterCollection | undefined
        if (!collection) continue
        for (let j = 0; j < collection.length; j++) params.push(collection[j])
        const entries = typeof collection.named === 'function'
          ? collection.named()
          : collection.map((param, index) => [String(index), param] as const)
        for (let j = 0; j < entries.length; j++) {
          named.push([`${i}.${entries[j][0]}`, entries[j][1]])
        }
      }
      const frozen = params.slice() as unknown as ParameterCollection
      ;(frozen as any).named = function(): ReadonlyArray<ParamEntry> {
        return Object.freeze(named.map(([name, param]) => Object.freeze([name, param] as [string, Tensor])))
      }
      return Object.freeze(frozen)
    }
    Object.defineProperty(list, 'parameters', {
      get() {
        return collectListParameters()
      },
      enumerable: true,
    })
    ;(list as any).mode = function(nextMode?: 'train' | 'eval'): ModuleList | 'train' | 'eval' {
      return owner.mode(nextMode as any) as ModuleList | 'train' | 'eval'
    }
    ;(list as any).train = function(): void {
      owner.train()
    }
    ;(list as any).eval = function(): void {
      owner.eval()
    }
    return list
  }
  nn.init = {
    xavier_uniform(tensor: Tensor): Tensor {
      const { fan_in, fan_out } = fanInOut(tensor)
      const bound = Math.sqrt(6.0 / (fan_in + fan_out))
      return fillUniform_(tensor, -bound, bound)
    },
    xavier_normal(tensor: Tensor): Tensor {
      const { fan_in, fan_out } = fanInOut(tensor)
      const std = Math.sqrt(2.0 / (fan_in + fan_out))
      return fillNormal_(tensor, std)
    },
    kaiming_uniform(tensor: Tensor): Tensor {
      const { fan_in } = fanInOut(tensor)
      const bound = Math.sqrt(6.0 / fan_in)
      return fillUniform_(tensor, -bound, bound)
    },
    kaiming_normal(tensor: Tensor): Tensor {
      const { fan_in } = fanInOut(tensor)
      const std = Math.sqrt(2.0 / fan_in)
      return fillNormal_(tensor, std)
    },
    zeros(tensor: Tensor): Tensor {
      return fillScalar_(tensor, 0.0)
    },
    ones(tensor: Tensor): Tensor {
      return fillScalar_(tensor, 1.0)
    }
  }

  // ============================================================
  // nn.Embedding
  // ============================================================
  nn.Embedding = function(num_embeddings: number, embedding_dim: number, opts?: EmbeddingOptions): EmbeddingModule {
    if (!Number.isInteger(num_embeddings) || num_embeddings <= 0) {
      throw new AffonError('invalid_arg', 'Embedding num_embeddings must be a positive integer')
    }
    if (!Number.isInteger(embedding_dim) || embedding_dim <= 0) {
      throw new AffonError('invalid_arg', 'Embedding embedding_dim must be a positive integer')
    }

    const dtype = opts?.dtype ?? defaultModuleDType()
    const weight = computeParameter([num_embeddings, embedding_dim], { dtype, axes: opts?.axes ?? [axis.vocab, axis.feature] })
    nn.init.kaiming_uniform(weight)

    function encodeFlat(weightTensor: Tensor, ids: Tensor): Tensor {
      if (ids.ndim !== 1) {
        throw new AffonError('invalid_shape', 'Embedding internal encodeFlat expects input shaped [tokens]')
      }
      return index_select(weightTensor, 0, ids)
    }

    const state = { weight }
    return computeModule(state, (state, ids: Tensor): Tensor => {
      if (ids.ndim === 1) {
        return encodeFlat(state.weight, ids)
      }
      if (ids.ndim === 2) {
        const batch = ids.shape[0] as number
        const tokens = ids.shape[1] as number
        const flatIds = reshape(contiguous(ids), [batch * tokens], { axes: [axis.token] })
        const flatOut = encodeFlat(state.weight, flatIds)
        return reshape(contiguous(flatOut), [batch, tokens, embedding_dim], { axes: [axis.batch, axis.token, axis.feature] })
      }
      throw new AffonError('invalid_shape', 'Embedding expects input shaped [tokens] or [batch, tokens]')
    }) as EmbeddingModule
  }

  // ============================================================
  // nn.Linear
  // ============================================================
  nn.Linear = function(in_features: number, out_features: number, opts?: { dtype?: TensorDType }): LinearModule {
    if (!Number.isInteger(in_features) || in_features <= 0) {
      throw new AffonError('invalid_arg', 'Linear in_features must be a positive integer')
    }
    if (!Number.isInteger(out_features) || out_features <= 0) {
      throw new AffonError('invalid_arg', 'Linear out_features must be a positive integer')
    }
    const dtype = opts?.dtype ?? defaultModuleDType()
    const weight = computeParameter([in_features, out_features], { dtype, axes: [axis.feature, axis.hidden] })
    const bias = trainableZeros([1, out_features], dtype, [axis.batch, axis.hidden])

    nn.init.kaiming_uniform(weight)

    const state = { weight, bias }
    return computeModule(state, (state, x: Tensor): Tensor => {
      const out = matmul(x, state.weight)
      return add(out, state.bias)
    }) as LinearModule
  }

  // ============================================================
  // nn.SimpleRNN
  // ============================================================
  nn.SimpleRNN = function(
    input_size: number,
    hidden_size: number,
    opts?: { num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu' }
  ): SimpleRNNModule {
    if (!Number.isInteger(input_size) || input_size <= 0) {
      throw new AffonError('invalid_arg', 'SimpleRNN input_size must be a positive integer')
    }
    if (!Number.isInteger(hidden_size) || hidden_size <= 0) {
      throw new AffonError('invalid_arg', 'SimpleRNN hidden_size must be a positive integer')
    }
    const num_layers = (opts && opts.num_layers !== undefined) ? opts.num_layers : 1
    const use_bias = (opts && opts.bias !== undefined) ? opts.bias : true
    const nonlinearity = (opts && opts.nonlinearity !== undefined) ? opts.nonlinearity : 'tanh'

    if (num_layers <= 0 || !Number.isInteger(num_layers)) {
      throw new AffonError('invalid_arg', 'SimpleRNN num_layers must be a positive integer')
    }
    if (nonlinearity !== 'tanh' && nonlinearity !== 'relu') {
      throw new AffonError('invalid_arg', "SimpleRNN nonlinearity must be 'tanh' or 'relu'")
    }

    const activation = nonlinearity === 'relu'
      ? (x: Tensor): Tensor => relu(x)
      : (x: Tensor): Tensor => tanh(x)

    function makeLayer(layerInputSize: number) {
      const weight_ih = computeParameter([layerInputSize, hidden_size], { dtype: defaultModuleDType(), axes: [axis.feature, axis.hidden] })
      const weight_hh = computeParameter([hidden_size, hidden_size], { dtype: defaultModuleDType(), axes: [axis.hidden, axis.hidden] })
      const bias = use_bias ? trainableZeros([1, hidden_size], defaultModuleDType(), [axis.batch, axis.hidden]) : null

      nn.init.kaiming_uniform(weight_ih)
      nn.init.kaiming_uniform(weight_hh)

      let hidden_state: Tensor | null = null

      const state = {
        weight_ih,
        weight_hh,
        bias,
        hidden_state() {
          return hidden_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor): Tensor => {
        if (X.ndim !== 2) {
          throw new AffonError('invalid_shape', 'SimpleRNN layer expects input shaped [seq_len, input_size]')
        }
        if (X.shape[1] !== layerInputSize) {
          throw new AffonError('shape_mismatch', 'SimpleRNN layer input_size mismatch: expected ' + layerInputSize + ', got ' + X.shape[1])
        }
        if (h0 && (h0.ndim !== 2 || h0.shape[0] !== 1 || h0.shape[1] !== hidden_size)) {
          throw new AffonError('invalid_shape', 'SimpleRNN layer h0 must have shape [1, hidden_size]')
        }

        const seqLen = X.shape[0] as number
        let h = h0 ?? zerosLike([1, hidden_size], X, [axis.batch, axis.hidden])
        const outputs: Tensor[] = []

        for (let i = 0; i < seqLen; i++) {
          const x = index_select(X, 0, axis0Index(i))
          const inputTerm = matmul(x, state.weight_ih)
          const hiddenTerm = matmul(h, state.weight_hh)
          let next = add(inputTerm, hiddenTerm)
          if (state.bias !== null) next = add(next, state.bias)
          h = activation(next)
          outputs.push(h)
        }

        hidden_state = h
        return squeeze(stack(outputs, 0), 1)
      }) as Module
    }

    const layers = nn.module_list(num_layers, (index: number) =>
      makeLayer(index === 0 ? input_size : hidden_size)
    )
    let hidden_state: Tensor | null = null

    const state = {
      layers,
      hidden_state() {
        return hidden_state
      },
    }
    return computeModule(state, (state, X: Tensor, h0?: Tensor): Tensor => {
      if (X.ndim !== 2) {
        throw new AffonError('invalid_shape', 'SimpleRNN expects input shaped [seq_len, input_size]')
      }
      if (X.shape[1] !== input_size) {
        throw new AffonError('shape_mismatch', 'SimpleRNN input_size mismatch: expected ' + input_size + ', got ' + X.shape[1])
      }
      if (h0 && (h0.ndim !== 2 || h0.shape[0] !== num_layers || h0.shape[1] !== hidden_size)) {
        throw new AffonError('invalid_shape', 'SimpleRNN h0 must have shape [num_layers, hidden_size]')
      }

      let output = X
      const finalStates: Tensor[] = []

      for (let i = 0; i < num_layers; i++) {
        const layer = state.layers[i] as any
        const h0_i = h0 ? index_select(h0, 0, axis0Index(i)) : undefined
        output = h0_i ? layer(output, h0_i) : layer(output)
        const layer_hidden = layer.hidden_state()
        if (layer_hidden === null) {
          throw new AffonError('internal', 'SimpleRNN internal error: layer hidden state missing after forward')
        }
        finalStates.push(layer_hidden)
      }

      hidden_state = squeeze(stack(finalStates, 0), 1)
      return output
    }) as SimpleRNNModule
  }

  // ============================================================
  // nn.RNN
  // ============================================================
  nn.RNN = function(
    input_size: number,
    hidden_size: number,
    opts?: { num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu'; batch_first?: boolean }
  ): RNNModule {
    if (!Number.isInteger(input_size) || input_size <= 0) {
      throw new AffonError('invalid_arg', 'RNN input_size must be a positive integer')
    }
    if (!Number.isInteger(hidden_size) || hidden_size <= 0) {
      throw new AffonError('invalid_arg', 'RNN hidden_size must be a positive integer')
    }
    const num_layers = (opts && opts.num_layers !== undefined) ? opts.num_layers : 1
    const use_bias = (opts && opts.bias !== undefined) ? opts.bias : true
    const nonlinearity = (opts && opts.nonlinearity !== undefined) ? opts.nonlinearity : 'tanh'
    const batch_first = (opts && opts.batch_first !== undefined) ? opts.batch_first : false

    if (num_layers <= 0 || !Number.isInteger(num_layers)) {
      throw new AffonError('invalid_arg', 'RNN num_layers must be a positive integer')
    }
    if (nonlinearity !== 'tanh' && nonlinearity !== 'relu') {
      throw new AffonError('invalid_arg', "RNN nonlinearity must be 'tanh' or 'relu'")
    }

    const activation = nonlinearity === 'relu'
      ? (x: Tensor): Tensor => relu(x)
      : (x: Tensor): Tensor => tanh(x)

    function makeSequenceFirstLayer(layerInputSize: number) {
      const weight_ih = computeParameter([layerInputSize, hidden_size], { dtype: defaultModuleDType(), axes: [axis.feature, axis.hidden] })
      const weight_hh = computeParameter([hidden_size, hidden_size], { dtype: defaultModuleDType(), axes: [axis.hidden, axis.hidden] })
      const bias = use_bias ? trainableZeros([1, hidden_size], defaultModuleDType(), [axis.batch, axis.hidden]) : null

      nn.init.kaiming_uniform(weight_ih)
      nn.init.kaiming_uniform(weight_hh)

      let hidden_state: Tensor | null = null

      const state = {
        weight_ih,
        weight_hh,
        bias,
        hidden_state() {
          return hidden_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor): Tensor => {
        if (X.ndim !== 3) {
          throw new AffonError('invalid_shape', 'RNN layer expects input shaped [seq_len, batch, input_size]')
        }
        if (X.shape[2] !== layerInputSize) {
          throw new AffonError('shape_mismatch', 'RNN layer input_size mismatch: expected ' + layerInputSize + ', got ' + X.shape[2])
        }
        const batch = X.shape[1]
        if (h0 && (h0.ndim !== 2 || h0.shape[0] !== batch || h0.shape[1] !== hidden_size)) {
          throw new AffonError('invalid_shape', 'RNN layer h0 must have shape [batch, hidden_size]')
        }

        const seqLen = X.shape[0] as number
        let h = h0 ?? zerosLike([batch, hidden_size], X, [axis.batch, axis.hidden])
        const outputs: Tensor[] = []

        for (let i = 0; i < seqLen; i++) {
          const x = squeeze(index_select(X, 0, axis0Index(i)), 0)
          const inputTerm = matmul(x, state.weight_ih)
          const hiddenTerm = matmul(h, state.weight_hh)
          let next = add(inputTerm, hiddenTerm)
          if (state.bias !== null) next = add(next, state.bias)
          h = activation(next)
          outputs.push(h)
        }

        hidden_state = h
        return stack(outputs, 0)
      }) as Module
    }

    function makeBatchFirstLayer(layerInputSize: number) {
      const weight_ih = computeParameter([layerInputSize, hidden_size], { dtype: defaultModuleDType(), axes: [axis.feature, axis.hidden] })
      const weight_hh = computeParameter([hidden_size, hidden_size], { dtype: defaultModuleDType(), axes: [axis.hidden, axis.hidden] })
      const bias = use_bias ? trainableZeros([1, hidden_size], defaultModuleDType(), [axis.batch, axis.hidden]) : null

      nn.init.kaiming_uniform(weight_ih)
      nn.init.kaiming_uniform(weight_hh)

      let hidden_state: Tensor | null = null

      const state = {
        weight_ih,
        weight_hh,
        bias,
        hidden_state() {
          return hidden_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor): Tensor => {
        if (X.ndim !== 3) {
          throw new AffonError('invalid_shape', 'RNN layer expects input shaped [batch, seq_len, input_size] when batch_first=true')
        }
        if (X.shape[2] !== layerInputSize) {
          throw new AffonError('shape_mismatch', 'RNN layer input_size mismatch: expected ' + layerInputSize + ', got ' + X.shape[2])
        }
        const batch = X.shape[0]
        if (h0 && (h0.ndim !== 2 || h0.shape[0] !== batch || h0.shape[1] !== hidden_size)) {
          throw new AffonError('invalid_shape', 'RNN layer h0 must have shape [batch, hidden_size]')
        }

        const seqLen = X.shape[1] as number
        let h = h0 ?? zerosLike([batch, hidden_size], X, [axis.batch, axis.hidden])
        const outputs: Tensor[] = []

        for (let i = 0; i < seqLen; i++) {
          const x = contiguous(squeeze(X.slice([':', `${i}:${i + 1}`, ':']), 1))
          const inputTerm = matmul(x, state.weight_ih)
          const hiddenTerm = matmul(h, state.weight_hh)
          let next = add(inputTerm, hiddenTerm)
          if (state.bias !== null) next = add(next, state.bias)
          h = activation(next)
          outputs.push(h)
        }

        hidden_state = h
        return stack(outputs, 1)
      }) as Module
    }

    function makeRnnSequenceFirst(): RNNModule {
      const layers = nn.module_list(num_layers, (index: number) =>
        makeSequenceFirstLayer(index === 0 ? input_size : hidden_size)
      )
      let hidden_state: Tensor | null = null

      const state = {
        layers,
        hidden_state() {
          return hidden_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor): Tensor => {
          if (X.ndim !== 3) {
            throw new AffonError('invalid_shape', 'RNN expects input shaped [seq_len, batch, input_size]')
          }
          if (X.shape[2] !== input_size) {
            throw new AffonError('shape_mismatch', 'RNN input_size mismatch: expected ' + input_size + ', got ' + X.shape[2])
          }
          const batch = X.shape[1]
          if (h0 && (h0.ndim !== 3 || h0.shape[0] !== num_layers || h0.shape[1] !== batch || h0.shape[2] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'RNN h0 must have shape [num_layers, batch, hidden_size]')
          }

          let output = X
          const finalStates: Tensor[] = []

          for (let i = 0; i < num_layers; i++) {
            const layer = state.layers[i] as any
            const h0_i = h0 ? squeeze(index_select(h0, 0, axis0Index(i)), 0) : undefined
            output = h0_i ? layer(output, h0_i) : layer(output)
            const layer_hidden = layer.hidden_state()
            if (layer_hidden === null) {
              throw new AffonError('internal', 'RNN internal error: layer hidden state missing after forward')
            }
            finalStates.push(layer_hidden)
          }

          hidden_state = stack(finalStates, 0)
          return output
      }) as RNNModule
    }

    function makeRnnBatchFirst(): RNNModule {
      const layers = nn.module_list(num_layers, (index: number) =>
        makeBatchFirstLayer(index === 0 ? input_size : hidden_size)
      )
      let hidden_state: Tensor | null = null

      const state = {
        layers,
        hidden_state() {
          return hidden_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor): Tensor => {
          if (X.ndim !== 3) {
            throw new AffonError('invalid_shape', 'RNN expects input shaped [batch, seq_len, input_size] when batch_first=true')
          }
          if (X.shape[2] !== input_size) {
            throw new AffonError('shape_mismatch', 'RNN input_size mismatch: expected ' + input_size + ', got ' + X.shape[2])
          }
          const batch = X.shape[0]
          if (h0 && (h0.ndim !== 3 || h0.shape[0] !== num_layers || h0.shape[1] !== batch || h0.shape[2] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'RNN h0 must have shape [num_layers, batch, hidden_size]')
          }

          let output = X
          const finalStates: Tensor[] = []

          for (let i = 0; i < num_layers; i++) {
            const layer = state.layers[i] as any
            const h0_i = h0 ? squeeze(index_select(h0, 0, axis0Index(i)), 0) : undefined
            output = h0_i ? layer(output, h0_i) : layer(output)
            const layer_hidden = layer.hidden_state()
            if (layer_hidden === null) {
              throw new AffonError('internal', 'RNN internal error: layer hidden state missing after forward')
            }
            finalStates.push(layer_hidden)
          }

          hidden_state = stack(finalStates, 0)
          return output
      }) as RNNModule
    }

    return batch_first ? makeRnnBatchFirst() : makeRnnSequenceFirst()
  }

  // ============================================================
  // nn.LSTM
  // ============================================================
  nn.LSTM = function(
    input_size: number,
    hidden_size: number,
    opts?: { num_layers?: number; bias?: boolean; batch_first?: boolean }
  ): LSTMModule {
    if (!Number.isInteger(input_size) || input_size <= 0) {
      throw new AffonError('invalid_arg', 'LSTM input_size must be a positive integer')
    }
    if (!Number.isInteger(hidden_size) || hidden_size <= 0) {
      throw new AffonError('invalid_arg', 'LSTM hidden_size must be a positive integer')
    }
    const num_layers = (opts && opts.num_layers !== undefined) ? opts.num_layers : 1
    const use_bias = (opts && opts.bias !== undefined) ? opts.bias : true
    const batch_first = (opts && opts.batch_first !== undefined) ? opts.batch_first : false

    if (num_layers <= 0 || !Number.isInteger(num_layers)) {
      throw new AffonError('invalid_arg', 'LSTM num_layers must be a positive integer')
    }

    function sliceGate(gates: Tensor, gateIndex: number): Tensor {
      const start = gateIndex * hidden_size
      const end = start + hidden_size
      return contiguous(gates.slice([':', `${start}:${end}`]))
    }

    function makeSequenceFirstLayer(layerInputSize: number) {
      const weight_ih = computeParameter([layerInputSize, hidden_size * 4], { dtype: defaultModuleDType(), axes: [axis.feature, axis.gate] })
      const weight_hh = computeParameter([hidden_size, hidden_size * 4], { dtype: defaultModuleDType(), axes: [axis.hidden, axis.gate] })
      const bias = use_bias ? trainableZeros([1, hidden_size * 4], defaultModuleDType(), [axis.batch, axis.gate]) : null

      nn.init.kaiming_uniform(weight_ih)
      nn.init.kaiming_uniform(weight_hh)

      let hidden_state: Tensor | null = null
      let cell_state: Tensor | null = null

      const state = {
        weight_ih,
        weight_hh,
        bias,
        hidden_state() {
          return hidden_state
        },
        cell_state() {
          return cell_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor, c0?: Tensor): Tensor => {
          if (X.ndim !== 3) {
            throw new AffonError('invalid_shape', 'LSTM layer expects input shaped [seq_len, batch, input_size]')
          }
          if (X.shape[2] !== layerInputSize) {
            throw new AffonError('shape_mismatch', 'LSTM layer input_size mismatch: expected ' + layerInputSize + ', got ' + X.shape[2])
          }
          const batch = X.shape[1]
          if (h0 && (h0.ndim !== 2 || h0.shape[0] !== batch || h0.shape[1] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM layer h0 must have shape [batch, hidden_size]')
          }
          if (c0 && (c0.ndim !== 2 || c0.shape[0] !== batch || c0.shape[1] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM layer c0 must have shape [batch, hidden_size]')
          }

          const seqLen = X.shape[0] as number
          let h = h0 ?? zerosLike([batch, hidden_size], X, [axis.batch, axis.hidden])
          let c = c0 ?? zerosLike([batch, hidden_size], X, [axis.batch, axis.hidden])
          const outputs: Tensor[] = []

          // Gate order: input, forget, candidate, output.
          for (let i = 0; i < seqLen; i++) {
            const x = squeeze(index_select(X, 0, axis0Index(i)), 0)
            const inputTerm = matmul(x, state.weight_ih)
            const hiddenTerm = matmul(h, state.weight_hh)
            let gates = add(inputTerm, hiddenTerm)
            if (state.bias !== null) gates = add(gates, state.bias)

            const inputGate = sigmoid(sliceGate(gates, 0))
            const forgetGate = sigmoid(sliceGate(gates, 1))
            const candidate = tanh(sliceGate(gates, 2))
            const outputGate = sigmoid(sliceGate(gates, 3))

            c = add(mul(forgetGate, c), mul(inputGate, candidate))
            h = mul(outputGate, tanh(c))
            outputs.push(h)
          }

          hidden_state = h
          cell_state = c
          return stack(outputs, 0)
      }) as Module
    }

    function makeBatchFirstLayer(layerInputSize: number) {
      const weight_ih = computeParameter([layerInputSize, hidden_size * 4], { dtype: defaultModuleDType(), axes: [axis.feature, axis.gate] })
      const weight_hh = computeParameter([hidden_size, hidden_size * 4], { dtype: defaultModuleDType(), axes: [axis.hidden, axis.gate] })
      const bias = use_bias ? trainableZeros([1, hidden_size * 4], defaultModuleDType(), [axis.batch, axis.gate]) : null

      nn.init.kaiming_uniform(weight_ih)
      nn.init.kaiming_uniform(weight_hh)

      let hidden_state: Tensor | null = null
      let cell_state: Tensor | null = null

      const state = {
        weight_ih,
        weight_hh,
        bias,
        hidden_state() {
          return hidden_state
        },
        cell_state() {
          return cell_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor, c0?: Tensor): Tensor => {
          if (X.ndim !== 3) {
            throw new AffonError('invalid_shape', 'LSTM layer expects input shaped [batch, seq_len, input_size] when batch_first=true')
          }
          if (X.shape[2] !== layerInputSize) {
            throw new AffonError('shape_mismatch', 'LSTM layer input_size mismatch: expected ' + layerInputSize + ', got ' + X.shape[2])
          }
          const batch = X.shape[0]
          if (h0 && (h0.ndim !== 2 || h0.shape[0] !== batch || h0.shape[1] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM layer h0 must have shape [batch, hidden_size]')
          }
          if (c0 && (c0.ndim !== 2 || c0.shape[0] !== batch || c0.shape[1] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM layer c0 must have shape [batch, hidden_size]')
          }

          const seqLen = X.shape[1] as number
          let h = h0 ?? zerosLike([batch, hidden_size], X, [axis.batch, axis.hidden])
          let c = c0 ?? zerosLike([batch, hidden_size], X, [axis.batch, axis.hidden])
          const outputs: Tensor[] = []

          for (let i = 0; i < seqLen; i++) {
            // Batch-first slices across the middle axis are not BLAS-friendly views here,
            // so we materialize a contiguous [batch, input_size] buffer before matmul.
            const x = contiguous(squeeze(X.slice([':', `${i}:${i + 1}`, ':']), 1))
            const inputTerm = matmul(x, state.weight_ih)
            const hiddenTerm = matmul(h, state.weight_hh)
            let gates = add(inputTerm, hiddenTerm)
            if (state.bias !== null) gates = add(gates, state.bias)

            const inputGate = sigmoid(sliceGate(gates, 0))
            const forgetGate = sigmoid(sliceGate(gates, 1))
            const candidate = tanh(sliceGate(gates, 2))
            const outputGate = sigmoid(sliceGate(gates, 3))

            c = add(mul(forgetGate, c), mul(inputGate, candidate))
            h = mul(outputGate, tanh(c))
            outputs.push(h)
          }

          hidden_state = h
          cell_state = c
          return stack(outputs, 1)
      }) as Module
    }

    function makeLstmSequenceFirst(): LSTMModule {
      const layers = nn.module_list(num_layers, (index: number) =>
        makeSequenceFirstLayer(index === 0 ? input_size : hidden_size)
      )
      let hidden_state: Tensor | null = null
      let cell_state: Tensor | null = null

      const state = {
        layers,
        hidden_state() {
          return hidden_state
        },
        cell_state() {
          return cell_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor, c0?: Tensor): Tensor => {
          if (X.ndim !== 3) {
            throw new AffonError('invalid_shape', 'LSTM expects input shaped [seq_len, batch, input_size]')
          }
          if (X.shape[2] !== input_size) {
            throw new AffonError('shape_mismatch', 'LSTM input_size mismatch: expected ' + input_size + ', got ' + X.shape[2])
          }
          const batch = X.shape[1]
          if (h0 && (h0.ndim !== 3 || h0.shape[0] !== num_layers || h0.shape[1] !== batch || h0.shape[2] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM h0 must have shape [num_layers, batch, hidden_size]')
          }
          if (c0 && (c0.ndim !== 3 || c0.shape[0] !== num_layers || c0.shape[1] !== batch || c0.shape[2] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM c0 must have shape [num_layers, batch, hidden_size]')
          }

          let output = X
          const finalHiddenStates: Tensor[] = []
          const finalCellStates: Tensor[] = []

          for (let i = 0; i < num_layers; i++) {
            const layer = state.layers[i] as any
            const h0_i = h0 ? squeeze(index_select(h0, 0, axis0Index(i)), 0) : undefined
            const c0_i = c0 ? squeeze(index_select(c0, 0, axis0Index(i)), 0) : undefined
            output = h0_i && c0_i ? layer(output, h0_i, c0_i) : layer(output)
            const layer_hidden = layer.hidden_state()
            const layer_cell = layer.cell_state()
            if (layer_hidden === null || layer_cell === null) {
              throw new AffonError('internal', 'LSTM internal error: layer state missing after forward')
            }
            finalHiddenStates.push(layer_hidden)
            finalCellStates.push(layer_cell)
          }

          hidden_state = stack(finalHiddenStates, 0)
          cell_state = stack(finalCellStates, 0)
          return output
      }) as LSTMModule
    }

    function makeLstmBatchFirst(): LSTMModule {
      const layers = nn.module_list(num_layers, (index: number) =>
        makeBatchFirstLayer(index === 0 ? input_size : hidden_size)
      )
      let hidden_state: Tensor | null = null
      let cell_state: Tensor | null = null

      const state = {
        layers,
        hidden_state() {
          return hidden_state
        },
        cell_state() {
          return cell_state
        },
      }
      return computeModule(state, (state, X: Tensor, h0?: Tensor, c0?: Tensor): Tensor => {
          if (X.ndim !== 3) {
            throw new AffonError('invalid_shape', 'LSTM expects input shaped [batch, seq_len, input_size] when batch_first=true')
          }
          if (X.shape[2] !== input_size) {
            throw new AffonError('shape_mismatch', 'LSTM input_size mismatch: expected ' + input_size + ', got ' + X.shape[2])
          }
          const batch = X.shape[0]
          if (h0 && (h0.ndim !== 3 || h0.shape[0] !== num_layers || h0.shape[1] !== batch || h0.shape[2] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM h0 must have shape [num_layers, batch, hidden_size]')
          }
          if (c0 && (c0.ndim !== 3 || c0.shape[0] !== num_layers || c0.shape[1] !== batch || c0.shape[2] !== hidden_size)) {
            throw new AffonError('invalid_shape', 'LSTM c0 must have shape [num_layers, batch, hidden_size]')
          }

          let output = X
          const finalHiddenStates: Tensor[] = []
          const finalCellStates: Tensor[] = []

          for (let i = 0; i < num_layers; i++) {
            const layer = state.layers[i] as any
            const h0_i = h0 ? squeeze(index_select(h0, 0, axis0Index(i)), 0) : undefined
            const c0_i = c0 ? squeeze(index_select(c0, 0, axis0Index(i)), 0) : undefined
            output = h0_i && c0_i ? layer(output, h0_i, c0_i) : layer(output)
            const layer_hidden = layer.hidden_state()
            const layer_cell = layer.cell_state()
            if (layer_hidden === null || layer_cell === null) {
              throw new AffonError('internal', 'LSTM internal error: layer state missing after forward')
            }
            finalHiddenStates.push(layer_hidden)
            finalCellStates.push(layer_cell)
          }

          hidden_state = stack(finalHiddenStates, 0)
          cell_state = stack(finalCellStates, 0)
          return output
      }) as LSTMModule
    }

    return batch_first ? makeLstmBatchFirst() : makeLstmSequenceFirst()
  }

  // ============================================================
  // nn.Sequential
  // ============================================================
  nn.Sequential = function(...layers: (Module | ((x: Tensor) => Tensor))[]): SequentialModule {
    const state = { layers }
    return computeModule(state, (state, x: Tensor): Tensor => {
      let out = x
      for (let i = 0; i < state.layers.length; i++) {
        out = state.layers[i](out)
      }
      return out
    }) as SequentialModule
  }

  // Squeeze prediction [N,1] → [N] when target is [N] to avoid broadcasting to [N,N]
  function alignPred(prediction: Tensor, target: Tensor): Tensor {
    if (prediction.ndim === 2 && target.ndim === 1 && prediction.shape[1] === 1) {
      return sum(prediction, 1, false)
    }
    return prediction
  }

  // ============================================================
  // nn.MSELoss
  // ============================================================
  nn.MSELoss = function(_opts?: { reduction?: string }) {
    return function(prediction: Tensor, target: Tensor): Tensor {
      const pred = alignPred(prediction, target)
      assertLossShape('MSELoss', pred, target)
      const diff = sub(pred, target)
      const sq = mul(diff, diff)
      return mean(sq)
    }
  }

  // ============================================================
  // nn.BCELoss
  // ============================================================
  nn.BCELoss = function(_opts?: { reduction?: string }) {
    // Formula: -mean(target * log(pred) + (1 - target) * log(1 - pred)),
    // with PyTorch-style probability clamping near 0 and 1 for stability.
    return function(prediction: Tensor, target: Tensor): Tensor {
      const one = scalarLike(1, target)
      const eps = probabilityClampEpsilon(target)
      const pred = clamp(alignPred(prediction, target), eps, 1 - eps)
      assertLossShape('BCELoss', pred, target)
      assertProbabilityInputs('BCELoss', prediction)
      assertBinaryTargets('BCELoss', target)
      return mean(neg(add(
        mul(target, log(pred)),
        mul(sub(one, target), log(sub(one, pred)))
      )))
    }
  }

  // ============================================================
  // nn.BCEWithLogitsLoss
  // ============================================================
  nn.BCEWithLogitsLoss = function(_opts?: { reduction?: string }) {
    // Stable logits-space BCE:
    // mean(max(x, 0) - x * target + log(1 + exp(-abs(x)))).
    return function(logits: Tensor, targets: Tensor): Tensor {
      const one = scalarLike(1, targets)
      const logit = alignPred(logits, targets)
      assertLossShape('BCEWithLogitsLoss', logit, targets)
      assertBinaryTargets('BCEWithLogitsLoss', targets)
      const relu_term = relu(logit)
      const linear_term = mul(logit, targets)
      const log_term = log(add(one, exp(neg(abs(logit)))))
      return mean(add(sub(relu_term, linear_term), log_term))
    }
  }

  // ============================================================
  // nn.CrossEntropyLoss
  // ============================================================
  nn.CrossEntropyLoss = function(opts?: CrossEntropyLossOptions) {
    const reduction = opts?.reduction ?? 'mean'
    const target = opts?.target ?? 'auto'
    const axis = opts?.axis ?? 1
    if (reduction !== 'mean') {
      throw new AffonError('invalid_arg', 'CrossEntropyLoss currently supports reduction="mean"')
    }
    if (target !== 'auto' && target !== 'index' && target !== 'probability' && target !== 'one_hot') {
      throw new AffonError('invalid_arg', 'CrossEntropyLoss target must be "auto", "index", "probability", or "one_hot"')
    }
    return function(logits: Tensor, targets: Tensor): Tensor {
      if (target === 'index' || (target === 'auto' && looksLikeIndexedCrossEntropyTargets(logits, targets, axis))) {
        assertCrossEntropyIndexedShape('CrossEntropyLoss', logits, targets, axis)
        return graphSupport.captureCrossEntropyIndexed(logits, targets, axis) as Tensor
      }
      if (axis !== 1) {
        throw new AffonError('invalid_arg', 'CrossEntropyLoss currently supports dense targets with axis=1')
      }
      return denseCrossEntropyLoss(logits, targets)
    }
  }

  nn.causal_mask = causal_mask
  nn.apply_causal_mask = apply_causal_mask
  nn.sinusoidal_encoding = sinusoidal_encoding
  nn.position_ids = position_ids

  // ============================================================
  // nn.Dropout
  // ============================================================
  nn.Dropout = function(p?: number): DropoutModule {
    const prob = (p !== undefined) ? p : 0.5
    if (!Number.isFinite(prob) || prob < 0 || prob >= 1) {
      throw new AffonError('invalid_arg', 'Dropout p must be in the range [0, 1)')
    }

    const state = { training: true }
    return computeModule(state, (_state, x: Tensor): Tensor => {
        const mask = cast(gt_scalar(randLike(x.shape, x), prob), x.dtype) as Tensor
        const scale = scalarLike(1.0 / (1.0 - prob), x)
        return mul(mul(x, mask), scale)
    }, (_state, x: Tensor): Tensor => x) as DropoutModule
  }

  // ============================================================
  // nn.BatchNorm
  // ============================================================
  nn.BatchNorm = function(num_features: number, opts?: { momentum?: number; eps?: number }): BatchNormModule {
    const momentum = (opts && opts.momentum !== undefined) ? opts.momentum : 0.1
    const eps = (opts && opts.eps !== undefined) ? opts.eps : 1e-5
    if (!Number.isInteger(num_features) || num_features <= 0) {
      throw new AffonError('invalid_arg', 'BatchNorm num_features must be a positive integer')
    }
    if (!Number.isFinite(momentum) || momentum < 0 || momentum > 1) {
      throw new AffonError('invalid_arg', 'BatchNorm momentum must be finite and in the range [0, 1]')
    }
    if (!Number.isFinite(eps) || eps <= 0) {
      throw new AffonError('invalid_arg', 'BatchNorm eps must be a positive finite number')
    }
    const gamma = trainableOnes([1, num_features], defaultModuleDType(), [axis.batch, axis.feature])
    const beta = trainableZeros([1, num_features], defaultModuleDType(), [axis.batch, axis.feature])
    const running_mean = zeros([1, num_features], { dtype: defaultModuleDType(), axes: [axis.batch, axis.feature] }) as Tensor
    const running_var = ones([1, num_features], { dtype: defaultModuleDType(), axes: [axis.batch, axis.feature] }) as Tensor

    const state = {
      gamma,
      beta,
      running_mean,
      running_var,
      training: true,
    }

    return computeModule(state, (state, x: Tensor): Tensor => {
        if (graphSupport.isCapturing()) {
          throw graphCaptureAdapterError('invalid_state', 'BatchNorm training forward is not supported during graph capture')
        }
        const batch_mean = mean(x, 0, true)
        const diff = sub(x, batch_mean)
        const batch_var = mean(mul(diff, diff), 0, true)

        no_grad((): void => {
          copy(
            state.running_mean,
            add(
              mul(state.running_mean, scalarLike(1.0 - momentum, state.running_mean)),
              mul(batch_mean, scalarLike(momentum, batch_mean)),
            ),
          )
          copy(
            state.running_var,
            add(
              mul(state.running_var, scalarLike(1.0 - momentum, state.running_var)),
              mul(batch_var, scalarLike(momentum, batch_var)),
            ),
          )
        })

        const x_norm = div(diff, sqrt(add(batch_var, scalarLike(eps, batch_var))))
        return add(mul(state.gamma, x_norm), state.beta)
    }, (state, x: Tensor): Tensor => {
      const diff = sub(x, state.running_mean)
      const x_norm = div(diff, sqrt(add(state.running_var, scalarLike(eps, state.running_var))))
      return add(mul(state.gamma, x_norm), state.beta)
    }) as BatchNormModule
  }

  // ============================================================
  // nn.LayerNorm
  // ============================================================
  nn.LayerNorm = function(normalized_shape: number, opts?: { eps?: number; dtype?: TensorDType }): LayerNormModule {
    const eps = (opts && opts.eps !== undefined) ? opts.eps : 1e-5
    if (!Number.isInteger(normalized_shape) || normalized_shape <= 0) {
      throw new AffonError('invalid_arg', 'LayerNorm normalized_shape must be a positive integer')
    }
    if (!Number.isFinite(eps) || eps <= 0) {
      throw new AffonError('invalid_arg', 'LayerNorm eps must be a positive finite number')
    }
    const dtype = opts?.dtype ?? defaultModuleDType()
    const gamma = trainableOnes([normalized_shape], dtype, [axis.feature])
    const beta = trainableZeros([normalized_shape], dtype, [axis.feature])

    const state = {
      gamma,
      beta,
    }

    return computeModule(state, (state, x: Tensor): Tensor => {
        if (x.ndim < 1) {
          throw new AffonError('invalid_shape', 'LayerNorm expects a tensor with at least one dimension')
        }
        const last_dim = x.shape[x.ndim - 1]
        if (last_dim !== normalized_shape) {
          throw new AffonError('shape_mismatch', 'LayerNorm shape mismatch: expected last dimension ' + normalized_shape + ', got ' + last_dim)
        }

        const axis = x.ndim - 1
        const captureAware = isCapturedGraphTensor(x) || isCapturedGraphTensor(state.gamma) || isCapturedGraphTensor(state.beta)
        let gammaView: Tensor = captureAware || state.gamma.device === x.device ? state.gamma : state.gamma.to(x.device)
        let betaView: Tensor = captureAware || state.beta.device === x.device ? state.beta : state.beta.to(x.device)
        for (let i = 0; i < axis; i++) {
          gammaView = unsqueeze(gammaView, 0)
          betaView = unsqueeze(betaView, 0)
        }
        const meanValue = mean(x, axis, true)
        const varianceValue = variance(x, axis, true)
        const centered = sub(x, meanValue)
        const denom = sqrt(add(varianceValue, scalarLike(eps, varianceValue)))
        const normalized = div(centered, denom)
        return add(mul(normalized, gammaView), betaView)
    }) as LayerNormModule
  }

export default nn
