import native from 'affon:compute/native'

class AffonError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = 'AffonError'
    this.code = code
  }
}

type NativeTensor = {
  shape: number[]
  dtype: 'f32' | 'f64' | 'i64'
}
type TensorDType = 'f32' | 'f64' | 'i64'
type Nested = number | Nested[]
type SliceRangeSpec = number | string

const GRAPH_VALUE = Symbol.for('affon.tensor.graph_value')

type UnaryKind = 'neg' | 'relu' | 'abs' | 'exp' | 'log' | 'sqrt' | 'sigmoid' | 'silu' | 'tanh' | 'sign'
type BinaryKind = 'add' | 'sub' | 'mul' | 'div' | 'gt'
type ReductionKind = 'sum' | 'mean' | 'std' | 'variance' | 'min' | 'max'
type IndexReductionKind = 'argmin' | 'argmax'
type MatmulExecutionHint = 'projection' | 'attention_scores' | 'attention_values'
type MatmulExecutionSource = 'higher_level_module'
type MatmulExecutionOptions = { hint?: MatmulExecutionHint; source?: MatmulExecutionSource }
type GraphNodeMetadata = { module_path?: string }
type CanonicalSliceRange = { start: number; stop: number; step: number }

type GraphNode =
  | ({ id: number; kind: 'input'; index: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'constant'; data: Nested; dtype: TensorDType } & GraphNodeMetadata)
  | ({ id: number; kind: 'rand' | 'randn'; shape: number[]; dtype: TensorDType } & GraphNodeMetadata)
  | ({ id: number; kind: UnaryKind; input: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'clamp'; input: number; min: number; max: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'reshape'; input: number; shape: number[] } & GraphNodeMetadata)
  | ({ id: number; kind: 'slice'; input: number; ranges: SliceRangeSpec[]; slice_ranges?: CanonicalSliceRange[] } & GraphNodeMetadata)
  | ({ id: number; kind: 'squeeze'; input: number; axis?: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'unsqueeze'; input: number; axis: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'transpose'; input: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'permute'; input: number; axes: number[] } & GraphNodeMetadata)
  | ({ id: number; kind: 'contiguous'; input: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'cast'; input: number; dtype: TensorDType } & GraphNodeMetadata)
  | ({ id: number; kind: 'gelu'; input: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'dot'; left: number; right: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'matmul'; left: number; right: number; execution?: MatmulExecutionOptions } & GraphNodeMetadata)
  | ({ id: number; kind: 'cat'; inputs: number[]; dim: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'stack'; inputs: number[]; dim: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'where'; cond: number; onTrue: number; onFalse: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'masked_fill'; input: number; mask: number; value: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'softmax'; input: number; dim: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'cross_entropy_indexed'; logits: number; targets: number; axis: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'one_hot'; input: number; numClasses: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'index_select'; input: number; dim: number; index: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'gather'; input: number; dim: number; index: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'topk_values'; input: number; k: number; dim: number } & GraphNodeMetadata)
  | ({ id: number; kind: 'topk_indices'; input: number; k: number; dim: number } & GraphNodeMetadata)
  | ({ id: number; kind: ReductionKind; input: number; axis?: number; keepdim?: boolean } & GraphNodeMetadata)
  | ({ id: number; kind: IndexReductionKind; input: number; axis?: number; keepdim?: boolean } & GraphNodeMetadata)
  | ({ id: number; kind: BinaryKind; left: number; right: number } & GraphNodeMetadata)

interface CaptureState {
  readonly nodes: GraphNode[]
  nextId: number
  inputArity: number
  boundInputs: Array<() => NativeTensor>
  boundInputIds: Map<unknown, number>
  modulePathStack: string[]
}

interface StaticTensorMeta {
  shape: number[]
  dtype: TensorDType
}

const GRAPH_META = Symbol.for('affon.tensor.graph_meta')

interface GraphPlan {
  inputCount: number
  outputId: number
  nodes: GraphNode[]
}

interface CapturedProgram {
  inputArity: number
  boundInputCount: number
  inputCount: number
  outputId: number
  nodes: GraphNode[]
}

type GraphExportBoundary = 'ad_hoc' | 'run' | 'epoch' | 'step' | 'forward' | 'backward'
type GraphExportPhase = 'unspecified' | 'forward' | 'backward' | 'optimizer' | 'evaluation'
type GraphExportMode = 'summary' | 'annotated'

interface GraphExportOptions {
  mode?: GraphExportMode
  boundary?: GraphExportBoundary
  phase?: GraphExportPhase
  runId?: string
  graphId?: string
  epoch?: number
  step?: number
  batch?: number
  note?: string
}

interface ExecutableRegion {
  nodeIds: number[]
}

interface GraphRegionSummary {
  nodeIds: number[]
  instructionCount: number
  broadcastModes: string[]
  lowering: string
}

interface GraphTemplateSummary {
  kind: string
  nodeIds: number[]
  lowering: string
}

interface GraphSummary {
  inputCount: number
  output: number
  nodes: ReturnType<typeof summarizeNodes>
  staticMetadata: {
    kind: 'provisional_capture_metadata'
    source: 'ts_capture'
    fields: string[]
  }
  stats: ReturnType<typeof summarizeStats>
  regions: GraphRegionSummary[]
  templates: GraphTemplateSummary[]
  specialized: boolean
  runtime?: 'native-graph'
  summaryKind?: 'execution'
  loweringAnalysis?: GraphLoweringAnalysis
  nativeState?: unknown
}

type GraphExecutionRuntime = 'native-graph'
type GraphCaptureFailureStage = 'syntax' | 'capture_adapter' | 'semantic_validation' | 'unknown'
type GraphLoweringAnalysis = {
  lowerable: boolean
  failure?: {
    nodeId?: number
    nodeKind?: string
    reason: string
  } | null
}

interface CapturedProgramSummary {
  inputArity: number
  boundInputCount: number
  inputCount: number
  outputId: number
  nodeCount: number
  nodeKinds: string[]
}

interface ExecutionPlanExport {
  kind: 'graph_plan'
  id: string
  version: number
  context?: unknown
  steps: unknown[]
  regions: unknown[]
  outputs: number[]
}

function graphObservationError(member: string, kind: 'method' | 'property' = 'method'): never {
  const target = kind === 'method' ? `${member}()` : member
  throw graphCaptureError('capture_adapter', 'invalid_state', `${target} is not supported during graph capture`)
}

function graphCaptureError(stage: GraphCaptureFailureStage, code: string, message: string): AffonError {
  const error = new AffonError(code as any, message)
  ;(error as any).graphCaptureStage = stage
  return error
}

function graphCaptureSyntaxError(message: string): AffonError {
  return graphCaptureError('syntax', 'invalid_arg', message)
}

function graphCaptureAdapterError(message: string): AffonError {
  return graphCaptureError('capture_adapter', 'invalid_arg', message)
}

class CapturedValue {
  readonly $graph = true as const
  readonly [GRAPH_VALUE] = true as const
  readonly [GRAPH_META]?: StaticTensorMeta

  constructor(readonly id: number, meta?: StaticTensorMeta) {
    if (meta) {
      ;(this as any)[GRAPH_META] = { shape: [...meta.shape], dtype: meta.dtype }
    }
    const self = this as any
    const installProperty = (name: string) => {
      Object.defineProperty(self, name, {
        configurable: true,
        enumerable: false,
        get() { return graphObservationError(name, 'property') },
      })
    }
    const installMethod = (name: string) => {
      self[name] = () => graphObservationError(name)
    }

    Object.defineProperty(self, 'shape', {
      configurable: true,
      enumerable: false,
      get() {
        const meta = getStaticMeta(self)
        if (meta) return [...meta.shape]
        return graphObservationError('shape', 'property')
      },
    })
    Object.defineProperty(self, 'dtype', {
      configurable: true,
      enumerable: false,
      get() {
        const meta = getStaticMeta(self)
        if (meta) return meta.dtype
        return graphObservationError('dtype', 'property')
      },
    })
    Object.defineProperty(self, 'ndim', {
      configurable: true,
      enumerable: false,
      get() {
        const meta = getStaticMeta(self)
        if (meta) return meta.shape.length
        return graphObservationError('ndim', 'property')
      },
    })
    installProperty('device')

    installMethod('detach')
    installMethod('reshape')
    installMethod('transpose')
    installMethod('permute')
    installMethod('contiguous')
    installMethod('to')
    installMethod('sum')
    installMethod('mean')
    installMethod('var')
    installMethod('variance')
    installMethod('std')
    installMethod('min')
    installMethod('max')
    installMethod('argmin')
    installMethod('argmax')
    installMethod('to_array')
    installMethod('item')
    self.slice = (ranges: unknown) => captureSlice(self, ranges)
    installMethod('toString')
  }
}

let currentCapture: CaptureState | null = null

function currentModulePath(state: CaptureState): string | undefined {
  return state.modulePathStack.length > 0 ? state.modulePathStack[state.modulePathStack.length - 1] : undefined
}

function pushNode<T extends GraphNode>(state: CaptureState, node: T): void {
  const module_path = currentModulePath(state)
  if (module_path && node.module_path === undefined) {
    state.nodes.push({ ...node, module_path })
    return
  }
  state.nodes.push(node)
}

function withCaptureModulePath<T>(modulePath: string | undefined, fn: () => T): T {
  const state = currentCapture
  if (!state || !modulePath) return fn()
  state.modulePathStack.push(modulePath)
  try {
    return fn()
  } finally {
    state.modulePathStack.pop()
  }
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isTensorLike(value: unknown): value is NativeTensor {
  return !!value
    && typeof value === 'object'
    && Array.isArray((value as any).shape)
    && typeof (value as any).item === 'function'
}

function cloneNested(data: Nested): Nested {
  if (typeof data === 'number') return data
  return data.map((item) => cloneNested(item)) as Nested
}

function inferShape(data: Nested): number[] {
  if (typeof data === 'number') return []
  const array = data as Nested[]
  if (array.length === 0) return [0]
  return [array.length, ...inferShape(array[0])]
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

function broadcastShapeRightAligned(a: readonly number[], b: readonly number[]): number[] | null {
  const rank = Math.max(a.length, b.length)
  const out = new Array<number>(rank)
  for (let i = 0; i < rank; i++) {
    const left = i < rank - a.length ? 1 : a[i - (rank - a.length)]
    const right = i < rank - b.length ? 1 : b[i - (rank - b.length)]
    if (left === right) {
      out[i] = left
      continue
    }
    if (left === 1) {
      out[i] = right
      continue
    }
    if (right === 1) {
      out[i] = left
      continue
    }
    return null
  }
  return out
}

function inferMatmulShape(a: readonly number[], b: readonly number[]): number[] | null {
  if (a.length === 0 || b.length === 0) return null
  if (a.length === 1 && b.length === 1) {
    return a[0] === b[0] ? [] : null
  }

  const leftContract = a[a.length - 1]
  const rightContract = b.length === 1 ? b[0] : b[b.length - 2]
  if (leftContract !== rightContract) return null

  const leftBatch = a.length <= 2 ? [] : a.slice(0, -2)
  const rightBatch = b.length <= 2 ? [] : b.slice(0, -2)
  const batch = broadcastShapeRightAligned(leftBatch, rightBatch)
  if (!batch) return null

  if (a.length === 1) {
    return [...batch, b[b.length - 1]!]
  }
  if (b.length === 1) {
    return [...batch, a[a.length - 2]!]
  }
  return [...batch, a[a.length - 2]!, b[b.length - 1]!]
}

function reduceShape(shape: readonly number[], axis?: number, keepdim?: boolean): number[] | null {
  if (axis == null) return [1]
  if (!Number.isInteger(axis) || axis < 0 || axis >= shape.length) return null
  if (keepdim) {
    const out = [...shape]
    out[axis] = 1
    return out
  }
  const out = shape.filter((_, index) => index !== axis)
  return out.length === 0 ? [1] : out
}

function normalizeSliceIndex(index: number, dim: number): number | null {
  const normalized = index < 0 ? dim + index : index
  if (!Number.isInteger(normalized) || normalized < 0 || normalized > dim) return null
  return normalized
}

function canonicalizeSliceRange(dim: number, range: SliceRangeSpec): CanonicalSliceRange | null {
  if (Number.isInteger(range)) {
    const index = normalizeSliceIndex(range, dim)
    return index == null || index >= dim ? null : { start: index, stop: index + 1, step: 1 }
  }
  if (typeof range !== 'string') return null
  const trimmed = range.trim()
  if (trimmed === ':') return { start: 0, stop: dim, step: 1 }
  if (!trimmed.includes(':')) {
    const index = Number(trimmed)
    const normalized = Number.isInteger(index) ? normalizeSliceIndex(index, dim) : null
    return normalized == null || normalized >= dim ? null : { start: normalized, stop: normalized + 1, step: 1 }
  }

  const parts = trimmed.split(':')
  if (parts.length > 3) return null
  const parsePart = (part: string, fallback: number): number | null => {
    if (part === '') return fallback
    const value = Number(part)
    return Number.isInteger(value) ? normalizeSliceIndex(value, dim) : null
  }
  const step = parts[2] == null || parts[2] === '' ? 1 : Number(parts[2])
  if (!Number.isInteger(step) || step <= 0) return null
  const start = parsePart(parts[0]!, 0)
  const stop = parsePart(parts[1]!, dim)
  if (start == null || stop == null || stop < start) return null
  return { start, stop, step }
}

function inferSliceMeta(shape: readonly number[], ranges: readonly SliceRangeSpec[]): { shape: number[]; ranges: CanonicalSliceRange[] } | null {
  if (ranges.length > shape.length) return null
  const out: number[] = []
  const canonical: CanonicalSliceRange[] = []
  for (let i = 0; i < shape.length; i++) {
    const range = ranges[i] ?? ':'
    const normalized = canonicalizeSliceRange(shape[i]!, range)
    if (!normalized) return null
    canonical.push(normalized)
    out.push(Math.ceil((normalized.stop - normalized.start) / normalized.step))
  }
  return { shape: out, ranges: canonical }
}

function getCaptureState(opName: string): CaptureState {
  const state = currentCapture
  if (!state) throw graphCaptureError('capture_adapter', 'invalid_state', `${opName} is only valid during graph capture`)
  return state
}

function getStaticMeta(value: unknown): StaticTensorMeta | null {
  if (!isCapturedTensor(value)) return null
  const meta = (value as any)[GRAPH_META] as StaticTensorMeta | undefined
  if (!meta) return null
  return { shape: [...meta.shape], dtype: meta.dtype }
}

function unsupportedGraphOp(opName: string): never {
  throw graphCaptureError('capture_adapter', 'invalid_state', `${opName} is not supported in graph capture`)
}

function makeInput(state: CaptureState, index: number, meta?: StaticTensorMeta): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'input', index })
  return new CapturedValue(id, meta)
}

function makeBoundInput(state: CaptureState, key: unknown, resolve: () => NativeTensor): CapturedValue {
  const tensor = resolve()
  const existing = state.boundInputIds.get(key)
  if (existing != null) {
    return new CapturedValue(existing, { shape: [...tensor.shape], dtype: tensor.dtype as TensorDType })
  }
  const id = state.nextId++
  const index = state.inputArity + state.boundInputs.length
  pushNode(state, { id, kind: 'input', index })
  state.boundInputs.push(resolve)
  state.boundInputIds.set(key, id)
  return new CapturedValue(id, { shape: [...tensor.shape], dtype: tensor.dtype as TensorDType })
}

function pushUnary(state: CaptureState, kind: UnaryKind, input: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind, input: input.id })
  return new CapturedValue(id, getStaticMeta(input) ?? undefined)
}

function pushRandom(state: CaptureState, kind: 'rand' | 'randn', shape: readonly number[], dtype: TensorDType): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind, shape: [...shape], dtype })
  return new CapturedValue(id, { shape: [...shape], dtype })
}

function pushClamp(state: CaptureState, input: CapturedValue, min: number, max: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'clamp', input: input.id, min, max })
  return new CapturedValue(id, getStaticMeta(input) ?? undefined)
}

function pushReshape(state: CaptureState, input: CapturedValue, shape: number[]): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'reshape', input: input.id, shape: [...shape] })
  const meta = getStaticMeta(input)
  return new CapturedValue(id, meta ? { shape: [...shape], dtype: meta.dtype } : undefined)
}

function pushSlice(state: CaptureState, input: CapturedValue, ranges: readonly SliceRangeSpec[]): CapturedValue {
  const id = state.nextId++
  const meta = getStaticMeta(input)
  const sliceMeta = meta ? inferSliceMeta(meta.shape, ranges) : null
  pushNode(state, {
    id,
    kind: 'slice',
    input: input.id,
    ranges: [...ranges],
    ...(sliceMeta ? { slice_ranges: sliceMeta.ranges } : {}),
  })
  return new CapturedValue(id, meta && sliceMeta ? { shape: sliceMeta.shape, dtype: meta.dtype } : undefined)
}

function pushSqueeze(state: CaptureState, input: CapturedValue, axis?: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'squeeze', input: input.id, axis })
  const meta = getStaticMeta(input)
  let squeezed: StaticTensorMeta | undefined
  if (meta) {
    let shape: number[]
    if (axis == null) {
      shape = meta.shape.filter((dim) => dim !== 1)
      if (shape.length === 0) shape = [1]
    } else if (axis >= 0 && axis < meta.shape.length && meta.shape[axis] === 1) {
      shape = meta.shape.filter((_, index) => index !== axis)
      if (shape.length === 0) shape = [1]
    } else {
      shape = [...meta.shape]
    }
    squeezed = { shape, dtype: meta.dtype }
  }
  return new CapturedValue(id, squeezed)
}

function pushUnsqueeze(state: CaptureState, input: CapturedValue, axis: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'unsqueeze', input: input.id, axis })
  const meta = getStaticMeta(input)
  let expanded: StaticTensorMeta | undefined
  if (meta && axis >= 0 && axis <= meta.shape.length) {
    const shape = meta.shape.slice()
    shape.splice(axis, 0, 1)
    expanded = { shape, dtype: meta.dtype }
  }
  return new CapturedValue(id, expanded)
}

function pushTranspose(state: CaptureState, input: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'transpose', input: input.id })
  const meta = getStaticMeta(input)
  return new CapturedValue(id, meta && meta.shape.length === 2
    ? { shape: [meta.shape[1]!, meta.shape[0]!], dtype: meta.dtype }
    : undefined)
}

function pushPermute(state: CaptureState, input: CapturedValue, axes: number[]): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'permute', input: input.id, axes: [...axes] })
  const meta = getStaticMeta(input)
  return new CapturedValue(id, meta && axes.length === meta.shape.length
    ? { shape: axes.map((axis) => meta.shape[axis]!), dtype: meta.dtype }
    : undefined)
}

function pushContiguous(state: CaptureState, input: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'contiguous', input: input.id })
  return new CapturedValue(id, getStaticMeta(input) ?? undefined)
}

function pushCast(state: CaptureState, input: CapturedValue, dtype: TensorDType): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'cast', input: input.id, dtype })
  const meta = getStaticMeta(input)
  return new CapturedValue(id, meta ? { shape: [...meta.shape], dtype } : undefined)
}

function pushGelu(state: CaptureState, input: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'gelu', input: input.id })
  return new CapturedValue(id, getStaticMeta(input) ?? undefined)
}

function pushDot(state: CaptureState, left: CapturedValue, right: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'dot', left: left.id, right: right.id })
  const leftMeta = getStaticMeta(left)
  const rightMeta = getStaticMeta(right)
  return new CapturedValue(id, leftMeta && rightMeta ? { shape: [], dtype: leftMeta.dtype } : undefined)
}

function pushMatmul(state: CaptureState, left: CapturedValue, right: CapturedValue, execution?: MatmulExecutionOptions): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'matmul', left: left.id, right: right.id, execution: normalizeMatmulExecution(execution) })
  const leftMeta = getStaticMeta(left)
  const rightMeta = getStaticMeta(right)
  let meta: StaticTensorMeta | undefined
  if (leftMeta && rightMeta) {
    const shape = inferMatmulShape(leftMeta.shape, rightMeta.shape)
    if (shape) {
      meta = { shape, dtype: leftMeta.dtype }
    }
  }
  return new CapturedValue(id, meta)
}

function pushCat(state: CaptureState, inputs: readonly CapturedValue[], dim: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'cat', inputs: inputs.map((input) => input.id), dim })
  return new CapturedValue(id)
}

function pushStack(state: CaptureState, inputs: readonly CapturedValue[], dim: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'stack', inputs: inputs.map((input) => input.id), dim })
  return new CapturedValue(id)
}

function pushWhere(state: CaptureState, cond: CapturedValue, onTrue: CapturedValue, onFalse: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'where', cond: cond.id, onTrue: onTrue.id, onFalse: onFalse.id })
  const condMeta = getStaticMeta(cond)
  const trueMeta = getStaticMeta(onTrue)
  const falseMeta = getStaticMeta(onFalse)
  let meta: StaticTensorMeta | undefined
  if (condMeta && trueMeta && falseMeta) {
    const condTrue = broadcastShapeRightAligned(condMeta.shape, trueMeta.shape)
    const shape = condTrue ? broadcastShapeRightAligned(condTrue, falseMeta.shape) : null
    if (shape) meta = { shape, dtype: trueMeta.dtype }
  }
  return new CapturedValue(id, meta)
}

function pushMaskedFill(state: CaptureState, input: CapturedValue, mask: CapturedValue, value: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'masked_fill', input: input.id, mask: mask.id, value })
  const inputMeta = getStaticMeta(input)
  const maskMeta = getStaticMeta(mask)
  let meta: StaticTensorMeta | undefined
  if (inputMeta && maskMeta) {
    const shape = broadcastShapeRightAligned(inputMeta.shape, maskMeta.shape)
    if (shape) meta = { shape, dtype: inputMeta.dtype }
  }
  return new CapturedValue(id, meta)
}

function pushSoftmax(state: CaptureState, input: CapturedValue, dim: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'softmax', input: input.id, dim })
  return new CapturedValue(id, getStaticMeta(input) ?? undefined)
}

function pushCrossEntropyIndexed(state: CaptureState, logits: CapturedValue, targets: CapturedValue, axis: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'cross_entropy_indexed', logits: logits.id, targets: targets.id, axis })
  return new CapturedValue(id, { shape: [1], dtype: 'f32' })
}

function pushOneHot(state: CaptureState, input: CapturedValue, numClasses: number): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'one_hot', input: input.id, numClasses })
  const meta = getStaticMeta(input)
  return new CapturedValue(id, meta ? { shape: [...meta.shape, numClasses], dtype: 'f32' } : undefined)
}

function pushIndexSelect(state: CaptureState, input: CapturedValue, dim: number, index: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'index_select', input: input.id, dim, index: index.id })
  const inputMeta = getStaticMeta(input)
  const indexMeta = getStaticMeta(index)
  let meta: StaticTensorMeta | undefined
  if (inputMeta && indexMeta && Number.isInteger(dim) && dim >= 0 && dim < inputMeta.shape.length) {
    const shape = inputMeta.shape.slice(0, dim).concat(indexMeta.shape, inputMeta.shape.slice(dim + 1))
    meta = { shape, dtype: inputMeta.dtype }
  }
  return new CapturedValue(id, meta)
}

function pushGather(state: CaptureState, input: CapturedValue, dim: number, index: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'gather', input: input.id, dim, index: index.id })
  const indexMeta = getStaticMeta(index)
  const inputMeta = getStaticMeta(input)
  return new CapturedValue(id, indexMeta && inputMeta ? { shape: [...indexMeta.shape], dtype: inputMeta.dtype } : undefined)
}

function pushTopK(state: CaptureState, input: CapturedValue, k: number, dim: number): { values: CapturedValue; indices: CapturedValue } {
  const valuesId = state.nextId++
  pushNode(state, { id: valuesId, kind: 'topk_values', input: input.id, k, dim })
  const indicesId = state.nextId++
  pushNode(state, { id: indicesId, kind: 'topk_indices', input: input.id, k, dim })
  return {
    values: new CapturedValue(valuesId),
    indices: new CapturedValue(indicesId),
  }
}

function pushReduction(
  state: CaptureState,
  kind: ReductionKind,
  input: CapturedValue,
  axis?: number,
  keepdim?: boolean,
): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind, input: input.id, axis, keepdim })
  const meta = getStaticMeta(input)
  let reduced: StaticTensorMeta | undefined
  if (meta) {
    const shape = reduceShape(meta.shape, axis, keepdim)
    if (shape) reduced = { shape, dtype: meta.dtype }
  }
  return new CapturedValue(id, reduced)
}

function pushIndexReduction(
  state: CaptureState,
  kind: IndexReductionKind,
  input: CapturedValue,
  axis?: number,
  keepdim?: boolean,
): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind, input: input.id, axis, keepdim })
  const meta = getStaticMeta(input)
  let reduced: StaticTensorMeta | undefined
  if (meta) {
    const shape = reduceShape(meta.shape, axis, keepdim)
    if (shape) reduced = { shape, dtype: 'i64' as TensorDType }
  }
  return new CapturedValue(id, reduced)
}

function pushBinary(state: CaptureState, kind: BinaryKind, left: CapturedValue, right: CapturedValue): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind, left: left.id, right: right.id })
  const leftMeta = getStaticMeta(left)
  const rightMeta = getStaticMeta(right)
  let meta: StaticTensorMeta | undefined
  if (leftMeta && rightMeta) {
    const shape = broadcastShapeRightAligned(leftMeta.shape, rightMeta.shape)
    if (shape) meta = { shape, dtype: kind === 'gt' ? 'i64' : leftMeta.dtype }
  }
  return new CapturedValue(id, meta)
}

function pushConstant(state: CaptureState, data: Nested, dtype: TensorDType): CapturedValue {
  const id = state.nextId++
  pushNode(state, { id, kind: 'constant', data: cloneNested(data), dtype })
  return new CapturedValue(id, { shape: inferShape(data), dtype })
}

function normalizeBinaryCaptureOperand(state: CaptureState, opName: string, value: unknown): CapturedValue {
  if (isCapturedTensor(value)) return value
  if (isTensorLike(value)) return makeBoundInput(state, value, () => value)
  if (isFiniteNumber(value)) return pushConstant(state, value, 'f32')
  throw graphCaptureError('capture_adapter', 'invalid_arg', `${opName} graph capture currently supports graph values and finite scalar constants only`)
}

function canonicalize(nodes: GraphNode[], rootId: number): { nodes: GraphNode[]; rootId: number } {
  const byId = new Map<number, GraphNode>()
  for (const node of nodes) byId.set(node.id, node)

  const rewritten = new Map<number, number>()

  function isScalarConstant(nodeId: number): boolean {
    const node = byId.get(nodeId)
    if (!node || node.kind !== 'constant') return false
    return typeof node.data === 'number'
  }

  function isZeroScalarConstant(nodeId: number): boolean {
    if (!isScalarConstant(nodeId)) return false
    const node = byId.get(nodeId) as Extract<GraphNode, { kind: 'constant' }>
    return node.data === 0
  }

  function isOneScalarConstant(nodeId: number): boolean {
    if (!isScalarConstant(nodeId)) return false
    const node = byId.get(nodeId) as Extract<GraphNode, { kind: 'constant' }>
    return node.data === 1
  }

  function canonicalBinaryOperands(kind: BinaryKind, left: number, right: number): { left: number; right: number } {
    if (kind !== 'add' && kind !== 'mul') return { left, right }

    const leftScalar = isScalarConstant(left)
    const rightScalar = isScalarConstant(right)
    if (leftScalar && !rightScalar) return { left: right, right: left }
    if (!leftScalar && rightScalar) return { left, right }
    if (left > right) return { left: right, right: left }
    return { left, right }
  }

  function rewrite(id: number): number {
    const cached = rewritten.get(id)
    if (cached != null) return cached
    const node = byId.get(id)
    if (!node) throw new AffonError('invalid_state', 'graph canonicalization encountered an unknown node')

    let result = id
    switch (node.kind) {
      case 'neg': {
        const input = rewrite(node.input)
        const inner = byId.get(input)
        if (inner && inner.kind === 'neg') result = rewrite(inner.input)
        break
      }
      case 'relu': {
        const input = rewrite(node.input)
        const inner = byId.get(input)
        if (inner && inner.kind === 'relu') result = input
        break
      }
      case 'add':
      case 'sub':
      case 'mul':
      case 'div': {
        const operands = canonicalBinaryOperands(node.kind, rewrite(node.left), rewrite(node.right))
        const left = operands.left
        const right = operands.right
        if ((node.kind === 'add' || node.kind === 'sub') && isZeroScalarConstant(right)) result = left
        if (node.kind === 'add' && isZeroScalarConstant(left)) result = right
        if ((node.kind === 'mul' || node.kind === 'div') && isOneScalarConstant(right)) result = left
        if (node.kind === 'mul' && isOneScalarConstant(left)) result = right
        break
      }
      case 'gt':
        break
      case 'matmul':
      case 'dot':
        break
      default:
        break
    }
    rewritten.set(id, result)
    return result
  }

  const reachable = new Set<number>()
  function mark(id: number): void {
    const resolved = rewrite(id)
    if (reachable.has(resolved)) return
    reachable.add(resolved)
    const node = byId.get(resolved)
    if (!node) throw new AffonError('invalid_state', 'graph canonicalization encountered an unknown node')
    switch (node.kind) {
      case 'neg':
      case 'relu':
      case 'abs':
      case 'exp':
      case 'log':
      case 'sqrt':
      case 'sigmoid':
      case 'silu':
      case 'tanh':
      case 'sign':
      case 'clamp':
      case 'reshape':
      case 'slice':
      case 'squeeze':
      case 'unsqueeze':
      case 'transpose':
      case 'permute':
      case 'contiguous':
      case 'cast':
      case 'gelu':
        mark(node.input)
        break
      case 'add':
      case 'sub':
      case 'mul':
      case 'div':
      case 'gt':
        mark(node.left)
        mark(node.right)
        break
      case 'matmul':
      case 'dot':
        mark(node.left)
        mark(node.right)
        break
      case 'cat':
      case 'stack':
        for (const input of node.inputs) mark(input)
        break
      case 'where':
        mark(node.cond)
        mark(node.onTrue)
        mark(node.onFalse)
        break
      case 'masked_fill':
        mark(node.input)
        mark(node.mask)
        break
      case 'softmax':
      case 'one_hot':
      case 'topk_values':
        mark(node.input)
        break
      case 'topk_indices': {
        mark(node.input)
        const sibling = nodes.find((candidate) =>
          candidate.kind === 'topk_values'
          && candidate.input === node.input
          && candidate.k === node.k
          && candidate.dim === node.dim)
        if (sibling) mark(sibling.id)
        break
      }
      case 'cross_entropy_indexed':
        mark(node.logits)
        mark(node.targets)
        break
      case 'index_select':
        mark(node.input)
        mark(node.index)
        break
      case 'gather':
        mark(node.input)
        mark(node.index)
        break
      case 'sum':
      case 'mean':
      case 'std':
      case 'variance':
      case 'min':
      case 'max':
      case 'argmin':
      case 'argmax':
        mark(node.input)
        break
      default:
        break
    }
  }

  const nextRoot = rewrite(rootId)
  mark(nextRoot)

  const nextNodes: GraphNode[] = []
  for (const node of nodes) {
    const rewrittenId = rewrite(node.id)
    if (rewrittenId !== node.id || !reachable.has(node.id)) continue
    switch (node.kind) {
      case 'neg':
      case 'relu':
      case 'abs':
      case 'exp':
      case 'log':
      case 'sqrt':
      case 'sigmoid':
      case 'silu':
      case 'tanh':
      case 'sign':
      case 'clamp':
      case 'reshape':
      case 'slice':
      case 'squeeze':
      case 'unsqueeze':
      case 'transpose':
      case 'permute':
      case 'contiguous':
      case 'cast':
      case 'gelu':
        nextNodes.push({ ...node, input: rewrite(node.input) })
        break
      case 'add':
      case 'sub':
      case 'mul':
      case 'div':
      case 'gt':
        nextNodes.push({ ...node, ...canonicalBinaryOperands(node.kind, rewrite(node.left), rewrite(node.right)) })
        break
      case 'matmul':
      case 'dot':
        nextNodes.push({ ...node, left: rewrite(node.left), right: rewrite(node.right) })
        break
      case 'cat':
      case 'stack':
        nextNodes.push({ ...node, inputs: node.inputs.map((input) => rewrite(input)) })
        break
      case 'where':
        nextNodes.push({ ...node, cond: rewrite(node.cond), onTrue: rewrite(node.onTrue), onFalse: rewrite(node.onFalse) })
        break
      case 'masked_fill':
        nextNodes.push({ ...node, input: rewrite(node.input), mask: rewrite(node.mask) })
        break
      case 'softmax':
        nextNodes.push({ ...node, input: rewrite(node.input) })
        break
      case 'cross_entropy_indexed':
        nextNodes.push({ ...node, logits: rewrite(node.logits), targets: rewrite(node.targets) })
        break
      case 'one_hot':
        nextNodes.push({ ...node, input: rewrite(node.input) })
        break
      case 'index_select':
        nextNodes.push({ ...node, input: rewrite(node.input), index: rewrite(node.index) })
        break
      case 'gather':
        nextNodes.push({ ...node, input: rewrite(node.input), index: rewrite(node.index) })
        break
      case 'topk_values':
      case 'topk_indices':
        nextNodes.push({ ...node, input: rewrite(node.input) })
        break
      case 'sum':
      case 'mean':
      case 'std':
      case 'variance':
      case 'min':
      case 'max':
      case 'argmin':
      case 'argmax':
        nextNodes.push({ ...node, input: rewrite(node.input) })
        break
      default:
        nextNodes.push(node)
        break
    }
  }

  return { nodes: nextNodes, rootId: nextRoot }
}

function summarizeNodes(nodes: GraphNode[]) {
  const byId = new Map<number, GraphNode>()
  for (const node of nodes) byId.set(node.id, node)

  const shapeCache = new Map<number, number[] | null>()
  function inferNodeShape(nodeId: number): number[] | null {
    const cached = shapeCache.get(nodeId)
    if (cached !== undefined || shapeCache.has(nodeId)) return cached ?? null

    const node = byId.get(nodeId)
    if (!node) throw new AffonError('invalid_state', 'graph summary encountered an unknown node')

    let shape: number[] | null = null
    switch (node.kind) {
      case 'input':
        shape = null
        break
      case 'constant':
        shape = inferShape(node.data)
        break
      case 'neg':
      case 'relu':
      case 'abs':
      case 'exp':
      case 'log':
      case 'sqrt':
      case 'sigmoid':
      case 'silu':
      case 'tanh':
      case 'sign':
      case 'clamp':
      case 'contiguous':
      case 'cast':
      case 'gelu':
        shape = inferNodeShape(node.input)
        break
      case 'reshape':
        shape = [...node.shape]
        break
      case 'slice':
        shape = null
        break
      case 'squeeze': {
        const inputShape = inferNodeShape(node.input)
        if (inputShape) {
          if (node.axis == null) {
            const squeezed = inputShape.filter((dim) => dim !== 1)
            shape = squeezed.length === 0 ? [1] : squeezed
          } else if (node.axis >= 0 && node.axis < inputShape.length) {
            if (inputShape[node.axis] === 1) {
              const squeezed = inputShape.filter((_, index) => index !== node.axis)
              shape = squeezed.length === 0 ? [1] : squeezed
            }
          }
        }
        break
      }
      case 'unsqueeze': {
        const inputShape = inferNodeShape(node.input)
        if (inputShape && node.axis >= 0 && node.axis <= inputShape.length) {
          shape = [...inputShape]
          shape.splice(node.axis, 0, 1)
        }
        break
      }
      case 'squeeze': {
        const inputShape = inferNodeShape(node.input)
        if (inputShape) {
          if (node.axis == null) {
            const squeezed = inputShape.filter((dim) => dim !== 1)
            shape = squeezed.length === 0 ? [1] : squeezed
          } else if (node.axis >= 0 && node.axis < inputShape.length) {
            if (inputShape[node.axis] === 1) {
              const squeezed = inputShape.filter((_, index) => index !== node.axis)
              shape = squeezed.length === 0 ? [1] : squeezed
            }
          }
        }
        break
      }
      case 'unsqueeze': {
        const inputShape = inferNodeShape(node.input)
        if (inputShape && node.axis >= 0 && node.axis <= inputShape.length) {
          shape = [...inputShape]
          shape.splice(node.axis, 0, 1)
        }
        break
      }
      case 'transpose': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape ? [...inputShape].reverse() : null
        break
      }
      case 'permute': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape && inputShape.length === node.axes.length
          ? node.axes.map((axis) => inputShape[axis]!)
          : null
        break
      }
      case 'add':
      case 'sub':
      case 'mul':
      case 'div': {
        const leftShape = inferNodeShape(node.left)
        const rightShape = inferNodeShape(node.right)
        if (leftShape && rightShape) {
          if (sameShape(leftShape, rightShape)) {
            shape = [...leftShape]
            break
          }
          const broadcast = broadcastShapeRightAligned(leftShape, rightShape)
          shape = broadcast ? [...broadcast] : null
          break
        }
        shape = null
        break
      }
      case 'matmul': {
        const leftShape = inferNodeShape(node.left)
        const rightShape = inferNodeShape(node.right)
        if (leftShape && rightShape && leftShape.length === 2 && rightShape.length === 2 && leftShape[1] === rightShape[0]) {
          shape = [leftShape[0]!, rightShape[1]!]
        }
        break
      }
      case 'dot': {
        const leftShape = inferNodeShape(node.left)
        const rightShape = inferNodeShape(node.right)
        if (leftShape && rightShape && leftShape.length === 1 && rightShape.length === 1 && leftShape[0] === rightShape[0]) {
          shape = [1]
        }
        break
      }
      case 'cat': {
        const inputShapes = node.inputs.map((input) => inferNodeShape(input))
        if (inputShapes.every((s): s is number[] => Array.isArray(s)) && inputShapes.length > 0) {
          const rank = inputShapes[0]!.length
          if (node.dim >= 0 && node.dim < rank && inputShapes.every((s) => s.length === rank)) {
            const out = [...inputShapes[0]!]
            let compatible = true
            let dimSize = 0
            for (const s of inputShapes) {
              dimSize += s[node.dim]!
              for (let i = 0; i < rank; i++) {
                if (i === node.dim) continue
                if (s[i] !== out[i]) compatible = false
              }
            }
            if (compatible) {
              out[node.dim] = dimSize
              shape = out
            }
          }
        }
        break
      }
      case 'stack': {
        const inputShapes = node.inputs.map((input) => inferNodeShape(input))
        if (inputShapes.every((s): s is number[] => Array.isArray(s)) && inputShapes.length > 0) {
          const base = inputShapes[0]!
          if (inputShapes.every((s) => sameShape(s, base)) && node.dim >= 0 && node.dim <= base.length) {
            shape = [...base]
            shape.splice(node.dim, 0, inputShapes.length)
          }
        }
        break
      }
      case 'dot': {
        const leftShape = inferNodeShape(node.left)
        const rightShape = inferNodeShape(node.right)
        if (leftShape && rightShape && leftShape.length === 1 && rightShape.length === 1 && leftShape[0] === rightShape[0]) {
          shape = [1]
        }
        break
      }
      case 'cat': {
        const inputShapes = node.inputs.map((input) => inferNodeShape(input))
        if (inputShapes.every((s): s is number[] => Array.isArray(s)) && inputShapes.length > 0) {
          const rank = inputShapes[0]!.length
          if (node.dim >= 0 && node.dim < rank && inputShapes.every((s) => s.length === rank)) {
            const out = [...inputShapes[0]!]
            let compatible = true
            let dimSize = 0
            for (const s of inputShapes) {
              dimSize += s[node.dim]!
              for (let i = 0; i < rank; i++) {
                if (i === node.dim) continue
                if (s[i] !== out[i]) compatible = false
              }
            }
            if (compatible) {
              out[node.dim] = dimSize
              shape = out
            }
          }
        }
        break
      }
      case 'stack': {
        const inputShapes = node.inputs.map((input) => inferNodeShape(input))
        if (inputShapes.every((s): s is number[] => Array.isArray(s)) && inputShapes.length > 0) {
          const base = inputShapes[0]!
          if (inputShapes.every((s) => sameShape(s, base)) && node.dim >= 0 && node.dim <= base.length) {
            shape = [...base]
            shape.splice(node.dim, 0, inputShapes.length)
          }
        }
        break
      }
      case 'where': {
        const condShape = inferNodeShape(node.cond)
        const trueShape = inferNodeShape(node.onTrue)
        const falseShape = inferNodeShape(node.onFalse)
        if (condShape && trueShape && falseShape) {
          const condTrue = broadcastShapeRightAligned(condShape, trueShape)
          shape = condTrue ? broadcastShapeRightAligned(condTrue, falseShape) : null
        }
        break
      }
      case 'softmax':
        shape = inferNodeShape(node.input)
        break
      case 'cross_entropy_indexed':
        shape = [1]
        break
      case 'one_hot': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape ? [...inputShape, node.numClasses] : null
        break
      }
      case 'index_select': {
        const inputShape = inferNodeShape(node.input)
        const indexShape = inferNodeShape(node.index)
        if (inputShape && indexShape && indexShape.length === 1 && node.dim >= 0 && node.dim < inputShape.length) {
          shape = [...inputShape]
          shape[node.dim] = indexShape[0]!
        }
        break
      }
      case 'gather':
        shape = inferNodeShape(node.index)
        break
      case 'topk_values':
      case 'topk_indices': {
        const inputShape = inferNodeShape(node.input)
        if (inputShape && node.dim >= 0 && node.dim < inputShape.length) {
          shape = [...inputShape]
          shape[node.dim] = node.k
        }
        break
      }
      case 'argmin':
      case 'argmax': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape ? reduceShape(inputShape, node.axis, node.keepdim) : null
        break
      }
      case 'sum':
      case 'mean':
      case 'std':
      case 'variance':
      case 'min':
      case 'max':
      case 'argmin':
      case 'argmax': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape ? reduceShape(inputShape, node.axis, node.keepdim) : null
        break
      }
      case 'masked_fill':
        {
          const inputShape = inferNodeShape(node.input)
          const maskShape = inferNodeShape(node.mask)
          shape = inputShape && maskShape ? broadcastShapeRightAligned(inputShape, maskShape) : null
        }
        break
      case 'softmax':
        shape = inferNodeShape(node.input)
        break
      case 'cross_entropy_indexed':
        shape = [1]
        break
      case 'one_hot': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape ? [...inputShape, node.numClasses] : null
        break
      }
      case 'index_select': {
        const inputShape = inferNodeShape(node.input)
        const indexShape = inferNodeShape(node.index)
        if (inputShape && indexShape && indexShape.length === 1 && node.dim >= 0 && node.dim < inputShape.length) {
          shape = [...inputShape]
          shape[node.dim] = indexShape[0]!
        }
        break
      }
      case 'gather':
        shape = inferNodeShape(node.index)
        break
      case 'topk_values':
      case 'topk_indices': {
        const inputShape = inferNodeShape(node.input)
        if (inputShape && node.dim >= 0 && node.dim < inputShape.length) {
          shape = [...inputShape]
          shape[node.dim] = node.k
        }
        break
      }
      case 'argmin':
      case 'argmax': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape ? reduceShape(inputShape, node.axis, node.keepdim) : null
        break
      }
      case 'sum':
      case 'mean':
      case 'std':
      case 'variance':
      case 'min':
      case 'max':
      case 'argmin':
      case 'argmax': {
        const inputShape = inferNodeShape(node.input)
        shape = inputShape ? reduceShape(inputShape, node.axis, node.keepdim) : null
        break
      }
    }

    shapeCache.set(nodeId, shape)
    return shape
  }

  return nodes.map((node) => {
    switch (node.kind) {
      case 'constant':
        return {
          id: node.id,
          kind: node.kind,
          shape: inferShape(node.data),
          outputShape: inferNodeShape(node.id),
          dtype: node.dtype,
        }
      case 'input':
        return {
          id: node.id,
          kind: node.kind,
          index: node.index,
          outputShape: inferNodeShape(node.id),
        }
      case 'neg':
      case 'relu':
      case 'abs':
      case 'exp':
      case 'log':
      case 'sqrt':
      case 'sigmoid':
      case 'silu':
      case 'tanh':
      case 'sign':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          outputShape: inferNodeShape(node.id),
        }
      case 'clamp':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          min: node.min,
          max: node.max,
          outputShape: inferNodeShape(node.id),
        }
      case 'reshape':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          shape: [...node.shape],
          outputShape: inferNodeShape(node.id),
        }
      case 'slice':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          ranges: [...node.ranges],
          outputShape: inferNodeShape(node.id),
        }
      case 'squeeze':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          axis: node.axis,
          outputShape: inferNodeShape(node.id),
        }
      case 'unsqueeze':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          axis: node.axis,
          outputShape: inferNodeShape(node.id),
        }
      case 'transpose':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          outputShape: inferNodeShape(node.id),
        }
      case 'permute':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          axes: [...node.axes],
          outputShape: inferNodeShape(node.id),
        }
      case 'contiguous':
      case 'cast':
      case 'gelu':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          outputShape: inferNodeShape(node.id),
        }
      case 'add':
      case 'sub':
      case 'mul':
      case 'div':
      case 'gt':
        return {
          id: node.id,
          kind: node.kind,
          left: node.left,
          right: node.right,
          outputShape: inferNodeShape(node.id),
        }
      case 'matmul':
      case 'dot':
        return {
          id: node.id,
          kind: node.kind,
          left: node.left,
          right: node.right,
          outputShape: inferNodeShape(node.id),
        }
      case 'cat':
      case 'stack':
        return {
          id: node.id,
          kind: node.kind,
          inputs: [...node.inputs],
          dim: node.dim,
          outputShape: inferNodeShape(node.id),
        }
      case 'where':
        return {
          id: node.id,
          kind: node.kind,
          cond: node.cond,
          onTrue: node.onTrue,
          onFalse: node.onFalse,
          outputShape: inferNodeShape(node.id),
        }
      case 'masked_fill':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          mask: node.mask,
          value: node.value,
          outputShape: inferNodeShape(node.id),
        }
      case 'softmax':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          dim: node.dim,
          outputShape: inferNodeShape(node.id),
        }
      case 'cross_entropy_indexed':
        return {
          id: node.id,
          kind: node.kind,
          logits: node.logits,
          targets: node.targets,
          axis: node.axis,
          outputShape: inferNodeShape(node.id),
        }
      case 'one_hot':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          numClasses: node.numClasses,
          outputShape: inferNodeShape(node.id),
        }
      case 'index_select':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          dim: node.dim,
          index: node.index,
          outputShape: inferNodeShape(node.id),
        }
      case 'gather':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          dim: node.dim,
          index: node.index,
          outputShape: inferNodeShape(node.id),
        }
      case 'topk_values':
      case 'topk_indices':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          k: node.k,
          dim: node.dim,
          outputShape: inferNodeShape(node.id),
        }
      case 'sum':
      case 'mean':
      case 'std':
      case 'variance':
      case 'min':
      case 'max':
      case 'argmin':
      case 'argmax':
        return {
          id: node.id,
          kind: node.kind,
          input: node.input,
          axis: node.axis,
          keepdim: node.keepdim,
          outputShape: inferNodeShape(node.id),
        }
    }
  })
}

function isExecutableNode(node: GraphNode): boolean {
  return node.kind === 'clamp'
    || node.kind === 'neg'
    || node.kind === 'relu'
    || node.kind === 'abs'
    || node.kind === 'exp'
    || node.kind === 'log'
    || node.kind === 'sqrt'
    || node.kind === 'sigmoid'
    || node.kind === 'silu'
    || node.kind === 'tanh'
    || node.kind === 'sign'
    || node.kind === 'add'
    || node.kind === 'sub'
    || node.kind === 'mul'
    || node.kind === 'div'
    || node.kind === 'gt'
}

function executableDependencies(node: GraphNode): number[] {
  switch (node.kind) {
    case 'neg':
    case 'relu':
    case 'abs':
    case 'exp':
    case 'log':
    case 'sqrt':
    case 'sigmoid':
    case 'silu':
    case 'tanh':
    case 'sign':
    case 'clamp':
      return [node.input]
    case 'add':
    case 'sub':
    case 'mul':
    case 'div':
    case 'gt':
      return [node.left, node.right]
    case 'matmul':
    case 'dot':
      return [node.left, node.right]
    case 'cat':
    case 'stack':
      return [...node.inputs]
    case 'where':
      return [node.cond, node.onTrue, node.onFalse]
    case 'masked_fill':
      return [node.input, node.mask]
    case 'softmax':
    case 'one_hot':
    case 'topk_values':
    case 'topk_indices':
      return [node.input]
    case 'cross_entropy_indexed':
      return [node.logits, node.targets]
    case 'index_select':
      return [node.input, node.index]
    case 'gather':
      return [node.input, node.index]
    case 'sum':
    case 'mean':
    case 'max':
      return [node.input]
    default:
      return []
  }
}

function buildExecutableRegions(nodes: GraphNode[]): ExecutableRegion[] {
  const byId = new Map<number, GraphNode>()
  for (const node of nodes) byId.set(node.id, node)

  const executableIds = nodes
    .filter(isExecutableNode)
    .map((node) => node.id)
  if (executableIds.length === 0) return []

  const adjacency = new Map<number, number[]>()
  for (const id of executableIds) adjacency.set(id, [])

  for (const node of nodes) {
    if (!isExecutableNode(node)) continue
    const deps = executableDependencies(node)
    for (const depId of deps) {
      const dep = byId.get(depId)
      if (!dep || !isExecutableNode(dep)) continue
      adjacency.get(node.id)!.push(depId)
      adjacency.get(depId)!.push(node.id)
    }
  }

  const visited = new Set<number>()
  const regions: ExecutableRegion[] = []

  for (const startId of executableIds) {
    if (visited.has(startId)) continue
    const stack = [startId]
    const region: number[] = []
    visited.add(startId)
    while (stack.length > 0) {
      const id = stack.pop()!
      region.push(id)
      for (const next of adjacency.get(id) ?? []) {
        if (visited.has(next)) continue
        visited.add(next)
        stack.push(next)
      }
    }
    region.sort((a, b) => a - b)
    regions.push({ nodeIds: region })
  }

  regions.sort((a, b) => a.nodeIds[0]! - b.nodeIds[0]!)
  return regions
}

function summarizeStats(nodes: GraphNode[]) {
  const executableNodeCount = nodes.filter(isExecutableNode).length
  const regions = buildExecutableRegions(nodes)
  return {
    nodeCount: nodes.length,
    executableNodeCount,
    fusedRegionCount: regions.length,
    largestRegionNodeCount: regions.reduce((max, region) => Math.max(max, region.nodeIds.length), 0),
    elementwiseOnly: !nodes.some((node) =>
      node.kind === 'matmul'
      || node.kind === 'dot'
      || node.kind === 'cat'
      || node.kind === 'stack'
      || node.kind === 'masked_fill'
      || node.kind === 'softmax'
      || node.kind === 'one_hot'
      || node.kind === 'topk_values'
      || node.kind === 'topk_indices'
      || node.kind === 'index_select'
      || node.kind === 'gather'
      || node.kind === 'sum'
      || node.kind === 'mean'
      || node.kind === 'max'
      || node.kind === 'gelu'),
  }
}

function validatePlan(plan: GraphPlan): void {
  const byId = new Map<number, GraphNode>()
  for (let i = 0; i < plan.nodes.length; i++) {
    const node = plan.nodes[i]
    if (byId.has(node.id)) {
      throw new AffonError('invalid_state', `graph plan contains duplicate node id ${node.id}`)
    }
    byId.set(node.id, node)
  }

  if (!byId.has(plan.outputId)) {
    throw new AffonError('invalid_state', `graph plan output ${plan.outputId} does not exist`)
  }

  for (let i = 0; i < plan.nodes.length; i++) {
    const node = plan.nodes[i]
    switch (node.kind) {
      case 'input':
        if (!Number.isInteger(node.index) || node.index < 0 || node.index >= plan.inputCount) {
          throw new AffonError('invalid_state', `graph plan input node ${node.id} has invalid index ${node.index}`)
        }
        break
      case 'constant':
        break
      case 'neg':
      case 'relu':
      case 'abs':
      case 'exp':
      case 'log':
      case 'sqrt':
      case 'sigmoid':
      case 'silu':
      case 'tanh':
      case 'sign':
      case 'clamp':
      case 'reshape':
      case 'slice':
      case 'squeeze':
      case 'unsqueeze':
      case 'transpose':
      case 'permute':
      case 'contiguous':
      case 'cast':
      case 'gelu':
        if (!byId.has(node.input)) {
          throw new AffonError('invalid_state', `graph plan unary node ${node.id} references missing input ${node.input}`)
        }
        if (node.input >= node.id) {
          throw new AffonError('invalid_state', `graph plan unary node ${node.id} is not in dependency order`)
        }
        break
      case 'add':
      case 'sub':
      case 'mul':
      case 'div':
      case 'gt':
        if (!byId.has(node.left)) {
          throw new AffonError('invalid_state', `graph plan binary node ${node.id} references missing left input ${node.left}`)
        }
        if (!byId.has(node.right)) {
          throw new AffonError('invalid_state', `graph plan binary node ${node.id} references missing right input ${node.right}`)
        }
        if (node.left >= node.id || node.right >= node.id) {
          throw new AffonError('invalid_state', `graph plan binary node ${node.id} is not in dependency order`)
        }
        break
      case 'matmul':
      case 'dot':
        if (!byId.has(node.left)) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} references missing left input ${node.left}`)
        }
        if (!byId.has(node.right)) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} references missing right input ${node.right}`)
        }
        if (node.left >= node.id || node.right >= node.id) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} is not in dependency order`)
        }
        break
      case 'cat':
      case 'stack':
        if (node.inputs.length === 0) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} requires at least one input`)
        }
        for (const input of node.inputs) {
          if (!byId.has(input)) {
            throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} references missing input ${input}`)
          }
          if (input >= node.id) {
            throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} is not in dependency order`)
          }
        }
        break
      case 'where':
        if (!byId.has(node.cond)) {
          throw new AffonError('invalid_state', `graph plan where node ${node.id} references missing condition ${node.cond}`)
        }
        if (!byId.has(node.onTrue)) {
          throw new AffonError('invalid_state', `graph plan where node ${node.id} references missing true input ${node.onTrue}`)
        }
        if (!byId.has(node.onFalse)) {
          throw new AffonError('invalid_state', `graph plan where node ${node.id} references missing false input ${node.onFalse}`)
        }
        if (node.cond >= node.id || node.onTrue >= node.id || node.onFalse >= node.id) {
          throw new AffonError('invalid_state', `graph plan where node ${node.id} is not in dependency order`)
        }
        break
      case 'masked_fill':
        if (!byId.has(node.input)) {
          throw new AffonError('invalid_state', `graph plan masked_fill node ${node.id} references missing input ${node.input}`)
        }
        if (!byId.has(node.mask)) {
          throw new AffonError('invalid_state', `graph plan masked_fill node ${node.id} references missing mask ${node.mask}`)
        }
        if (node.input >= node.id || node.mask >= node.id) {
          throw new AffonError('invalid_state', `graph plan masked_fill node ${node.id} is not in dependency order`)
        }
        break
      case 'softmax':
      case 'one_hot':
        if (!byId.has(node.input)) {
          throw new AffonError('invalid_state', `graph plan softmax node ${node.id} references missing input ${node.input}`)
        }
        if (node.input >= node.id) {
          throw new AffonError('invalid_state', `graph plan softmax node ${node.id} is not in dependency order`)
        }
        break
      case 'cross_entropy_indexed':
        if (!byId.has(node.logits)) {
          throw new AffonError('invalid_state', `graph plan cross_entropy_indexed node ${node.id} references missing logits ${node.logits}`)
        }
        if (!byId.has(node.targets)) {
          throw new AffonError('invalid_state', `graph plan cross_entropy_indexed node ${node.id} references missing targets ${node.targets}`)
        }
        if (node.logits >= node.id || node.targets >= node.id) {
          throw new AffonError('invalid_state', `graph plan cross_entropy_indexed node ${node.id} is not in dependency order`)
        }
        break
      case 'topk_values':
      case 'topk_indices':
        if (!byId.has(node.input)) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} references missing input ${node.input}`)
        }
        if (node.input >= node.id) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} is not in dependency order`)
        }
        break
      case 'index_select':
        if (!byId.has(node.input)) {
          throw new AffonError('invalid_state', `graph plan index_select node ${node.id} references missing input ${node.input}`)
        }
        if (!byId.has(node.index)) {
          throw new AffonError('invalid_state', `graph plan index_select node ${node.id} references missing index ${node.index}`)
        }
        if (node.input >= node.id || node.index >= node.id) {
          throw new AffonError('invalid_state', `graph plan index_select node ${node.id} is not in dependency order`)
        }
        break
      case 'gather':
        if (!byId.has(node.input)) {
          throw new AffonError('invalid_state', `graph plan gather node ${node.id} references missing input ${node.input}`)
        }
        if (!byId.has(node.index)) {
          throw new AffonError('invalid_state', `graph plan gather node ${node.id} references missing index ${node.index}`)
        }
        if (node.input >= node.id || node.index >= node.id) {
          throw new AffonError('invalid_state', `graph plan gather node ${node.id} is not in dependency order`)
        }
        break
      case 'sum':
      case 'mean':
      case 'std':
      case 'variance':
      case 'min':
      case 'max':
      case 'argmin':
      case 'argmax':
        if (!byId.has(node.input)) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} references missing input ${node.input}`)
        }
        if (node.input >= node.id) {
          throw new AffonError('invalid_state', `graph plan ${node.kind} node ${node.id} is not in dependency order`)
        }
        break
      default: {
        const exhaustive: never = node
        throw new AffonError('invalid_state', `graph plan contains unsupported node ${(exhaustive as any).kind}`)
      }
    }
  }
}

function createCapturedProgram(state: CaptureState, output: CapturedValue): CapturedProgram {
  const canonical = canonicalize(state.nodes, output.id)
  return {
    inputArity: state.inputArity,
    boundInputCount: state.boundInputs.length,
    inputCount: state.inputArity + state.boundInputs.length,
    outputId: canonical.rootId,
    nodes: canonical.nodes,
  }
}

function createPlan(program: CapturedProgram): GraphPlan {
  const plan = {
    inputCount: program.inputCount,
    outputId: program.outputId,
    nodes: program.nodes,
  }
  validatePlan(plan)
  return plan
}

function summarizeCapturedProgram(program: CapturedProgram): CapturedProgramSummary {
  return {
    inputArity: program.inputArity,
    boundInputCount: program.boundInputCount,
    inputCount: program.inputCount,
    outputId: program.outputId,
    nodeCount: program.nodes.length,
    nodeKinds: Array.from(new Set(program.nodes.map((node) => node.kind))),
  }
}

function summarizeExecutionGraph(program: CapturedProgram, specialized: boolean, loweringAnalysis: GraphLoweringAnalysis): GraphSummary {
  const plan = createPlan(program)
  return {
    inputCount: plan.inputCount,
    output: plan.outputId,
    nodes: summarizeNodes(plan.nodes),
    staticMetadata: {
      kind: 'provisional_capture_metadata',
      source: 'ts_capture',
      fields: ['nodes.outputShape'],
    },
    stats: summarizeStats(plan.nodes),
    regions: [],
    templates: [],
    specialized,
    runtime: 'native-graph',
    summaryKind: 'execution',
    loweringAnalysis,
  }
}

function captureGraph(fn: (...inputs: CapturedValue[]) => CapturedValue, arityOverride?: number, inputMetas?: StaticTensorMeta[]) {
  if (typeof fn !== 'function') {
    throw graphCaptureError('syntax', 'invalid_arg', 'graph() requires a capture function')
  }

  const arity = arityOverride ?? fn.length
  if (!Number.isInteger(arity) || arity < 0) {
    throw graphCaptureError('syntax', 'invalid_arg', 'graph() requires a fixed-arity function')
  }

  const state: CaptureState = {
    nodes: [],
    nextId: 0,
    inputArity: arity,
    boundInputs: [],
    boundInputIds: new Map(),
    modulePathStack: [],
  }
  const previous = currentCapture
  currentCapture = state
  let output: unknown
  try {
    const inputs = Array.from({ length: arity }, (_, index) => makeInput(state, index, inputMetas?.[index]))
    output = fn(...inputs)
  } finally {
    currentCapture = previous
  }

  if (!isCapturedTensor(output)) {
    throw graphCaptureSyntaxError('graph() capture function must return a graph value')
  }

  const capturedProgram = createCapturedProgram(state, output)
  const capturedProgramJson = JSON.stringify(capturedProgram)
  const nativeExecutable = (native as any).$create_compiled_executable_native(capturedProgramJson)
  const program: any = {}
  const materializeInputs = (inputs: NativeTensor[]) => inputs.concat(state.boundInputs.map((resolve) => resolve()))
  const loweringAnalysis = () => nativeExecutable.analyze() as GraphLoweringAnalysis
  let lastExecutionRuntime: GraphExecutionRuntime = 'native-graph'
  const executionRuntime = () => lastExecutionRuntime
  const tryNativeRun = (inputs: NativeTensor[]) => nativeExecutable.run(inputs)
  const exportNativeBundle = (inputs: NativeTensor[], opts?: GraphExportOptions) => nativeExecutable.exportBundle(inputs, opts)
  const extractExecutionPlan = (inputs: NativeTensor[]) => {
    createPlan(capturedProgram)
    return {
      kind: 'graph_plan',
      id: 'captured-graph',
      version: 1,
      context: { inputCount: inputs.length },
      steps: capturedProgram.nodes
        .filter((node) => node.kind !== 'input' && node.kind !== 'constant' && node.kind !== 'rand' && node.kind !== 'randn')
        .map((node) => {
          if (node.kind === 'matmul' && node.execution?.hint) {
            return {
              nodeId: node.id,
              matmul: {
                family: `gemm_${node.execution.hint}`,
                hint: node.execution.hint,
                hint_source: node.execution.source ?? 'api_execution_arg',
              },
            }
          }
          return { nodeId: node.id }
        }),
      regions: [],
      outputs: [capturedProgram.outputId],
    } satisfies ExecutionPlanExport
  }
  program.run = (...inputs: NativeTensor[]) => {
    if (inputs.length !== arity) {
      throw new AffonError('invalid_arg', `graph.run expected ${arity} tensor inputs, received ${inputs.length}`)
    }
    const allInputs = materializeInputs(inputs)
    const result = tryNativeRun(allInputs)
    lastExecutionRuntime = 'native-graph'
    return result
  }
  program.summary = (...inputs: NativeTensor[]) => {
    if (inputs.length !== 0 && inputs.length !== arity) {
      throw new AffonError('invalid_arg', `graph.summary expected ${arity} tensor inputs, received ${inputs.length}`)
    }
    return {
      ...summarizeExecutionGraph(capturedProgram, inputs.length !== 0, loweringAnalysis()),
      nativeState: nativeExecutable.summary(),
    }
  }
  program.plan = (...inputs: NativeTensor[]) => {
    if (inputs.length !== arity) {
      throw new AffonError('invalid_arg', `graph.plan expected ${arity} tensor inputs, received ${inputs.length}`)
    }
    return extractExecutionPlan(materializeInputs(inputs))
  }
  program.exportBundle = (...args: any[]) => {
    let opts: GraphExportOptions | undefined
    let inputs = args as NativeTensor[]
    if (args.length === arity + 1) {
      opts = args[arity] as GraphExportOptions | undefined
      inputs = args.slice(0, arity) as NativeTensor[]
    }
    if (inputs.length !== arity) {
      throw new AffonError('invalid_arg', `graph.exportBundle expected ${arity} tensor inputs, received ${inputs.length}`)
    }
    return exportNativeBundle(materializeInputs(inputs), opts)
  }
  program.exportReport = program.exportBundle
  program.executionRuntime = executionRuntime
  program.loweringAnalysis = loweringAnalysis
  program.nativeState = () => nativeExecutable.summary()
  program.$recordNativePlanOutcome = (outcome: string, kind?: string | null, reason?: string | null, signature?: string | null) =>
    nativeExecutable.recordPlanOutcome(outcome, kind ?? null, reason ?? null, signature ?? null)
  program.$recordNativeFallback = (kind?: string | null, reason?: string | null, signature?: string | null) =>
    nativeExecutable.recordFallback(kind ?? null, reason ?? null, signature ?? null)
  program.$recordNativeSpecializationCapture = (signature?: string | null) =>
    nativeExecutable.recordSpecializationCapture(signature ?? null)
  program.$recordNativeSpecializationReuse = (signature?: string | null) =>
    nativeExecutable.recordSpecializationReuse(signature ?? null)
  program.capturedProgram = () => ({
    inputArity: capturedProgram.inputArity,
    boundInputCount: capturedProgram.boundInputCount,
    inputCount: capturedProgram.inputCount,
    outputId: capturedProgram.outputId,
    nodes: capturedProgram.nodes.map((node) => ({ ...node })),
  })
  program.captureSummary = () => summarizeCapturedProgram(capturedProgram)
  return Object.freeze(program)
}

function graph(fn: (...inputs: CapturedValue[]) => CapturedValue) {
  return captureGraph(fn)
}

function graphWithArity(arity: number, fn: (...inputs: CapturedValue[]) => CapturedValue) {
  return captureGraph(fn, arity)
}

function graphWithInputMetas(
  arity: number,
  inputMetas: StaticTensorMeta[],
  fn: (...inputs: CapturedValue[]) => CapturedValue,
) {
  return captureGraph(fn, arity, inputMetas)
}

function isCapturing(): boolean {
  return currentCapture !== null
}

function captureBoundTensor(key: unknown, resolve: () => NativeTensor): NativeTensor | CapturedValue {
  if (!currentCapture) return resolve()
  return makeBoundInput(currentCapture, key, resolve)
}

function captureUnary(kind: UnaryKind, opName: string, value: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(value)) return pushUnary(getCaptureState(opName), kind, value)
  return (native as any)[opName](value)
}

function captureRandom(kind: 'rand' | 'randn', shape: unknown, dtype: unknown): NativeTensor | CapturedValue | null {
  if (!currentCapture) return null
  if (!Array.isArray(shape) || shape.some((dim) => !Number.isInteger(dim) || dim < 0)) {
    throw graphCaptureSyntaxError(`${kind} expects a non-negative integer shape array`)
  }
  if (dtype !== 'f32' && dtype !== 'f64' && dtype !== 'i64') {
    throw graphCaptureSyntaxError(`${kind} expects dtype to be "f32", "f64", or "i64"`)
  }
  return pushRandom(getCaptureState(kind), kind, shape as number[], dtype)
}

function captureClamp(value: unknown, min: unknown, max: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(min) || isCapturedTensor(max)) {
    throw graphCaptureAdapterError('clamp graph capture requires finite scalar min and max')
  }
  if (!isFiniteNumber(min) || !isFiniteNumber(max)) {
    throw graphCaptureSyntaxError('clamp graph capture requires finite scalar min and max')
  }
  if (isCapturedTensor(value)) return pushClamp(getCaptureState('clamp'), value, min, max)
  return native.clamp(value as any, min, max)
}

function captureReshape(value: unknown, shape: unknown, opts?: unknown): NativeTensor | CapturedValue {
  if (!Array.isArray(shape) || !shape.every((dim) => Number.isInteger(dim) && dim >= 0)) {
    throw graphCaptureSyntaxError('reshape graph capture requires a non-negative integer shape array')
  }
  if (isCapturedTensor(value)) return pushReshape(getCaptureState('reshape'), value, shape as number[])
  if (opts === undefined) return native.reshape(value as any, shape as number[])
  return native.reshape(value as any, shape as number[], opts as any)
}

function captureSlice(value: unknown, ranges: unknown): NativeTensor | CapturedValue {
  if (!Array.isArray(ranges) || !ranges.every((entry) => typeof entry === 'string' || Number.isInteger(entry))) {
    throw graphCaptureSyntaxError('slice graph capture requires an array of number or string ranges')
  }
  if (isCapturedTensor(value)) return pushSlice(getCaptureState('slice'), value, ranges as SliceRangeSpec[])
  return native.slice(value as any, ranges as SliceRangeSpec[])
}

function captureSqueeze(value: unknown, axis?: unknown): NativeTensor | CapturedValue {
  if (axis != null && !Number.isInteger(axis)) {
    throw graphCaptureSyntaxError('squeeze graph capture requires axis to be an integer when provided')
  }
  if (isCapturedTensor(value)) return pushSqueeze(getCaptureState('squeeze'), value, axis as number | undefined)
  if (axis === undefined) return native.squeeze(value as any)
  return native.squeeze(value as any, axis as any)
}

function captureUnsqueeze(value: unknown, axis: unknown): NativeTensor | CapturedValue {
  if (!Number.isInteger(axis)) {
    throw graphCaptureSyntaxError('unsqueeze graph capture requires an integer axis')
  }
  if (isCapturedTensor(value)) return pushUnsqueeze(getCaptureState('unsqueeze'), value, axis as number)
  return native.unsqueeze(value as any, axis as any)
}

function captureTranspose(value: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(value)) return pushTranspose(getCaptureState('transpose'), value)
  return native.transpose(value as any)
}

function capturePermute(value: unknown, axes: unknown): NativeTensor | CapturedValue {
  if (!Array.isArray(axes) || !axes.every((axis) => Number.isInteger(axis) && axis >= 0)) {
    throw graphCaptureSyntaxError('permute graph capture requires a non-negative integer axes array')
  }
  if (isCapturedTensor(value)) return pushPermute(getCaptureState('permute'), value, axes as number[])
  return native.permute(value as any, axes as number[])
}

function captureContiguous(value: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(value)) return pushContiguous(getCaptureState('contiguous'), value)
  return native.contiguous(value as any)
}

function captureCast(value: unknown, dtype: unknown): NativeTensor | CapturedValue {
  if (dtype !== 'f32' && dtype !== 'f64') {
    throw graphCaptureAdapterError('cast graph capture currently supports f32 and f64 dtypes only')
  }
  if (isCapturedTensor(value)) return pushCast(getCaptureState('cast'), value, dtype)
  return native.cast(value as any, dtype as any)
}

function captureGelu(value: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(value)) return pushGelu(getCaptureState('gelu'), value)
  return native.gelu(value as any)
}

function captureDot(left: unknown, right: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(left) || isCapturedTensor(right)) {
    const state = getCaptureState('dot')
    return pushDot(
      state,
      normalizeBinaryCaptureOperand(state, 'dot', left),
      normalizeBinaryCaptureOperand(state, 'dot', right),
    )
  }
  return native.dot(left as any, right as any)
}

function captureMatmul(left: unknown, right: unknown, execution?: MatmulExecutionOptions): NativeTensor | CapturedValue {
  if (isCapturedTensor(left) || isCapturedTensor(right)) {
    const state = getCaptureState('matmul')
    return pushMatmul(
      state,
      normalizeBinaryCaptureOperand(state, 'matmul', left),
      normalizeBinaryCaptureOperand(state, 'matmul', right),
      execution,
    )
  }
  if (execution === undefined) return native.matmul(left as any, right as any)
  return native.matmul(left as any, right as any, normalizeMatmulExecution(execution))
}

function normalizeMatmulExecution(execution?: MatmulExecutionOptions): MatmulExecutionOptions | undefined {
  if (execution == null) return undefined
  if (typeof execution !== 'object') {
    throw graphCaptureSyntaxError('matmul execution metadata must be an object')
  }
  if (execution.hint == null) {
    if (execution.source != null) {
      throw graphCaptureSyntaxError('matmul execution source requires an execution hint')
    }
    return undefined
  }
  if (execution.hint !== 'projection' && execution.hint !== 'attention_scores' && execution.hint !== 'attention_values') {
    throw graphCaptureSyntaxError('matmul execution hint must be one of projection, attention_scores, attention_values')
  }
  if (execution.source != null && execution.source !== 'higher_level_module') {
    throw graphCaptureSyntaxError('matmul execution source must be higher_level_module')
  }
  return execution.source == null ? { hint: execution.hint } : { hint: execution.hint, source: execution.source }
}

function normalizeTensorListCaptureOperand(opName: string, values: unknown): CapturedValue[] | null {
  if (!Array.isArray(values)) {
    throw graphCaptureSyntaxError(`${opName} graph capture requires an input array`)
  }
  const hasGraph = values.some((value) => isCapturedTensor(value))
  if (!hasGraph) return null
  if (values.length === 0) {
    throw graphCaptureSyntaxError(`${opName} graph capture requires at least one input tensor`)
  }
  if (!values.every((value) => isCapturedTensor(value))) {
    throw graphCaptureAdapterError(`${opName} graph capture requires all inputs to be graph values when any input is captured`)
  }
  return values as CapturedValue[]
}

function captureCat(inputs: unknown, dim?: unknown): NativeTensor | CapturedValue {
  if (dim === undefined) {
    if (!currentCapture) return native.cat(inputs as any)
    dim = 0
  }
  if (!Number.isInteger(dim)) {
    throw graphCaptureSyntaxError('cat graph capture requires an integer dim')
  }
  if (!currentCapture) return native.cat(inputs as any, dim as any)
  const captured = normalizeTensorListCaptureOperand('cat', inputs)
  if (captured) return pushCat(getCaptureState('cat'), captured, dim as number)
  return native.cat(inputs as any, dim as any)
}

function captureStack(inputs: unknown, dim?: unknown): NativeTensor | CapturedValue {
  if (dim === undefined) {
    if (!currentCapture) return native.stack(inputs as any)
    dim = 0
  }
  if (!Number.isInteger(dim)) {
    throw graphCaptureSyntaxError('stack graph capture requires an integer dim')
  }
  if (!currentCapture) return native.stack(inputs as any, dim as any)
  const captured = normalizeTensorListCaptureOperand('stack', inputs)
  if (captured) return pushStack(getCaptureState('stack'), captured, dim as number)
  return native.stack(inputs as any, dim as any)
}

function captureWhere(cond: unknown, onTrue: unknown, onFalse: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(cond) || isCapturedTensor(onTrue) || isCapturedTensor(onFalse)) {
    const state = getCaptureState('where')
    return pushWhere(
      state,
      normalizeBinaryCaptureOperand(state, 'where', cond),
      normalizeBinaryCaptureOperand(state, 'where', onTrue),
      normalizeBinaryCaptureOperand(state, 'where', onFalse),
    )
  }
  return native.where(cond as any, onTrue as any, onFalse as any)
}

function captureMaskedFill(input: unknown, mask: unknown, value: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(input) || isCapturedTensor(mask)) {
    if (!isFiniteNumber(value)) {
      throw graphCaptureSyntaxError('masked_fill graph capture requires a finite scalar fill value')
    }
    const state = getCaptureState('masked_fill')
    return pushMaskedFill(
      state,
      normalizeBinaryCaptureOperand(state, 'masked_fill', input),
      normalizeBinaryCaptureOperand(state, 'masked_fill', mask),
      value,
    )
  }
  return native.masked_fill(input as any, mask as any, value as any)
}

function captureSoftmax(input: unknown, dim: unknown): NativeTensor | CapturedValue {
  if (!Number.isInteger(dim)) {
    throw graphCaptureSyntaxError('softmax graph capture requires an integer dim')
  }
  if (isCapturedTensor(input)) return pushSoftmax(getCaptureState('softmax'), input, dim)
  return native.softmax(input as any, dim)
}

function captureCrossEntropyIndexed(logits: unknown, targets: unknown, axis: unknown): NativeTensor | CapturedValue {
  if (!Number.isInteger(axis)) {
    throw graphCaptureSyntaxError('cross_entropy_indexed graph capture requires an integer axis')
  }
  if ((axis as number) !== 1) {
    throw graphCaptureAdapterError('cross_entropy_indexed graph capture currently supports axis=1 only')
  }
  if (isCapturedTensor(logits) || isCapturedTensor(targets)) {
    const state = getCaptureState('cross_entropy_indexed')
    return pushCrossEntropyIndexed(
      state,
      normalizeBinaryCaptureOperand(state, 'cross_entropy_indexed', logits),
      normalizeBinaryCaptureOperand(state, 'cross_entropy_indexed', targets),
      axis as number,
    )
  }
  return (native as any).cross_entropy_indexed(logits, targets)
}

function captureOneHot(input: unknown, numClasses: unknown): NativeTensor | CapturedValue {
  if (!Number.isInteger(numClasses) || (numClasses as number) < 0) {
    throw graphCaptureSyntaxError('one_hot graph capture requires a non-negative integer num_classes')
  }
  if (isCapturedTensor(input)) return pushOneHot(getCaptureState('one_hot'), input, numClasses as number)
  return native.one_hot(input as any, numClasses as any)
}

function captureTopK(input: unknown, k: unknown, dim?: unknown): TopKResult | { values: CapturedValue; indices: CapturedValue } {
  if (!Number.isInteger(k) || (k as number) <= 0) {
    throw graphCaptureSyntaxError('topk graph capture requires a positive integer k')
  }
  const normalizedDim = dim == null ? -1 : dim
  if (!Number.isInteger(normalizedDim)) {
    throw graphCaptureSyntaxError('topk graph capture requires dim to be an integer when provided')
  }
  if (isCapturedTensor(input)) return pushTopK(getCaptureState('topk'), input, k as number, normalizedDim as number)
  if (dim === undefined) return native.topk(input as any, k as any)
  return native.topk(input as any, k as any, dim as any)
}

function captureIndexSelect(input: unknown, dim: unknown, index: unknown): NativeTensor | CapturedValue {
  if (!Number.isInteger(dim)) {
    throw graphCaptureSyntaxError('index_select graph capture requires an integer dim')
  }
  if (isCapturedTensor(input) || isCapturedTensor(index)) {
    const state = getCaptureState('index_select')
    return pushIndexSelect(
      state,
      normalizeBinaryCaptureOperand(state, 'index_select', input),
      dim as number,
      normalizeBinaryCaptureOperand(state, 'index_select', index),
    )
  }
  return native.index_select(input as any, dim as any, index as any)
}

function captureGather(input: unknown, dim: unknown, index: unknown): NativeTensor | CapturedValue {
  if (!Number.isInteger(dim)) {
    throw graphCaptureSyntaxError('gather graph capture requires an integer dim')
  }
  if (isCapturedTensor(input) || isCapturedTensor(index)) {
    const state = getCaptureState('gather')
    return pushGather(
      state,
      normalizeBinaryCaptureOperand(state, 'gather', input),
      dim,
      normalizeBinaryCaptureOperand(state, 'gather', index),
    )
  }
  return native.gather(input as any, dim, index as any)
}

function captureReduction(kind: ReductionKind, input: unknown, axis?: unknown, keepdim?: unknown): NativeTensor | CapturedValue {
  if (axis != null && !Number.isInteger(axis)) {
    throw graphCaptureSyntaxError(`${kind} graph capture requires axis to be an integer when provided`)
  }
  if (keepdim != null && typeof keepdim !== 'boolean') {
    throw graphCaptureSyntaxError(`${kind} graph capture requires keepdim to be boolean when provided`)
  }
  if (isCapturedTensor(input)) return pushReduction(getCaptureState(kind), kind, input, axis as number | undefined, keepdim as boolean | undefined)
  if (axis === undefined) return (native as any)[kind](input)
  if (keepdim === undefined) return (native as any)[kind](input, axis)
  return (native as any)[kind](input, axis, keepdim)
}

function captureIndexReduction(kind: IndexReductionKind, input: unknown, axis?: unknown, keepdim?: unknown): NativeTensor | CapturedValue {
  if (axis != null && !Number.isInteger(axis)) {
    throw graphCaptureSyntaxError(`${kind} graph capture requires axis to be an integer when provided`)
  }
  if (keepdim != null && typeof keepdim !== 'boolean') {
    throw graphCaptureSyntaxError(`${kind} graph capture requires keepdim to be boolean when provided`)
  }
  if (isCapturedTensor(input)) return pushIndexReduction(getCaptureState(kind), kind, input, axis as number | undefined, keepdim as boolean | undefined)
  if (axis === undefined) return (native as any)[kind](input)
  if (keepdim === undefined) return (native as any)[kind](input, axis)
  return (native as any)[kind](input, axis, keepdim)
}

function captureBinary(kind: BinaryKind, opName: string, left: unknown, right: unknown): NativeTensor | CapturedValue {
  if (isCapturedTensor(left) || isCapturedTensor(right)) {
    const state = getCaptureState(opName)
    return pushBinary(
      state,
      kind,
      normalizeBinaryCaptureOperand(state, opName, left),
      normalizeBinaryCaptureOperand(state, opName, right),
    )
  }
  return (native as any)[opName](left, right)
}

function rejectIfGraphArgs(opName: string, values: unknown[]): void {
  for (let i = 0; i < values.length; i++) {
    if (isCapturedTensor(values[i])) unsupportedGraphOp(opName)
  }
}

function graphTensor(data: any, opts?: { dtype?: TensorDType }): CapturedValue | undefined {
  if (!currentCapture) return undefined
  if (isCapturedTensor(data)) {
    throw graphCaptureAdapterError('graph capture does not support wrapping graph values with tensor()')
  }
  if (typeof data === 'number') {
    return pushConstant(currentCapture, data as Nested, opts?.dtype ?? 'f32')
  }
  if (Array.isArray(data)) {
    return makeBoundInput(currentCapture, data, () => native.tensor(data as any, { dtype: opts?.dtype ?? 'f32' }))
  }
  return undefined
}

function isCapturedTensor(value: unknown): value is CapturedValue {
  return !!value && typeof value === 'object' && (value as any)[GRAPH_VALUE] === true
}

export { graph, graphWithArity, graphWithInputMetas, graphTensor, isCapturing, withCaptureModulePath, captureBoundTensor, captureUnary, captureBinary, captureClamp, captureReshape, captureSlice, captureSqueeze, captureUnsqueeze, captureTranspose, capturePermute, captureContiguous, captureCast, captureGelu, captureDot, captureMatmul, captureCat, captureStack, captureWhere, captureMaskedFill, captureSoftmax, captureCrossEntropyIndexed, captureOneHot, captureTopK, captureIndexSelect, captureGather, captureReduction, captureIndexReduction, captureRandom, rejectIfGraphArgs }

export default {
  graph,
  graphWithArity,
  graphWithInputMetas,
  graphTensor,
  isCapturing,
  withCaptureModulePath,
  captureBoundTensor,
  captureUnary,
  captureBinary,
  captureClamp,
  captureReshape,
  captureSlice,
  captureSqueeze,
  captureUnsqueeze,
  captureTranspose,
  capturePermute,
  captureContiguous,
  captureCast,
  captureGelu,
  captureDot,
  captureMatmul,
  captureCat,
  captureStack,
  captureWhere,
  captureMaskedFill,
  captureSoftmax,
  captureCrossEntropyIndexed,
  captureOneHot,
  captureTopK,
  captureIndexSelect,
  captureGather,
  captureReduction,
  captureIndexReduction,
  captureRandom,
  rejectIfGraphArgs,
}
