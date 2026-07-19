class AffonError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = 'AffonError'
    this.code = code
  }
}

type ComputeTensorLike = {
  shape: number[]
  dtype: 'f32' | 'f64' | 'i64'
}

type CompilableProgram = {
  kind: string
  fn: (...args: any[]) => any
  params: Record<string, ComputeTensorLike> | null
  source?: any
}

type GraphSupport = {
  graph: (fn: (...inputs: any[]) => any) => any
  graphWithArity: (arity: number, fn: (...inputs: any[]) => any) => any
  graphWithInputMetas: (
    arity: number,
    metas: Array<{ shape: number[]; dtype: 'f32' | 'f64' | 'i64' }>,
    fn: (...inputs: any[]) => any,
  ) => any
  isCapturing: () => boolean
}

type CompileHelpers = {
  graphSupport: GraphSupport
  isTensorLike: (x: any) => x is ComputeTensorLike
  programStateKey: (source: any) => string
  programArity: (source: any, fallback: (...args: any[]) => any) => number
  replayCapturedProgram: (source: any, userInputs: any[]) => any
  isProgramSource: (source: any) => boolean
  isReplayableProgramSource: (source: any) => boolean
  finalizeExecutable?: (executable: any, original: any) => any
}

type TensorSignature = ReadonlyArray<Readonly<{ dtype: 'f32' | 'f64' | 'i64'; shape: readonly number[] }>>
type FallbackKind = 'tensor_signature_changed' | 'program_state_changed' | 'graph_capture_failed' | 'no_graph_available'
type GraphCaptureFailureStage = 'syntax' | 'capture_adapter' | 'semantic_validation' | 'unknown'
type GraphCaptureFailure = {
  stage: GraphCaptureFailureStage
  reason: string
}

function cloneTensorSignature(signature: TensorSignature | null): Array<{ dtype: 'f32' | 'f64' | 'i64'; shape: number[] }> | null {
  if (signature == null) return null
  return signature.map((entry) => ({ dtype: entry.dtype, shape: [...entry.shape] }))
}

function tensorSignatureLabel(signature: TensorSignature | null): string | null {
  const cloned = cloneTensorSignature(signature)
  return cloned == null ? null : JSON.stringify(cloned)
}

function currentProgramStateKeyFor(source: any, programStateKey: (source: any) => string): string {
  if (source == null) return 'default'
  return programStateKey(source)
}

function describeCompileError(error: unknown): string {
  if (error instanceof Error && typeof error.message === 'string' && error.message.length > 0) {
    return error.message
  }
  return String(error)
}

function classifyGraphCaptureFailure(error: unknown): GraphCaptureFailure {
  const reason = describeCompileError(error)
  const taggedStage = (error as any)?.graphCaptureStage
  if (
    taggedStage === 'syntax' ||
    taggedStage === 'capture_adapter' ||
    taggedStage === 'semantic_validation' ||
    taggedStage === 'unknown'
  ) {
    return { stage: taggedStage, reason }
  }
  if (reason.includes('graph() requires a capture function')) {
    return { stage: 'syntax', reason }
  }
  if (
    reason.includes('graph capture') ||
    reason.includes('during graph capture') ||
    reason.includes('finite_summary') ||
    reason.includes('expects compute values/parameters')
  ) {
    return { stage: 'capture_adapter', reason }
  }
  return { stage: 'capture_adapter', reason }
}

export function compileWithHelpers(programOrFn: any, helpers: CompileHelpers): any {
  const {
    graphSupport,
    isTensorLike,
    programStateKey,
    programArity,
    replayCapturedProgram,
    isProgramSource,
    isReplayableProgramSource,
    finalizeExecutable,
  } = helpers

  const specializationKey = (source: any, inputs: ComputeTensorLike[]): string =>
    `${currentProgramStateKeyFor(source, programStateKey)}|${inputs.map((input) => `${input.dtype}:${input.shape.join('x')}`).join('|')}`

  const tryGraphProgram = (program: CompilableProgram): { graphProgram: any | null; failure: GraphCaptureFailure | null } => {
    try {
      if (
        isReplayableProgramSource(program.source) &&
        programStateKey(program.source) === 'default'
      ) {
        const runArity = programArity(program.source, program.fn)
        return {
          graphProgram: graphSupport.graphWithArity(runArity, (...capturedInputs: any[]) =>
            replayCapturedProgram(program.source, capturedInputs)),
          failure: null,
        }
      }
      if (
        isProgramSource(program.source)
      ) {
        const runArity = programArity(program.source, program.fn)
        return {
          graphProgram: graphSupport.graphWithArity(runArity, (...capturedInputs: any[]) => program.fn(...capturedInputs)),
          failure: null,
        }
      }
      const runArity = programArity(program.source, program.fn)
      return {
        graphProgram: graphSupport.graphWithArity(runArity, (...capturedInputs: any[]) => program.fn(...capturedInputs)),
        failure: null,
      }
    } catch (error) {
      return { graphProgram: null, failure: classifyGraphCaptureFailure(error) }
    }
  }

  const makeExecutable = (program: CompilableProgram, original: any) => {
    const initialGraphAttempt = tryGraphProgram(program)
    const graphProgram = initialGraphAttempt.graphProgram
    let lastSpecializedGraph: any | null = null
    let lastSpecializationKey: string | null = null
    let lastRuntimeTensorSignature: TensorSignature | null = null
    let lastGraphCaptureFailure: GraphCaptureFailure | null = initialGraphAttempt.failure
    let lastGraphCaptureError: string | null = initialGraphAttempt.failure?.reason ?? null
    let specializationCaptureCount = 0
    let specializationReuseCount = 0
    let eagerFallbackCount = 0
    let lastExecutionMode: 'graph' | 'eager-forward' = graphProgram ? 'graph' : 'eager-forward'
    let lastFallbackKind: FallbackKind | null = initialGraphAttempt.failure ? 'graph_capture_failed' : null
    let lastFallbackReason: string | null = initialGraphAttempt.failure?.reason ?? null
    const hasTensorArgs = (args: any[]) => args.length > 0 && args.every((arg) => isTensorLike(arg))
    const tensorSignatureForArgs = (args: any[]): TensorSignature | null =>
      hasTensorArgs(args)
        ? (args as ComputeTensorLike[]).map((arg) => ({ dtype: arg.dtype, shape: [...arg.shape] }))
        : null
    const sameTensorSignature = (left: TensorSignature | null, right: TensorSignature | null): boolean => {
      if (left == null || right == null) return false
      if (left.length !== right.length) return false
      for (let i = 0; i < left.length; i++) {
        if (left[i].dtype !== right[i].dtype) return false
        if (left[i].shape.length !== right[i].shape.length) return false
        for (let j = 0; j < left[i].shape.length; j++) {
          if (left[i].shape[j] !== right[i].shape[j]) return false
        }
      }
      return true
    }
    const specializationKeyForArgs = (args: any[]) =>
      hasTensorArgs(args) ? specializationKey(program.source, args as ComputeTensorLike[]) : null
    const fallbackReasonForTensorArgs = (): string => {
      if (lastSpecializationKey != null) {
        return 'program state changed'
      }
      return lastGraphCaptureError ?? 'no compatible specialization available'
    }
    const fallbackKindForTensorArgs = (): FallbackKind => {
      if (lastSpecializationKey != null) {
        return 'program_state_changed'
      }
      return 'graph_capture_failed'
    }
    const graphNativeState = (graph: any | null) => graph?.nativeState?.() ?? null
    const lastKnownNativeState = () => graphNativeState(lastSpecializedGraph) ?? graphNativeState(graphProgram)
    const recordNativeEagerFallbackOutcome = (graph: any | null, kind: FallbackKind, reason: string | null, signature: TensorSignature | null) => {
      graph?.$recordNativePlanOutcome?.('eager-forward', kind, reason, tensorSignatureLabel(signature))
      graph?.$recordNativeFallback?.(kind, reason, tensorSignatureLabel(signature))
    }
    const recordNativeSpecializationCapture = (graph: any | null, signature: TensorSignature | null) => {
      graph?.$recordNativeSpecializationCapture?.(tensorSignatureLabel(signature))
    }
    const recordNativeSpecializationReuse = (graph: any | null, signature: TensorSignature | null) => {
      graph?.$recordNativeSpecializationReuse?.(tensorSignatureLabel(signature))
    }
    const exactSpecializedGraphForArgs = (args: any[]) => {
      const key = specializationKeyForArgs(args)
      if (key == null) return null
      return key === lastSpecializationKey ? lastSpecializedGraph : null
    }
    const currentGraphProgram = (args?: any[]) => {
      if (args) {
        if (hasTensorArgs(args)) {
          return exactSpecializedGraphForArgs(args) ?? trySpecializedGraphProgram(args)
        }
        return exactSpecializedGraphForArgs(args) ?? graphProgram
      }
      if (graphProgram) return graphProgram
      return lastSpecializedGraph
    }
    const trySpecializedGraphProgram = (args: any[]) => {
      const key = specializationKeyForArgs(args)
      if (key == null) return null
      if (lastSpecializationKey != null && key !== lastSpecializationKey) {
        return null
      }
      const tensorArgs = args as ComputeTensorLike[]
      try {
        const captured = graphSupport.graphWithInputMetas(
          tensorArgs.length,
          tensorArgs.map((input) => ({ shape: [...input.shape], dtype: input.dtype as 'f32' | 'f64' | 'i64' })),
          (...capturedInputs: any[]) => {
            if (isReplayableProgramSource(program.source)) {
              return replayCapturedProgram(program.source, capturedInputs)
            }
            return program.fn(...capturedInputs)
          },
        )
        lastSpecializedGraph = captured
        lastSpecializationKey = key
        lastGraphCaptureError = null
        specializationCaptureCount += 1
        recordNativeSpecializationCapture(captured, tensorSignatureForArgs(args))
        return captured
      } catch (error) {
        lastGraphCaptureFailure = classifyGraphCaptureFailure(error)
        lastGraphCaptureError = lastGraphCaptureFailure.reason
        lastSpecializedGraph = null
        lastSpecializationKey = null
        return null
      }
    }
    const executeInline = (...args: any[]) => {
      if (graphSupport.isCapturing() && isReplayableProgramSource(program.source)) {
        return replayCapturedProgram(program.source, args)
      }
      return program.fn(...args)
    }
    const executeCompiled = (...args: any[]) => {
      if (graphSupport.isCapturing()) return executeInline(...args)
      if (hasTensorArgs(args)) {
        const signature = tensorSignatureForArgs(args)
        if (lastRuntimeTensorSignature != null && !sameTensorSignature(signature, lastRuntimeTensorSignature)) {
          eagerFallbackCount += 1
          lastExecutionMode = 'eager-forward'
          lastFallbackKind = 'tensor_signature_changed'
          lastFallbackReason = 'tensor signature changed'
          recordNativeEagerFallbackOutcome(lastSpecializedGraph ?? graphProgram, lastFallbackKind, lastFallbackReason, signature)
          return program.fn(...args)
        }
        const cached = exactSpecializedGraphForArgs(args)
        if (cached) {
          specializationReuseCount += 1
          lastExecutionMode = 'graph'
          lastFallbackKind = null
          lastFallbackReason = null
          lastRuntimeTensorSignature = signature
          recordNativeSpecializationReuse(cached, signature)
          return cached.run(...args)
        }
        const specialized = trySpecializedGraphProgram(args)
        if (specialized) {
          lastRuntimeTensorSignature = signature
          lastExecutionMode = 'graph'
          lastFallbackKind = null
          lastFallbackReason = null
          return specialized.run(...args)
        }
        eagerFallbackCount += 1
        lastExecutionMode = 'eager-forward'
        lastFallbackKind = fallbackKindForTensorArgs()
        lastFallbackReason = fallbackReasonForTensorArgs()
        recordNativeEagerFallbackOutcome(lastSpecializedGraph ?? graphProgram, lastFallbackKind, lastFallbackReason, signature)
        return program.fn(...args)
      }
      const graph = currentGraphProgram(args)
      if (graph) {
        lastExecutionMode = 'graph'
        lastFallbackKind = null
        lastFallbackReason = null
        return graph.run(...args)
      }
      eagerFallbackCount += 1
      lastExecutionMode = 'eager-forward'
      lastFallbackKind = 'no_graph_available'
      lastFallbackReason = lastGraphCaptureError ?? 'no graph available'
      recordNativeEagerFallbackOutcome(graphProgram ?? lastSpecializedGraph, lastFallbackKind, lastFallbackReason, null)
      return program.fn(...args)
    }
    const executable: any = (...args: any[]) => executeCompiled(...args)
    executable.run = (...args: any[]) => executeCompiled(...args)
    executable.computation = () => program
    executable.graph = (...args: any[]) => currentGraphProgram(args)
    executable.plan = (...args: any[]) => currentGraphProgram(args)?.plan?.(...args) ?? null
    executable.capturedProgram = (...args: any[]) => currentGraphProgram(args)?.capturedProgram?.()
    executable.captureSummary = (...args: any[]) => currentGraphProgram(args)?.captureSummary?.()
    executable.exportBundle = (...args: any[]) => {
      let graphArgs = args
      let maybeOpts: any = undefined
      if (args.length >= 1) {
        const tensorPrefix = args.slice(0, -1)
        if (hasTensorArgs(tensorPrefix) && !isTensorLike(args[args.length - 1])) {
          graphArgs = tensorPrefix
          maybeOpts = args[args.length - 1]
        }
      }
      const graph = currentGraphProgram(graphArgs)
      if (graph && maybeOpts === undefined && args.length >= 1 && args.length === graph.capturedProgram().inputArity + 1) {
        maybeOpts = args[args.length - 1]
        graphArgs = args.slice(0, -1)
      }
      const resolved = currentGraphProgram(graphArgs)
      return resolved?.exportBundle?.(...graphArgs, maybeOpts)
    }
    executable.exportReport = (...args: any[]) => executable.exportBundle(...args)
    executable.summary = (...args: any[]) => {
      const graph = args.length > 0 ? currentGraphProgram(args) : currentGraphProgram()
      const plan = args.length > 0 ? graph?.plan?.(...args) ?? null : null
      return {
        kind: program.kind,
        hasDefinedParams: !!program.params,
        paramCount: program.params ? Object.keys(program.params).length : 0,
        mode: graph ? 'graph' : 'eager-forward',
        graphRuntime: graph?.executionRuntime?.() ?? null,
        graphLoweringAnalysis: graph?.loweringAnalysis?.() ?? null,
        graphCaptureError: graph ? null : lastGraphCaptureError,
        graphCaptureFailure: graph ? null : lastGraphCaptureFailure,
        planAvailable: plan !== null,
        planStepCount: plan?.steps.length ?? null,
        planRegionCount: plan?.regions.length ?? null,
        planOutputCount: plan?.outputs.length ?? null,
        specializationActive: lastSpecializationKey !== null,
        activeTensorSignature: cloneTensorSignature(lastRuntimeTensorSignature),
        specializationCaptureCount,
        specializationReuseCount,
        eagerFallbackCount,
        nativeState: graphNativeState(graph) ?? lastKnownNativeState(),
        lastExecutionMode,
        lastFallbackKind,
        lastFallbackReason,
      }
    }
    return finalizeExecutable ? finalizeExecutable(executable, original) : executable
  }

  if (
    isProgramSource(programOrFn) &&
    typeof programOrFn.run === 'function'
  ) {
    return makeExecutable({
      kind: programOrFn.kind || 'defined',
      fn: typeof programOrFn === 'function' ? programOrFn : programOrFn.run.bind(programOrFn),
      params: programOrFn.params || null,
      source: programOrFn,
    }, programOrFn)
  }

  if (typeof programOrFn === 'function') {
    return makeExecutable({ kind: 'pure', fn: programOrFn, params: null, source: programOrFn }, programOrFn)
  }

  throw new AffonError('invalid_arg', 'compile: expected a callable program or program object')
}
