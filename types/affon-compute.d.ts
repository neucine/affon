declare module "affon:compute" {
  /**
   * @summary Tensor shape tuple.
   * @category Core
   * @semantics
   * Describes the public shape of a compute tensor as a readonly tuple of axis
   * sizes.
   */
  type Shape = readonly number[];
  /**
   * @summary Semantic tensor axis names.
   * @category Core
   * @semantics
   * Optional labels over positional tensor axes. Axis names preserve human
   * meaning while execution still lowers to numeric axis indices.
   */
  type AxisName = string;
  /**
   * @summary Conventional axis-name constants.
   * @category Core
   * @semantics
   * Reusable string labels for common tensor axes. These are conventions, not a
   * closed vocabulary; custom axis names remain valid.
   */
  const axes: Readonly<{
    batch: "batch";
    token: "token";
    sequence: "sequence";
    feature: "feature";
    hidden: "hidden";
    channel: "channel";
    height: "height";
    width: "width";
    head: "head";
    vocab: "vocab";
  }>;
  /**
   * @summary Tensor creation options.
   * @category Creation
   * @semantics
   * Configures dtype, device placement, and optional semantic axis names.
   */
  interface TensorOptions<D extends DType = DType, V extends Device = Device> {
    dtype?: D;
    device?: V;
    axes?: readonly AxisName[];
  }
  /**
   * @summary Public compute tensor dtypes.
   * @category Core
   * @semantics
   * Enumerates the element dtypes currently exposed by `affon:compute`.
   */
  type DType = "f32" | "f64" | "i64";
  /**
   * @summary Public compute execution devices.
   * @category Core
   * @semantics
   * Enumerates the device placements currently exposed by `affon:compute`.
   */
  type Device = "cpu" | "metal";
  /**
   * @summary Callable compute program.
   * @category Core
   * @semantics
   * A `Program` is the primary callable computation boundary for compilation in
   * `affon:compute`.
   */
  type Program<Args extends readonly unknown[] = readonly unknown[], Out = unknown> = (...args: Args) => Out;
  /**
   * @summary Public execution mode for modeful compute modules.
   * @category Modules
   * @semantics
   * Distinguishes training and evaluation behavior for stateful callable
   * modules.
   */
  type ModuleMode = "train" | "eval";
  type MatmulExecutionHint = "projection" | "attention_scores" | "attention_values";
  type MatmulExecutionSource = "higher_level_module";
  type MatmulExecutionOptions = { hint?: MatmulExecutionHint; source?: MatmulExecutionSource };
  /**
   * @summary Concrete numeric tensor in `affon:compute`.
   * @category Core
   * @semantics
   * `Tensor` is the public numeric object of the compute API. It carries shape,
   * dtype, device, and data access behavior, but it is not the public training
   * target concept. Trainable state is represented by {@link Parameter}.
   * @usecase
   * Use `Tensor` values for data, activations, metrics, and intermediate
   * computation results.
   */
  interface Tensor<S extends Shape = number[], D extends DType = DType, V extends Device = Device> {
    /** Tensor shape. */
    readonly shape: S;
    /** Optional semantic names for each positional axis. */
    readonly axes?: readonly AxisName[];
    /** Optional alias for rank when preserved by the runtime wrapper. */
    readonly rank?: number;
    /** Element dtype. */
    readonly dtype: D;
    /** Current device placement. */
    readonly device: V;
    /** Retained gradient when this tensor was explicitly tracked by runtime internals. */
    readonly grad?: Tensor<S, D> | null;
    /** Number of dimensions. */
    readonly ndim: number;
    /**
     * @summary Read a scalar tensor as a JavaScript number.
     * @category Core
     * @semantics
     * Materializes a rank-0 tensor as a plain JavaScript number.
     */
    item(): number;
    /**
     * @summary Move a tensor to another device.
     * @category Core
     * @semantics
     * Materializes an equivalent tensor on the requested execution device.
     */
    to<T extends Device>(device: T): Tensor<S, D, T>;
    /**
     * @summary Convert tensor contents to JavaScript arrays.
     * @category Core
     * @semantics
     * Materializes the tensor payload into nested JavaScript arrays for
     * debugging, inspection, or interop.
     */
    to_array(): unknown;
    /**
     * @summary Slice with explicit selector strings or numeric indices.
     * @category Shape
     * @semantics
     * Applies explicit axis selectors and returns the resulting tensor view or
     * materialized slice.
     */
    slice(selectors: readonly (number | string)[]): Tensor;
    /**
     * @summary Compact text representation.
     * @category Debugging
     * @semantics
     * Returns a concise string representation suitable for logs and REPL use.
     */
    toString(): string;
    /**
     * @summary Rich text or HTML representation.
     * @category Debugging
     * @semantics
     * Returns a richer representation for notebook, terminal, or HTML display.
     */
    repr(): string;
    repr(opts: { mode?: "text" | "html"; sparse?: boolean; max_rows?: number; max_cols?: number }): string | { mime: string; data: string };
    /**
     * @summary Export a backward-connected structural report for this tensor.
     * @category Observability
     * @semantics
     * Returns either a text rendering or canonical JSON report payload
     * describing the derived backward graph rooted at this tensor.
     */
    exportReport(opts?: BundleExportOptions): string;
    /** @deprecated Prefer `exportReport(...)`. */
    exportBundle(opts?: BundleExportOptions): string;
    /**
     * @summary Backward compatibility hook for direct tensor-style backward.
     * @category Training
     * @deprecated Prefer `grad(loss, params)` for public training code.
     */
    backward(): void;
  }

  /**
   * @summary Trainable tensor state.
   * @category Training
   * @semantics
   * `Parameter` is the public gradient-bearing leaf used in normal training
   * code. Parameters are returned by `module.parameters`, participate in
   * `grad(loss, params)`, and retain gradient buffers across training steps
   * until cleared with `clear_grad(...)`.
   * @usecase
   * Use `Parameter` for persistent module weights, affine terms, embeddings,
   * and other trainable state.
   */
  interface Parameter<S extends Shape = number[], D extends DType = DType, V extends Device = Device> extends Tensor<S, D, V> {
    /** Gradient accumulated by `grad(loss, params)`. */
    readonly grad: Tensor<S, D, V> | null;
    /**
     * @summary Fill a parameter with zeros.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    zeros(): Parameter<S, D>;
    /**
     * @summary Fill a parameter with ones.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    ones(): Parameter<S, D>;
    /**
     * @summary Fill a parameter with a scalar value.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    full(fill: number): Parameter<S, D>;
    /**
     * @summary Uniform random initialization in `[0, 1)`.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    rand(): Parameter<S, D>;
    /**
     * @summary Standard normal initialization.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    randn(): Parameter<S, D>;
    /**
     * @summary Xavier/Glorot uniform initialization.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    xavier_uniform(): Parameter<S, D>;
    /**
     * @summary Xavier/Glorot normal initialization.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    xavier_normal(): Parameter<S, D>;
    /**
     * @summary Kaiming/He uniform initialization.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    kaiming_uniform(): Parameter<S, D>;
    /**
     * @summary Kaiming/He normal initialization.
     * @category Training
     * @semantics Initializes the parameter in place and returns the same parameter.
     */
    kaiming_normal(): Parameter<S, D>;
  }

  /**
   * @summary Named parameter entry.
   * @category Training
   * @semantics
   * Produced by `parameters.named()` to preserve both a stable structural name
   * and the parameter it refers to.
   */
  type ParamEntry<S extends Shape = number[], D extends DType = DType> = readonly [string, Parameter<S, D>];

  /**
   * @summary Readonly parameter traversal view.
   * @category Training
   * @semantics
   * Returned by `module.parameters`. Preserves a stable traversal order and
   * can surface structural names for checkpoints, logging, and inspection.
   */
  interface ParameterCollection<S extends Shape = number[], D extends DType = DType> extends ReadonlyArray<Parameter<S, D>> {
    /** Return `[name, parameter]` entries in traversal order. */
    named(): ReadonlyArray<ParamEntry<S, D>>;
  }

  /**
   * @summary Explicit range selector.
   * @category Shape
   * @semantics
   * Selects a contiguous or strided axis range in `slice(...)`.
   */
  type RangeSelector = Readonly<{ kind: "range"; start: number; end: number; step?: number }>;
  /**
   * @summary Explicit full-axis selector.
   * @category Shape
   * @semantics
   * Selects every position along one axis in `slice(...)`.
   */
  type AllSelector = Readonly<{ kind: "all" }>;
  /**
   * @summary Public selector union for `slice(...)`.
   * @category Shape
   * @semantics Combines scalar, range, and all-axis selectors.
   */
  type Selector = number | RangeSelector | AllSelector;
  /**
   * @summary Serializable compute state tree.
   * @category Modules
   * @semantics
   * Represents the nested state structure returned by `module.state()` and
   * consumed by `module.restore(...)` or `checkpoint.restore(...)`.
   */
  type ComputeState = Tensor | number | string | boolean | null | ComputeState[] | { [key: string]: ComputeState };

  /**
   * @summary Callable stateful compute module.
   * @category Modules
   * @semantics
   * A `Module` is a callable compute program with persistent state, trainable
   * parameter traversal, and explicit execution-mode control.
   */
  interface Module<
    Args extends readonly Tensor<any, any>[] = readonly Tensor<any, any>[],
    Out extends Tensor<any, any> = Tensor,
  > extends Program<Args, Out> {
    /**
     * @summary Invoke the module.
     * @category Modules
     * @semantics
     * Runs the module in its current training or eval mode and returns the
     * computed tensor output.
     */
    /** Current training/eval mode. */
    readonly training: boolean;
    /** Logical module-tree path used by graph export and diagnostics. */
    readonly module_path: string | null;
    /** Trainable parameter view. */
    readonly parameters: ParameterCollection;
    /** Materialize the full persistent state tree. */
    state(): ComputeState;
    /** Restore a previously captured state tree. */
    restore(state: ComputeState): void;
    /** Read or update the module execution mode. */
    mode(): ModuleMode;
    mode(mode: ModuleMode): this;
    /** Switch to training mode. */
    train(): void;
    /** Switch to eval mode. */
    eval(): void;
    /** Save state to disk. */
    save(path: string): void;
    /** Load state from disk. */
    load(path: string): void;
    /**
     * Annotate the module with a logical path used by graph export and
     * observability tooling to group captured ops under module ownership.
     */
    metadata(path: string): this;
    /**
     * @summary Export a structural report for this module invocation.
     * @category Observability
     * @semantics
     * Returns either a text rendering or canonical JSON report payload for
     * the captured native forward graph and execution plan of this invocation.
     */
    exportReport(...args: [...Args, BundleExportOptions?]): string;
    /** @deprecated Prefer `exportReport(...)`. */
    exportReport(...args: [...Args, BundleExportOptions?]): string;
    /** @deprecated Prefer `exportReport(...)`. */
    exportBundle(...args: [...Args, BundleExportOptions?]): string;
  }

  /**
   * @summary Low-level graph-build node record.
   * @category Observability
   * @semantics
   * Raw schema node returned by compatibility helpers such as
   * `capturedProgram()`. This is the graph-build record, not the primary
   * execution-graph inspection surface.
   */
  type CapturedProgramNode = {
    id: number;
    kind: string;
    module_path?: string;
    index?: number;
    data?: unknown;
    dtype?: DType;
    input?: number;
    left?: number;
    right?: number;
    inputs?: number[];
    cond?: number;
    onTrue?: number;
    onFalse?: number;
    mask?: number;
    value?: number;
    min?: number;
    max?: number;
    shape?: number[];
    ranges?: (number | string)[];
    axis?: number;
    keepdim?: boolean;
    axes?: number[];
    dim?: number;
    logits?: number;
    targets?: number;
    numClasses?: number;
    k?: number;
    execution?: { hint?: MatmulExecutionHint };
  };

  /**
   * @summary Low-level graph-build schema view.
   * @category Observability
   * @semantics
   * Compact structural record returned by compatibility helpers that expose the
   * raw graph-build schema directly.
   */
  interface CapturedProgramView {
    inputArity: number;
    boundInputCount: number;
    inputCount: number;
    outputId: number;
    nodes: CapturedProgramNode[];
  }

  /**
   * @summary Low-level graph-build schema summary.
   * @category Observability
   * @semantics
   * Compact summary for tooling that consumes the raw graph-build schema
   * directly.
   */
  interface CapturedProgramSummary {
    inputArity: number;
    boundInputCount: number;
    inputCount: number;
    outputId: number;
    nodeCount: number;
    nodeKinds: string[];
  }

  interface GraphLoweringAnalysis {
    lowerable: boolean;
    failure?: {
      category: "unsupported_captured_node_kind" | "invalid_adapter_metadata" | "invalid_execution_metadata" | "native_semantic_rejection";
      nodeId?: number;
      nodeKind?: string;
      reason: string;
    } | null;
  }

  interface GraphRegionSummary {
    nodeIds: number[];
    instructionCount: number;
    broadcastModes: string[];
    lowering: string;
  }

  interface GraphTemplateSummary {
    kind: string;
    nodeIds: number[];
    lowering: string;
  }

  interface GraphStatsSummary {
    nodeCount: number;
    executableNodeCount: number;
    fusedRegionCount: number;
    largestRegionNodeCount: number;
    elementwiseOnly: boolean;
  }

  interface GraphStaticMetadataSummary {
    kind: "provisional_capture_metadata";
    source: "ts_capture";
    fields: string[];
  }

  interface GraphSummaryNode {
    id: number;
    kind: string;
    outputShape: number[] | null;
    index?: number;
    input?: number;
    left?: number;
    right?: number;
    inputs?: number[];
    cond?: number;
    onTrue?: number;
    onFalse?: number;
    mask?: number;
    value?: number;
    min?: number;
    max?: number;
    shape?: number[];
    ranges?: (number | string)[];
    axis?: number;
    keepdim?: boolean;
    axes?: number[];
    dim?: number;
    logits?: number;
    targets?: number;
    numClasses?: number;
    k?: number;
    dtype?: DType;
  }

  interface GraphSummary {
    inputCount: number;
    output: number;
    nodes: GraphSummaryNode[];
    staticMetadata: GraphStaticMetadataSummary;
    stats: GraphStatsSummary;
    regions: GraphRegionSummary[];
    templates: GraphTemplateSummary[];
    specialized: boolean;
    runtime?: "native-graph";
    summaryKind?: "execution";
    loweringAnalysis?: GraphLoweringAnalysis;
    nativeState?: NativeCompiledExecutableState;
  }

  interface NativeCompiledExecutableState {
    mode: "none" | "graph" | "eager-forward";
    graphRuntime: "native-graph" | null;
    planOutcome: "none" | "native-graph" | "eager-forward";
    planOutcomeReason: string | null;
    specializationCaptureCount: number;
    specializationReuseCount: number;
    eagerFallbackCount: number;
    lastFallbackKind: CompileFallbackKind | null;
    lastFallbackReason: string | null;
    lastTensorSignature: string | null;
    lastErrorDetail: string | null;
  }

  interface ExecutionPlanContext {
    boundary: BundleExportBoundary;
    phase: BundleExportPhase;
    run_id?: string | null;
    graph_id?: string | null;
    epoch_index?: number | null;
    step_index?: number | null;
    batch_index?: number | null;
    note?: string | null;
  }

  type ExecutionPlanStepKind = "single_op" | "fused_elementwise_region";
  type ExecutionPlanRegionKind = "fusable_run" | "matmul_epilogue";
  type ExecutionPlanExecutionKind =
    | "elementwise_binary"
    | "elementwise_unary"
    | "elementwise_generic"
    | "reduction_all"
    | "reduction"
    | "index"
    | "view";
  type ExecutionPlanAllocationIntent = "new_storage" | "view_only";
  type ExecutionPlanInputRequirement = "preserve" | "require_storage" | "require_contiguous_input";
  type ExecutionPlanInputLayoutDecision = "accept" | "pack_to_dense" | "unsupported";
  type ExecutionPlanMatmulExecutionFamily =
    | "gemm_2d"
    | "gemm_batched"
    | "gemm_projection"
    | "gemm_attention_scores"
    | "gemm_attention_values"
    | "gemm_generic_unresolved";
  type ExecutionPlanClassificationConfidence = "high" | "medium" | "low";
  type ExecutionPlanMatmulHint = "none" | "projection" | "attention_scores" | "attention_values";
  type ExecutionPlanHintSource = "none" | "graph_context" | "higher_level_module" | "api_execution_arg";
  type ExecutionPlanMatmulEpilogueActivation = "none" | "gelu" | "relu" | "sigmoid";

  interface ExecutionPlanPlannerHint {
    input_layout_decision: ExecutionPlanInputLayoutDecision;
  }

  interface ExecutionPlanMatmulDescriptor {
    family: ExecutionPlanMatmulExecutionFamily;
    confidence: ExecutionPlanClassificationConfidence;
    hint: ExecutionPlanMatmulHint;
    hint_source: ExecutionPlanHintSource;
    lhs_rank: number;
    rhs_rank: number;
    out_rank: number;
    batch_rank: number;
    flattenable_leading_batch: boolean;
  }

  interface ExecutionPlanStep {
    step_id: number;
    node_id: number;
    kind: ExecutionPlanStepKind;
    output_value_ids: number[];
    execution_kind: ExecutionPlanExecutionKind;
    allocation: ExecutionPlanAllocationIntent;
    input_requirement: ExecutionPlanInputRequirement;
    planner_hint?: ExecutionPlanPlannerHint | null;
    plan_region_id?: number | null;
    matmul?: ExecutionPlanMatmulDescriptor | null;
  }

  interface ExecutionPlanRegion {
    region_id: number;
    kind: ExecutionPlanRegionKind;
    step_start: number;
    step_end: number;
    matmul_epilogue_activation?: ExecutionPlanMatmulEpilogueActivation | null;
  }

  interface GraphPlanReport {
    kind: "graph_plan";
    id: string;
    version: number;
    context?: ExecutionPlanContext | null;
    steps: ExecutionPlanStep[];
    regions: ExecutionPlanRegion[];
    outputs: number[];
  }

  type ExecutionPlan = GraphPlanReport;

  interface Report {
    kind: "report";
    version: number;
    context: ExecutionPlanContext;
    graph?: unknown | null;
    graph_plan?: GraphPlanReport | null;
    derived_graph?: unknown | null;
    links: {
      plan_step_forward_node: Array<{ plan_step_id: number; forward_node_id: number }>;
      derived_node_provenance: Array<{ derived_node_id: number; provenance_node_id: number }>;
      runtime_value_derived_output: Array<{ runtime_value_id: number; derived_output_value_id: number }>;
    };
  }

  interface ExecutionTensorSignatureEntry {
    dtype: DType;
    shape: number[];
  }

  type CompileFallbackKind =
    | "tensor_signature_changed"
    | "program_state_changed"
    | "graph_capture_failed"
    | "no_graph_available";

  type GraphCaptureFailureStage =
    | "syntax"
    | "capture_adapter"
    | "semantic_validation"
    | "unknown";

  interface GraphCaptureFailure {
    stage: GraphCaptureFailureStage;
    reason: string;
  }

  /**
   * @summary Execution-graph view for a compiled program invocation.
   * @category Observability
   * @semantics
   * Returned by `compiled.graph(...args)`. This is the structural graph-level
   * inspection surface for a compiled program. Prefer this view when you need
   * native lowering analysis or graph-shaped export data.
   */
  interface ExecutionGraphProgram<Args extends readonly unknown[] = readonly unknown[]> {
    run(...args: Args): unknown;
    summary(...args: Args): GraphSummary;
    plan(...args: Args): ExecutionPlan | null;
    exportReport(...args: [...Args, BundleExportOptions?]): string;
    /** @deprecated Prefer `exportReport(...)`. */
    exportBundle(...args: [...Args, BundleExportOptions?]): string;
    executionRuntime(): "native-graph";
    loweringAnalysis(): GraphLoweringAnalysis;
    nativeState(): NativeCompiledExecutableState;
    /**
     * @summary Low-level captured-program schema view.
     * @category Observability
     * @semantics
     * Compatibility helper for tooling that needs the raw captured-program
     * schema. Prefer `graph()`, `plan()`, or `summary()` for normal inspection.
     */
    capturedProgram(): CapturedProgramView;
    /**
     * @summary Low-level captured-program summary.
     * @category Observability
     * @semantics
     * Compatibility helper for tooling that consumes the compact captured
     * schema summary directly. Prefer `summary()` for the normal quick view.
     */
    captureSummary(): CapturedProgramSummary;
  }

  /**
   * @summary Compatibility alias for the execution-graph view.
   * @category Observability
   * @semantics
   * Deprecated in spirit but kept for compatibility. Prefer
   * {@link ExecutionGraphProgram}.
   */
  type CapturedGraphProgram<Args extends readonly unknown[] = readonly unknown[]> = ExecutionGraphProgram<Args>;

  /**
   * @summary Compiled callable program.
   * @category Compilation
   * @semantics
   * Compiled programs preserve the original call signature while surfacing the
   * execution graph and executable summary through stable runtime helpers.
   */
  interface ExecutableProgram<
    Args extends readonly unknown[] = readonly unknown[],
    Out = unknown,
    Source = unknown,
  > extends Program<Args, Out> {
    readonly __affon_compute_compiled: true;
    readonly source: Source;
    run(...args: Args): Out;
    /**
     * @summary Compatibility view of the underlying compiled source.
     * @category Compilation
     * @semantics
     * Exposes the legacy internal compiled-program descriptor. Prefer the
     * callable executable surface plus `summary()`, `graph()`, and `plan()`
     * for normal public use.
     */
    computation(): unknown;
    /**
     * @summary Execution-graph inspection view.
     * @category Observability
     * @semantics
     * Returns the canonical graph-level inspection view for the provided
     * invocation arguments. This is the main structural inspection surface for
     * compiled programs.
     */
    graph(...args: Args): ExecutionGraphProgram<Args> | null;
    /**
     * @summary Execution plan inspection view.
     * @category Observability
     * @semantics
     * Returns the canonical execution-plan view for the provided invocation
     * arguments.
     */
    plan(...args: Args): ExecutionPlan | null;
    /**
     * @summary Low-level captured-program schema view.
     * @category Observability
     * @semantics
     * Compatibility helper for tooling that still consumes the raw captured
     * schema directly. Prefer `graph()`, `plan()`, or `summary()` for normal
     * inspection.
     */
    capturedProgram(...args: Args): CapturedProgramView | null;
    /**
     * @summary Low-level captured-program summary.
     * @category Observability
     * @semantics
     * Compatibility helper for tooling that consumes the compact captured
     * schema summary directly. Prefer `summary()` for the normal quick view.
     */
    captureSummary(...args: Args): CapturedProgramSummary | null;
    exportBundle(...args: [...Args, BundleExportOptions?]): string;
    /**
     * @summary Quick compiled-program status view.
     * @category Observability
     * @semantics
     * Returns the canonical human/debug summary for a compiled program without
     * requiring concrete invocation arguments.
     */
    summary(): {
      kind: string;
      hasDefinedParams: boolean;
      paramCount: number;
      mode: string;
      graphRuntime: "native-graph" | null;
      graphLoweringAnalysis: GraphLoweringAnalysis | null;
      graphCaptureError: string | null;
      graphCaptureFailure: GraphCaptureFailure | null;
      planAvailable: boolean;
      planStepCount: number | null;
      planRegionCount: number | null;
      planOutputCount: number | null;
      specializationActive: boolean;
      activeTensorSignature: ExecutionTensorSignatureEntry[] | null;
      specializationCaptureCount: number;
      specializationReuseCount: number;
      eagerFallbackCount: number;
      nativeState: NativeCompiledExecutableState | null;
      lastExecutionMode: "graph" | "eager-forward";
      lastFallbackKind: CompileFallbackKind | null;
      lastFallbackReason: string | null;
    };
    /**
     * @summary Quick compiled-program status view for a concrete invocation.
     * @category Observability
     * @semantics
     * Returns the canonical human/debug summary for a specific invocation,
     * including plan counts and specialization status derived from those args.
     */
    summary(...args: Args): {
      kind: string;
      hasDefinedParams: boolean;
      paramCount: number;
      mode: string;
      graphRuntime: "native-graph" | null;
      graphLoweringAnalysis: GraphLoweringAnalysis | null;
      graphCaptureError: string | null;
      graphCaptureFailure: GraphCaptureFailure | null;
      planAvailable: boolean;
      planStepCount: number | null;
      planRegionCount: number | null;
      planOutputCount: number | null;
      specializationActive: boolean;
      activeTensorSignature: ExecutionTensorSignatureEntry[] | null;
      specializationCaptureCount: number;
      specializationReuseCount: number;
      eagerFallbackCount: number;
      nativeState: NativeCompiledExecutableState | null;
      lastExecutionMode: "graph" | "eager-forward";
      lastFallbackKind: CompileFallbackKind | null;
      lastFallbackReason: string | null;
    };
  }

  /**
   * @summary Compiled modeful compute module.
   * @category Compilation
   * @semantics
   * A compiled module preserves the mode/state/parameter surface of a callable
   * module while routing execution through a compiled program executable.
   */
  interface CompiledModule<
    Args extends readonly Tensor<any, any>[] = readonly Tensor<any, any>[],
    Out extends Tensor<any, any> = Tensor,
  > extends Module<Args, Out>, ExecutableProgram<Args, Out, Module<Args, Out>> {}

  /**
   * @summary Stateful optimizer step callable.
   * @category Optim
   * @semantics
   * A compute optimizer is a callable step object that updates parameters in
   * place, exposes a mutable learning rate, and can save and restore its own
   * optimizer state.
   */
  interface ComputeStep {
    /**
     * @summary Update a parameter list in place.
     * @category Optim
     * @semantics
     * Applies one optimizer step to the supplied parameters using their current
     * gradient buffers.
     */
    (params: readonly Parameter[]): void;
    /**
     * @summary Mutable learning rate.
     * @category Optim
     * @semantics
     * The current step learning rate, which can be updated directly or driven
     * by a schedule wrapper.
     */
    lr: number;
    /**
     * @summary Serialize optimizer state.
     * @category Optim
     * @semantics Returns a checkpointable optimizer state payload.
     */
    state(): unknown;
    /**
     * @summary Restore optimizer state.
     * @category Optim
     * @semantics Restores a previously serialized optimizer state payload.
     */
    restore(state: unknown): void;
  }

  /**
   * @summary Schedule context passed to learning-rate schedules.
   * @category Optim
   * @semantics
   * Carries the current epoch and optimizer step when evaluating a schedule.
   */
  interface TrainContext {
    readonly epoch: number;
    readonly step: number;
  }

  /**
   * @summary Structural export boundary labels for compute observability.
   * @category Observability
   * @semantics
   * Identifies the caller-selected lifecycle boundary associated with a graph
   * or backward export snapshot.
   */
  type BundleExportBoundary = "ad_hoc" | "run" | "epoch" | "step" | "forward" | "backward";

  /**
   * @summary Structural export phase labels for compute observability.
   * @category Observability
   * @semantics
   * Distinguishes forward, backward, optimizer, and evaluation exports when
   * attaching runtime context to a report.
   */
  type BundleExportPhase = "unspecified" | "forward" | "backward" | "optimizer" | "evaluation";

  /**
   * @summary Text rendering modes for report exports.
   * @category Observability
   * @semantics
   * Chooses between a compact summary and a more detailed annotated text
   * rendering when `format` is left as `"text"`.
   */
  type BundleExportMode = "summary" | "annotated";

  /**
   * @summary Output formats for report exports.
   * @category Observability
   * @semantics
   * `"text"` returns the native human-readable rendering, while `"json"`
   * returns the canonical report as raw JSON text suitable
   * for offline visualization or analysis tooling.
   */
  type BundleExportFormat = "text" | "json";

  /**
   * @summary Context and rendering controls for compute report export.
   * @category Observability
   * @semantics
   * Carries optional runtime context and output-format selection for forward
   * and backward structural exports.
   */
  interface BundleExportOptions {
    mode?: BundleExportMode;
    format?: BundleExportFormat;
    boundary?: BundleExportBoundary;
    phase?: BundleExportPhase;
    runId?: string;
    graphId?: string;
    epoch?: number;
    step?: number;
    batch?: number;
    note?: string;
  }

  /**
   * @summary File-writing options for exported compute reports.
   * @category Observability
   * @semantics
   * Extends `BundleExportOptions` with file naming and directory controls for
   * helper APIs that persist report payloads to disk.
   */
  interface BundleFileOptions extends BundleExportOptions {
    dir?: string;
    filename?: string;
    extension?: string;
    prefix?: string;
  }

  /**
   * @summary Duration expressed in optimizer steps.
   * @category Optim
   * @semantics
   * Marks a finite duration that should be interpreted in optimizer-step units.
   */
  interface StepDuration {
    unit: "step";
    value: number;
  }

  /**
   * @summary Duration expressed in epochs.
   * @category Optim
   * @semantics
   * Marks a finite duration that should be interpreted in epoch units.
   */
  interface EpochDuration {
    unit: "epoch";
    value: number;
  }

  /**
   * @summary Learning-rate schedule function.
   * @category Optim
   * @semantics
   * Maps the current training context to a scalar learning rate.
   */
  type LRSchedule = (ctx: TrainContext) => number;

  /**
   * @summary Scheduled optimizer step with mutable schedule context.
   * @category Optim
   * @semantics
   * Extends a compute optimizer step with explicit epoch progression and a
   * readable schedule context.
   */
  interface ScheduledStep extends ComputeStep {
    /**
     * @summary Read current schedule context.
     * @category Optim
     * @semantics Exposes the current epoch and optimizer-step counters.
     */
    readonly context: TrainContext;
    /**
     * @summary Advance or set the current epoch.
     * @category Optim
     * @semantics Updates the epoch component used by epoch-based schedules.
     */
    epoch(value: number): void;
  }

  /**
   * @summary Result bundle returned by `topk(...)`.
   * @category Selection
   * @semantics
   * Holds the selected top values and their integer indices along the queried axis.
   */
  interface TopKResult<S extends Shape = number[], D extends DType = DType> {
    /**
     * @summary Top values.
     * @category Selection
     * @semantics Contains the selected top values along the queried axis.
     */
    values: Tensor<S, D>;
    /**
     * @summary Top indices.
     * @category Selection
     * @semantics Contains the integer indices of the selected top values.
     */
    indices: Tensor<S, "i64">;
  }

  /**
   * @summary Non-finite summary returned by `finite_summary(...)`.
   * @category Debugging
   * @semantics
   * Reports whether a tensor is finite and, when it is not, the first bad flat
   * index plus its numeric value.
   */
  interface FiniteSummary {
    /**
     * @summary Finite-status flag.
     * @category Debugging
     * @semantics `true` when every tensor element is finite.
     */
    ok: boolean;
    /**
     * @summary First non-finite flat index.
     * @category Debugging
     * @semantics Index of the first non-finite element in flat row-major order.
     */
    first_bad_flat_index: number;
    /**
     * @summary First non-finite numeric value.
     * @category Debugging
     * @semantics The first offending value found in the tensor.
     */
    first_bad_value: number;
  }

  /**
   * @summary Maximum absolute value when a tensor is fully finite.
   * @category Debugging
   * @semantics
   * Returns the largest absolute value in a tensor when every element is
   * finite. Returns `null` if a non-finite value is present or if the runtime
   * cannot safely complete the reduction.
   */
  type FiniteAbsMax = number | null;

  /**
   * @summary Construct a concrete tensor from JavaScript numeric data.
   * @category Creation
   * @semantics
   * `tensor(...)` is the default public constructor for concrete numeric data.
   * It does not represent trainable state; use `parameter(...)` for that.
   * @input
   * Accepts a scalar number or nested JavaScript arrays of numbers.
   * @output
   * Returns a concrete `Tensor` on the requested device and dtype.
   * @param data Scalar or nested numeric array data.
   * @param opts Optional dtype, device placement, and axis names.
   * @example
   * const x = tensor([[1, 2], [3, 4]], { dtype: "f32" })
   */
  function tensor<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(data: number | number[] | number[][] | number[][][] | number[][][][], opts?: TensorOptions<D, V>): Tensor<S, D, V>;
  /**
   * @summary Allocate an uninitialized tensor buffer.
   * @category Creation
   * @semantics
   * Allocates storage without initializing element values.
   * @input
   * Accepts an explicit shape and optional dtype/device placement.
   * @output
   * Returns a concrete `Tensor` with unspecified contents.
   */
  function empty<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(shape: S, opts?: TensorOptions<D, V>): Tensor<S, D, V>;
  /**
   * @summary Construct a zero-filled tensor.
   * @category Creation
   * @semantics Returns a concrete tensor filled with `0`.
   * @input Accepts an explicit shape and optional dtype/device placement.
   * @output Returns a zero-filled `Tensor`.
   */
  function zeros<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(shape: S, opts?: TensorOptions<D, V>): Tensor<S, D, V>;
  /**
   * @summary Construct a one-filled tensor.
   * @category Creation
   * @semantics Returns a concrete tensor filled with `1`.
   * @input Accepts an explicit shape and optional dtype/device placement.
   * @output Returns a one-filled `Tensor`.
   */
  function ones<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(shape: S, opts?: TensorOptions<D, V>): Tensor<S, D, V>;
  /**
   * @summary Construct a tensor filled with a scalar value.
   * @category Creation
   * @semantics Returns a concrete tensor where every element is `fill`.
   * @param shape Output tensor shape.
   * @param fill Scalar fill value.
   * @param opts Optional dtype, device placement, and axis names.
   */
  function full<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(shape: S, fill: number, opts?: TensorOptions<D, V>): Tensor<S, D, V>;
  /**
   * @summary Construct a uniform random tensor.
   * @category Creation
   * @semantics Samples elements from a uniform distribution in `[0, 1)`.
   * @input Accepts an explicit shape and optional dtype/device placement.
   * @output Returns a concrete random `Tensor`.
   */
  function rand<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(shape: S, opts?: TensorOptions<D, V>): Tensor<S, D, V>;
  /**
   * @summary Construct a standard normal random tensor.
   * @category Creation
   * @semantics Samples elements from a standard normal distribution.
   * @input Accepts an explicit shape and optional dtype/device placement.
   * @output Returns a concrete random `Tensor`.
   */
  function randn<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(shape: S, opts?: TensorOptions<D, V>): Tensor<S, D, V>;
  /**
   * @summary Construct a one-dimensional numeric range.
   * @category Creation
   * @semantics
   * Builds a rank-1 tensor from `start`, `end`, and `step` in the usual range
   * style. When `end` is omitted, the single argument form is treated as
   * `arange(0, start)`.
   */
  function arange<D extends DType = DType, V extends Device = Device>(start: number, end?: number, step?: number, opts?: TensorOptions<D, V>): Tensor<[number], D, V>;
  /**
   * @summary Construct a one-dimensional linear interpolation.
   * @category Creation
   * @semantics
   * Returns a rank-1 tensor of evenly spaced values between `start` and `end`.
   * @param start Inclusive start value.
   * @param end Inclusive end value.
   * @param steps Number of interpolation steps.
   * @param opts Optional dtype and device placement.
   */
  function linspace<D extends DType = DType, V extends Device = Device>(start: number, end: number, steps?: number, opts?: TensorOptions<D, V>): Tensor<[number], D, V>;
  /**
   * @summary Seed the public random constructors.
   * @category Creation
   * @semantics
   * Controls the random stream used by constructors like `rand(...)`,
   * `randn(...)`, and parameter initializers built on those constructors.
   */
  function seed(value: number): void;

  /**
   * @summary Construct a trainable parameter tensor from shape.
   * @category Training
   * @semantics
   * `parameter(...)` is the public entrypoint for gradient-bearing trainable
   * state. It creates a `Parameter`, which can then be initialized fluently
   * and optimized with `grad(loss, params)` plus a compute optimizer step.
   * @input
   * Accepts an explicit shape and optional dtype/device placement.
   * @output
   * Returns an uninitialized `Parameter`.
   * @example
   * const weight = parameter([128, 64]).xavier_uniform()
   */
  function parameter<S extends Shape = number[], D extends DType = DType, V extends Device = Device>(shape: S, opts?: TensorOptions<D, V>): Parameter<S, D, V>;

  /** Set the default device used by subsequent tensor constructors. */
  function setDevice(device: Device): void;

  /**
   * @summary Change tensor dtype without changing shape or device.
   * @category Core
   * @input One tensor and the destination dtype.
   * @output A tensor with the same shape on the same device.
   */
  function cast<S extends Shape = Shape, D extends DType = DType, E extends DType = DType>(x: Tensor<S, D>, dtype: E): Tensor<S, E>;
  /**
   * @summary Move a tensor to another device.
   * @category Core
   * @input One tensor and the destination device.
   * @output A tensor with the same shape and dtype on the target device.
   */
  function move<S extends Shape = Shape, D extends DType = DType, V extends Device = Device>(x: Tensor<S, D>, device: V): Tensor<S, D, V>;

  /**
   * @summary Negate a tensor elementwise.
   * @category Arithmetic
   * @semantics Multiplies every element by `-1` and preserves shape.
   */
  function neg<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Absolute value.
   * @category Arithmetic
   * @semantics Applies absolute value elementwise and preserves shape.
   */
  function abs<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Elementwise sign function.
   * @category Arithmetic
   * @semantics Maps each element to `-1`, `0`, or `1` and preserves shape.
   */
  function sign<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Elementwise addition with broadcasting.
   * @category Arithmetic
   * @semantics Broadcasts both operands using compute broadcast rules before adding.
   */
  function add(a: Tensor, b: Tensor): Tensor;
  /**
   * @summary Elementwise subtraction with broadcasting.
   * @category Arithmetic
   * @semantics Broadcasts both operands using compute broadcast rules before subtracting.
   */
  function sub(a: Tensor, b: Tensor): Tensor;
  /**
   * @summary Elementwise multiplication with broadcasting.
   * @category Arithmetic
   * @semantics Broadcasts both operands using compute broadcast rules before multiplying.
   */
  function mul(a: Tensor, b: Tensor): Tensor;
  /**
   * @summary Elementwise division with broadcasting.
   * @category Arithmetic
   * @semantics Broadcasts both operands using compute broadcast rules before dividing.
   */
  function div(a: Tensor, b: Tensor): Tensor;
  /**
   * @summary Matrix multiplication.
   * @category Linear Algebra
   * @semantics
   * Performs standard matrix multiplication over the trailing dimensions,
   * broadcasting any leading batch dimensions.
   */
  function matmul(a: Tensor, b: Tensor, execution?: MatmulExecutionOptions): Tensor;
  /**
   * @summary Dot product reduced to a scalar tensor.
   * @category Linear Algebra
   * @output A scalar tensor `[]`.
   */
  function dot<S extends Shape = Shape, T extends Shape = Shape, D extends DType = DType>(a: Tensor<S, D>, b: Tensor<T, D>): Tensor<[], D>;

  /**
   * @summary Elementwise exponential.
   * @category Arithmetic
   * @semantics Applies `exp(...)` elementwise and preserves shape.
   */
  function exp<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Elementwise natural logarithm.
   * @category Arithmetic
   * @semantics Applies `log(...)` elementwise and preserves shape.
   */
  function log<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Elementwise square root.
   * @category Arithmetic
   * @semantics Applies `sqrt(...)` elementwise and preserves shape.
   */
  function sqrt<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Elementwise square.
   * @category Arithmetic
   * @semantics Multiplies each element by itself and preserves shape.
   */
  function square<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Clamp tensor values into a closed interval.
   * @category Arithmetic
   * @param min Lower bound.
   * @param max Upper bound.
   */
  function clamp<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>, min: number, max: number): Tensor<S, D>;
  /**
   * @summary ReLU activation.
   * @category Activation
   * @semantics Applies rectified linear activation elementwise and preserves shape.
   */
  function relu<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary GELU activation.
   * @category Activation
   * @semantics Applies Gaussian error linear activation elementwise and preserves shape.
   */
  function gelu<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Sigmoid activation.
   * @category Activation
   * @semantics Applies logistic activation elementwise and preserves shape.
   */
  function sigmoid<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary SiLU activation.
   * @category Activation
   * @semantics Applies the sigmoid linear unit elementwise and preserves shape.
   */
  function silu<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Swish activation alias.
   * @category Activation
   * @semantics Alias of `silu(...)`.
   */
  function swish<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Hyperbolic tangent activation.
   * @category Activation
   * @semantics Applies `tanh(...)` elementwise and preserves shape.
   */
  function tanh<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Softmax normalization along one axis.
   * @category Activation
   * @axis Normalizes along `dim`.
   */
  function softmax<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>, dim: number): Tensor<S, D>;

  /**
   * @summary Indexed cross-entropy loss.
   * @category Loss
   * @semantics Computes cross-entropy from class logits and integer class
   * targets along the selected class axis.
   */
  function cross_entropy_indexed<S extends Shape = Shape, D extends DType = DType>(
    logits: Tensor<S, D>,
    targets: Tensor,
    axis?: number,
  ): Tensor;

  /**
   * @summary Sum reduction.
   * @category Reduction
   * @semantics Reduces the selected axes by summation and can optionally keep singleton dimensions.
   */
  function sum(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Mean reduction.
   * @category Reduction
   * @semantics Reduces the selected axes by arithmetic mean and can optionally keep singleton dimensions.
   */
  function mean(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Maximum reduction.
   * @category Reduction
   * @semantics Reduces the selected axes by maximum value and can optionally keep singleton dimensions.
   */
  function max(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Minimum reduction.
   * @category Reduction
   * @semantics Reduces the selected axes by minimum value and can optionally keep singleton dimensions.
   */
  function min(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Variance reduction.
   * @category Reduction
   * @semantics Reduces the selected axes by variance and can optionally keep singleton dimensions.
   */
  function variance(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Standard deviation reduction.
   * @category Reduction
   * @semantics Reduces the selected axes by standard deviation and can optionally keep singleton dimensions.
   */
  function std(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Argmin reduction.
   * @category Reduction
   * @semantics Returns index positions of minimum values along the selected axes.
   */
  function argmin(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Argmax reduction.
   * @category Reduction
   * @semantics Returns index positions of maximum values along the selected axes.
   */
  function argmax(x: Tensor, dim?: number | readonly number[], keep?: boolean): Tensor;
  /**
   * @summary Elementwise conditional selection.
   * @category Selection
   * @semantics Selects values from `a` or `b` according to `cond`.
   */
  function where(cond: Tensor, a: Tensor, b: Tensor): Tensor;
  /**
   * @summary Fill masked positions with a scalar value.
   * @category Selection
   * @semantics Returns a tensor where masked positions are replaced by `value`.
   */
  function masked_fill(input: Tensor, mask: Tensor, value: number): Tensor;
  /**
   * @summary Concatenate tensors along one axis.
   * @category Selection
   * @semantics Reassembles a list of tensors by joining them along `dim`.
   */
  function cat(values: readonly Tensor[], dim?: number): Tensor;
  /**
   * @summary Stack tensors along a new axis.
   * @category Selection
   * @semantics Reassembles a list of tensors by inserting a new dimension at `dim`.
   */
  function stack(values: readonly Tensor[], dim?: number): Tensor;
  /**
   * @summary One-hot encode integer indices.
   * @category Selection
   * @semantics Expands index values into one-hot rows with `numClasses` columns.
   */
  function one_hot(indices: Tensor, numClasses: number): Tensor;
  /**
   * @summary Gather values by index along one axis.
   * @category Selection
   * @semantics Selects values from `input` using `index` along `dim`.
   */
  function gather(input: Tensor, dim: number, index: Tensor): Tensor;
  /**
   * @summary Select slices by index along one axis.
   * @category Selection
   * @semantics Selects full slices from `input` along `dim` using `index`.
   */
  function index_select(input: Tensor, dim: number, index: Tensor): Tensor;
  /**
   * @summary Top-k values and indices.
   * @category Selection
   * @semantics Returns both top values and their integer indices along one axis.
   */
  function topk<S extends Shape = Shape, D extends DType = DType>(input: Tensor<S, D>, k: number, dim?: number): TopKResult<S, D>;
  /**
   * @summary Materialize contiguous layout.
   * @category Selection
   * @semantics Returns a tensor with contiguous storage suitable for downstream kernels.
   */
  function contiguous<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>): Tensor<S, D>;

  /**
   * @summary Reshape a tensor.
   * @category Shape
   * @semantics Reinterprets tensor shape without changing element values.
   */
  function reshape<S extends Shape = Shape, T extends Shape = number[], D extends DType = DType>(x: Tensor<S, D>, shape: T, opts?: { axes?: readonly AxisName[] }): Tensor<[...T], D>;
  /**
   * @summary Swap two tensor axes.
   * @category Shape
   * @semantics Transposes `dim1` and `dim2` while preserving element values.
   */
  function transpose<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>, dim1: number, dim2: number): Tensor<S, D>;
  /**
   * @summary Permute tensor axes.
   * @category Shape
   * @semantics Reorders axes according to `dims` while preserving element values.
   */
  function permute<S extends Shape = Shape, D extends DType = DType>(x: Tensor<S, D>, dims: readonly number[]): Tensor<S, D>;
  /**
   * @summary Remove size-1 dimensions.
   * @category Shape
   * @semantics Removes singleton dimensions, optionally only at one axis.
   */
  function squeeze<D extends DType = DType>(x: Tensor, dim?: number): Tensor<number[], D>;
  /**
   * @summary Insert a size-1 dimension.
   * @category Shape
   * @semantics Inserts a singleton dimension at the requested axis.
   */
  function unsqueeze<D extends DType = DType>(x: Tensor, dim?: number): Tensor<number[], D>;

  /**
   * @summary Select a scalar position by explicit indices.
   * @category Shape
   * @semantics Returns the tensor element at the requested multidimensional index.
   */
  function at(x: Tensor, ...selectors: number[]): Tensor;
  /**
   * @summary Select a tensor slice with explicit selectors.
   * @category Shape
   * @semantics Applies scalar, range, and full-axis selectors across tensor axes.
   */
  function slice(x: Tensor, ...selectors: Selector[]): Tensor;
  /**
   * @summary Selector for taking all positions along one axis.
   * @category Shape
   * @semantics
   * Use in `slice(...)` to keep every position along the selected axis.
   */
  const all: AllSelector;
  /**
   * @summary Build an explicit range selector for `slice(...)`.
   * @category Shape
   * @semantics
   * Constructs a start/end[/step] selector that can be passed to `slice(...)`.
   */
  function range(start: number, end: number, step?: number): RangeSelector;

  /**
   * @summary Compute gradients with respect to trainable parameters.
   * @category Training
   * @semantics
   * Executes reverse-mode autodiff from `loss` into the provided parameter
   * list. This is the primary public differentiation entrypoint in
   * `affon:compute`.
   * @input
   * Accepts a scalar loss tensor and an explicit parameter list.
   * @param loss Scalar-valued loss tensor.
   * @param params Trainable parameters to differentiate.
   * @param opts Optional accumulation behavior.
   * @returns No value; writes gradients onto `params[i].grad`.
   * @example
   * grad(loss, model.parameters)
   */
  function grad(loss: Tensor, params: readonly Parameter[], opts?: { assign?: "replace" | "accumulate" }): void;
  /**
   * @summary Clear parameter gradients in place.
   * @category Training
   * @semantics
   * Resets parameter gradients to `null`, preparing a parameter list for the
   * next optimization step.
   * @param params Parameters whose gradients should be cleared.
   */
  function clear_grad(params: readonly Parameter[]): void;
  /**
   * @summary Clip parameter gradients by global norm.
   * @category Training
   * @semantics
   * Computes the total gradient norm across `params` and scales gradients in
   * place when the norm exceeds `max_norm`.
   * @param params Parameters whose gradients should be clipped.
   * @param max_norm Maximum allowed global norm.
   * @param eps Numerical stability epsilon.
   * @returns The pre-clipped global norm.
   */
  function clip_grad_norm(params: readonly Parameter[], max_norm: number, eps?: number): number;
  /**
   * @summary Temporarily disable gradient tracking within a callback.
   * @category Training
   * @semantics
   * Runs `fn` with gradient tracking disabled for the duration of the callback.
   * This is primarily useful for eval-time state updates and advanced runtime
   * control.
   */
  function no_grad<T>(fn: () => T): T;
  /**
   * @summary Copy tensor contents into an existing target tensor.
   * @category Core
   * @semantics
   * Overwrites the contents of `target` with the contents of `source` while
   * preserving the identity of `target`.
   * @param target Destination tensor.
   * @param source Source tensor.
   * @returns The destination tensor after the copy.
   */
  function copy<S extends Shape = Shape, D extends DType = DType>(target: Tensor<S, D>, source: Tensor<S, D>): Tensor<S, D>;
  /**
   * @summary Report whether a tensor is finite and where the first bad value occurs.
   * @category Debugging
   * @semantics
   * Summarizes the first non-finite value in a tensor for debugging numeric
   * instability.
   */
  function finite_summary(x: Tensor): FiniteSummary;
  /**
   * @summary Maximum absolute value for a fully finite tensor.
   * @category Debugging
   * @semantics
   * Returns the largest absolute value in a tensor when every element is
   * finite. Returns `null` if a non-finite value is present or if the runtime
   * cannot safely complete the reduction.
   */
  function finite_abs_max(x: Tensor): FiniteAbsMax;
  /**
   * @summary Elementwise threshold comparison.
   * @category Selection
   * @semantics
   * Computes `x > threshold` elementwise and returns an `i64` mask tensor.
   */
  function gt_scalar<S extends Shape = Shape>(x: Tensor<S>, threshold: number): Tensor<S, "i64">;
  /**
   * @summary Classification accuracy metric.
   * @category Metrics
   * @semantics
   * Computes multiclass accuracy from `[N, C]` logits plus `[N]` labels, or
   * binary accuracy from `[N]` / `[N, 1]` predictions plus binary targets.
   */
  function accuracy(pred: Tensor, target: Tensor, opts?: { threshold?: number }): number;
  /**
   * @summary Binary precision metric.
   * @category Metrics
   * @semantics
   * Measures `tp / (tp + fp)` after thresholding binary predictions.
   */
  function precision(pred: Tensor, target: Tensor, opts?: { threshold?: number }): number;
  /**
   * @summary Binary recall metric.
   * @category Metrics
   * @semantics
   * Measures `tp / (tp + fn)` after thresholding binary predictions.
   */
  function recall(pred: Tensor, target: Tensor, opts?: { threshold?: number }): number;
  /**
   * @summary Binary F1 metric.
   * @category Metrics
   * @semantics
   * Computes the harmonic mean of binary precision and recall.
   */
  function f1(pred: Tensor, target: Tensor, opts?: { threshold?: number }): number;
  /**
   * @summary Mean squared error metric.
   * @category Metrics
   * @semantics
   * Computes the average squared difference between predictions and targets.
   */
  function mse(pred: Tensor, target: Tensor): number;
  /**
   * @summary Mean absolute error metric.
   * @category Metrics
   * @semantics
   * Computes the average absolute difference between predictions and targets.
   */
  function mae(pred: Tensor, target: Tensor): number;
  /**
   * @summary Coefficient of determination metric.
   * @category Metrics
   * @semantics
   * Computes the regression `R^2` score from predictions and targets.
   */
  function r2(pred: Tensor, target: Tensor): number;

  /**
   * @summary Construct a callable compute module from state plus apply functions.
   * @category Modules
   * @semantics
   * Creates a `Module` from explicit persistent state and one or two apply
   * functions. The optional `evalApply` path is used when the module is
   * switched to eval mode.
   */
  function module<
    Args extends readonly Tensor[] = readonly Tensor[],
    Out extends Tensor = Tensor,
  >(
    state: ComputeState,
    apply: (state: any, ...args: Args) => Out,
    evalApply?: (state: any, ...args: Args) => Out,
  ): Module<Args, Out>;

  /**
   * @summary Compile a compute program or modeful module without changing its call shape.
   * @category Compilation
   * @semantics
   * Returns a callable executable wrapper that preserves the original call
   * signature while allowing the runtime to capture and optimize the
   * computation path underneath.
   */
  function compile<
    Args extends readonly Tensor[] = readonly Tensor[],
    Out extends Tensor = Tensor,
  >(f: Module<Args, Out>): CompiledModule<Args, Out>;
  function compile<
    Args extends readonly Tensor[] = readonly Tensor[],
    Out extends Tensor = Tensor,
  >(f: Program<Args, Out>): ExecutableProgram<Args, Out, Program<Args, Out>>;

  /**
   * @summary Persist a compute export report to disk.
   * @category Observability
   * @semantics
   * Calls `source.exportReport(...)`, writes the returned text or raw JSON
   * payload to a file, and returns the written path for offline tooling.
   */
  function exportReportFile(source: { exportReport?: (...args: any[]) => string; exportBundle?: (...args: any[]) => string }, ...args: any[]): string;
  /** @deprecated Prefer `exportReportFile(...)`. */
  function exportBundleFile(source: { exportBundle: (...args: any[]) => string }, ...args: any[]): string;

  /**
   * @summary Construct an SGD optimizer step.
   * @category Optim
   * @semantics
   * Creates a stateful stochastic gradient descent step object with mutable
   * learning rate and serializable optimizer state.
   */
  function sgd(opts?: { lr?: number; momentum?: number }): ComputeStep;
  /**
   * @summary Construct an Adam optimizer step.
   * @category Optim
   * @semantics
   * Creates a stateful Adam optimizer step with serializable optimizer state.
   */
  function adam(opts?: { lr?: number; beta1?: number; beta2?: number; eps?: number }): ComputeStep;
  /**
   * @summary Construct an AdamW optimizer step.
   * @category Optim
   * @semantics
   * Creates a stateful AdamW optimizer step with decoupled weight decay and
   * serializable optimizer state.
   */
  function adamw(opts?: { lr?: number; beta1?: number; beta2?: number; eps?: number; weight_decay?: number }): ComputeStep;
  /**
   * @summary Helpers for schedule durations.
   * @category Optim
   * @semantics
   * Provides explicit duration builders for step-based and epoch-based
   * learning-rate schedules.
   */
  const Duration: {
    /**
     * @summary Step-based duration.
     * @category Optim
     * @semantics Interprets `value` as a count of optimizer steps.
     */
    steps(value: number): StepDuration;
    /**
     * @summary Epoch-based duration.
     * @category Optim
     * @semantics Interprets `value` as a count of training epochs.
     */
    epochs(value: number): EpochDuration;
  };
  /**
   * @summary Built-in learning-rate schedule constructors.
   * @category Optim
   * @semantics
   * Provides reusable schedule builders for constant, linear, cosine, step,
   * and composed learning-rate schedules.
   */
  const schedules: {
    /**
     * @summary Constant learning-rate schedule.
     * @category Optim
     * @semantics Always returns the same learning rate.
     */
    constant(lr: number): LRSchedule;
    /**
     * @summary Linear learning-rate schedule.
     * @category Optim
     * @semantics Interpolates linearly from `start` to `end` over a finite duration.
     */
    linear(opts: {
      start: number;
      end: number;
      duration: StepDuration | EpochDuration;
    }): LRSchedule;
    /**
     * @summary Cosine learning-rate schedule.
     * @category Optim
     * @semantics Interpolates smoothly from `start` to `end` with cosine decay over a finite duration.
     */
    cosine(opts: {
      start: number;
      end: number;
      duration: StepDuration | EpochDuration;
    }): LRSchedule;
    /**
     * @summary Piecewise step-decay schedule.
     * @category Optim
     * @semantics Applies multiplicative decay by `gamma` every requested interval.
     */
    step(opts: {
      base: number;
      gamma: number;
      every: StepDuration | EpochDuration;
      duration?: StepDuration | EpochDuration;
    }): LRSchedule;
    /**
     * @summary Sequential schedule composition.
     * @category Optim
     * @semantics Evaluates the provided schedules in sequence over their declared durations.
     */
    sequence(...parts: LRSchedule[]): LRSchedule;
  };
  /**
   * @summary Compose an optimizer step with a learning-rate schedule.
   * @category Optim
   * @semantics
   * Wraps a `ComputeStep` so that `step.lr` is driven by the supplied
   * schedule over training context.
   */
  function scheduled(step: ComputeStep, schedule: LRSchedule): ScheduledStep;

  /**
   * @summary Default `affon:compute` namespace object.
   * @category Core
   * @semantics
   * Mirrors the named compute exports on a single namespace object for users
   * who prefer `compute.foo(...)` style imports.
   */
  const compute: {
    axes: typeof axes;
    tensor: typeof tensor;
    empty: typeof empty;
    zeros: typeof zeros;
    ones: typeof ones;
    full: typeof full;
    rand: typeof rand;
    randn: typeof randn;
    arange: typeof arange;
    linspace: typeof linspace;
    seed: typeof seed;
    setDevice: typeof setDevice;
    parameter: typeof parameter;
    cast: typeof cast;
    move: typeof move;
    neg: typeof neg;
    abs: typeof abs;
    sign: typeof sign;
    add: typeof add;
    sub: typeof sub;
    mul: typeof mul;
    div: typeof div;
    matmul: typeof matmul;
    dot: typeof dot;
    exp: typeof exp;
    log: typeof log;
    sqrt: typeof sqrt;
    square: typeof square;
    clamp: typeof clamp;
    relu: typeof relu;
    gelu: typeof gelu;
    silu: typeof silu;
    swish: typeof swish;
    sigmoid: typeof sigmoid;
    tanh: typeof tanh;
    softmax: typeof softmax;
    sum: typeof sum;
    mean: typeof mean;
    max: typeof max;
    min: typeof min;
    variance: typeof variance;
    std: typeof std;
    argmin: typeof argmin;
    argmax: typeof argmax;
    where: typeof where;
    masked_fill: typeof masked_fill;
    cross_entropy_indexed: typeof cross_entropy_indexed;
    cat: typeof cat;
    stack: typeof stack;
    one_hot: typeof one_hot;
    gather: typeof gather;
    index_select: typeof index_select;
    topk: typeof topk;
    contiguous: typeof contiguous;
    reshape: typeof reshape;
    transpose: typeof transpose;
    permute: typeof permute;
    squeeze: typeof squeeze;
    unsqueeze: typeof unsqueeze;
    at: typeof at;
    slice: typeof slice;
    all: typeof all;
    range: typeof range;
    grad: typeof grad;
    clear_grad(params: readonly Parameter[]): void;
    clip_grad_norm: typeof clip_grad_norm;
    no_grad: typeof no_grad;
    copy: typeof copy;
    finite_summary: typeof finite_summary;
    finite_abs_max: typeof finite_abs_max;
    gt_scalar: typeof gt_scalar;
    accuracy: typeof accuracy;
    precision: typeof precision;
    recall: typeof recall;
    f1: typeof f1;
    mse: typeof mse;
    mae: typeof mae;
    r2: typeof r2;
    module: typeof module;
    compile: typeof compile;
    exportReportFile: typeof exportReportFile;
    exportBundleFile: typeof exportBundleFile;
    sgd: typeof sgd;
    adam: typeof adam;
    adamw: typeof adamw;
    Duration: typeof Duration;
    schedules: typeof schedules;
    scheduled: typeof scheduled;
  };

  export {
    type Shape,
    type AxisName,
    type TensorOptions,
    type DType,
    type Device,
    type Program,
    type ModuleMode,
    type Tensor,
    type Parameter,
    type ParameterCollection,
    type RangeSelector,
    type AllSelector,
    type Selector,
    type ComputeState,
    type Module,
    type CapturedProgramNode,
    type CapturedProgramView,
    type CapturedProgramSummary,
    type GraphLoweringAnalysis,
    type GraphRegionSummary,
    type GraphTemplateSummary,
    type GraphStatsSummary,
    type GraphStaticMetadataSummary,
    type GraphSummaryNode,
    type GraphSummary,
    type NativeCompiledExecutableState,
    type ExecutionPlanContext,
    type ExecutionPlanStepKind,
    type ExecutionPlanRegionKind,
    type ExecutionPlanExecutionKind,
    type ExecutionPlanAllocationIntent,
    type ExecutionPlanInputRequirement,
    type ExecutionPlanInputLayoutDecision,
    type ExecutionPlanMatmulExecutionFamily,
    type ExecutionPlanClassificationConfidence,
    type ExecutionPlanMatmulHint,
    type ExecutionPlanHintSource,
    type ExecutionPlanMatmulEpilogueActivation,
    type ExecutionPlanPlannerHint,
    type ExecutionPlanMatmulDescriptor,
    type ExecutionPlanStep,
    type ExecutionPlanRegion,
    type GraphPlanReport,
    type ExecutionPlan,
    type Report,
    type ExecutionTensorSignatureEntry,
    type CompileFallbackKind,
    type GraphCaptureFailureStage,
    type GraphCaptureFailure,
    type ExecutionGraphProgram,
    type CapturedGraphProgram,
    type ExecutableProgram,
    type CompiledModule,
    type ComputeStep,
    type ScheduledStep,
    type TrainContext,
    type BundleExportBoundary,
    type BundleExportPhase,
    type BundleExportMode,
    type BundleExportFormat,
    type BundleExportOptions,
    type BundleFileOptions,
    type LRSchedule,
    type StepDuration,
    type EpochDuration,
    type TopKResult,
    type FiniteSummary,
    axes,
    tensor,
    empty,
    zeros,
    ones,
    full,
    rand,
    randn,
    arange,
    linspace,
    seed,
    setDevice,
    parameter,
    cast,
    move,
    neg,
    abs,
    sign,
    add,
    sub,
    mul,
    div,
    matmul,
    dot,
    exp,
    log,
    sqrt,
    square,
    clamp,
    relu,
    gelu,
    sigmoid,
    silu,
    swish,
    tanh,
    softmax,
    cross_entropy_indexed,
    sum,
    mean,
    max,
    min,
    variance,
    std,
    argmin,
    argmax,
    where,
    masked_fill,
    cat,
    stack,
    one_hot,
    gather,
    index_select,
    topk,
    contiguous,
    reshape,
    transpose,
    permute,
    squeeze,
    unsqueeze,
    at,
    slice,
    all,
    range,
    grad,
    clear_grad,
    clip_grad_norm,
    no_grad,
    copy,
    finite_summary,
    finite_abs_max,
    gt_scalar,
    accuracy,
    precision,
    recall,
    f1,
    mse,
    mae,
    r2,
    module,
    compile,
    exportReportFile,
    exportBundleFile,
    sgd,
    adam,
    adamw,
    Duration,
    schedules,
    scheduled,
  };

  export default compute;
}
