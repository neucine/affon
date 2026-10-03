/**
 * Declarative tensor Programs, evaluated tensors, compilation, differentiation,
 * optimization, and Session-owned execution.
 *
 * @summary Build inspectable compute Programs and execute them in an explicit Session.
 */
declare module "affon:compute" {
  import type { Optimizer } from "affon:optim";
  /** Numeric element types supported by the Program API. */
  export type ProgramDType = "f32" | "f64" | "i64";
  /** Compute device selected when a Session is created. */
  export type Device = "cpu" | "metal" | "cuda" | `cuda:${number}`;
  /** Immutable tensor dimensions in row-major order. */
  export type ProgramShape = readonly number[];
  /** Scalar or rectangular nested-array tensor data. */
  export type TensorData = number | readonly TensorData[];
  /** Data accepted when restoring a parameter or model-state value. */
  export type TensorInitializerValue = TensorData | Readonly<{
    shape: readonly number[];
    dtype: ProgramDType;
    to_array(): TensorData;
  }>;
  /** Named evaluated inputs supplied to Executable.run(). */
  export type ProgramArguments = Readonly<Record<string, Tensor>>;
  /** Half-open slice range. A negative step reverses traversal. */
  export type SliceRange = Readonly<{ start: number; stop: number; step?: number }>;
  /** Static dtype, shape, and optional semantic-axis contract for a formal tensor. */
  export type TensorSpec = Readonly<{ dtype: ProgramDType; shape: ProgramShape; axes?: readonly string[] }>;
  /** Options shared by default-Session evaluated-tensor constructors. */
  export type TensorValueOptions = Readonly<{ dtype?: ProgramDType; axes?: readonly string[] }>;
  /** Immutable parameter or state initialization recipe. */
  export type Initializer =
    | Readonly<{ kind: "zeros" | "ones" }>
    | Readonly<{ kind: "constant"; value: number }>
    | Readonly<{ kind: "uniform"; min: number; max: number }>
    | Readonly<{ kind: "normal"; mean?: number; standard_deviation?: number }>
    | Readonly<{ kind: "xavier_uniform" | "xavier_normal" }>;
  /** Inspected formal input or state declaration. */
  export type ProgramFormal = Readonly<{
    name: string;
    role: "argument" | "parameter" | "state" | "constant";
    spec: TensorSpec;
    provenance: string;
  }>;
  /** One automatically generated step in a node's nested Program-composition path. */
  export type ProgramPathSegment = Readonly<{
    program: string;
    instance: string;
  }>;
  /** Outermost-to-innermost Program uses containing an inspected node. */
  export type ProgramPath = readonly ProgramPathSegment[];
  /** Serializable structural view returned by Program.inspect(). */
  export type ProgramInspection = Readonly<{
    name: string;
    provenance: string;
    kind: string;
    arguments: readonly ProgramFormal[];
    parameters: readonly ProgramFormal[];
    state: readonly ProgramFormal[];
    constants: readonly ProgramFormal[];
    nodes: readonly Readonly<{
      id: number;
      kind: "argument" | "parameter" | "state" | "constant" | "intermediate" | "operation" | "composition" | "gradient";
      path: ProgramPath;
      role?: "argument" | "parameter" | "state" | "constant";
      name?: string;
      provenance?: string;
      spec: TensorSpec;
      op?: string;
      operands?: readonly number[];
      options?: Readonly<Record<string, unknown>>;
      value?: unknown;
    }>[];
    outputs: readonly number[];
    transitions: readonly Readonly<{ kind: "optimize"; optimizer: Optimizer; parameters: readonly string[] }>[];
  }>;

  /**
   * Tensor specification factories and convenient evaluated-value constructors.
   * Specification factories are used while authoring Programs. Value constructors
   * allocate in one hidden, lazily created default Session.
   */
  export const Tensor: Readonly<{
    /** Create a formal tensor specification with an explicit dtype. */
    spec(dtype: ProgramDType, shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
    /** Create an f32 formal tensor specification. */
    f32(shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
    /** Create an f64 formal tensor specification. */
    f64(shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
    /** Create an i64 formal tensor specification. */
    i64(shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
    /** Create an evaluated tensor from scalar or rectangular nested-array data. */
    from(values: TensorData, options?: TensorValueOptions): Tensor;
    /** Create an evaluated tensor filled with one scalar value. */
    full(shape: readonly number[], value: number, options?: TensorValueOptions): Tensor;
    /** Create a zero-filled evaluated tensor. */
    zeros(shape: readonly number[], options?: TensorValueOptions): Tensor;
    /** Create a one-filled evaluated tensor. */
    ones(shape: readonly number[], options?: TensorValueOptions): Tensor;
    /** Create evenly spaced values in the half-open interval. */
    arange(start: number, end?: number, step?: number, options?: TensorValueOptions): Tensor;
    /** Create a fixed number of evenly spaced values including both endpoints. */
    linspace(start: number, end: number, steps?: number, options?: TensorValueOptions): Tensor;
    /** Create reproducible uniform random values in [0, 1). */
    rand(shape: readonly number[], options?: TensorValueOptions & { seed?: number }): Tensor;
    /** Create reproducible standard-normal random values. */
    randn(shape: readonly number[], options?: TensorValueOptions & { seed?: number }): Tensor;
  }>;
  /** Evaluated, Session-owned tensor data. */
  export interface Tensor {
    /** Immutable dimensions in row-major order. */
    readonly shape: readonly number[];
    /** Optional semantic name for each dimension. */
    readonly axes?: readonly string[];
    /** Number of dimensions. */
    readonly ndim: number;
    /** Element dtype. */
    readonly dtype: "f32" | "f64" | "i64";
    /** Device of the owning Session. */
    readonly device: Device;
    readonly disposed: boolean;
    /** Return the scalar value of a one-element tensor. */
    item(): number;
    /** Copy tensor contents into scalar or nested JavaScript arrays. */
    to_array(): TensorData;
    /** Human-readable summary. */
    toString(): string;
    /** Detailed representation including shape, dtype, device, and values. */
    repr(): string;
    /** Release this tensor's native resources. Safe to call repeatedly. */
    dispose(): void;
    [Symbol.dispose](): void;
  }

  /** Symbolic tensor used only while authoring a Program. */
  export interface FormalTensor<S extends TensorSpec = TensorSpec> {
    /** Static dtype, shape, and axis contract. */
    readonly spec: S;
    /** Stable node identifier within the Program graph. */
    readonly id: number;
    /** Declared name, when this node represents an argument or state value. */
    readonly name?: string;
    /** Role assigned during authoring. */
    readonly role: "argument" | "parameter" | "state" | "constant" | "intermediate";
  }

  /** Immutable, inspectable compute graph. */
  export interface Program<Bindings extends Record<string, FormalTensor> = Record<string, FormalTensor>, Out = FormalTensor | readonly FormalTensor[]> {
    /** Compose this Program with named bindings and an optional parent-local instance name. */
    (bindings: Bindings, instance?: string): Out;
    /** Stable author-supplied Program name. */
    readonly name: string;
    /** Provenance identifier distinguishing authored and transformed Programs. */
    readonly provenance: string;
    /** Return an immutable structural description suitable for tooling. */
    inspect(): ProgramInspection;
  }
  /** Deferred built-in loss that is materialized from a model's output specification. */
  export type LossProgramTemplate = Readonly<{
    kind: "cross_entropy" | "mean_squared_error" | "binary_cross_entropy" | "binary_cross_entropy_with_logits";
    target: string;
  }>;
  /** Convenient built-in loss Program templates. Custom losses can be authored with program(). */
  export const losses: Readonly<{
    /** Infer indexed cross-entropy logits and label specifications from the model. */
    cross_entropy(options?: { target?: string }): LossProgramTemplate;
    /** Infer a same-shaped floating-point regression target from the model. */
    mean_squared_error(options?: { target?: string }): LossProgramTemplate;
    /** Infer a same-shaped floating-point binary target for probability predictions. */
    binary_cross_entropy(options?: { target?: string }): LossProgramTemplate;
    /** Infer a same-shaped floating-point binary target for unnormalized logits. */
    binary_cross_entropy_with_logits(options?: { target?: string }): LossProgramTemplate;
  }>;
  /** Reporting metrics for evaluated tensors. Metrics never mutate ExecutionState. */
  export const metrics: Readonly<{
    /** Multiclass logits accuracy or thresholded binary accuracy. */
    accuracy(prediction: Tensor, target: Tensor, options?: { threshold?: number }): number;
    precision(prediction: Tensor, target: Tensor, options?: { threshold?: number }): number;
    recall(prediction: Tensor, target: Tensor, options?: { threshold?: number }): number;
    f1(prediction: Tensor, target: Tensor, options?: { threshold?: number }): number;
    mean_squared_error(prediction: Tensor, target: Tensor): number;
    mean_absolute_error(prediction: Tensor, target: Tensor): number;
    r2_score(prediction: Tensor, target: Tensor): number;
  }>;
  /** Map formal Program outputs to their evaluated run-time tensor shape. */
  export type EvaluatedProgramOutput<Out> = Out extends FormalTensor
    ? Tensor
    : Out extends readonly FormalTensor[]
      ? readonly Tensor[]
      : never;

  /** Authoring context passed to program(). */
  export interface ProgramBuilder {
    /** Declare a named value that must be supplied to every execution. */
    argument(name: string, spec: TensorSpec): FormalTensor;
    /** Declare named trainable state with an optional initialization recipe. */
    parameter(name: string, spec: TensorSpec, options?: { initializer?: Initializer }): FormalTensor;
    /** Declare named persistent, non-trainable model state. */
    state(name: string, spec: TensorSpec, options?: { initializer?: Initializer }): FormalTensor;
    /** Embed immutable validated data in the Program graph. */
    constant(name: string, value: TensorData, spec: TensorSpec): FormalTensor;
  }

  /**
   * Author a named, immutable compute graph.
   * @semantics The callback runs once during authoring. Use affon:ops with the returned FormalTensor values to build graph nodes.
   * @output Returns an inspectable Program. Calling it is only valid while composing another authored Program.
   * @example const square = program("square", p => mul(p.argument("x", Tensor.f32([4])), p.argument("x", Tensor.f32([4]))))
   */
  export function program<Out extends FormalTensor | readonly FormalTensor[]>(name: string, author: (p: ProgramBuilder) => Out): Program<Record<string, FormalTensor>, Out>;
  /**
   * Transform a scalar loss Program into a Program that returns derivatives.
   * @input independent_variables names one value or an ordered list of values to differentiate.
   * @errors Throws when the source is not scalar, a requested name is absent, or an operation has no gradient rule.
   */
  export function gradient(loss: Program<Record<string, FormalTensor>, FormalTensor>, independent_variables: string): Program<Record<string, FormalTensor>, FormalTensor>;
  export function gradient(loss: Program<Record<string, FormalTensor>, FormalTensor>, independent_variables: readonly string[]): Program;
  /**
   * Combine a reusable model Program with a scalar loss Program or built-in loss template and transform the result into a training step.
   * @input Model outputs bind positionally to the loss Program's leading arguments. Remaining loss arguments become training inputs.
   * @semantics Preserves the model's parameter and state provenance so an ExecutionState initialized for the training step can run the standalone model for evaluation or inference.
   * @output Returns a Program compiled and run like any other Program, using an ExecutionState for parameters and optimizer state.
   */
  export function optimize(model: Program<Record<string, FormalTensor>, FormalTensor | readonly FormalTensor[]>, loss: Program<Record<string, FormalTensor>, FormalTensor> | LossProgramTemplate, optimizer: Optimizer): Program<Record<string, FormalTensor>, FormalTensor>;

  /** Session-owned mutable execution data for parameters, model state, optimizer state, and RNG state. */
  export interface ExecutionState {
    /** Materialized trainable parameters keyed by fully qualified Program name. */
    readonly parameters: Record<string, Tensor>;
    /** Materialized persistent non-parameter state. */
    readonly model_state: Record<string, Tensor>;
    /** Optimizer slots and step counters. */
    readonly optimizer_state: Record<string, Tensor | number>;
    /** Reproducible initialization and execution RNG counters. */
    readonly rng_state: Record<string, number>;
    /** Session that owns this state. */
    readonly session: Session;
    /** Whether dispose() has been called. */
    readonly disposed: boolean;
    /** Release resources owned by this state. Safe to call repeatedly. */
    dispose(): void;
    [Symbol.dispose](): void;
  }
  /** Compiled, Session-bound form of a Program. */
  export interface Executable<Out extends FormalTensor | readonly FormalTensor[] = FormalTensor | readonly FormalTensor[]> {
    /** Source Program compiled by the Session. */
    readonly program: Program<Record<string, FormalTensor>, Out>;
    /** Session that owns this executable and its returned tensors. */
    readonly session: Session;
    /** Names of evaluated arguments accepted by run(). */
    readonly argument_names: readonly string[];
    /** Complete dtype, shape, and axis contract for each argument. */
    readonly argument_specs: readonly ProgramFormal[];
    /** Fully qualified parameter names consumed by this executable. */
    readonly parameter_names: readonly string[];
    /** Fully qualified model-state names consumed by this executable. */
    readonly state_names: readonly string[];
    /** Stable native input order used after named validation. */
    readonly native_input_order: readonly string[];
    readonly disposed: boolean;
    /** Validate named arguments and execute with optional initialized state. */
    run(arguments_: ProgramArguments, state?: ExecutionState): EvaluatedProgramOutput<Out>;
    dispose(): void;
    [Symbol.dispose](): void;
  }
  /** Owns evaluated tensors, compiled executable caches, and execution state for one device. */
  export class Session {
    /** Create a Session on the requested device, or on the runtime-selected default device. */
    constructor(options?: { device?: Device });
    /** Device used for all tensors and executions owned by this Session. */
    readonly device: Device;
    readonly disposed: boolean;
    /** Create an evaluated tensor owned by this Session. */
    tensor(values: TensorData, options?: { dtype?: ProgramDType; axes?: readonly string[] }): Tensor;
    /** Compile a Program, returning the cached executable for repeated compilation of the same Program. */
    compile<Out extends FormalTensor | readonly FormalTensor[]>(source: Program<Record<string, FormalTensor>, Out>): Executable<Out>;
    /** Materialize Program parameters and model state using deterministic initialization. */
    initialize(source: Program, options?: { seed?: number; parameters?: Readonly<Record<string, TensorInitializerValue>>; model_state?: Readonly<Record<string, TensorInitializerValue>> }): ExecutionState;
    /** Return whether this Session owns the supplied Tensor, Executable, or ExecutionState. */
    owns(value: unknown): boolean;
    /** Dispose this Session. Child objects retain their native storage until they are also released. */
    dispose(): void;
    [Symbol.dispose](): void;
  }
}
