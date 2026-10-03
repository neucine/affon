declare module "affon:compute" {
  import type { Optimizer } from "affon:optim";
  export type ProgramDType = "f32" | "f64" | "i64";
  export type Device = "cpu" | "metal" | "cuda" | `cuda:${number}`;
  export type ProgramShape = readonly number[];
  export type TensorData = number | readonly TensorData[];
  export type TensorInitializerValue = TensorData | Readonly<{
    shape: readonly number[];
    dtype: ProgramDType;
    to_array(): TensorData;
  }>;
  export type ProgramArguments = Readonly<Record<string, Tensor>>;
  export type SliceRange = Readonly<{ start: number; stop: number; step?: number }>;
  export type TensorSpec = Readonly<{ dtype: ProgramDType; shape: ProgramShape; axes?: readonly string[] }>;
  export type Initializer =
    | Readonly<{ kind: "zeros" | "ones" }>
    | Readonly<{ kind: "constant"; value: number }>
    | Readonly<{ kind: "uniform"; min: number; max: number }>
    | Readonly<{ kind: "normal"; mean?: number; standard_deviation?: number }>
    | Readonly<{ kind: "xavier_uniform" | "xavier_normal" }>;
  export type ProgramFormal = Readonly<{
    name: string;
    role: "argument" | "parameter" | "state" | "constant";
    spec: TensorSpec;
    provenance: string;
  }>;
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

  export const Tensor: Readonly<{
    spec(dtype: ProgramDType, shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
    f32(shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
    f64(shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
    i64(shape: readonly number[], options?: { axes?: readonly string[] }): TensorSpec;
  }>;
  export interface Tensor {
    readonly shape: readonly number[];
    readonly axes?: readonly string[];
    readonly ndim: number;
    readonly dtype: "f32" | "f64" | "i64";
    readonly device: Device;
    readonly disposed: boolean;
    item(): number;
    to_array(): TensorData;
    toString(): string;
    repr(): string;
    dispose(): void;
    [Symbol.dispose](): void;
  }

  export interface FormalTensor<S extends TensorSpec = TensorSpec> {
    readonly spec: S;
    readonly id: number;
    readonly name?: string;
    readonly role: "argument" | "parameter" | "state" | "constant" | "intermediate";
  }

  export interface Program<Args extends readonly unknown[] = readonly FormalTensor[], Out = FormalTensor | readonly FormalTensor[]> {
    (...arguments_: Args): Out;
    (arguments_: Record<string, FormalTensor>): Out;
    readonly name: string;
    readonly provenance: string;
    inspect(): ProgramInspection;
  }
  export type EvaluatedProgramOutput<Out> = Out extends FormalTensor
    ? Tensor
    : Out extends readonly FormalTensor[]
      ? readonly Tensor[]
      : never;

  export interface ProgramNN {
    linear(value: FormalTensor, options: { name: string; out_features: number; bias?: boolean }): FormalTensor;
    embedding(indices: FormalTensor, options: { name: string; num_embeddings: number; embedding_dim: number; dtype?: ProgramDType }): FormalTensor;
    layer_norm(value: FormalTensor, options: { name: string; normalized_shape?: number; epsilon?: number; affine?: boolean }): FormalTensor;
    cross_entropy(logits: FormalTensor, labels: FormalTensor): FormalTensor;
  }

  export interface ProgramBuilder {
    readonly nn: ProgramNN;
    argument(name: string, spec: TensorSpec): FormalTensor;
    parameter(name: string, spec: TensorSpec, options?: { initializer?: Initializer }): FormalTensor;
    state(name: string, spec: TensorSpec, options?: { initializer?: Initializer }): FormalTensor;
    constant(name: string, value: TensorData, spec: TensorSpec): FormalTensor;
    use(child: Program, options: { as: string; [name: string]: unknown }): FormalTensor | readonly FormalTensor[];
  }

  export function program<Out extends FormalTensor | readonly FormalTensor[]>(name: string, author: (p: ProgramBuilder) => Out): Program<readonly FormalTensor[], Out>;
  export function gradient(loss: Program<readonly FormalTensor[], FormalTensor>, independent_variables: string): Program<readonly FormalTensor[], FormalTensor>;
  export function gradient(loss: Program<readonly FormalTensor[], FormalTensor>, independent_variables: readonly string[]): Program;
  export function optimize(loss: Program<readonly FormalTensor[], FormalTensor>, optimizer: Optimizer): Program<readonly FormalTensor[], FormalTensor>;

  export interface ExecutionState {
    readonly parameters: Record<string, Tensor>;
    readonly model_state: Record<string, Tensor>;
    readonly optimizer_state: Record<string, Tensor | number>;
    readonly rng_state: Record<string, number>;
    readonly session: Session;
    readonly disposed: boolean;
    dispose(): void;
    [Symbol.dispose](): void;
  }
  export interface Executable<Out extends FormalTensor | readonly FormalTensor[] = FormalTensor | readonly FormalTensor[]> {
    readonly program: Program<readonly FormalTensor[], Out>;
    readonly session: Session;
    readonly argument_names: readonly string[];
    readonly argument_specs: readonly ProgramFormal[];
    readonly parameter_names: readonly string[];
    readonly state_names: readonly string[];
    readonly native_input_order: readonly string[];
    readonly disposed: boolean;
    run(arguments_: ProgramArguments, state?: ExecutionState): EvaluatedProgramOutput<Out>;
    dispose(): void;
    [Symbol.dispose](): void;
  }
  export class Session {
    constructor(options?: { device?: Device });
    readonly device: Device;
    readonly disposed: boolean;
    tensor(values: TensorData, options?: { dtype?: ProgramDType; axes?: readonly string[] }): Tensor;
    compile<Out extends FormalTensor | readonly FormalTensor[]>(source: Program<readonly FormalTensor[], Out>): Executable<Out>;
    initialize(source: Program, options?: { seed?: number; parameters?: Readonly<Record<string, TensorInitializerValue>>; model_state?: Readonly<Record<string, TensorInitializerValue>> }): ExecutionState;
    owns(value: unknown): boolean;
    dispose(): void;
    [Symbol.dispose](): void;
  }
}
