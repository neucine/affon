declare module "affon:core/native" {
  export type NativeDType = 1 | 2 | 3 | 4 | 5;
  export type NativeProgramNodeKind = 0 | 1 | 2 | 3 | 4 | 5;

  export interface NativeSessionOptions {
    readonly schema_version: 1;
    /** 0 = strict, 1 = allow_nondeterministic. */
    readonly determinism: 0 | 1;
    readonly telemetry_enabled: 0 | 1;
    /** 0 = none, 1 = invocation, 2 = kernel. */
    readonly compute_timing: 0 | 1 | 2;
    readonly hardware_metrics: 0 | 1;
    readonly profiler_markers: 0 | 1;
    readonly bridge_timing: 0 | 1;
    readonly capture_records?: number;
  }

  export interface NativeCompileOptions {
    readonly schema_version: 1;
    /** 0 = none, 1 = safe. */
    readonly optimization_level: 0 | 1;
    /** 1 = memory, 2 = balanced, 3 = latency, 4 = throughput. */
    readonly optimization_goal: 1 | 2 | 3 | 4;
    /** 0 = exact_only, 1 = backend_equivalent, 2 = approximate. */
    readonly numerical_policy: 0 | 1 | 2;
    /** 0 = summary, 1 = detailed. */
    readonly explanation_level: 0 | 1;
    readonly dense_inputs: 0 | 1;
    /** 0 = retain, 1 = automatic, 2 = recompute. */
    readonly residual_policy: 0 | 1 | 2;
    readonly structural_states?: number;
    readonly structural_expansions?: number;
    readonly complete_evaluations?: number;
    readonly physical_work?: number;
    readonly ready_work?: number;
    readonly max_peak_active_bytes?: number;
    readonly max_invocation_scratch_bytes?: number;
  }

  export interface NativeProgramNode {
    readonly id: number;
    readonly kind: NativeProgramNodeKind;
    readonly dtype: NativeDType;
    readonly shape: readonly number[];
    readonly op?: number;
    readonly operands?: readonly number[];
    readonly axis?: number;
    readonly keep_dims?: 0 | 1;
    readonly min?: number;
    readonly max?: number;
    readonly fill_value?: number;
    readonly epsilon?: number;
    readonly k?: number;
    readonly num_classes?: number;
    readonly class_axis?: number;
    readonly bits?: number;
    readonly group_size?: number;
    readonly axes?: readonly number[];
    readonly permutation?: readonly number[];
    readonly ranges?: readonly Readonly<{ start: number; stop: number; step: number }>[];
    readonly largest?: 0 | 1;
    readonly sorted?: 0 | 1;
    readonly constant?: number;
    readonly constant_bytes?: ArrayBuffer;
    readonly gradient_target?: number;
  }

  export interface NativeProgram {
    readonly schema_version: 2;
    /** 0 = authored, 1 = gradient. */
    readonly kind: 0 | 1;
    readonly nodes: readonly NativeProgramNode[];
    readonly outputs?: readonly number[];
  }

  interface NativeTensor {
    readonly shape: readonly number[];
    readonly ndim: number;
    readonly dtype: "bf16" | "f32" | "f64" | "u32" | "i64";
    readonly device: "cpu" | "metal" | "cuda" | `cuda:${number}`;
    item(): number;
    to_array(): unknown;
    to_bytes(): ArrayBuffer;
    toString(): string;
    repr(): string;
    dispose(): void;
  }

  interface NativeSession { dispose(): void }
  interface NativeExecutable { dispose(): void }

  const native: {
    defaultDevice(): "cpu" | "metal" | "cuda";
    createSession(device: "cpu" | "metal" | "cuda" | `cuda:${number}`, options?: NativeSessionOptions): NativeSession;
    sessionTensor(session: NativeSession, values: unknown, dtype: "bf16" | "f32" | "f64" | "u32" | "i64", shape?: readonly number[]): NativeTensor;
    sessionTensorBytes(session: NativeSession, bytes: ArrayBuffer, dtype: "bf16" | "f32" | "f64" | "u32" | "i64", shape: readonly number[]): NativeTensor;
    sessionTensorCopy(session: NativeSession, source: unknown): NativeTensor | undefined;
    sessionFull(session: NativeSession, shape: readonly number[], value: number, dtype: "bf16" | "f32" | "f64" | "u32" | "i64"): NativeTensor;
    compileProgram(session: NativeSession, program: NativeProgram, options?: NativeCompileOptions): NativeExecutable;
    explainExecutable(executable: NativeExecutable): string;
    cynaSelectedSteps(executable: NativeExecutable): string;
    cynaSelectedPhysicalNodes(executable: NativeExecutable): string;
    selectedPhysicalValues(executable: NativeExecutable): string;
    selectedPhysicalLiveness(executable: NativeExecutable): string;
    cynaResidualDecisions(executable: NativeExecutable): string;
    cynaSelectionComparison(executable: NativeExecutable): string;
    cynaSearchEvidence(executable: NativeExecutable): string;
    executableRouting(executable: NativeExecutable): string;
    executableBackend(executable: NativeExecutable): "cyna";
    runExecutable(executable: NativeExecutable, inputs: readonly NativeTensor[]): NativeTensor[];
    resetEvidenceCapture(session: NativeSession): void;
    exportEvidenceCapture(session: NativeSession, executable: NativeExecutable): string;
    stftPower(signal: NativeTensor, window: NativeTensor, hop: number, paddedLength: number, frames: number): NativeTensor;
    filterbank(spectrum: NativeTensor, filters: NativeTensor): NativeTensor;
  };

  export default native;
}
