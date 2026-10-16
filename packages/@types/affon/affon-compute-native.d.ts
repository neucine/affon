declare module "affon:compute/native" {
  interface NativeTensor {
    readonly shape: readonly number[];
    readonly ndim: number;
    readonly dtype: "bf16" | "f32" | "f64" | "u32" | "i64";
    readonly device: "cpu" | "metal" | "cuda";
    item(): number;
    to_array(): unknown;
    toString(): string;
    repr(): string;
    dispose(): void;
  }

  interface NativeSession { dispose(): void }
  interface NativeExecutable { dispose(): void }

  const native: {
    defaultDevice(): "cpu" | "metal" | "cuda";
    createSession(device: "cpu" | "metal" | "cuda" | `cuda:${number}`, options_json?: string): NativeSession;
    sessionTensor(session: NativeSession, values: unknown, dtype: "bf16" | "f32" | "f64" | "u32" | "i64", shape?: readonly number[]): NativeTensor;
    sessionTensorCopy(session: NativeSession, source: unknown): NativeTensor | undefined;
    sessionFull(session: NativeSession, shape: readonly number[], value: number, dtype: "bf16" | "f32" | "f64" | "u32" | "i64"): NativeTensor;
    compileProgram(session: NativeSession, program_json: string, options_json?: string): NativeExecutable;
    explainExecutable(executable: NativeExecutable): string;
    runExecutable(executable: NativeExecutable, inputs: readonly NativeTensor[]): NativeTensor[];
    stftPower(signal: NativeTensor, window: NativeTensor, hop: number, paddedLength: number, frames: number): NativeTensor;
    filterbank(spectrum: NativeTensor, filters: NativeTensor): NativeTensor;
    saveNative(entries: readonly { name: string; value: NativeTensor }[], path: string): void;
    loadNative(path: string, names?: readonly string[]): Record<string, NativeTensor>;
    inspectCheckpoint(path: string): Record<string, { dtype: "BF16" | "F32" | "F64" | "U32" | "I64"; shape: number[] }>;
  };

  export default native;
}
