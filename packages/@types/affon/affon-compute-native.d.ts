declare module "affon:compute/native" {
  interface NativeTensor {
    readonly shape: readonly number[];
    readonly ndim: number;
    readonly dtype: "f32" | "f64" | "i64";
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
    tensor(values: number | readonly unknown[]): NativeTensor;
    add(lhs: NativeTensor, rhs: NativeTensor): NativeTensor;
    sub(lhs: NativeTensor, rhs: NativeTensor): NativeTensor;
    mul(lhs: NativeTensor, rhs: NativeTensor): NativeTensor;
    div(lhs: NativeTensor, rhs: NativeTensor): NativeTensor;
    matmul(lhs: NativeTensor, rhs: NativeTensor): NativeTensor;
    dot(lhs: NativeTensor, rhs: NativeTensor): NativeTensor;
    createSession(device: "cpu" | "metal" | "cuda" | `cuda:${number}`): NativeSession;
    sessionTensor(session: NativeSession, values: unknown, dtype: "f32" | "f64" | "i64"): NativeTensor;
    compileProgram(session: NativeSession, program_json: string): NativeExecutable;
    runExecutable(executable: NativeExecutable, inputs: readonly NativeTensor[]): NativeTensor[];
    stftPower(signal: NativeTensor, window: NativeTensor, hop: number, paddedLength: number, frames: number): NativeTensor;
    filterbank(spectrum: NativeTensor, filters: NativeTensor): NativeTensor;
    saveNative(entries: readonly { name: string; value: NativeTensor }[], path: string): void;
    loadNative(path: string, names?: readonly string[]): Record<string, NativeTensor>;
    inspectCheckpoint(path: string): Record<string, { dtype: "F32" | "F64" | "I64" | "BF16"; shape: number[] }>;
    $with_graph_execution<T>(callback: () => T): { value: T };
  };

  export default native;
}
