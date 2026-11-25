declare module "affon:checkpoint/native" {
  interface NativeCheckpointTensor {
    readonly shape: readonly number[];
    readonly dtype: "bf16" | "f32" | "f64" | "u32" | "i64";
    readonly device: "cpu" | "metal" | "cuda" | `cuda:${number}`;
    item(): number;
    to_array(): unknown;
    to_bytes(): ArrayBuffer;
    toString(): string;
    repr(): string;
    dispose(): void;
  }

  const native: {
    saveNative(
      entries: readonly {
        name: string;
        value: NativeCheckpointTensor;
      }[],
      path: string,
    ): void;
    loadNative(
      path: string,
      names?: readonly string[],
    ): Record<string, NativeCheckpointTensor>;
    inspectCheckpoint(path: string): Record<
      string,
      {
        dtype: "BF16" | "F32" | "F64" | "U32" | "I64";
        shape: number[];
      }
    >;
  };

  export default native;
}
