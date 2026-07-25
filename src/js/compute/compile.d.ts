declare module "affon:compute/compile.ts" {
export type DType = "f32" | "f64" | "i64"

export type ComputeTensorLike = {
  shape: number[]
  dtype: DType
}

export type GraphSupport = {
  graph(fn: (...inputs: any[]) => any): any
  graphWithArity(arity: number, fn: (...inputs: any[]) => any): any
  graphWithInputMetas(
    arity: number,
    metas: Array<{ shape: number[]; dtype: DType }>,
    fn: (...inputs: any[]) => any,
  ): any
  isCapturing(): boolean
}

export type CompileHelpers = {
  graphSupport: GraphSupport
  isTensorLike(x: any): x is ComputeTensorLike
  programStateKey(source: any): string
  programArity(source: any, fallback: (...args: any[]) => any): number
  replayCapturedProgram(source: any, userInputs: any[]): any
  isProgramSource(source: any): boolean
  isReplayableProgramSource(source: any): boolean
  finalizeExecutable?(executable: any, original: any): any
}

export function compileWithHelpers(programOrFn: any, helpers: CompileHelpers): any
}
