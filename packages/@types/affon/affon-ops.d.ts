declare module "affon:ops" {
  import type { FormalTensor, ProgramDType, SliceRange, Tensor } from "affon:compute"
  type AnyTensor = Tensor | FormalTensor
  type Same<T extends AnyTensor> = T extends FormalTensor ? FormalTensor : Tensor
  export function add<T extends AnyTensor>(a: T, b: T): Same<T>
  export function sub<T extends AnyTensor>(a: T, b: T): Same<T>
  export function mul<T extends AnyTensor>(a: T, b: T): Same<T>
  export function div<T extends AnyTensor>(a: T, b: T): Same<T>
  export function matmul<T extends AnyTensor>(a: T, b: T): Same<T>
  export function dot<T extends AnyTensor>(a: T, b: T): Same<T>
  export function neg<T extends AnyTensor>(x: T): Same<T>
  export function abs<T extends AnyTensor>(x: T): Same<T>
  export function exp<T extends AnyTensor>(x: T): Same<T>
  export function log<T extends AnyTensor>(x: T): Same<T>
  export function sqrt<T extends AnyTensor>(x: T): Same<T>
  export function relu<T extends AnyTensor>(x: T): Same<T>
  export function sigmoid<T extends AnyTensor>(x: T): Same<T>
  export function silu<T extends AnyTensor>(x: T): Same<T>
  export function tanh<T extends AnyTensor>(x: T): Same<T>
  export function erf<T extends AnyTensor>(x: T): Same<T>
  export function gelu<T extends AnyTensor>(x: T): Same<T>
  export function softmax<T extends AnyTensor>(x: T, axis: number): Same<T>
  export function sum<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function mean<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function reshape<T extends AnyTensor>(x: T, shape: readonly number[], axes?: readonly string[]): Same<T>
  export function contiguous<T extends AnyTensor>(x: T): Same<T>
  export function transpose<T extends AnyTensor>(x: T, permutation?: readonly number[]): Same<T>
  export function cast<T extends AnyTensor>(x: T, dtype: ProgramDType): Same<T>
  export function slice<T extends AnyTensor>(x: T, ranges: readonly SliceRange[]): Same<T>
  export function squeeze<T extends AnyTensor>(x: T, axis?: number): Same<T>
  export function unsqueeze<T extends AnyTensor>(x: T, axis: number, name?: string): Same<T>
  export function layer_norm<T extends AnyTensor>(x: T, axis: number, epsilon?: number): Same<T>
  export function masked_fill<T extends AnyTensor>(x: T, mask: T, value: number): Same<T>
  export function embedding<T extends AnyTensor>(table: T, indices: T): Same<T>
  export function index_select<T extends AnyTensor>(x: T, axis: number, indices: T): Same<T>
  export function cat<T extends AnyTensor>(values: readonly T[], axis?: number): Same<T>
}
