/**
 * Tensor operations shared by declarative Programs and evaluated tensors.
 * FormalTensor operands append graph nodes; evaluated Tensor operands execute
 * immediately in their common owning Session.
 *
 * @summary One operation vocabulary for symbolic authoring and immediate computation.
 */
declare module "affon:ops" {
  import type { FormalTensor, ProgramDType, SliceRange, Tensor } from "affon:compute"
  type AnyTensor = Tensor | FormalTensor
  type Same<T extends AnyTensor> = T extends FormalTensor ? FormalTensor : Tensor
  /** Elementwise addition with NumPy-style broadcasting. */
  export function add<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Elementwise subtraction with NumPy-style broadcasting. */
  export function sub<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Elementwise multiplication with NumPy-style broadcasting. */
  export function mul<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Elementwise division with NumPy-style broadcasting. */
  export function div<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Elementwise equality, returning an i64 mask. */
  export function eq<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Elementwise less-than comparison, returning an i64 mask. */
  export function lt<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Elementwise greater-than comparison, returning an i64 mask. */
  export function gt<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Compare each element with a scalar threshold, returning an i64 mask. */
  export function gt_scalar<T extends AnyTensor>(x: T, threshold: number): Same<T>
  /** Batched matrix multiplication over the final two dimensions. */
  export function matmul<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Inner product of two one-dimensional tensors. */
  export function dot<T extends AnyTensor>(a: T, b: T): Same<T>
  /** Elementwise arithmetic negation. */
  export function neg<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise absolute value. */
  export function abs<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise natural exponential. */
  export function exp<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise natural logarithm. */
  export function log<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise square root. */
  export function sqrt<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise square. */
  export function square<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise sign. */
  export function sign<T extends AnyTensor>(x: T): Same<T>
  /** Clamp each element to the inclusive range. */
  export function clamp<T extends AnyTensor>(x: T, minimum: number, maximum: number): Same<T>
  /** Elementwise rectified linear activation. */
  export function relu<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise logistic sigmoid activation. */
  export function sigmoid<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise SiLU (swish) activation. */
  export function silu<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise hyperbolic tangent. */
  export function tanh<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise error function. */
  export function erf<T extends AnyTensor>(x: T): Same<T>
  /** Elementwise Gaussian error linear unit. */
  export function gelu<T extends AnyTensor>(x: T): Same<T>
  /** Normalize exponentials along one axis. */
  export function softmax<T extends AnyTensor>(x: T, axis: number): Same<T>
  /** Sum all values or reduce one axis, optionally retaining a size-one dimension. */
  export function sum<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  /** Average all values or reduce one axis, optionally retaining a size-one dimension. */
  export function mean<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function min<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function max<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function variance<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function std<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function argmin<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  export function argmax<T extends AnyTensor>(x: T, axis?: number, keep_dims?: boolean): Same<T>
  /** Return a view with a compatible shape and optional replacement axis names. */
  export function reshape<T extends AnyTensor>(x: T, shape: readonly number[], axes?: readonly string[]): Same<T>
  /** Materialize a row-major contiguous tensor with unchanged logical values. */
  export function contiguous<T extends AnyTensor>(x: T): Same<T>
  /** Permute dimensions; omitting permutation reverses their order. */
  export function transpose<T extends AnyTensor>(x: T, permutation?: readonly number[]): Same<T>
  /** Convert tensor elements to another supported dtype. */
  export function cast<T extends AnyTensor>(x: T, dtype: ProgramDType): Same<T>
  /** Slice every dimension with explicit half-open ranges. */
  export function slice<T extends AnyTensor>(x: T, ranges: readonly SliceRange[]): Same<T>
  /** Remove one size-one dimension, or all size-one dimensions when axis is omitted. */
  export function squeeze<T extends AnyTensor>(x: T, axis?: number): Same<T>
  /** Insert a size-one dimension with an optional semantic axis name. */
  export function unsqueeze<T extends AnyTensor>(x: T, axis: number, name?: string): Same<T>
  /** Normalize values from axis onward using population variance. */
  export function layer_norm<T extends AnyTensor>(x: T, axis: number, epsilon?: number): Same<T>
  /** Replace values selected by a broadcast-compatible mask. */
  export function masked_fill<T extends AnyTensor>(x: T, mask: T, value: number): Same<T>
  /** Gather rows from an embedding table using integer indices. */
  export function embedding<T extends AnyTensor>(table: T, indices: T): Same<T>
  /** Gather positions from one axis using integer indices. */
  export function index_select<T extends AnyTensor>(x: T, axis: number, indices: T): Same<T>
  /** Gather values using an i64 index tensor. */
  export function gather<T extends AnyTensor>(x: T, axis: number, indices: T): Same<T>
  /** Convert i64 indices to an f32 one-hot tensor. */
  export function one_hot<T extends AnyTensor>(indices: T, classes: number): Same<T>
  /** Return mean indexed cross-entropy for floating-point logits and i64 class labels. */
  export function cross_entropy<T extends AnyTensor>(logits: T, labels: T): Same<T>
  export function mean_squared_error<T extends AnyTensor>(prediction: T, target: T): Same<T>
  export function mean_absolute_error<T extends AnyTensor>(prediction: T, target: T): Same<T>
  export function binary_cross_entropy<T extends AnyTensor>(prediction: T, target: T): Same<T>
  export function binary_cross_entropy_with_logits<T extends AnyTensor>(logits: T, target: T): Same<T>
  /** Concatenate same-rank tensors along one axis. */
  export function cat<T extends AnyTensor>(values: readonly T[], axis?: number): Same<T>
  /** Insert an axis and join identical tensor specs along it. */
  export function stack<T extends AnyTensor>(values: readonly T[], axis?: number): Same<T>
  /** Select values elementwise using an i64 condition. */
  export function where<T extends AnyTensor>(condition: T, on_true: T, on_false: T): Same<T>
}
