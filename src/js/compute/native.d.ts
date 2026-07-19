declare module "affon:compute/native" {
  interface Tensor {
    readonly shape: readonly number[]
    readonly ndim: number
    readonly dtype: "f32" | "f64" | "i64"
    readonly device: "cpu" | "metal"
    readonly grad: Tensor | undefined
    readonly grad_device: "cpu" | "metal" | undefined
    item(): number
    to_array(): unknown
    toString(): string
    repr(): string
    to(device: "cpu" | "metal"): Tensor
    backward(): void
  }

  const native: {
    tensor(values: number | readonly number[] | readonly (number | readonly number[])[], options?: { dtype?: "f32" | "f64" | "i64"; device?: "cpu" | "metal" }): Tensor
    empty(shape: readonly number[], options?: { dtype?: "f32" | "f64" | "i64"; device?: "cpu" | "metal" }): Tensor
    zeros(shape: readonly number[], options?: { dtype?: "f32" | "f64" | "i64"; device?: "cpu" | "metal" }): Tensor
    ones(shape: readonly number[], options?: { dtype?: "f32" | "f64" | "i64"; device?: "cpu" | "metal" }): Tensor
    full(shape: readonly number[], fill: number, options?: { dtype?: "f32" | "f64" | "i64"; device?: "cpu" | "metal" }): Tensor
    parameter(shape: readonly number[], options?: { dtype?: "f32" | "f64"; device?: "cpu" | "metal" }): Tensor
    setDevice(device: "cpu" | "metal"): void
    copy<T extends Tensor>(target: T, source: Tensor): T
    $muladd_(target: Tensor, scale: number, addend: Tensor): void
    grad(loss: Tensor, parameters: readonly Tensor[]): void
    clip_grad_norm(parameters: readonly Tensor[], max_norm: number, eps?: number): number
    clear_grad(parameters: readonly Tensor[]): void
    no_grad<T>(fn: () => T): T
    rand(shape: readonly number[], options?: { dtype?: "f32" | "f64"; axes?: readonly string[] }): Tensor
    randn(shape: readonly number[], options?: { dtype?: "f32" | "f64"; axes?: readonly string[] }): Tensor
    seed(value: number): void
    arange(start: number, end?: number, step?: number): Tensor
    linspace(start: number, end: number, steps?: number, options?: { dtype?: "f32" | "f64" | "i64"; axes?: readonly string[] }): Tensor
    add(lhs: Tensor, rhs: Tensor): Tensor
    sub(lhs: Tensor, rhs: Tensor): Tensor
    mul(lhs: Tensor, rhs: Tensor): Tensor
    div(lhs: Tensor, rhs: Tensor): Tensor
    matmul(lhs: Tensor, rhs: Tensor): Tensor
    dot(lhs: Tensor, rhs: Tensor): Tensor
    square(value: Tensor): Tensor
    gt_scalar(value: Tensor, threshold: number): Tensor
    cast(value: Tensor, dtype: "f32" | "f64" | "i64"): Tensor
    abs(value: Tensor): Tensor
    exp(value: Tensor): Tensor
    log(value: Tensor): Tensor
    neg(value: Tensor): Tensor
    sqrt(value: Tensor): Tensor
    sign(value: Tensor): Tensor
    relu(value: Tensor): Tensor
    sigmoid(value: Tensor): Tensor
    silu(value: Tensor): Tensor
    tanh(value: Tensor): Tensor
    gelu(value: Tensor): Tensor
    clamp(value: Tensor, min: number, max: number): Tensor
    softmax(value: Tensor, axis: number): Tensor
    sum(value: Tensor): Tensor
    mean(value: Tensor, axis?: number): Tensor
    min(value: Tensor, axis?: number): Tensor
    max(value: Tensor, axis?: number): Tensor
    variance(value: Tensor, axis?: number): Tensor
    std(value: Tensor, axis?: number): Tensor
    argmin(value: Tensor, axis?: number): Tensor
    argmax(value: Tensor, axis?: number): Tensor
    reshape(value: Tensor, shape: readonly number[]): Tensor
    slice(value: Tensor, selectors: readonly (number | string)[]): Tensor
    contiguous(value: Tensor): Tensor
    permute(value: Tensor, axes: readonly number[]): Tensor
    transpose(value: Tensor, axisA: number, axisB: number): Tensor
    squeeze(value: Tensor, axis?: number): Tensor
    unsqueeze(value: Tensor, axis: number): Tensor
    cat(values: readonly Tensor[], axis?: number): Tensor
    stack(values: readonly Tensor[], axis?: number): Tensor
    one_hot(indices: Tensor, classes: number): Tensor
    gather(value: Tensor, axis: number, index: Tensor): Tensor
    index_select(value: Tensor, axis: number, index: Tensor): Tensor
    topk(value: Tensor, k: number, axis?: number): { values: Tensor; indices: Tensor }
    setDevice(device: 'cpu' | 'metal'): void
    $axpy_(target: Tensor, scale: number, addend: Tensor): void
    $zero_grad_(parameter: Tensor): void
    $backward_(loss: Tensor): void
    where(condition: Tensor, onTrue: Tensor, onFalse: Tensor): Tensor
    masked_fill(value: Tensor, mask: Tensor, fill: number): Tensor
    cross_entropy_indexed(logits: Tensor, targets: Tensor, axis?: number): Tensor
  }

  export default native
}
