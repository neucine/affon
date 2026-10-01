declare module "affon:compute/native" {
  type Device = "cpu" | "metal" | "cuda"
  interface Tensor {
    readonly shape: readonly number[]
    readonly ndim: number
    readonly dtype: "f32" | "f64" | "i64"
    readonly device: Device
    item(): number
    to_array(): unknown
    toString(): string
    repr(): string
  }
  const native: {
    tensor(values: number | readonly unknown[]): Tensor
    add(lhs: Tensor, rhs: Tensor): Tensor
    sub(lhs: Tensor, rhs: Tensor): Tensor
    mul(lhs: Tensor, rhs: Tensor): Tensor
    div(lhs: Tensor, rhs: Tensor): Tensor
    matmul(lhs: Tensor, rhs: Tensor): Tensor
    dot(lhs: Tensor, rhs: Tensor): Tensor
  }
  export default native
}
