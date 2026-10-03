import { Session, type Device, type Tensor } from 'affon:compute'

export function internal_tensor(
  data: number | number[] | number[][] | number[][][],
  opts?: { dtype?: 'f32' | 'f64' | 'i64'; device?: Device; axes?: readonly string[] },
): Tensor {
  const session = new Session({ device: opts?.device ?? 'cpu' })
  return session.tensor(data as any, { dtype: opts?.dtype, axes: opts?.axes })
}
