import { copy, parameter, tensor } from 'affon:compute'

export function internal_tensor(
  data: number | number[] | number[][] | number[][][],
  opts?: { dtype?: 'f32' | 'f64' | 'i64'; device?: 'cpu' | 'metal'; axes?: readonly string[] },
): any {
  const source = tensor(data as any, opts)
  const tracked = parameter((source.shape as number[]).slice(), {
    dtype: source.dtype as 'f32' | 'f64' | 'i64',
    device: opts?.device,
    axes: opts?.axes,
  }).to(source.device) as any
  delete tracked.$compute
  copy(tracked, source)
  return tracked
}
