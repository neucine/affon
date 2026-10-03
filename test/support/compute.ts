import { copy, parameter, tensor } from 'affon:compute/legacy'

export function internal_tensor(
  data: number | number[] | number[][] | number[][][],
  opts?: { dtype?: 'f32' | 'f64' | 'i64'; device?: 'cpu' | 'metal' },
): any {
  const source = tensor(data as any, opts)
  const tracked = parameter((source.shape as number[]).slice(), {
    dtype: source.dtype as 'f32' | 'f64' | 'i64',
    device: opts?.device,
  }).to(source.device) as any
  delete tracked.$compute
  copy(tracked, source)
  return tracked
}

export function computeParityDTypes(device: 'cpu' | 'metal'): ReadonlyArray<{ dtype: 'f32' | 'f64' }> {
  return device === 'metal'
    ? [{ dtype: 'f32' }]
    : [{ dtype: 'f32' }, { dtype: 'f64' }]
}

export function torchDType(dtype: 'f32' | 'f64' | 'i64'): string {
  switch (dtype) {
    case 'f64':
      return 'torch.float64'
    case 'i64':
      return 'torch.int64'
    default:
      return 'torch.float32'
  }
}
