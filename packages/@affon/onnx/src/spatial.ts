import { Session } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'

export type Conv = {
  kernel: number[]
  strides?: number[]
  dilations?: number[]
  pads?: number[]
  group?: number
}

export type HostTensor = { shape: number[]; values: number[] }

function product(shape: readonly number[]) { return shape.reduce((size, value) => size * value, 1) }

export function hostTensor(value: Tensor): HostTensor {
  return { shape: [...value.shape], values: (value.to_array() as any[]).flat(Infinity).map(Number) }
}

export function nested(values: readonly number[], shape: readonly number[]): any {
  let offset = 0
  const build = (axis: number): any => axis === shape.length
    ? values[offset++]
    : Array.from({ length: shape[axis] }, () => build(axis + 1))
  return build(0)
}

export function padHost(input: HostTensor, pads: readonly number[]): HostTensor {
  if (input.shape.length !== 4 || pads.length !== 4) throw new Error('Pad requires NCHW input and four spatial pads')
  const [batch, channels, height, width] = input.shape
  const oh = height + pads[0] + pads[2]
  const ow = width + pads[1] + pads[3]
  const values = Array(batch * channels * oh * ow).fill(0)
  for (let n = 0; n < batch; n++) for (let c = 0; c < channels; c++)
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const source = ((n * channels + c) * height + y) * width + x
      const target = ((n * channels + c) * oh + y + pads[0]) * ow + x + pads[1]
      values[target] = input.values[source]
    }
  return { shape: [batch, channels, oh, ow], values }
}

export function convHost(input: HostTensor, weight: HostTensor, bias: HostTensor | undefined, options: Conv, clip?: readonly number[]): HostTensor {
  if (input.shape.length !== 4 || weight.shape.length !== 4) throw new Error('Conv requires NCHW input and OIHW weights')
  const [batch, channels, height, width] = input.shape
  const [outputs, channelsPerGroup, kh, kw] = weight.shape
  const strides = options.strides ?? options.kernel
  const dilations = options.dilations ?? [1, 1]
  const pads = options.pads ?? [0, 0, 0, 0]
  const groups = options.group ?? 1
  const oh = Math.floor((height + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0]) + 1
  const ow = Math.floor((width + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1]) + 1
  if (groups < 1 || channels !== channelsPerGroup * groups || outputs % groups || oh < 1 || ow < 1) throw new Error('Invalid convolution dimensions')
  const outputsPerGroup = outputs / groups
  const values = Array(batch * outputs * oh * ow).fill(0)
  for (let n = 0; n < batch; n++) for (let oc = 0; oc < outputs; oc++) {
    const group = Math.floor(oc / outputsPerGroup)
    for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
      let value = bias?.values[oc] ?? 0
      for (let icg = 0; icg < channelsPerGroup; icg++) for (let ky = 0; ky < kh; ky++) for (let kx = 0; kx < kw; kx++) {
        const iy = oy * strides[0] + ky * dilations[0] - pads[0]
        const ix = ox * strides[1] + kx * dilations[1] - pads[1]
        if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue
        const ic = group * channelsPerGroup + icg
        const inputIndex = ((n * channels + ic) * height + iy) * width + ix
        const weightIndex = ((oc * channelsPerGroup + icg) * kh + ky) * kw + kx
        value += input.values[inputIndex] * weight.values[weightIndex]
      }
      if (clip) value = Math.min(Math.max(value, clip[0]), clip[1])
      values[((n * outputs + oc) * oh + oy) * ow + ox] = value
    }
  }
  return { shape: [batch, outputs, oh, ow], values }
}

function materializer(device: Device) {
  const session = new Session({ device })
  return (value: HostTensor) => session.tensor(nested(value.values, value.shape)) as Tensor
}

export function prepare_pad(shape: number[], pads: number[], device: Device) {
  const output = materializer(device)
  return (input: Tensor) => {
    if (JSON.stringify(input.shape) !== JSON.stringify(shape)) throw new Error('Pad input shape mismatch')
    return output(padHost(hostTensor(input), pads))
  }
}

export function prepare_conv(shape: number[], weight: Tensor, bias: Tensor | undefined, options: Conv, device: Device, clip?: number[]) {
  const output = materializer(device)
  const hostWeight = hostTensor(weight)
  const hostBias = bias ? hostTensor(bias) : undefined
  return (input: Tensor) => {
    if (JSON.stringify(input.shape) !== JSON.stringify(shape)) throw new Error('Conv input shape mismatch')
    return output(convHost(hostTensor(input), hostWeight, hostBias, options, clip))
  }
}

export function reshapeHost(input: HostTensor, shape: number[]): HostTensor {
  if (product(input.shape) !== product(shape)) throw new Error('Reshape element count mismatch')
  return { shape: [...shape], values: [...input.values] }
}
