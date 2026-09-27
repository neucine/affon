import {
  abs, add, contiguous, div, exp, masked_fill, matmul, mean, mul, neg,
  permute, reshape, sign, softmax, sqrt, sub, tensor, transpose,
} from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'


export interface AffineWeights { weight: Tensor; bias: Tensor }

export function encoder_ops(device: Device) {
  const scalar = (value: number) => tensor(value, { dtype: 'f32', device })
  const dense = (x: Tensor, params: AffineWeights) => add(matmul(x, params.weight), params.bias)
  function norm(x: Tensor, params: AffineWeights, eps: number) {
    const centered = sub(x, mean(x, x.ndim - 1, true))
    const variance = mean(mul(centered, centered), x.ndim - 1, true)
    return add(mul(div(centered, sqrt(add(variance, scalar(eps)))), params.weight), params.bias)
  }
  function attention(q: Tensor, k: Tensor, v: Tensor, heads: number, mask?: Tensor) {
    const [batch, length, dim] = q.shape
    const width = dim / heads
    const split = (x: Tensor) => permute(reshape(contiguous(x), [batch, length, heads, width]), [0, 2, 1, 3])
    let scores = div(matmul(split(q), permute(split(k), [0, 1, 3, 2])), scalar(Math.sqrt(width)))
    if (mask) scores = masked_fill(scores, mask, -3.4028234663852886e38)
    const context = matmul(softmax(scores, 3), split(v))
    return reshape(contiguous(permute(context, [0, 2, 1, 3])), [batch, length, dim])
  }
  return { scalar, dense, norm, attention }
}

// Affon's native gelu currently uses the tanh approximation. BERT/ViT request
// erf GELU. This Abramowitz-Stegun 7.1.26 approximation is checked
// against torch.nn.functional.gelu independently; it is not a native erf kernel.
export function erf_gelu(x: Tensor): Tensor {
  const scalar = (value: number) => tensor(value, { dtype: 'f32', device: x.device })
  const z = div(x, scalar(Math.sqrt(2)))
  const t = div(scalar(1), add(scalar(1), mul(scalar(0.3275911), abs(z))))
  let polynomial = add(mul(scalar(1.061405429), t), scalar(-1.453152027))
  polynomial = add(mul(polynomial, t), scalar(1.421413741))
  polynomial = add(mul(polynomial, t), scalar(-0.284496736))
  polynomial = add(mul(polynomial, t), scalar(0.254829592))
  const erf = mul(sign(z), sub(scalar(1), mul(mul(polynomial, t), exp(neg(mul(z, z))))))
  return mul(mul(scalar(0.5), x), add(scalar(1), erf))
}
