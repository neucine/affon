import { describe, expect, test, values } from 'std:test'
import {
  abs,
  add,
  clamp,
  div,
  exp,
  grad,
  log,
  matmul,
  mean,
  mul,
  neg,
  relu,
  silu,
  sigmoid,
  softmax,
  sqrt,
  sub,
  sum,
  tanh,
  tensor,
} from 'affon:compute'
import { internal_tensor } from '../../support/compute.ts'

describe('compute math ops', () => {
  test('supports common arithmetic and activations', () => {
    expect(values(add(tensor([[1, 2], [3, 4]]), tensor([[10, 20], [30, 40]])))).toEqual([[11, 22], [33, 44]])
    expect(values(sub(tensor([[5, 6], [7, 8]]), tensor([[1, 2], [3, 4]])))).toEqual([[4, 4], [4, 4]])
    expect(values(mul(tensor([[1, 2], [3, 4]]), tensor([[2, 3], [4, 5]])))).toEqual([[2, 6], [12, 20]])
    expect(values(div(tensor([[2, 6], [12, 20]]), tensor([[2, 3], [4, 5]])))).toEqual([[1, 2], [3, 4]])
    expect(values(neg(tensor([1, -2, 3])))).toEqual([-1, 2, -3])
    expect(values(abs(tensor([-1, -2, 3])))).toEqual([1, 2, 3])
    expect(values(clamp(tensor([-1, 0.5, 2]), 0, 1))).toEqual([0, 0.5, 1])
    expect(values(relu(tensor([-1, 0, 2])))).toEqual([0, 0, 2])

    const sig = values(sigmoid(tensor([0]))) as number[]
    const sil = values(silu(tensor([1]))) as number[]
    const th = values(tanh(tensor([0]))) as number[]
    expect(Math.abs(sig[0] - 0.5) < 1e-6).toBe(true)
    expect(Math.abs(sil[0] - (1 / (1 + Math.exp(-1)))) < 1e-6).toBe(true)
    expect(Math.abs(th[0]) < 1e-6).toBe(true)
  })

  test('supports linalg, reductions, and autograd on compute-native tracked values', () => {
    const x = internal_tensor([[1, 2], [3, 4]])
    const w = internal_tensor([[2, 0], [1, 3]])
    const y = matmul(x, w)
    expect(values(y)).toEqual([[4, 6], [10, 12]])

    const probs = softmax(tensor([[1, 2, 3], [0, -1, 1]]), 1)
    expect(values(probs)).toBeAllFinite()
    expect(values(sum(probs, 1))).toBeAllClose([1, 1])

    const stats = sqrt(exp(log(tensor([[1, 4], [9, 16]]))))
    expect(values(stats)).toEqual([[1, 2], [3, 4]])
    expect(mean(tensor([[1, 2], [3, 4]])).item()).toBe(2.5)

    grad(sum(y), [x as any, w as any])
    expect(values(x.grad)).toBeAllFinite()
    expect(values(w.grad)).toBeAllFinite()
  })

  test('supports autograd through silu', () => {
    const x = internal_tensor([-1, 0, 1])
    const loss = sum(silu(x))
    grad(loss, [x as any])
    expect(values(x.grad)).toBeAllFinite()
  })
})
