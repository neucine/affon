import { describe, expect, test } from 'std:test'
import { range, copy, tensor, parameter, module, add, sub, square, mean, sum, matmul, relu, gelu, gt_scalar, where, masked_fill, cast, dot, cat, stack, slice, transpose, contiguous, setDevice, clip_grad_norm, grad, clear_grad, sgd, adam, adamw, compile, cross_entropy_indexed, gather, one_hot } from 'affon:compute/legacy'

function close(actual: any, expected: any, tolerance = 1e-4) {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true)
    expect(actual.length).toBe(expected.length)
    for (let i = 0; i < expected.length; i++) close(actual[i], expected[i], tolerance)
  } else expect(Math.abs(actual - expected) <= tolerance).toBe(true)
}

describe('CUDA training and operation parity', () => {
  test('reads scalars and arrays directly from CUDA', () => {
    setDevice('cuda')
    try { expect(tensor(3).item()).toBe(3); close(tensor([[1, 2], [3, 4]]).to_array(), [[1, 2], [3, 4]]) }
    finally { setDevice('cpu') }
  })
  test('copy overwrites non-finite destinations and supports integer tensors', () => {
    for (const device of ['cpu', 'cuda'] as const) {
      setDevice(device)
      try {
        const target = tensor([Number.NaN, Number.POSITIVE_INFINITY])
        copy(target, tensor([1, 2]))
        close(target.to_array(), [1, 2])
        const initialized = parameter([2, 4])
        for (const initializer of ['zeros', 'ones', 'rand', 'randn', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) {
          copy(initialized, tensor([Array(4).fill(Number.NaN), Array(4).fill(Number.NaN)]))
          ;(initialized as any)[initializer]()
          expect(initialized.to_array().flat().every(Number.isFinite)).toBe(true)
        }
        const indices = tensor([0, 0], { dtype: 'i64' })
        copy(indices, tensor([2, 1], { dtype: 'i64' }))
        close(indices.to_array(), [2, 1])
      } finally { setDevice('cpu') }
    }
  })
  test('comparison, where, broadcast mask, casts and shape operations', () => {
    setDevice('cuda')
    try {
      const a = tensor([[1, -2, 3], [4, 5, -6]])
      close(gt_scalar(a, 0).to_array(), [[1, 0, 1], [1, 1, 0]])
      close(where(gt_scalar(a, 0), a, tensor(0)).to_array(), [[1, 0, 3], [4, 5, 0]])
      close(masked_fill(a, tensor([0, 1, 0]), -9).to_array(), [[1, -9, 3], [4, -9, -6]])
      close(cast(tensor([1.9, -2.8]), 'i64').to_array(), [1, -2])
      close(dot(tensor([1, 2, 3]), tensor([4, 5, 6])).item(), 32)
      close(cat([a, a], 1).to_array(), [[1, -2, 3, 1, -2, 3], [4, 5, -6, 4, 5, -6]])
      close(stack([tensor([1, 2]), tensor([3, 4])], 1).to_array(), [[1, 3], [2, 4]])
      close(slice(a, range(0, 2), range(1, 3)).to_array(), [[-2, 3], [5, -6]])
      close(one_hot(tensor([2, 0]), 3).to_array(), [[0, 0, 1], [1, 0, 0]])
    } finally { setDevice('cpu') }
  })
  for (const kind of ['sgd', 'adam', 'adamw']) {
    test(kind + ' matches CPU parameters and gradients across multiple steps', () => {
      function train(device: 'cpu' | 'cuda') {
        setDevice(device)
        const model = module({ w: parameter([2, 1]).ones(), b: parameter([1]).zeros() }, ({w, b}, x) => add(matmul(x, w), b))
        const step = kind === 'sgd' ? sgd({ lr: 0.01 }) : kind === 'adam' ? adam({ lr: 0.01 }) : adamw({ lr: 0.01, weight_decay: 0.1 })
        const params = model.parameters
        const x = tensor([[1, 2], [3, 4]]), y = tensor([[1], [2]])
        let loss
        for (let i = 0; i < 4; i++) {
          clear_grad(params)
          loss = mean(square(sub(model(x), y)))
          grad(loss, params)
          step(params)
        }
        return [loss.item(), params.map(p => p.to_array()), params.map(p => p.grad.to_array())]
      }
      try { const cpu = train('cpu'); const cuda = train('cuda'); close(cuda, cpu, 2e-4) }
      finally { setDevice('cpu') }
    })
  }
  test('indexed loss backward and gradient clipping match CPU', () => {
    function evaluate(device: 'cpu' | 'cuda') {
      setDevice(device)
      const w = parameter([2, 3]).ones(), params = [w]
      const x = tensor([[1, 2], [3, 1]]), targets = tensor([1, 2], { dtype: 'i64' })
      const loss = cross_entropy_indexed(matmul(x, w), targets)
      grad(loss, params)
      const norm = clip_grad_norm(params, 0.5)
      return [loss.item(), norm, w.grad.to_array()]
    }
    try { close(evaluate('cuda'), evaluate('cpu'), 1e-4) }
    finally { setDevice('cpu') }
  })
  test('compiled matmul bias GELU and unary chains match eager CUDA', () => {
    setDevice('cuda')
    try {
      const model = module({ w: parameter([2, 2]).ones(), b: parameter([2]).ones() }, ({ w, b }, x) => relu(gelu(add(matmul(x, w), b))))
      const x = tensor([[1, 2], [-3, 1]])
      const compiled = compile(model)
      close(compiled(x).to_array(), model(x).to_array())
      expect((compiled as any).summary().graphRuntime).toBe("native-graph")
      close(compiled(x).to_array(), model(x).to_array())
    } finally { setDevice('cpu') }
  })
})
