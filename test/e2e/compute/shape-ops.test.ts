import { describe, test, expect, values } from 'std:test'
import { all, cast, cat, contiguous, grad, permute, randn, reshape, seed, slice, squeeze, stack, sum, tensor, transpose, unsqueeze } from 'affon:compute/legacy'
import { internal_tensor } from '../../support/compute.ts'

describe('compute shape ops', () => {
  test('supports cat stack squeeze unsqueeze reshape transpose permute and transfers', () => {
    const a = tensor([[1, 2], [3, 4]])
    const b = tensor([[5, 6]])

    expect(values(cat([a, b], 0))).toEqual([[1, 2], [3, 4], [5, 6]])
    expect(values(stack([a, a], 0))).toEqual([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    expect(values(squeeze(tensor([[[1], [2]]])))).toEqual([1, 2])
    expect(values(unsqueeze(tensor([1, 2, 3]), 0))).toEqual([[1, 2, 3]])
    expect(values(reshape(tensor([[1, 2], [3, 4]]), [4]))).toEqual([1, 2, 3, 4])
    expect(values(transpose(tensor([[1, 2], [3, 4]]), 0, 1))).toEqual([[1, 3], [2, 4]])
    expect(values(permute(tensor([[[1, 2], [3, 4], [5, 6]]]), [0, 2, 1]))).toEqual([[[1, 3, 5], [2, 4, 6]]])
    expect(values(contiguous(tensor([[1, 2], [3, 4]])))).toEqual([[1, 2], [3, 4]])
    expect(tensor([[1, 2], [3, 4]]).device).toBe('cpu')
    expect(values(tensor([[1, 2], [3, 4]]).to('cpu'))).toEqual([[1, 2], [3, 4]])
    expect(tensor([[1, 2], [3, 4]]).to('metal').device).toBe('metal')
    expect(values(tensor([[1, 2], [3, 4]]).to('metal').to('cpu'))).toEqual([[1, 2], [3, 4]])
  })

  test('supports no-op to() and cast() through retained wrappers', () => {
    const cpu = tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    const sameDevice = cpu.to('cpu')
    const sameDType = cast(cpu, 'f32')

    expect(sameDevice).not.toBe(cpu)
    expect(sameDevice.device).toBe('cpu')
    expect(sameDevice.dtype).toBe('f32')
    expect(values(sameDevice)).toEqual([[1, 2], [3, 4]])

    expect(sameDType).not.toBe(cpu)
    expect(sameDType.device).toBe('cpu')
    expect(sameDType.dtype).toBe('f32')
    expect(values(sameDType)).toEqual([[1, 2], [3, 4]])

    const metal = cpu.to('metal')
    const sameMetal = metal.to('metal')
    const sameMetalDType = cast(metal, 'f32')

    expect(sameMetal).not.toBe(metal)
    expect(sameMetal.device).toBe('metal')
    expect(sameMetal.dtype).toBe('f32')
    expect(values(sameMetal.to('cpu'))).toEqual([[1, 2], [3, 4]])

    expect(sameMetalDType).not.toBe(metal)
    expect(sameMetalDType.device).toBe('metal')
    expect(sameMetalDType.dtype).toBe('f32')
    expect(values(sameMetalDType.to('cpu'))).toEqual([[1, 2], [3, 4]])
  })

  test('supports no-op contiguous() through retained wrappers for contiguous tensors', () => {
    const cpu = tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    const sameCpu = contiguous(cpu)

    expect(sameCpu).not.toBe(cpu)
    expect(sameCpu.device).toBe('cpu')
    expect(sameCpu.dtype).toBe('f32')
    expect(values(sameCpu)).toEqual([[1, 2], [3, 4]])

    const metal = cpu.to('metal')
    const sameMetal = contiguous(metal)

    expect(sameMetal).not.toBe(metal)
    expect(sameMetal.device).toBe('metal')
    expect(sameMetal.dtype).toBe('f32')
    expect(values(sameMetal.to('cpu'))).toEqual([[1, 2], [3, 4]])
  })

  test('supports no-op reshape() through retained wrappers for same-shape tensors', () => {
    const cpu = tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    const sameCpu = reshape(cpu, [2, 2])

    expect(sameCpu).not.toBe(cpu)
    expect(sameCpu.device).toBe('cpu')
    expect(sameCpu.dtype).toBe('f32')
    expect(values(sameCpu)).toEqual([[1, 2], [3, 4]])

    const metal = cpu.to('metal')
    const sameMetal = reshape(metal, [2, 2])

    expect(sameMetal).not.toBe(metal)
    expect(sameMetal.device).toBe('metal')
    expect(sameMetal.dtype).toBe('f32')
    expect(values(sameMetal.to('cpu'))).toEqual([[1, 2], [3, 4]])
  })

  test('supports no-op permute() through retained wrappers for identity axes', () => {
    const cpu = tensor([[[1, 2], [3, 4]]], { dtype: 'f32' })
    const sameCpu = permute(cpu, [0, 1, 2])

    expect(sameCpu).not.toBe(cpu)
    expect(sameCpu.device).toBe('cpu')
    expect(sameCpu.dtype).toBe('f32')
    expect(values(sameCpu)).toEqual([[[1, 2], [3, 4]]])

    const metal = cpu.to('metal')
    const sameMetal = permute(metal, [0, 1, 2])

    expect(sameMetal).not.toBe(metal)
    expect(sameMetal.device).toBe('metal')
    expect(sameMetal.dtype).toBe('f32')
    expect(values(sameMetal.to('cpu'))).toEqual([[[1, 2], [3, 4]]])
  })

  test('supports no-op transpose() through retained wrappers for rank-1 tensors', () => {
    const cpu = tensor([1, 2, 3], { dtype: 'f32' })
    const sameCpu = transpose(cpu, 0, 0)

    expect(sameCpu).not.toBe(cpu)
    expect(sameCpu.device).toBe('cpu')
    expect(sameCpu.dtype).toBe('f32')
    expect(values(sameCpu)).toEqual([1, 2, 3])

    const metal = cpu.to('metal')
    const sameMetal = transpose(metal, 0, 0)

    expect(sameMetal).not.toBe(metal)
    expect(sameMetal.device).toBe('metal')
    expect(sameMetal.dtype).toBe('f32')
    expect(values(sameMetal.to('cpu'))).toEqual([1, 2, 3])
  })

  test('supports no-op slice() through retained wrappers for full-range tensors', () => {
    const cpu = tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    const sameCpu = slice(cpu, all, all)

    expect(sameCpu).not.toBe(cpu)
    expect(sameCpu.device).toBe('cpu')
    expect(sameCpu.dtype).toBe('f32')
    expect(values(sameCpu)).toEqual([[1, 2], [3, 4]])

    const metal = cpu.to('metal')
    const sameMetal = slice(metal, all, all)

    expect(sameMetal).not.toBe(metal)
    expect(sameMetal.device).toBe('metal')
    expect(sameMetal.dtype).toBe('f32')
    expect(values(sameMetal.to('cpu'))).toEqual([[1, 2], [3, 4]])
  })

  test('supports no-op squeeze() through retained wrappers when no singleton dims exist', () => {
    const cpu = tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    const sameCpu = squeeze(cpu)

    expect(sameCpu).not.toBe(cpu)
    expect(sameCpu.device).toBe('cpu')
    expect(sameCpu.dtype).toBe('f32')
    expect(values(sameCpu)).toEqual([[1, 2], [3, 4]])

    const metal = cpu.to('metal')
    const sameMetal = squeeze(metal)

    expect(sameMetal).not.toBe(metal)
    expect(sameMetal.device).toBe('metal')
    expect(sameMetal.dtype).toBe('f32')
    expect(values(sameMetal.to('cpu'))).toEqual([[1, 2], [3, 4]])
  })

  test('supports autograd through reshape transpose and permute', () => {
    const x = internal_tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    grad(sum(reshape(x, [4])), [x as any])
    expect(values(x.grad)).toEqual([[1, 1], [1, 1]])

    const y = internal_tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    grad(sum(transpose(y, 0, 1)), [y as any])
    expect(values(y.grad)).toEqual([[1, 1], [1, 1]])

    const z = internal_tensor([[[1, 2], [3, 4], [5, 6]]], { dtype: 'f32' })
    grad(sum(permute(z, [0, 2, 1])), [z as any])
    expect(values(z.grad)).toEqual([[[1, 1], [1, 1], [1, 1]]])
  })

  test('keeps shape movement finite for large finite inputs', () => {
    const x = tensor([
      [[1e20, -1e20], [1e10, -1e10], [5, -5]],
    ], { dtype: 'f32' })

    expect(values(reshape(x, [2, 3]))).toBeAllFinite()
    expect(values(transpose(tensor([[1e20, -1e20], [1e10, -1e10]], { dtype: 'f32' }), 0, 1))).toBeAllFinite()
    expect(values(permute(x, [0, 2, 1]))).toBeAllFinite()
    expect(values(contiguous(permute(x, [0, 2, 1])))).toBeAllFinite()
  })

  test('supports deterministic random seeding', () => {
    seed(123)
    const a = values(randn([2, 2]))
    seed(123)
    const b = values(randn([2, 2]))
    seed(124)
    const c = values(randn([2, 2]))

    expect(a).toEqual(b)
    expect(c).not.toEqual(a)
  })

  test('supports negative slice indices', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]])
    expect(values(a.slice(['-1:', ':']))).toEqual([[4, 5, 6]])
    expect(values(a.slice([':-1', ':']))).toEqual([[1, 2, 3]])
    expect(values(a.slice(['-1', ':']))).toEqual([[4, 5, 6]])
  })

  test('supports step slices', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expect(values(a.slice(['::2', ':']))).toEqual([[1, 2, 3], [7, 8, 9]])
    expect(values(a.slice(['::-1', ':']))).toEqual([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
  })

  test('supports numeric index entries inside slice arrays', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expect(values(a.slice([1, ':']))).toEqual([[4, 5, 6]])
    expect(values(a.slice([2, ':']))).toEqual([[7, 8, 9]])
    expect(values(a.slice([-1, ':']))).toEqual([[7, 8, 9]])
  })

  test('materializes later contiguous first-axis slices correctly', () => {
    const a = tensor([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
      [[9, 10], [11, 12]],
    ])
    expect(values(contiguous(a.slice(['1:2', ':', ':'])))).toEqual([[[5, 6], [7, 8]]])
    expect(values(contiguous(squeeze(a.slice(['2:3', ':', ':']), 0)))).toEqual([[9, 10], [11, 12]])
  })
})
