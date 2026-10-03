import { describe, expect, test } from 'std:test'
import { compile, matmul, tensor } from 'affon:compute/legacy'
import { captureError } from '../../support/errors.ts'

describe('matmul execution hints', () => {
  test('accepts expert execution hints without changing math semantics', () => {
    const a = tensor([[1, 2], [3, 4]])
    const b = tensor([[5, 6], [7, 8]])

    expect(matmul(a, b, { hint: 'projection' }).to_array()).toEqual([[19, 22], [43, 50]])
    expect(matmul(a, b, { hint: 'attention_scores' }).to_array()).toEqual([[19, 22], [43, 50]])
    expect(matmul(a, b, { hint: 'attention_values' }).to_array()).toEqual([[19, 22], [43, 50]])
  })

  test('rejects invalid execution hints as invalid_arg', () => {
    const a = tensor([[1, 2], [3, 4]])
    const b = tensor([[5, 6], [7, 8]])
    const err = captureError(() => (matmul as any)(a, b, { hint: 'not_real' }))
  })

  test('preserves explicit hint provenance in compiled graph plans', () => {
    const program = compile((a, b) => matmul(a, b, { hint: 'projection' }))
    const a = tensor([[[1, 2], [3, 4]]], { dtype: 'f32' })
    const b = tensor([[5, 6], [7, 8]], { dtype: 'f32' })

    program(a, b)
    const plan = (program as any).plan(a, b)
    const matmulStep = plan?.steps.find((step: any) => step.matmul)

    expect(matmulStep?.matmul?.family).toBe('gemm_projection')
    expect(matmulStep?.matmul?.hint).toBe('projection')
    expect(matmulStep?.matmul?.hint_source).toBe('api_execution_arg')
  })

  test('preserves higher-level module hint provenance in compiled graph plans', () => {
    const program = compile((a, b) => matmul(a, b, { hint: 'attention_scores', source: 'higher_level_module' }))
    const a = tensor([[[[1, 2], [3, 4]]]], { dtype: 'f32' })
    const b = tensor([[[[5, 6], [7, 8]]]], { dtype: 'f32' })

    program(a, b)
    const plan = (program as any).plan(a, b)
    const matmulStep = plan?.steps.find((step: any) => step.matmul)

    expect(matmulStep?.matmul?.family).toBe('gemm_attention_scores')
    expect(matmulStep?.matmul?.hint).toBe('attention_scores')
    expect(matmulStep?.matmul?.hint_source).toBe('higher_level_module')
  })

  test('rejects invalid execution hint sources as invalid_arg', () => {
    const a = tensor([[1, 2], [3, 4]])
    const b = tensor([[5, 6], [7, 8]])
    const err = captureError(() => (matmul as any)(a, b, { hint: 'projection', source: 'not_real' }))
  })
})
