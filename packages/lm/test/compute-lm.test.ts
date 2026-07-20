import { describe, expect, test, values } from 'affon:test'
import { axes, compile, grad, sum, tensor } from 'affon:compute'
import telemetry from 'std:telemetry'
import type { Tensor } from 'affon:compute'

import { CausalLMLoss, DecoderModel, causal_lm_eval_loss_forward, generate } from '../src/index.ts'
import { internal_tensor } from './support/compute.ts'
import { metalAvailable } from './support/metal.ts'

function trackedF32(data: number[][][]): any {
  return internal_tensor(data, { dtype: 'f32', axes: [axes.batch, axes.token, axes.vocab] })
}

function metricValue(group: string, name: string): number {
  return telemetry.metrics()
    .filter((metric: any) => metric.scope === group && metric.name === name)
    .reduce((sum: number, metric: any) => sum + metric.value, 0)
}

describe('@affon/lm compute', () => {
  test('decoder model and loss work with compute-native values', () => {
    const model = DecoderModel(32, 6, {
      numLayers: 1,
      numHeads: 2,
      hiddenDim: 12,
      causal: true,
      positional: 'learned',
      maxSeqLen: 16,
    })

    const tokenIds = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] })
    expect(model.embedding.token_embedding.weight.axes).toEqual(['vocab', 'feature'])
    expect(model.embedding.position_embedding?.weight.axes).toEqual(['token', 'feature'])
    expect(model.final_norm.gamma.axes).toEqual(['feature'])
    const logits = model(tokenIds as Tensor<[number, number], 'f32'>)
    expect(logits.shape).toEqual([2, 3, 32])

    const lossTokens = tensor([
      [1, 2, 3, 4],
      [4, 5, 6, 7],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] })
    const lossInputs = lossTokens.slice([':', `0:${lossTokens.shape[1] as number - 1}`])
    const loss = CausalLMLoss()(model(lossInputs as any), lossTokens)
    expect(loss.shape).toEqual([1])

    const evalLoss = causal_lm_eval_loss_forward(model(lossInputs as any), lossTokens)
    expect(Math.abs(loss.item() - evalLoss.item()) < 1e-6).toBeTruthy()

    const generated = generate(model, tokenIds as Tensor<[number, number], 'f32'>, { max_new_tokens: 2 })
    expect(generated.shape).toEqual([2, 5])

    const banned = generate(model, tensor([
      [1, 2, 3],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] }), { max_new_tokens: 1, forbidden_token_ids: [0, 1] })
    expect(banned.shape).toEqual([1, 4])

    const sampled = generate(model, tensor([
      [1, 2, 3],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] }), { max_new_tokens: 2, temperature: 0.8, top_k: 5 })
    expect(sampled.shape).toEqual([1, 5])
  })

  test('causal lm loss keeps compute-native tracked logits finite', () => {
    const logits = trackedF32([
      [
        [1000, -1000, -1000],
        [-1000, 1000, -1000],
      ],
      [
        [-1000, -1000, 1000],
        [1000, -1000, -1000],
      ],
    ])
    const tokenIds = tensor([
      [0, 0, 1],
      [2, 2, 0],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] })

    const loss = CausalLMLoss()(logits, tokenIds)
    expect(Number.isFinite(loss.item())).toBe(true)
    grad(sum(loss), [logits as any])
    expect(values(logits.grad)).toBeAllFinite()
  })

  test('tags tied lm-head projection in compiled plans', () => {
    const model = DecoderModel(16, 4, {
      numLayers: 1,
      numHeads: 2,
      hiddenDim: 8,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    const compiled = compile(model as any)
    const tokenIds = tensor([[1, 2, 3]], { dtype: 'f32', axes: [axes.batch, axes.token] })

    compiled(tokenIds as any)
    const plan = (compiled as any).plan(tokenIds as any)
    const tiedProjection = plan.steps.find((step: any) =>
      step.matmul?.family === 'gemm_projection' &&
      step.matmul?.hint === 'projection' &&
      step.matmul?.hint_source === 'higher_level_module'
    )

    expect(tiedProjection).toBeTruthy()
  })

  test('compiles tied decoder lm loss through native graph planning', () => {
    const model = DecoderModel(16, 4, {
      numLayers: 1,
      numHeads: 2,
      hiddenDim: 8,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    const lossFn = CausalLMLoss()
    const compiled = compile((tokenIds: any) => {
      const inputSeqLen = (tokenIds.shape[1] as number) - 1
      const inputs = tokenIds.slice([':', `0:${inputSeqLen}`])
      return lossFn(model(inputs as any) as any, tokenIds as any)
    })
    const tokenIds = tensor([[1, 2, 3, 4]], { dtype: 'f32', axes: [axes.batch, axes.token] })
    const inputs = tokenIds.slice([':', '0:3'])

    const eager = lossFn(model(inputs as any) as any, tokenIds as any)
    const fusionFallbackBefore = metricValue('compute.execution', 'fusion_fallback_count')
    const callable = compiled(tokenIds as any)
    const lowered = compiled.run(tokenIds as any)
    const fusionFallbackAfter = metricValue('compute.execution', 'fusion_fallback_count')
    const summary = (compiled as any).summary(tokenIds as any)
    const plan = (compiled as any).plan(tokenIds as any)
    const tiedProjection = plan.steps.find((step: any) =>
      step.matmul?.family === 'gemm_projection' &&
      step.matmul?.hint === 'projection' &&
      step.matmul?.hint_source === 'higher_level_module'
    )

    expect(Math.abs(callable.item() - eager.item()) < 1e-6).toBe(true)
    expect(Math.abs(lowered.item() - eager.item()) < 1e-6).toBe(true)
    expect(summary.graphRuntime).toBe('native-graph')
    expect(summary.graphLoweringAnalysis).toEqual({ lowerable: true })
    expect(summary.eagerFallbackCount).toBe(0)
    expect(fusionFallbackAfter).toBe(fusionFallbackBefore)
    expect(plan.steps.length > 0).toBe(true)
    expect(tiedProjection).toBeTruthy()
  })

  test.skip(() => !metalAvailable())('keeps decoder model backward finite on metal for large finite activations', () => {
    setDevice('cpu')
    setDevice('metal')

    const model = DecoderModel(32, 8, {
      numLayers: 1,
      numHeads: 2,
      hiddenDim: 16,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    const tokenIds = tensor([[1, 2, 3]], { dtype: 'f32', axes: [axes.batch, axes.token] }).to('metal')

    const logits = model(tokenIds)
    grad(sum(logits), model.parameters)

    for (const param of model.parameters) {
      if (param.grad !== null) expect(values(param.grad.to('cpu'))).toBeAllFinite()
    }
    setDevice('cpu')
  })

  test.skip(() => !metalAvailable())('keeps decoder causal lm loss backward finite on metal', () => {
    setDevice('cpu')
    setDevice('metal')

    const model = DecoderModel(64, 16, {
      numLayers: 2,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 16,
      tieEmbeddings: true,
    })
    const tokenIds = tensor([
      [1, 2, 3, 4, 5, 6],
      [6, 5, 4, 3, 2, 1],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] }).to('metal')
    const inputs = tokenIds.slice([':', `0:${tokenIds.shape[1] as number - 1}`])
    const loss = CausalLMLoss()(model(inputs as any), tokenIds)

    expect(Number.isFinite(loss.item())).toBe(true)
    grad(loss, model.parameters)

    for (const param of model.parameters) {
      if (param.grad !== null) expect(values(param.grad.to('cpu'))).toBeAllFinite()
    }
    setDevice('cpu')
  })
})
