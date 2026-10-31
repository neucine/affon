import { abs, add, cast, cat, contiguous, div, embedding, eq, exp, gt, index_select, log, masked_fill, matmul, mean, mul, neg, reshape, slice, softmax, sqrt, sub, transpose, where } from 'affon:ops'
import { Tensor, program, type FormalTensor } from 'affon:compute'
import type { ModelTensor } from '../shared/parameters.ts'
import { positive_dimensions } from '../shared/parameters.ts'

export interface ApertusConfig {
  width: number; innerWidth: number; heads: number; kvHeads: number; layers: number
  contextLength: number; vocabSize: number; epsilon: number; ropeTheta: number; eosTokenId: number
  ropeScaling: { factor: number; lowFreqFactor: number; highFreqFactor: number; originalContext: number }
}
export interface ApertusBlock {
  attentionNorm: ModelTensor; feedForwardNorm: ModelTensor; queryNorm: ModelTensor; keyNorm: ModelTensor
  query: ModelTensor; key: ModelTensor; value: ModelTensor; attentionOutput: ModelTensor
  up: ModelTensor; down: ModelTensor; alphaP: ModelTensor; alphaN: ModelTensor; beta: ModelTensor; activationEpsilon: ModelTensor
}
export interface ApertusWeights {
  tokenEmbedding: ModelTensor; finalNorm: ModelTensor; lmHead: ModelTensor; blocks: ApertusBlock[]
}

/** Shape-specialized reference path for Apertus full-prefix logits. */
export function create_apertus(config: ApertusConfig, weights: ApertusWeights) {
  const { width, innerWidth, heads, kvHeads, layers, contextLength, vocabSize } = config
  positive_dimensions(width, innerWidth, heads, kvHeads, layers, contextLength, vocabSize)
  const headWidth = width / heads, kvWidth = kvHeads * headWidth
  if (width % heads || heads % kvHeads || headWidth % 2 || weights.blocks.length !== layers) throw Error('Invalid Apertus dimensions')
  const check = (value: ModelTensor, shape: number[]) => {
    if (!value || (value.dtype !== 'bf16' && value.dtype !== 'f32') || JSON.stringify(value.shape) !== JSON.stringify(shape))
      throw Error(`Expected BF16 or f32 Apertus tensor with shape ${JSON.stringify(shape)}`)
  }
  check(weights.tokenEmbedding, [vocabSize, width])
  check(weights.finalNorm, [width])
  check(weights.lmHead, [vocabSize, width])
  for (const block of weights.blocks) {
    for (const name of ['attentionNorm', 'feedForwardNorm'] as const) check(block[name], [width])
    for (const name of ['queryNorm', 'keyNorm'] as const) check(block[name], [headWidth])
    check(block.query, [width, width])
    check(block.key, [kvWidth, width])
    check(block.value, [kvWidth, width])
    check(block.attentionOutput, [width, width])
    check(block.up, [innerWidth, width])
    check(block.down, [width, innerWidth])
    for (const name of ['alphaP', 'alphaN'] as const) {
      try {
        check(block[name], [1])
      } catch {
        throw Error(`Invalid Apertus ${name} tensor: ${JSON.stringify(block[name]?.shape)}`)
      }
    }
    for (const name of ['beta', 'activationEpsilon'] as const) {
      try {
        check(block[name], [])
      } catch {
        throw Error(`Invalid Apertus ${name} tensor: ${JSON.stringify(block[name]?.shape)}`)
      }
    }
  }
  const parameters: Record<string, ModelTensor> = { token_embedding: weights.tokenEmbedding, final_norm: weights.finalNorm, lm_head: weights.lmHead }
  weights.blocks.forEach((block, i) => {
    for (const [name, value] of Object.entries(block)) parameters[`block_${i}_${name}`] = value
  })
  const computeDType = weights.tokenEmbedding.dtype
  const parameter = (p: any, name: string, shape: number[]) => {
    const dtype = parameters[name].dtype
    const raw = p.parameter(name, dtype === 'bf16' ? Tensor.bf16(shape) : Tensor.f32(shape)) as FormalTensor
    return dtype === computeDType ? raw : cast(raw, computeDType)
  }
  const linear = (p: any, name: string, output: number, input: number) => transpose(parameter(p, name, [output, input]), [1, 0])

  function author(mode: 'full' | 'prefill' | 'decode', length: number, outputStart: number, capacity: number) {
    return program('apertus', p => {
      const ids = p.argument('ids', Tensor.i64([length]))
      const position = mode === 'decode' ? p.argument('position', Tensor.i64([1])) : undefined
      const positions = mode === 'decode'
        ? p.constant('cache_positions', Array.from({ length: capacity }, (_, i) => i), Tensor.i64([capacity])) : undefined
      const mask = mode === 'decode' ? reshape(gt(positions!, position!), [1, 1, 1, capacity]) : p.argument('mask', Tensor.i64([1, 1, length, length]))
      const scalar = (name: string, value: number) => p.constant(name, value, computeDType === 'bf16' ? Tensor.bf16([1]) : Tensor.f32([1]))
      const one = scalar('one', 1), zero = scalar('zero', 0)
      const oneF32 = p.constant('one_f32', 1, Tensor.f32([1]))
      const epsilonF32 = p.constant('epsilon_f32', config.epsilon, Tensor.f32([1]))
      const scale = scalar('attention_scale', 1 / Math.sqrt(headWidth))
      const { factor, lowFreqFactor, highFreqFactor, originalContext } = config.ropeScaling
      const angles = Array.from({ length: mode === 'decode' ? capacity : length }, (_, token) => Array.from({ length: headWidth }, (_, j) => {
          const inverse = 1 / Math.pow(config.ropeTheta, (2 * (j % (headWidth / 2))) / headWidth)
          const wavelength = (2 * Math.PI) / inverse
          const low = originalContext / lowFreqFactor, high = originalContext / highFreqFactor
          const scaled = wavelength < high ? inverse : wavelength > low ? inverse / factor
            : (1 - (originalContext / wavelength - lowFreqFactor) / (highFreqFactor - lowFreqFactor)) * inverse / factor
              + ((originalContext / wavelength - lowFreqFactor) / (highFreqFactor - lowFreqFactor)) * inverse
          return token * scaled
        }))
      const angleSpec = (shape: number[]) => (computeDType === 'bf16' ? Tensor.bf16(shape) : Tensor.f32(shape))
      const cosine = mode === 'decode'
        ? reshape(index_select(p.constant('rope_cosine_table', angles.map(row => row.map(Math.cos)), angleSpec([capacity, headWidth])), 0, position!), [1, 1, 1, headWidth])
        : p.constant('rope_cosine', [angles.map(row => [row.map(Math.cos)])], angleSpec([1, length, 1, headWidth]))
      const sine = mode === 'decode'
        ? reshape(index_select(p.constant('rope_sine_table', angles.map(row => row.map(Math.sin)), angleSpec([capacity, headWidth])), 0, position!), [1, 1, 1, headWidth])
        : p.constant('rope_sine', [angles.map(row => [row.map(Math.sin)])], angleSpec([1, length, 1, headWidth]))
      const norm = (x: FormalTensor, w: FormalTensor, axis: number) => {
        const widened = computeDType === 'bf16' ? cast(x, 'f32') : x
        const scaled = mul(widened, div(oneF32, sqrt(add(mean(mul(widened, widened), axis, true), epsilonF32))))
        return mul(computeDType === 'bf16' ? cast(scaled, 'bf16') : scaled, w)
      }
      const rope = (x: FormalTensor, count: number) => {
        const first = contiguous(slice(x, [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start: 0, stop: count }, { start: 0, stop: headWidth / 2 }]))
        const second = contiguous(slice(x, [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start: 0, stop: count }, { start: headWidth / 2, stop: headWidth }]))
        return add(mul(x, cosine), mul(cat([neg(second), first], 3), sine))
      }
      const softplus = (x: FormalTensor) => add(where(gt(x, zero), x, zero), log(add(one, exp(neg(abs(x))))))
      const embed = parameter(p, 'token_embedding', [vocabSize, width])
      let x = reshape(embedding(embed, ids), [1, length, width])
      const hidden: FormalTensor[] = []
      const caches: FormalTensor[] = []
      const padding = mode === 'prefill' && capacity > length ? p.full('cache_padding', 0, angleSpec([1, kvHeads, capacity - length, headWidth])) : undefined
      const insertMask = mode === 'decode' ? reshape(eq(positions!, position!), [1, 1, capacity, 1]) : undefined
      for (let i = 0; i < layers; i++) {
        if (mode === 'full') hidden.push(x)
        const prefix = `block_${i}`
        const w = (name: string, shape: number[]) => parameter(p, `${prefix}_${name}`, shape)
        const matrix = (name: string, output: number, input: number) => linear(p, `${prefix}_${name}`, output, input)
        const normalized = norm(x, w('attentionNorm', [width]), 2)
        const q = reshape(matmul(normalized, matrix('query', width, width)), [1, length, heads, headWidth])
        const k = reshape(matmul(normalized, matrix('key', kvWidth, width)), [1, length, kvHeads, headWidth])
        const currentV = contiguous(transpose(reshape(matmul(normalized, matrix('value', kvWidth, width)), [1, length, kvHeads, headWidth]), [0, 2, 1, 3]))
        const qRot = contiguous(transpose(rope(norm(q, w('queryNorm', [headWidth]), 3), heads), [0, 2, 1, 3]))
        const currentK = contiguous(transpose(rope(norm(k, w('keyNorm', [headWidth]), 3), kvHeads), [0, 2, 1, 3]))
        let attentionK = currentK, attentionV = currentV
        if (mode === 'prefill') {
          caches.push(padding ? cat([currentK, padding], 2) : currentK, padding ? cat([currentV, padding], 2) : currentV)
        } else if (mode === 'decode') {
          const pastK = p.argument(`past_${i}_k`, angleSpec([1, kvHeads, capacity, headWidth]))
          const pastV = p.argument(`past_${i}_v`, angleSpec([1, kvHeads, capacity, headWidth]))
          attentionK = where(insertMask!, currentK, pastK)
          attentionV = where(insertMask!, currentV, pastV)
          caches.push(attentionK, attentionV)
        }
        const groupSize = heads / kvHeads
        const groupedQ = reshape(qRot, [1, kvHeads, groupSize, length, headWidth])
        const groupedK = reshape(attentionK, [1, kvHeads, 1, mode === 'decode' ? capacity : length, headWidth])
        const groupedV = reshape(attentionV, [1, kvHeads, 1, mode === 'decode' ? capacity : length, headWidth])
        const scores = masked_fill(mul(matmul(groupedQ, transpose(groupedK, [0, 1, 2, 4, 3])), scale), mask, computeDType === 'bf16' ? -1e30 : -3.4028234663852886e38)
        const probabilities = computeDType === 'bf16' ? cast(softmax(cast(scores, 'f32'), 4), 'bf16') : softmax(scores, 4)
        const groupedOutput = matmul(probabilities, groupedV)
        const attended = reshape(contiguous(transpose(reshape(groupedOutput, [1, heads, length, headWidth]), [0, 2, 1, 3])), [1, length, width])
        x = add(x, matmul(attended, matrix('attentionOutput', width, width)))
        const mlpInput = norm(x, w('feedForwardNorm', [width]), 2)
        const up = matmul(mlpInput, matrix('up', innerWidth, width))
        const alphaP = softplus(w('alphaP', [1])), beta = w('beta', [])
        const alphaN = add(beta, softplus(w('alphaN', [1])))
        const positive = add(mul(mul(alphaP, up), up), mul(beta, up))
        const activationEpsilon = w('activationEpsilon', [])
        const clipped = where(gt(up, activationEpsilon), activationEpsilon, up)
        const expMinusOne = computeDType === 'bf16' ? cast(sub(exp(cast(clipped, 'f32')), oneF32), 'bf16') : sub(exp(clipped), one)
        const negative = add(mul(sub(expMinusOne, up), alphaN), mul(beta, up))
        x = add(x, matmul(where(gt(up, zero), positive, negative), matrix('down', width, innerWidth)))
      }
      x = norm(x, parameter(p, 'final_norm', [width]), 2)
      if (mode === 'full') hidden.push(x)
      const suffix = (value: FormalTensor) => contiguous(slice(value, [{ start: 0, stop: 1 }, { start: outputStart, stop: length }, { start: 0, stop: width }]))
      return [matmul(suffix(x), linear(p, 'lm_head', vocabSize, width)), ...(mode === 'full' ? hidden.map(suffix) : caches)]
    })
  }
  function forward(length: number, outputStart = 0) {
    positive_dimensions(length)
    if (length > contextLength || !Number.isInteger(outputStart) || outputStart < 0 || outputStart >= length) throw Error('Invalid Apertus sequence')
    return author('full', length, outputStart, length)
  }
  const prefillPrograms = new Map<string, ReturnType<typeof program>>()
  function prefill(length: number, capacity = Math.min(contextLength, Math.max(length, 512))) {
    positive_dimensions(length, capacity)
    if (length > capacity || capacity > contextLength) throw Error('Invalid Apertus cache capacity')
    const key = `${length}:${capacity}`
    const cached = prefillPrograms.get(key)
    if (cached) return cached
    const source = author('prefill', length, length - 1, capacity)
    prefillPrograms.set(key, source)
    return source
  }
  const decodePrograms = new Map<number, ReturnType<typeof program>>()
  function decode(capacity: number) {
    positive_dimensions(capacity)
    if (capacity > contextLength) throw Error('Invalid Apertus cache capacity')
    const cached = decodePrograms.get(capacity)
    if (cached) return cached
    const source = author('decode', 1, 0, capacity)
    decodePrograms.set(capacity, source)
    return source
  }
  return { config, parameters, forward, prefill, decode, cacheMode: 'fixed' as const }
}
