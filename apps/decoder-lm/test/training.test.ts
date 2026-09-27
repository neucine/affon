import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { describe, expect, test } from 'std:test'
import { seed, tensor } from 'affon:compute'
import { run } from 'std:process'
import type { Tensor } from 'affon:compute'

import { DecoderModel } from '../src/model.ts'
import { createPackedTextCorpus, pack_token_windows } from '../src/data/index.ts'
import { createHFTokenizerFromFile, createLookupTokenizer } from '@affon/tokenizers'
import {
  evaluateDecoderLM,
  loadDecoderLMCheckpoint,
  perplexityFromLoss,
  saveDecoderLMCheckpoint,
  trainDecoderLM,
} from '../src/index.ts'

function tensorTreeToPlain(value: unknown): unknown {
  if (value && typeof value === 'object' && typeof (value as any).to_array === 'function') {
    return (value as any).to_array()
  }
  if (Array.isArray(value)) return value.map((entry) => tensorTreeToPlain(entry))
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      out[key] = tensorTreeToPlain((value as Record<string, unknown>)[key])
    }
    return out
  }
  return value
}

function expectLossPrefix(actual: readonly number[], expected: readonly number[], tolerance = 1e-5): void {
  expect(actual.length >= expected.length).toBeTruthy()
  for (let i = 0; i < expected.length; i++) {
    const delta = Math.abs(actual[i] - expected[i])
    if (delta > tolerance) {
      throw new Error(`loss curve mismatch at batch ${i + 1}: actual=${actual[i]} expected=${expected[i]} delta=${delta}`)
    }
  }
}

function flattenNumbers(value: unknown, out: number[] = []): number[] {
  if (Array.isArray(value)) {
    for (const entry of value) flattenNumbers(entry, out)
  } else {
    out.push(value as number)
  }
  return out
}

function maxAbsDelta(before: unknown, after: unknown): number {
  const beforeFlat = flattenNumbers(before)
  const afterFlat = flattenNumbers(after)
  let max = 0
  for (let i = 0; i < beforeFlat.length; i++) {
    max = Math.max(max, Math.abs(afterFlat[i] - beforeFlat[i]))
  }
  return max
}

function collectTensors(value: unknown, out: Tensor[] = []): Tensor[] {
  if (value && typeof value === 'object' && typeof (value as any).item === 'function' && Array.isArray((value as any).shape)) {
    out.push(value as Tensor)
    return out
  }
  if (Array.isArray(value)) {
    for (const entry of value) collectTensors(entry, out)
    return out
  }
  if (value && typeof value === 'object') {
    for (const entry of Object.values(value as Record<string, unknown>)) collectTensors(entry, out)
  }
  return out
}

function firstNonEmptyLines(path: string, limit: number): string[] {
  return fs.readFileSync(path)
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .slice(0, limit)
}

describe('@affon/decoder-lm training', () => {
  test('packs token rows into overlapping decoder windows', () => {
    const windows = pack_token_windows([
      [1, 2, 3, 4],
      [5, 6, 7],
    ], {
      seqLen: 2,
      stride: 1,
      joinWithTokenId: 9,
    })

    expect(windows).toEqual([
      [1, 2, 3],
      [2, 3, 4],
      [3, 4, 9],
      [4, 9, 5],
      [9, 5, 6],
      [5, 6, 7],
    ])
  })

  test('trains a tiny decoder lm and records checkpoints', () => {
    setDevice('cpu')
    seed(0)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    const initialTrainLoss = evaluateDecoderLM(model, windows, { batchSize: 2 })
    const result = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 10,
      lr: 0.01,
      checkpointEveryEpochs: 5,
      validationWindows: windows,
    })

    expect(result.steps > 0).toBeTruthy()
    expect(result.history.length).toBe(10)
    expect(result.checkpoints.length).toBe(2)
    expect(result.initialTrainLoss).toBe(initialTrainLoss)
    expect(result.initialTrainPerplexity !== null).toBeTruthy()
    expect(result.initialTrainLoss !== null).toBeTruthy()
    expect(result.finalTrainLoss < (result.initialTrainLoss as number)).toBeTruthy()
    expect(result.finalValLoss !== null).toBeTruthy()
  })

  test('overfits a deterministic tiny corpus to a regression quality floor', () => {
    setDevice('cpu')
    seed(21)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      alpha: 3,
      beta: 4,
      gamma: 5,
      delta: 6,
      omega: 7,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    const rows = Array.from({ length: 4 }, () =>
      tokenizer.encode('alpha beta gamma delta omega', { addBos: true, addEos: true }),
    )
    const windows = pack_token_windows(rows, {
      seqLen: 4,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const model = DecoderModel(tokenizer.vocabSize, 16, {
      numLayers: 1,
      numHeads: 2,
      hiddenDim: 32,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: false,
    })

    const batchLosses: number[] = []
    const result = trainDecoderLM(model, windows, {
      seqLen: 4,
      batchSize: 4,
      epochs: 36,
      lr: 0.012,
      maxGradNorm: 1,
      shuffle: false,
      validationWindows: windows,
      checkpointEveryEpochs: 0,
      onBatch: (metrics) => batchLosses.push(metrics.batchLoss),
    })

    expect(result.initialTrainLoss !== null).toBeTruthy()
    expect(result.steps).toBe(252)
    expect(result.history.length).toBe(36)
    expect(batchLosses.length).toBe(252)
    if (result.finalTrainLoss >= 0.35 || result.finalValLoss === null || result.finalValLoss >= 0.35) {
      throw new Error(`tiny corpus did not overfit enough: initial=${result.initialTrainLoss} final=${result.finalTrainLoss} val=${result.finalValLoss}`)
    }
    expect(result.finalTrainLoss < (result.initialTrainLoss as number) * 0.25).toBeTruthy()
    expect(batchLosses.every((loss) => Number.isFinite(loss))).toBeTruthy()
    expectLossPrefix(batchLosses, [
      2.8755221366882324,
      2.0429625511169434,
      1.6723483800888062,
      1.5234858989715576,
      1.2551047801971436,
      1.32273268699646,
      0.9983834624290466,
      0.8179523944854736,
      1.0814785957336426,
      0.5837952494621277,
      0.9183781147003174,
      0.4360317885875702,
    ])
  })

  test('runs a fixed WikiText subset canary with stable early metrics', () => {
    setDevice('cpu')
    seed(31)

    const tokenizer = createHFTokenizerFromFile('apps/decoder-lm/data/wikitext-gpt2-tokenizer.json', {
      specialTokens: { eos: '<|endoftext|>' },
    })
    const corpus = createPackedTextCorpus(
      tokenizer,
      firstNonEmptyLines('apps/decoder-lm/data/wikitext-2-train.txt', 12),
      {
        seqLen: 16,
        stride: 8,
        addBos: false,
        addEos: true,
        joinWithTokenId: tokenizer.specialTokenIds.eos,
        validationSplit: 0.25,
        shuffle: false,
        preprocess: {
          replacements: [
            { from: ' @-@ ', to: '-' },
            { from: ' @,@ ', to: ',' },
            { from: '<unk>', to: '' },
          ],
          normalizeWhitespace: true,
        },
      },
    )

    const model = DecoderModel(tokenizer.vocabSize, 8, {
      numLayers: 1,
      numHeads: 2,
      hiddenDim: 16,
      causal: true,
      positional: 'learned',
      maxSeqLen: 16,
      tieEmbeddings: true,
    })
    const trackedParam = (model.blocks[0] as any).self_attention.q_proj.weight as Tensor
    let previousTrackedParam = trackedParam.to_array()
    const batchLosses: number[] = []
    const gradNorms: number[] = []
    const updateDeltas: number[] = []

    const result = trainDecoderLM(model, corpus.trainWindows.slice(0, 8), {
      seqLen: 16,
      batchSize: 2,
      epochs: 1,
      lr: 0.0005,
      maxGradNorm: 1,
      shuffle: false,
      validationWindows: corpus.validationWindows.slice(0, 4),
      validationBatchSize: 2,
      maxEvalBatches: 2,
      checkpointEveryEpochs: 0,
      onBatch: (metrics) => {
        batchLosses.push(metrics.batchLoss)
        if (metrics.gradNorm !== null) gradNorms.push(metrics.gradNorm)
        const current = trackedParam.to_array()
        updateDeltas.push(maxAbsDelta(previousTrackedParam, current))
        previousTrackedParam = current
      },
    })

    expect(corpus.trainWindows.length >= 8).toBeTruthy()
    expect(corpus.validationWindows.length >= 4).toBeTruthy()
    expect(result.initialValLoss !== null).toBeTruthy()
    expect(result.finalValLoss !== null).toBeTruthy()
    expect(batchLosses.length).toBe(4)
    expect(gradNorms.length).toBe(4)
    expect(updateDeltas.length).toBe(4)
    expect(batchLosses.every((loss) => Number.isFinite(loss))).toBeTruthy()
    expect(gradNorms.every((norm) => Number.isFinite(norm) && norm > 0)).toBeTruthy()
    expect(updateDeltas.every((delta) => Number.isFinite(delta) && delta > 0)).toBeTruthy()
    expectLossPrefix([result.initialValLoss as number, result.finalValLoss as number], [
      10.820650100708008,
      10.81892204284668,
    ])
    expectLossPrefix(batchLosses, [
      10.830144882202148,
      10.827646255493164,
      10.819657325744629,
      10.82106876373291,
    ])
    expectLossPrefix(gradNorms, [
      0.5263403248203973,
      0.5206719017965002,
      0.5294269171684293,
      0.5520692046729742,
    ])
    expectLossPrefix(updateDeltas, [
      0.000500023365020752,
      0.0005006492137908936,
      0.00045168399810791016,
      0.0004553794860839844,
    ])
  })

  test('validates WikiText canary summaries with the standalone checker', async () => {
    const prefix = `/tmp/affon-wikitext-canary-check-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const goodPath = `${prefix}-good.json`
    const badPath = `${prefix}-bad.json`

    fs.writeFileSync(goodPath, JSON.stringify({
      initialValLoss: 10.25,
      finalValLoss: 5.8,
      history: [
        { epoch: 1, step: 1250, trainLoss: 10.1, valLoss: 9.4 },
        { epoch: 2, step: 2500, trainLoss: 8.8, valLoss: 8.2 },
        { epoch: 3, step: 3750, trainLoss: 7.6, valLoss: 7.1 },
        { epoch: 4, step: 5000, trainLoss: 6.5, valLoss: 6.2 },
        { epoch: 5, step: 6250, trainLoss: 5.9, valLoss: 5.8 },
      ],
    }))
    fs.writeFileSync(badPath, JSON.stringify({
      initialValLoss: 10.25,
      finalValLoss: 5.7,
      history: [
        { epoch: 1, step: 1250, trainLoss: 10.1, valLoss: 9.4 },
        { epoch: 2, step: 2500, trainLoss: 8.8, valLoss: 8.2 },
        { epoch: 3, step: 3750, trainLoss: 7.6, valLoss: 8.3 },
        { epoch: 4, step: 5000, trainLoss: 6.5, valLoss: 6.1 },
        { epoch: 5, step: 6250, trainLoss: 5.9, valLoss: 5.7 },
      ],
    }))

    const good = await run({
      cmd: 'sh',
      args: ['-c', `AFFON_WIKITEXT_CANARY_SUMMARY="${goodPath}" ./zig-out/bin/affon apps/decoder-lm/check-wikitext-canary.ts`],
    })
    expect(good.stdout.includes('WikiText canary passed')).toBe(true)

    const bad = await run({
      cmd: 'sh',
      args: ['-c', `AFFON_WIKITEXT_CANARY_SUMMARY="${badPath}" ./zig-out/bin/affon apps/decoder-lm/check-wikitext-canary.ts`],
      check: false,
    })
    expect(bad.exitCode === 0).toBe(false)
    expect(bad.stderr.includes('WikiText canary val loss rose')).toBe(true)
  })

  test('can skip initial train loss evaluation', () => {
    setDevice('cpu')
    seed(4)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    const result = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      evaluateInitialTrainLoss: false,
      validationWindows: windows,
    })

    expect(result.initialTrainLoss).toBeNull()
    expect(result.initialTrainPerplexity).toBeNull()
    expect(result.initialValLoss !== null).toBeTruthy()
  })

  test('uses a dedicated evaluation forward without consuming the training forward', () => {
    setDevice('cpu')
    seed(41)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })
    const windows = pack_token_windows([
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ], {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })
    const model = DecoderModel(tokenizer.vocabSize, 16, {
      numLayers: 1,
      numHeads: 2,
      hiddenDim: 32,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    let trainingForwardCalls = 0
    let evaluationForwardCalls = 0

    trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      maxTrainBatchesPerEpoch: 1,
      maxEvalBatches: 1,
      evaluateInitialTrainLoss: false,
      validationWindows: windows,
      forward: (tokenIds) => {
        trainingForwardCalls += 1
        return model(tokenIds)
      },
      evaluationForward: (tokenIds) => {
        evaluationForwardCalls += 1
        return model(tokenIds)
      },
    })

    expect(trainingForwardCalls).toBe(1)
    expect(evaluationForwardCalls).toBe(2)
  })

  test('can cap train and evaluation batches for faster debug runs', () => {
    setDevice('cpu')
    seed(11)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    const batchesSeen: number[] = []
    const evalStatuses: string[] = []
    const result = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      maxTrainBatchesPerEpoch: 2,
      maxEvalBatches: 1,
      validationWindows: windows,
      onBatch: (metrics) => batchesSeen.push(metrics.batch),
      onStatus: (status) => evalStatuses.push(status),
    })

    expect(result.steps).toBe(2)
    expect(batchesSeen).toEqual([1, 2])
    const detailedEvalStatuses = evalStatuses.filter((status) => /\d+\/\d+\.\.\.$/.test(status))
    expect(detailedEvalStatuses.length > 0).toBe(true)
    expect(detailedEvalStatuses.every((status) => status.includes('1/1'))).toBe(true)
  })

  test('applies lr schedules using current epoch and global step across resume', () => {
    setDevice('cpu')
    seed(15)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    const seen: Array<{ epoch: number, step: number }> = []
    const first = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      lrSchedule: (ctx) => {
        seen.push({ epoch: ctx.epoch, step: ctx.step })
        return 0.01
      },
      checkpointEverySteps: 2,
      evaluateInitialTrainLoss: false,
    })

    expect(seen.map((ctx) => ctx.step)).toEqual([0, 1, 2, 3, 4, 5])
    expect(seen.every((ctx) => ctx.epoch === 1)).toBe(true)

    const checkpoint = first.checkpoints[0]
    const resumedModel = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    resumedModel.restore(checkpoint.state)

    const resumedSeen: Array<{ epoch: number, step: number }> = []
    trainDecoderLM(resumedModel, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      lrSchedule: (ctx) => {
        resumedSeen.push({ epoch: ctx.epoch, step: ctx.step })
        return 0.01
      },
      resumeCheckpoint: checkpoint as any,
      initialEpoch: checkpoint.epoch,
      initialStep: checkpoint.step,
      evaluateInitialTrainLoss: false,
    })

    expect(resumedSeen.length > 0).toBe(true)
    expect(resumedSeen[0].epoch).toBe(1)
    expect(resumedSeen[0].step).toBe(2)
  })

  test('fails fast on non-finite batch loss with training context', () => {
    setDevice('cpu')
    seed(9)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const badModel = Object.assign(
      function(tokenIds: Tensor<[number, number], 'f32'>): Tensor<[number, number, number], 'f32'> {
        return tensor(
          Array.from({ length: tokenIds.shape[0] as number }, () =>
            Array.from({ length: tokenIds.shape[1] as number }, () =>
              Array.from({ length: tokenizer.vocabSize }, () => Number.NaN),
            ),
          ),
          { dtype: 'f32' },
        ) as Tensor<[number, number, number], 'f32'>
      },
      {
        parameters: [] as Tensor[],
        train(): void {},
        eval(): void {},
        state(): Record<string, Tensor> { return {} },
      },
    ) as any

    try {
      trainDecoderLM(badModel, windows, {
        seqLen: 3,
        batchSize: 2,
        epochs: 1,
        lr: 0.01,
        evaluateInitialTrainLoss: false,
      })
      expect(false).toBe(true)
    } catch (error) {
      const message = String(error)
      expect(message.includes('trainDecoderLM batch loss became non-finite')).toBe(true)
      expect(message.includes('logits_first_bad_flat_index=0')).toBe(true)
      expect(message.includes('logits_first_bad_value=NaN')).toBe(true)
      expect(message.includes('logits_abs_max=unavailable')).toBe(true)
      expect(message.includes('param_nonfinite=[none]')).toBe(true)
      expect(message.includes('batch_shape=')).toBe(true)
    }
  })


  test('fails fast on non-finite evaluation loss with evaluation context', () => {
    setDevice('cpu')
    seed(10)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const badModel = Object.assign(
      function(tokenIds: Tensor<[number, number], 'f32'>): Tensor<[number, number, number], 'f32'> {
        return tensor(
          Array.from({ length: tokenIds.shape[0] as number }, () =>
            Array.from({ length: tokenIds.shape[1] as number }, () =>
              Array.from({ length: tokenizer.vocabSize }, () => Number.NaN),
            ),
          ),
          { dtype: 'f32' },
        ) as Tensor<[number, number, number], 'f32'>
      },
      {
        parameters: [] as Tensor[],
        train(): void {},
        eval(): void {},
        state(): Record<string, Tensor> { return {} },
      },
    ) as any

    try {
      evaluateDecoderLM(badModel, windows, {
        batchSize: 2,
        seqLen: 3,
        statusLabel: 'initial validation loss',
      })
      expect(false).toBe(true)
    } catch (error) {
      const message = String(error)
      expect(message.includes('evaluateDecoderLM batch loss became non-finite')).toBe(true)
      expect(message.includes('logits_first_bad_flat_index=0')).toBe(true)
      expect(message.includes('logits_first_bad_value=NaN')).toBe(true)
      expect(message.includes('logits_abs_max=unavailable')).toBe(true)
      expect(message.includes('batch_shape=')).toBe(true)
    }
  })

  test('rejects non-finite perplexity inputs and overflowed outputs', () => {
    expect(() => perplexityFromLoss(Number.NaN)).toThrow('perplexityFromLoss input loss became non-finite')
    expect(() => perplexityFromLoss(1000)).toThrow('perplexityFromLoss result became non-finite')
  })

  test('saves and loads decoder lm checkpoints', () => {
    setDevice('cpu')
    seed(1)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })
    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })
    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    const result = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 4,
      lr: 0.01,
      checkpointEveryEpochs: 4,
    })

    const prefix = `/tmp/affon-transformers-checkpoint-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    saveDecoderLMCheckpoint(prefix, result.checkpoints[0], {
      metadata: { tokenizerFamily: tokenizer.family },
    })

    const restored = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    const loaded = loadDecoderLMCheckpoint(prefix, restored)
    const state = checkpoint.load(`${prefix}.safetensors`) as Record<string, Tensor>
    const originalLoss = evaluateDecoderLM(model, windows, { batchSize: 2, seqLen: 3 })
    const restoredLoss = evaluateDecoderLM(restored, windows, { batchSize: 2, seqLen: 3 })

    expect(loaded.epoch).toBe(4)
    expect(loaded.metadata?.tokenizerFamily).toBe(tokenizer.family)
    expect(Object.values(state).every((value) => value.dtype === 'f32')).toBeTruthy()
    expect(Object.values(state).every((value) => value.device === 'cpu')).toBeTruthy()
    expect(collectTensors(loaded.state).every((value) => value.device === 'cpu')).toBeTruthy()
    expect(Math.abs(originalLoss - restoredLoss) < 1e-6).toBeTruthy()
  })

  test('writes step-based checkpoints using global step paths', () => {
    setDevice('cpu')
    seed(12)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })
    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })
    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    const prefix = `/tmp/affon-transformers-step-checkpoint-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const result = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      checkpointEverySteps: 2,
      checkpointPathPrefix: prefix,
      evaluateInitialTrainLoss: false,
    })

    expect(result.steps).toBe(6)
    expect(result.checkpointPaths).toEqual([
      `${prefix}-step-2`,
      `${prefix}-step-4`,
      `${prefix}-step-6`,
    ])
    const loaded = loadDecoderLMCheckpoint(`${prefix}-step-6`, model)
    expect(loaded.step).toBe(6)
    expect(loaded.epoch).toBe(1)
    expect(loaded.valLoss).toBeNull()
  })

  test('prefers epoch checkpoints over duplicate step checkpoints at epoch end', () => {
    setDevice('cpu')
    seed(13)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })
    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })
    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    const prefix = `/tmp/affon-transformers-dedup-checkpoint-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const result = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      checkpointEveryEpochs: 1,
      checkpointEverySteps: 2,
      checkpointPathPrefix: prefix,
      evaluateInitialTrainLoss: false,
    })

    expect(result.steps).toBe(6)
    expect(result.checkpointPaths).toEqual([
      `${prefix}-step-2`,
      `${prefix}-step-4`,
      `${prefix}-epoch-1`,
    ])
  })

  test('can resume exactly from a mid-epoch step checkpoint', () => {
    setDevice('cpu')
    seed(14)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
      general: 6,
      kenobi: 7,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })
    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
      tokenizer.encode('general kenobi', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
      tokenizer.encode('general kenobi', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const createModel = () => DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    seed(21)
    const baseline = createModel()
    const baselineResult = trainDecoderLM(baseline, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 2,
      lr: 0.01,
      evaluateInitialTrainLoss: false,
      shuffle: true,
      shuffleSeed: 123,
    })
    const baselineLoss = evaluateDecoderLM(baseline, windows, { batchSize: 2, seqLen: 3 })

    seed(21)
    const interrupted = createModel()
    const prefix = `/tmp/affon-transformers-exact-resume-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    trainDecoderLM(interrupted, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 2,
      lr: 0.01,
      checkpointEverySteps: 2,
      checkpointPathPrefix: prefix,
      evaluateInitialTrainLoss: false,
      shuffle: true,
      shuffleSeed: 123,
    })

    const resumedModel = createModel()
    const resumedCheckpoint = loadDecoderLMCheckpoint(`${prefix}-step-2`, resumedModel)
    const resumed = trainDecoderLM(resumedModel, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 2,
      lr: 0.01,
      resumeCheckpoint: resumedCheckpoint,
      evaluateInitialTrainLoss: false,
      shuffle: true,
      shuffleSeed: 123,
    })
    const resumedLoss = evaluateDecoderLM(resumedModel, windows, { batchSize: 2, seqLen: 3 })

    expect(resumedCheckpoint.resumeEpoch).toBe(1)
    expect(resumedCheckpoint.resumeBatchIndex).toBe(2)
    expect(resumedCheckpoint.batchOrder !== null).toBeTruthy()
    expect(Math.abs(baselineLoss - resumedLoss) < 1e-6).toBeTruthy()
    expect(resumed.steps).toBe(baselineResult.steps)
  })

  test('round-trips optimizer state and matches the next resumed step exactly', () => {
    setDevice('cpu')
    seed(22)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
      general: 6,
      kenobi: 7,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })
    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
      tokenizer.encode('general kenobi', { addBos: true, addEos: true }),
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
      tokenizer.encode('general kenobi', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })

    const createModel = () => DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })

    const prefix = `/tmp/affon-transformers-optimizer-resume-${Date.now()}-${Math.floor(Math.random() * 1e6)}`

    seed(23)
    const firstModel = createModel()
    trainDecoderLM(firstModel, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      checkpointEverySteps: 2,
      checkpointPathPrefix: prefix,
      evaluateInitialTrainLoss: false,
      shuffle: true,
      shuffleSeed: 321,
    })

    const savedModel = createModel()
    const loaded = loadDecoderLMCheckpoint(`${prefix}-step-2`, savedModel)
    expect(loaded.optimizerState !== null).toBe(true)
    expect(loaded.resumeEpoch).toBe(1)
    expect(loaded.resumeBatchIndex).toBe(2)

    const manifest = JSON.parse(fs.readFileSync(`${prefix}-step-2.json`)) as {
      optimizerStateScalars?: Record<string, number> | null
    }
    expect(loaded.optimizerState?.scalars ?? {}).toEqual(manifest.optimizerStateScalars ?? {})

    const optimizerTensors = checkpoint.load(`${prefix}-step-2.optimizer.safetensors`) as Record<string, Tensor>
    expect(tensorTreeToPlain(loaded.optimizerState?.tensors ?? {})).toEqual(tensorTreeToPlain(optimizerTensors))
    expect(Object.values(optimizerTensors).every((value) => value.device === 'cpu')).toBeTruthy()
    expect(collectTensors(loaded.state).every((value) => value.device === 'cpu')).toBeTruthy()
    expect(collectTensors(loaded.optimizerState?.tensors ?? {}).every((value) => value.device === 'cpu')).toBeTruthy()

    seed(23)
    const baselineModel = createModel()
    const baseline = trainDecoderLM(baselineModel, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      evaluateInitialTrainLoss: false,
      shuffle: true,
      shuffleSeed: 321,
      maxTrainBatchesPerEpoch: 3,
    })

    const resumed = trainDecoderLM(savedModel, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 1,
      lr: 0.01,
      resumeCheckpoint: loaded,
      evaluateInitialTrainLoss: false,
      shuffle: true,
      shuffleSeed: 321,
      maxTrainBatchesPerEpoch: 3,
    })

    expect(baseline.steps).toBe(3)
    expect(resumed.steps).toBe(3)
    expect(baseline.history.length).toBe(1)
    expect(resumed.history.length).toBe(1)
    expect(Math.abs(baseline.history[0].trainLoss - resumed.history[0].trainLoss) < 1e-6).toBe(true)

    const baselineLoss = evaluateDecoderLM(baselineModel, windows, { batchSize: 2, seqLen: 3 })
    const resumedLoss = evaluateDecoderLM(savedModel, windows, { batchSize: 2, seqLen: 3 })
    expect(Math.abs(baselineLoss - resumedLoss) < 1e-6).toBe(true)
    expect(tensorTreeToPlain(savedModel.state())).toEqual(tensorTreeToPlain(baselineModel.state()))
  })

  test('loads a checkpoint with relative statePath in the manifest', () => {
    setDevice('cpu')
    seed(6)

    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
      there: 5,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })
    const rows = [
      tokenizer.encode('hello world', { addBos: true, addEos: true }),
      tokenizer.encode('hello there', { addBos: true, addEos: true }),
    ]
    const windows = pack_token_windows(rows, {
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.tokenId('<eos>'),
    })
    const model = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    const result = trainDecoderLM(model, windows, {
      seqLen: 3,
      batchSize: 2,
      epochs: 2,
      lr: 0.01,
      checkpointEveryEpochs: 2,
    })

    const srcPrefix = `/tmp/affon-transformers-relative-src-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    saveDecoderLMCheckpoint(srcPrefix, result.checkpoints[0], {
      metadata: { tokenizerFamily: tokenizer.family },
    })

    const srcManifest = JSON.parse(fs.readFileSync(`${srcPrefix}.json`)) as {
      format: string
      epoch: number
      step: number
      trainLoss: number
      valLoss: number | null
      statePath: string
      basePath?: string | null
      metadata: Record<string, unknown> | null
    }
    srcManifest.statePath = `${srcPrefix.split('/').pop()}.safetensors`
    srcManifest.basePath = '/tmp'
    fs.writeFileSync(`${srcPrefix}.json`, JSON.stringify(srcManifest, null, 2))

    const restored = DecoderModel(tokenizer.vocabSize, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 8,
      tieEmbeddings: true,
    })
    const loaded = loadDecoderLMCheckpoint(srcPrefix, restored)
    const originalLoss = evaluateDecoderLM(model, windows, { batchSize: 2, seqLen: 3 })
    const restoredLoss = evaluateDecoderLM(restored, windows, { batchSize: 2, seqLen: 3 })

    expect(loaded.epoch).toBe(2)
    expect(loaded.metadata?.tokenizerFamily).toBe(tokenizer.family)
    expect(Math.abs(originalLoss - restoredLoss) < 1e-6).toBeTruthy()
  })
})
