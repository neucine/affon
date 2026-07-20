import fs from 'affon:fs'
import { describe, expect, test } from 'affon:test'
import { seed } from 'affon:compute'

import {
  loadDecoderLMWorkflowConfig,
  trainDecoderLMFromConfig,
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

describe('@affon/decoder-lm workflow', () => {
  test('trains from a workflow config file', () => {
    setDevice('cpu')
    seed(2)

    const prefix = `/tmp/affon-transformers-workflow-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    const configPath = `${prefix}.json`
    const trainCachePath = `${prefix}-train-cache.json`
    const validationCachePath = `${prefix}-validation-cache.json`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')
    fs.writeFileSync(configPath, JSON.stringify({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        tokenCachePath: trainCachePath,
        validationTokenCachePath: validationCachePath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 4,
        batchSize: 2,
        lr: 0.01,
      },
      checkpoint: {
        prefix,
        everyNEpochs: 2,
        metadata: {
          source: 'workflow-test',
        },
      },
      report: {
        summaryPath: `${prefix}-summary.json`,
        sample: {
          prompt: 'hello',
          max_new_tokens: 1,
          everyNEpochs: 2,
          addBos: true,
          skipSpecialTokens: true,
          banSpecialTokens: true,
        },
      },
    }, null, 2))

    const config = loadDecoderLMWorkflowConfig(configPath)
    const seenEpochs: number[] = []
    const seenBatches: Array<{ epoch: number; batch: number; batches: number; step: number }> = []
    const result = trainDecoderLMFromConfig(config, {
      onBatch: (metrics) => {
        seenBatches.push({
          epoch: metrics.epoch,
          batch: metrics.batch,
          batches: metrics.batches,
          step: metrics.step,
        })
      },
      onEpoch: (metrics) => {
        seenEpochs.push(metrics.epoch)
      },
    })

    expect(result.corpus.trainRows.length > 0).toBeTruthy()
    expect(result.training.steps > 0).toBeTruthy()
    expect(result.training.checkpointPaths.length).toBe(2)
    expect(result.samples.length).toBe(2)
    expect(result.summary.samples.length).toBe(2)
    const summary = JSON.parse(fs.readFileSync(`${prefix}-summary.json`)) as { finalTrainPerplexity: number, samples: unknown[] }
    const trainCache = JSON.parse(fs.readFileSync(trainCachePath)) as number[][]
    const validationCache = JSON.parse(fs.readFileSync(validationCachePath)) as number[][]
    expect(summary.finalTrainPerplexity > 0).toBeTruthy()
    expect(summary.samples.length).toBe(2)
    expect(trainCache.length > 0).toBeTruthy()
    expect(validationCache.length > 0).toBeTruthy()
    expect(seenEpochs).toEqual([1, 2, 3, 4])
    expect(seenBatches.length > 0).toBeTruthy()
    expect(seenBatches[0].epoch).toBe(1)
    expect(seenBatches[0].batch).toBe(1)
    expect(seenBatches[0].batches > 0).toBeTruthy()
  })

  test('supports cached token rows in workflow config', () => {
    setDevice('cpu')
    seed(3)

    const prefix = `/tmp/affon-transformers-workflow-cache-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    const cachePath = `${prefix}.cache.json`
    const configPath = `${prefix}.json`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const bootstrap = loadDecoderLMWorkflowConfig('apps/decoder-lm/configs/train-decoder-lm-helloworld.config.json')
    const tokenizer = trainDecoderLMFromConfig({
      ...bootstrap,
      corpus: {
        ...bootstrap.corpus,
        path: corpusPath,
      },
      training: {
        ...bootstrap.training,
        epochs: 1,
      },
      checkpoint: undefined,
      report: undefined,
    }).tokenizer

    const tokenRows = tokenizer.encode('hello world', { addBos: true, addEos: true })
    fs.writeFileSync(cachePath, JSON.stringify([tokenRows, tokenRows]))
    fs.writeFileSync(configPath, JSON.stringify({
      device: 'cpu',
      tokenizer: bootstrap.tokenizer,
      corpus: {
        tokenCachePath: cachePath,
        seqLen: 3,
        stride: 1,
      },
      model: bootstrap.model,
      training: {
        epochs: 1,
        batchSize: 2,
        lr: 0.01,
      },
    }, null, 2))

    const result = trainDecoderLMFromConfig(loadDecoderLMWorkflowConfig(configPath))
    expect(result.corpus.trainRows.length).toBe(2)
    expect(result.training.steps > 0).toBeTruthy()
  })

  test('supports learning-rate schedules in workflow config', () => {
    setDevice('cpu')
    seed(16)

    const prefix = `/tmp/affon-transformers-workflow-schedule-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 2,
        batchSize: 2,
        lrSchedule: {
          kind: 'step',
          base: 0.01,
          gamma: 0.5,
          every: { unit: 'epoch', value: 1 },
        },
      },
    })

    expect(result.training.steps > 0).toBeTruthy()
    expect(result.training.history.length).toBe(2)
  })

  test('supports warmup-constant learning-rate schedules in workflow config', () => {
    setDevice('cpu')
    seed(17)

    const prefix = `/tmp/affon-transformers-workflow-warmup-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 2,
        batchSize: 2,
        lrSchedule: {
          kind: 'warmup_constant',
          start: 0.001,
          lr: 0.01,
          duration: { unit: 'step', value: 2 },
        },
      },
    })

    expect(result.training.steps > 0).toBeTruthy()
    expect(result.training.history.length).toBe(2)
  })

  test('supports warmup-cosine learning-rate schedules in workflow config', () => {
    setDevice('cpu')
    seed(19)

    const prefix = `/tmp/affon-transformers-workflow-warmup-cosine-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 2,
        batchSize: 2,
        lrSchedule: {
          kind: 'warmup_cosine',
          start: 0.001,
          peak: 0.01,
          end: 0.002,
          warmup: { unit: 'step', value: 2 },
          total: { unit: 'step', value: 6 },
        },
      },
    })

    expect(result.training.steps > 0).toBeTruthy()
    expect(result.training.history.length).toBe(2)
  })

  test('supports adamw optimizer config in workflow training', () => {
    setDevice('cpu')
    seed(31)

    const prefix = `/tmp/affon-transformers-workflow-adamw-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 1,
        batchSize: 2,
        lr: 0.01,
        optimizer: {
          kind: 'adamw',
          weightDecay: 0.05,
        },
      },
      checkpoint: {
        prefix,
        everyNEpochs: 1,
      },
    })

    expect(result.training.steps > 0).toBeTruthy()
    const manifest = JSON.parse(fs.readFileSync(`${prefix}-epoch-1.json`)) as {
      optimizerStateKind?: string | null
      optimizerStateScalars?: Record<string, number> | null
    }
    expect(manifest.optimizerStateKind).toBe('AdamW')
    expect(manifest.optimizerStateScalars?.weight_decay).toBe(0.05)
  })

  test('supports gradient accumulation in workflow training', () => {
    setDevice('cpu')
    seed(33)

    const corpusPath = `/tmp/affon-transformers-workflow-accum-${Date.now()}-${Math.floor(Math.random() * 1e6)}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\nhello there\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.25,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 1,
        batchSize: 1,
        gradientAccumulationSteps: 2,
        lr: 0.01,
        maxTrainBatchesPerEpoch: 4,
      },
    })

    expect(result.training.history.length).toBe(1)
    expect(result.training.steps).toBe(2)
    expect(result.training.finalTrainLoss > 0).toBeTruthy()
  })

  test('resumes with fresh optimizer state when checkpoint optimizer kind differs', () => {
    setDevice('cpu')
    seed(32)

    const prefix = `/tmp/affon-transformers-workflow-adamw-resume-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 1,
        batchSize: 2,
        lr: 0.01,
      },
      checkpoint: {
        prefix,
        everyNEpochs: 1,
      },
    })

    const statuses: string[] = []
    const resumed = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 1,
        batchSize: 2,
        lr: 0.01,
        optimizer: {
          kind: 'adamw',
          weightDecay: 0.05,
        },
      },
      checkpoint: {
        prefix,
        resumeFrom: `${prefix}-epoch-1`,
        resumeStrategy: 'exact',
        restoreOptimizerState: true,
      },
    }, {
      onStatus: (status) => {
        statuses.push(status)
      },
    })

    expect(resumed.training.steps > 0).toBeTruthy()
    expect(statuses.some((status) => status.includes('restoring model weights only with fresh optimizer state'))).toBeTruthy()
  })

  test('keeps compiled forward enabled when dropout is enabled', () => {
    setDevice('cpu')
    seed(18)

    const prefix = `/tmp/affon-transformers-workflow-dropout-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const statuses: string[] = []
    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
        dropout: 0.1,
      },
      training: {
        epochs: 1,
        batchSize: 2,
        lr: 0.01,
        compileModelForward: true,
      },
    }, {
      onStatus: (status) => {
        statuses.push(status)
      },
    })

    expect(result.training.steps > 0).toBeTruthy()
    expect(result.compiledModel).not.toBeNull()
    expect(result.summary.modelForwardMode).toBe(result.compiledModel ? 'graph' : 'eager')
    expect(statuses.some((status) => status.includes('dropout enabled; skipping compiled forward'))).toBeFalsy()
  })

  test('can resume from a saved checkpoint prefix', () => {
    setDevice('cpu')
    seed(5)

    const prefix = `/tmp/affon-transformers-workflow-resume-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const baseConfig = {
      device: 'cpu' as const,
      tokenizer: {
        family: 'lookup' as const,
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned' as const,
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
    }

    const first = trainDecoderLMFromConfig({
      ...baseConfig,
      training: {
        epochs: 2,
        batchSize: 2,
        lr: 0.01,
      },
      checkpoint: {
        prefix,
        everyNEpochs: 2,
      },
      report: undefined,
    })

    expect(first.training.checkpointPaths).toEqual([`${prefix}-epoch-2`])

    const resumed = trainDecoderLMFromConfig({
      ...baseConfig,
      training: {
        epochs: 2,
        batchSize: 2,
        lr: 0.01,
      },
      checkpoint: {
        prefix,
        everyNEpochs: 2,
        resumeFrom: `${prefix}-epoch-2`,
      },
      report: undefined,
    })

    expect(resumed.training.history.map((entry) => entry.epoch)).toEqual([3, 4])
    expect(resumed.training.steps > first.training.steps).toBeTruthy()
    expect(resumed.training.checkpointPaths).toEqual([`${prefix}-epoch-4`])
  })

  test('can resume from the latest best-loss epoch in workflow summary', () => {
    setDevice('cpu')
    seed(31)

    const prefix = `/tmp/affon-transformers-workflow-best-epoch-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    const summaryPath = `${prefix}-summary.json`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const baseConfig = {
      device: 'cpu' as const,
      tokenizer: {
        family: 'lookup' as const,
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned' as const,
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      report: {
        summaryPath,
      },
    }

    const first = trainDecoderLMFromConfig({
      ...baseConfig,
      training: {
        epochs: 4,
        batchSize: 2,
        lr: 0.01,
      },
      checkpoint: {
        prefix,
        everyNEpochs: 1,
      },
    })

    const summary = JSON.parse(fs.readFileSync(summaryPath)) as {
      history: Array<{ epoch: number; trainLoss: number; valLoss: number | null }>
    }
    summary.history = summary.history.map((entry) => ({
      ...entry,
      valLoss: entry.epoch === 2 ? 0.1 : 10 + entry.epoch,
    }))
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2))

    const resumed = trainDecoderLMFromConfig({
      ...baseConfig,
      training: {
        epochs: 2,
        batchSize: 2,
        lr: 0.01,
      },
      checkpoint: {
        prefix,
        everyNEpochs: 1,
        resumeStrategy: 'best_epoch',
      },
    })

    expect(first.training.history.map((entry) => entry.epoch)).toEqual([1, 2, 3, 4])
    expect(resumed.training.history.map((entry) => entry.epoch)).toEqual([3, 4])
    expect(resumed.training.checkpointPaths).toEqual([`${prefix}-epoch-3`, `${prefix}-epoch-4`])
  })

  test('supports step-based checkpoints in workflow config', () => {
    setDevice('cpu')
    seed(8)

    const prefix = `/tmp/affon-transformers-workflow-step-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 1,
        batchSize: 2,
        lr: 0.01,
        evaluateInitialTrainLoss: false,
      },
      checkpoint: {
        prefix,
        everyNSteps: 2,
      },
      report: undefined,
    })

    expect(result.training.checkpointPaths).toEqual([`${prefix}-step-2`])
  })

  test('can run workflow training through a compiled model forward', () => {
    setDevice('cpu')
    seed(11)

    const prefix = `/tmp/affon-transformers-workflow-compiled-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 1,
        batchSize: 2,
        compileModelForward: true,
        lr: 0.01,
      },
      report: undefined,
    })

    expect(result.compiledModel).not.toBe(null)
    expect(result.summary.modelForwardMode).toBe(result.compiledModel ? 'graph' : 'eager')
    expect(result.training.steps > 0).toBeTruthy()
    expect(result.training.finalTrainLoss > 0).toBeTruthy()
  })

  test('writes a json forward report when step export is enabled', () => {
    setDevice('cpu')
    seed(21)

    const prefix = `/tmp/affon-transformers-workflow-export-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 1,
        batchSize: 2,
        compileModelForward: true,
        lr: 0.01,
      },
      report: {
        export: {
          dir: '/tmp',
          everyNSteps: 1,
          includeForward: true,
          includeBackward: false,
          prefix: `affon-export-${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
          runId: 'workflow-test',
          graphIdPrefix: 'workflow-test',
        },
      },
    })

    expect(result.summary.bundleExportPaths.length).toBe(result.training.steps)
    const forwardExportPath = result.summary.bundleExportPaths[result.summary.bundleExportPaths.length - 1]
    expect(forwardExportPath.endsWith('.json')).toBe(true)
    const bundleJSON = fs.readFileSync(forwardExportPath)
    expect(bundleJSON).toContain('"kind":"report"')
    expect(bundleJSON).toContain('"module_nodes"')
    expect(bundleJSON).toContain('"module_edges"')
  })

  test('can resume exactly from a mid-epoch workflow step checkpoint', () => {
    setDevice('cpu')
    seed(15)

    const prefix = `/tmp/affon-transformers-workflow-exact-resume-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\ngeneral kenobi\nhello world\nhello there\ngeneral kenobi\n')

    const baseConfig = {
      device: 'cpu' as const,
      tokenizer: {
        family: 'lookup' as const,
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
          general: 6,
          kenobi: 7,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned' as const,
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 2,
        batchSize: 2,
        lr: 0.01,
        shuffleSeed: 123,
        evaluateInitialTrainLoss: false,
      },
      report: undefined,
    }

    seed(21)
    const baseline = trainDecoderLMFromConfig({
      ...baseConfig,
      checkpoint: undefined,
    })

    seed(21)
    const interrupted = trainDecoderLMFromConfig({
      ...baseConfig,
      checkpoint: {
        prefix,
        everyNSteps: 2,
      },
    })

    expect(interrupted.training.checkpointPaths).toEqual([
      `${prefix}-step-2`,
      `${prefix}-step-4`,
      `${prefix}-step-6`,
      `${prefix}-step-8`,
      `${prefix}-step-10`,
      `${prefix}-step-12`,
      `${prefix}-step-14`,
      `${prefix}-step-16`,
      `${prefix}-step-18`,
      `${prefix}-step-20`,
      `${prefix}-step-22`,
      `${prefix}-step-24`,
      `${prefix}-step-26`,
    ])

    seed(21)
    const resumed = trainDecoderLMFromConfig({
      ...baseConfig,
      checkpoint: {
        prefix,
        resumeFrom: `${prefix}-step-2`,
      },
    })

    expect(resumed.training.steps).toBe(baseline.training.steps)
    expect(Math.abs(resumed.training.finalTrainLoss - baseline.training.finalTrainLoss) < 1e-6).toBe(true)
    expect(resumed.training.history).toEqual(baseline.training.history)
  })

  test('workflow resume with optimizer restore matches uninterrupted training, while restore-off diverges', () => {
    setDevice('cpu')
    seed(41)

    const prefix = `/tmp/affon-transformers-workflow-restore-regime-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\ngeneral kenobi\nhello world\nhello there\ngeneral kenobi\n')

    const baseConfig = {
      device: 'cpu' as const,
      tokenizer: {
        family: 'lookup' as const,
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
          general: 6,
          kenobi: 7,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned' as const,
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 2,
        batchSize: 2,
        lr: 0.01,
        shuffleSeed: 123,
        evaluateInitialTrainLoss: false,
      },
      report: undefined,
    }

    seed(42)
    const baseline = trainDecoderLMFromConfig({
      ...baseConfig,
      checkpoint: undefined,
    })

    seed(42)
    const interrupted = trainDecoderLMFromConfig({
      ...baseConfig,
      training: {
        ...baseConfig.training,
        epochs: 1,
      },
      checkpoint: {
        prefix,
        everyNEpochs: 1,
      },
    })

    expect(interrupted.training.checkpointPaths).toEqual([`${prefix}-epoch-1`])

    seed(42)
    const resumedExact = trainDecoderLMFromConfig({
      ...baseConfig,
      training: {
        ...baseConfig.training,
        epochs: 1,
      },
      checkpoint: {
        prefix,
        resumeFrom: `${prefix}-epoch-1`,
        restoreOptimizerState: true,
      },
    })

    expect(resumedExact.training.history.map((entry) => entry.epoch)).toEqual([2])
    expect(Math.abs(resumedExact.training.finalTrainLoss - baseline.training.finalTrainLoss) < 1e-6).toBe(true)
    expect(tensorTreeToPlain(resumedExact.model.state())).toEqual(tensorTreeToPlain(baseline.model.state()))

    seed(42)
    const resumedFreshOpt = trainDecoderLMFromConfig({
      ...baseConfig,
      training: {
        ...baseConfig.training,
        epochs: 1,
      },
      checkpoint: {
        prefix,
        resumeFrom: `${prefix}-epoch-1`,
        restoreOptimizerState: false,
      },
    })

    expect(resumedFreshOpt.training.history.map((entry) => entry.epoch)).toEqual([2])
    expect(Math.abs(resumedFreshOpt.training.finalTrainLoss - baseline.training.finalTrainLoss) > 1e-9).toBe(true)
    expect(JSON.stringify(tensorTreeToPlain(resumedFreshOpt.model.state())) === JSON.stringify(tensorTreeToPlain(baseline.model.state()))).toBe(false)
  })

  test('captures bounded runtime monitor samples and writes them to an artifact', () => {
    setDevice('cpu')
    seed(7)

    const prefix = `/tmp/affon-transformers-workflow-monitor-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    const monitorPath = `${prefix}-monitor.json`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 2,
        batchSize: 1,
        lr: 0.01,
      },
      report: {
        monitor: {
          everyBatches: 1,
          maxSamples: 3,
          path: monitorPath,
        },
      },
    })

    const monitorArtifact = JSON.parse(fs.readFileSync(monitorPath)) as {
      everyBatches: number
      maxSamples: number
      droppedSamples: number
      path: string | null
      snapshots: Array<{
        epoch: number
        batch: number
        batches: number
        step: number
        runtimeMetrics: Array<{
          scope: string
          name: string
          value: number
        }>
        phase: 'batch' | 'epoch'
        unattributedBytes: number
      }>
    }

    expect(result.monitor).not.toBe(null)
    expect(result.summary.monitor).not.toBe(null)
    expect(result.monitor?.maxSamples).toBe(3)
    expect((result.monitor?.snapshots.length ?? 0) <= 3).toBeTruthy()
    expect((result.monitor?.snapshots.length ?? 0) > 0).toBeTruthy()
    expect((result.monitor?.droppedSamples ?? 0) >= 0).toBeTruthy()
    expect(monitorArtifact.everyBatches).toBe(1)
    expect(monitorArtifact.maxSamples).toBe(3)
    expect(monitorArtifact.path).toBe(monitorPath)
    expect(monitorArtifact.snapshots.length <= 3).toBeTruthy()
    expect(monitorArtifact.snapshots.length > 0).toBeTruthy()
    expect(monitorArtifact.droppedSamples >= 0).toBeTruthy()
    expect(monitorArtifact.snapshots[0].epoch >= 1).toBeTruthy()
    expect(['batch', 'epoch'].includes(monitorArtifact.snapshots[0].phase)).toBeTruthy()
    expect(monitorArtifact.snapshots[0].batch >= 1).toBeTruthy()
    expect(monitorArtifact.snapshots[0].batches > 0).toBeTruthy()
    expect(monitorArtifact.snapshots[0].step > 0).toBeTruthy()
    const metalLive = monitorArtifact.snapshots[0].runtimeMetrics.find(
      (entry) => entry.scope === 'compute.storage' && entry.name === 'live_metal_bytes',
    )
    const cpuLive = monitorArtifact.snapshots[0].runtimeMetrics.find(
      (entry) => entry.scope === 'compute.storage' && entry.name === 'live_cpu_bytes',
    )
    const qjsUsed = monitorArtifact.snapshots[0].runtimeMetrics.find(
      (entry) => entry.scope === 'runtime.memory' && entry.name === 'qjs_heap_used_bytes',
    )
    const metalPool = monitorArtifact.snapshots[0].runtimeMetrics.find(
      (entry) => entry.scope === 'compute.memory' && entry.name === 'metal_pool_live_bytes',
    )
    expect((metalLive?.value ?? 0) >= 0).toBeTruthy()
    expect((cpuLive?.value ?? 0) >= 0).toBeTruthy()
    expect((qjsUsed?.value ?? 0) >= 0).toBeTruthy()
    expect((metalPool?.value ?? 0) >= 0).toBeTruthy()
    expect(monitorArtifact.snapshots[0].unattributedBytes >= 0).toBeTruthy()
  })

  test('keeps repeated-batch runtime memory bounded after warmup', () => {
    setDevice('cpu')
    seed(17)

    const prefix = `/tmp/affon-transformers-workflow-memory-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    const monitorPath = `${prefix}-monitor.json`
    fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\nhello there\n')

    const result = trainDecoderLMFromConfig({
      device: 'cpu',
      tokenizer: {
        family: 'lookup',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
          there: 5,
        },
        specialTokens: {
          bos: '<bos>',
          eos: '<eos>',
          unk: '<unk>',
        },
      },
      corpus: {
        path: corpusPath,
        addBos: true,
        addEos: true,
        seqLen: 3,
        stride: 1,
        validationSplit: 0.34,
      },
      model: {
        dModel: 32,
        numLayers: 1,
        numHeads: 4,
        hiddenDim: 64,
        causal: true,
        positional: 'learned',
        maxSeqLen: 8,
        tieEmbeddings: true,
      },
      training: {
        epochs: 3,
        batchSize: 1,
        lr: 0.01,
        maxTrainBatchesPerEpoch: 8,
        maxEvalBatches: 1,
      },
      report: {
        monitor: {
          everyBatches: 1,
          maxSamples: 8,
          path: monitorPath,
        },
      },
    })

    const snapshots = result.monitor?.snapshots ?? []
    expect(snapshots.length).toBe(8)

    for (const metricName of ['live_cpu_bytes', 'live_metal_bytes', 'metal_pool_live_bytes', 'qjs_heap_used_bytes']) {
      const values = snapshots
        .map((snapshot) => snapshot.runtimeMetrics.find(
          (metric) => metric.name === metricName,
        )?.value ?? 0)
        .filter((value) => value > 0)
      if (values.length < 4) continue

      const warmup = values[0]
      const peakAfterWarmup = Math.max(...values.slice(1))
      expect(peakAfterWarmup <= warmup * 2 + 8 * 1024 * 1024).toBeTruthy()
    }
  })
})
