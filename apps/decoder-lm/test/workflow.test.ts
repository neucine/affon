import fs from 'std:fs'
import { describe, expect, test } from 'std:test'

import { loadDecoderLMWorkflowConfig, trainDecoderLMFromConfig } from '../src/index.ts'
import type { DecoderLMWorkflowConfig } from '../src/workflow.ts'

function fixture(name: string): { config: DecoderLMWorkflowConfig; prefix: string; configPath: string } {
  const prefix = `/tmp/affon-decoder-program-${name}-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
  const corpusPath = `${prefix}.txt`
  const configPath = `${prefix}.json`
  fs.writeFileSync(corpusPath, 'hello world\nhello there\nhello world\n')
  const config: DecoderLMWorkflowConfig = {
    device: 'cpu',
    tokenizer: {
      family: 'lookup',
      vocab: { '<bos>': 0, '<eos>': 1, '<unk>': 2, hello: 3, world: 4, there: 5 },
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    },
    corpus: { path: corpusPath, addBos: true, addEos: true, seqLen: 3, stride: 1, validationSplit: 0.34 },
    model: { dModel: 8, numLayers: 1, numHeads: 2, hiddenDim: 16, causal: true, positional: 'learned', maxSeqLen: 8, seed: 7 },
    training: { epochs: 1, batchSize: 2, lr: 0.01, maxTrainBatchesPerEpoch: 1, maxEvalBatches: 1 },
  }
  fs.writeFileSync(configPath, JSON.stringify(config))
  return { config, prefix, configPath }
}

describe('@affon/decoder-lm Program workflow', () => {
  test('loads config and trains through the canonical Program model', () => {
    const value = fixture('train')
    const loaded = loadDecoderLMWorkflowConfig(value.configPath)
    const result = trainDecoderLMFromConfig(loaded)
    expect(result.summary.modelForwardMode).toBe('graph')
    expect(result.summary.modelForwardGraphRuntime).toBe('program')
    expect(result.summary.history.length).toBe(1)
    expect(Number.isFinite(result.summary.finalTrainLoss)).toBe(true)
    expect(result.model.device).toBe('cpu')
    result.model.dispose()
  })

  test('uses the model itself for the requested compiled forward path', () => {
    const value = fixture('compiled')
    value.config.training.compileModelForward = true
    const result = trainDecoderLMFromConfig(value.config)
    expect(result.compiledModel).toBe(result.model)
    expect(result.summary.modelForwardGraphLoweringAnalysis).toEqual({ lowerable: true })
    result.model.dispose()
  })

  test('writes an epoch checkpoint and resumes it', () => {
    const value = fixture('resume')
    value.config.checkpoint = { prefix: value.prefix, everyNEpochs: 1 }
    const first = trainDecoderLMFromConfig(value.config)
    expect(first.training.checkpointPaths).toEqual([`${value.prefix}-epoch-1`])
    first.model.dispose()

    const resumedConfig = { ...value.config, checkpoint: { prefix: value.prefix, resumeFrom: `${value.prefix}-epoch-1` } }
    const resumed = trainDecoderLMFromConfig(resumedConfig)
    expect(resumed.training.history[0].epoch).toBe(2)
    expect(resumed.training.steps > first.training.steps).toBe(true)
    resumed.model.dispose()
  })

  test('generates samples and captures bounded runtime snapshots', () => {
    const value = fixture('report')
    value.config.report = {
      sample: { prompt: 'hello', max_new_tokens: 1, everyNEpochs: 1 },
      monitor: { everyBatches: 1, maxSamples: 2 },
    }
    const result = trainDecoderLMFromConfig(value.config)
    expect(result.samples.length).toBe(1)
    expect(result.samples[0].generatedIds.length).toBe(result.samples[0].promptIds.length + 1)
    expect(result.monitor?.snapshots.length).toBe(2)
    result.model.dispose()
  })
})
