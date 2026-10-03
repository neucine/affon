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
    model: { dModel: 8, numLayers: 1, numHeads: 2, hiddenDim: 16, causal: true, positional: 'learned', maxSeqLen: 8 },
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
    expect(result.summary.program.name).toBe('decoder_model')
    expect(result.summary.program.parameters > 0).toBe(true)
    expect(result.summary.history.length).toBe(1)
    expect(Number.isFinite(result.summary.finalTrainLoss)).toBe(true)
    expect(result.session.device).toBe('cpu')
    result.state.dispose(); result.session.dispose()
  })

  test('uses Session compilation for the requested Program', () => {
    const value = fixture('compiled')
    const result = trainDecoderLMFromConfig(value.config)
    expect(result.session.compile(result.model.forward(1, 3)).program.inspect().kind).toBe('authored')
    expect(result.summary.program.provenance).toBe(result.model.forward(2, 3).provenance)
    result.state.dispose(); result.session.dispose()
  })

  test('writes an epoch checkpoint and resumes it', () => {
    const value = fixture('resume')
    value.config.checkpoint = { prefix: value.prefix, everyNEpochs: 1 }
    const first = trainDecoderLMFromConfig(value.config)
    expect(first.training.checkpointPaths).toEqual([`${value.prefix}-epoch-1`])
    first.state.dispose(); first.session.dispose()

    const resumedConfig = { ...value.config, checkpoint: { prefix: value.prefix, resumeFrom: `${value.prefix}-epoch-1` } }
    const resumed = trainDecoderLMFromConfig(resumedConfig)
    expect(resumed.training.history[0].epoch).toBe(2)
    expect(resumed.training.steps > first.training.steps).toBe(true)
    resumed.state.dispose(); resumed.session.dispose()
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
    result.state.dispose(); result.session.dispose()
  })
})
