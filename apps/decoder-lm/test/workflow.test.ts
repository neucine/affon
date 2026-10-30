import fs from 'std:fs'
import { describe, expect, test } from 'std:test'
import { Tensor, program } from 'affon:compute'
import checkpoint from 'affon:checkpoint'

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
    expect(result.summary.program.name).toBe('decoder_lm')
    expect(result.summary.program.parameters > 0).toBe(true)
    expect(result.summary.history.length).toBe(1)
    expect(Number.isFinite(result.summary.finalTrainLoss)).toBe(true)
    expect(result.session.device).toBe('cpu')
    result.state.dispose(); result.session.dispose()
  })

  test('uses Session compilation for the requested Program', () => {
    const value = fixture('compiled')
    const result = trainDecoderLMFromConfig(value.config)
    const inference = program('workflow_inference', p => result.model({
      token_ids: p.argument('token_ids', Tensor.i64([1, 3], { axes: ['batch', 'token'] })),
    }, 'decoder'))
    expect(result.session.compile(inference).program.inspect().kind).toBe('authored')
    expect(result.summary.program.parameters).toBe(inference.inspect().parameters.length)
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

  test('resumed training matches continuous model and optimizer state', () => {
    const continuousFixture = fixture('continuous-state')
    continuousFixture.config.training.epochs = 2
    continuousFixture.config.checkpoint = { prefix: continuousFixture.prefix, everyNEpochs: 1 }
    const continuous = trainDecoderLMFromConfig(continuousFixture.config)
    const continuousPath = `${continuousFixture.prefix}-epoch-2`
    expect(continuous.training.steps).toBe(2)
    continuous.state.dispose(); continuous.session.dispose()

    const interruptedFixture = fixture('interrupted-state')
    interruptedFixture.config.checkpoint = { prefix: interruptedFixture.prefix, everyNEpochs: 1 }
    const interrupted = trainDecoderLMFromConfig(interruptedFixture.config)
    const firstPath = `${interruptedFixture.prefix}-epoch-1`
    expect(interrupted.training.steps).toBe(1)
    interrupted.state.dispose(); interrupted.session.dispose()


    const resumed = trainDecoderLMFromConfig({
      ...interruptedFixture.config,
      checkpoint: { prefix: interruptedFixture.prefix, everyNEpochs: 1, resumeFrom: firstPath },
    })
    expect(resumed.training.steps).toBe(continuous.training.steps)
    resumed.state.dispose(); resumed.session.dispose()

    const expected = checkpoint.loadBundle(continuousPath)
    const actual = checkpoint.loadBundle(`${interruptedFixture.prefix}-epoch-2`)
    const firstContinuous = checkpoint.loadBundle(`${continuousFixture.prefix}-epoch-1`)
    const firstInterrupted = checkpoint.loadBundle(firstPath)
    try {
      for (const name of Object.keys(firstContinuous.state)) {
        expect(firstInterrupted.state[name].to_array()).toEqual(firstContinuous.state[name].to_array())
      }
      expect(firstInterrupted.manifest.optimizerScalars).toEqual(firstContinuous.manifest.optimizerScalars)
      for (const name of Object.keys(firstContinuous.tensorGroups.optimizer)) {
        expect(firstInterrupted.tensorGroups.optimizer[name].to_array()).toEqual(firstContinuous.tensorGroups.optimizer[name].to_array())
      }
      expect(Object.keys(actual.state).sort()).toEqual(Object.keys(expected.state).sort())
      for (const name of Object.keys(expected.state)) {
        expect(actual.state[name].to_array()).toEqual(expected.state[name].to_array())
      }
      expect(actual.manifest.step).toBe(expected.manifest.step)
      expect(actual.manifest.optimizerKind).toBe(expected.manifest.optimizerKind)
      expect(actual.manifest.optimizerScalars).toEqual(expected.manifest.optimizerScalars)
      const expectedOptimizer = expected.tensorGroups.optimizer
      const actualOptimizer = actual.tensorGroups.optimizer
      expect(Object.keys(actualOptimizer).sort()).toEqual(Object.keys(expectedOptimizer).sort())
      expect(Object.keys(expectedOptimizer).length > 0).toBe(true)
      for (const name of Object.keys(expectedOptimizer)) {
        expect(actualOptimizer[name].to_array()).toEqual(expectedOptimizer[name].to_array())
      }
    } finally {
      for (const bundle of [expected, actual, firstContinuous, firstInterrupted]) {
        for (const tensor of Object.values(bundle.state)) tensor.dispose()
        for (const group of Object.values(bundle.tensorGroups)) for (const tensor of Object.values(group)) tensor.dispose()
      }
    }
  })

  test('resumes a saved step inside an epoch without repeating the first batch', () => {
    const value = fixture('mid-epoch')
    value.config.training.maxTrainBatchesPerEpoch = 2
    value.config.checkpoint = { prefix: value.prefix, everyNEpochs: 1, everyNSteps: 1 }
    const continuous = trainDecoderLMFromConfig(value.config)
    expect(continuous.training.steps).toBe(2)
    continuous.state.dispose(); continuous.session.dispose()

    const resumed = trainDecoderLMFromConfig({
      ...value.config,
      checkpoint: { prefix: `${value.prefix}-resumed`, everyNEpochs: 1, resumeFrom: `${value.prefix}-step-1` },
    })
    expect(resumed.training.steps).toBe(2)
    expect(resumed.training.history[0].epoch).toBe(1)
    resumed.state.dispose(); resumed.session.dispose()

    const expected = checkpoint.loadBundle(`${value.prefix}-epoch-1`)
    const actual = checkpoint.loadBundle(`${value.prefix}-resumed-epoch-1`)
    try {
      expect(actual.manifest.optimizerScalars).toEqual(expected.manifest.optimizerScalars)
      expect(actual.manifest.trainLoss).toBe(expected.manifest.trainLoss)
      for (const name of Object.keys(expected.state)) {
        expect(actual.state[name].to_array()).toEqual(expected.state[name].to_array())
      }
      for (const name of Object.keys(expected.tensorGroups.optimizer)) {
        expect(actual.tensorGroups.optimizer[name].to_array()).toEqual(expected.tensorGroups.optimizer[name].to_array())
      }
    } finally {
      for (const bundle of [expected, actual]) {
        for (const tensor of Object.values(bundle.state)) tensor.dispose()
        for (const group of Object.values(bundle.tensorGroups)) for (const tensor of Object.values(group)) tensor.dispose()
      }
    }
  })

  test('rejects a corrupt checkpoint and resumes after it is repaired', () => {
    const value = fixture('corrupt-resume')
    value.config.checkpoint = { prefix: value.prefix, everyNEpochs: 1 }
    const first = trainDecoderLMFromConfig(value.config)
    first.state.dispose(); first.session.dispose()
    const resumeFrom = `${value.prefix}-epoch-1`
    const manifestPath = `${resumeFrom}.json`
    const original = fs.readFileSync(manifestPath)
    const resumeConfig = { ...value.config, checkpoint: { prefix: `${value.prefix}-resumed`, resumeFrom } }
    try {
      fs.writeFileSync(manifestPath, '{broken')
      expect(() => trainDecoderLMFromConfig(resumeConfig)).toThrow()
    } finally { fs.writeFileSync(manifestPath, original) }

    const resumed = trainDecoderLMFromConfig(resumeConfig)
    expect(resumed.training.history[0].epoch).toBe(2)
    expect(resumed.training.steps).toBe(2)
    resumed.state.dispose(); resumed.session.dispose()
  })

  test('switching optimizer kind restores weights with a fresh optimizer step', () => {
    const value = fixture('optimizer-switch')
    value.config.checkpoint = { prefix: value.prefix, everyNEpochs: 1 }
    const first = trainDecoderLMFromConfig(value.config)
    first.state.dispose(); first.session.dispose()
    const messages: string[] = []
    const resumed = trainDecoderLMFromConfig({
      ...value.config,
      training: { ...value.config.training, lr: 0, optimizer: { kind: 'adamw', weightDecay: 0 } },
      checkpoint: { prefix: `${value.prefix}-adamw`, everyNEpochs: 1, resumeFrom: `${value.prefix}-epoch-1` },
    }, { onStatus: message => messages.push(message) })
    expect(messages.some(message => message.includes('restoring model weights only with fresh optimizer state'))).toBe(true)
    expect(resumed.training.steps).toBe(2)
    expect(resumed.state.optimizer_state.$step).toBe(1)
    const saved = checkpoint.loadBundle(`${value.prefix}-epoch-1`)
    try {
      for (const [name, tensor] of Object.entries(saved.state)) {
        expect(resumed.state.parameters[name].to_array()).toEqual(tensor.to_array())
      }
    } finally {
      for (const tensor of Object.values(saved.state)) tensor.dispose()
      for (const group of Object.values(saved.tensorGroups)) for (const tensor of Object.values(group)) tensor.dispose()
    }
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
