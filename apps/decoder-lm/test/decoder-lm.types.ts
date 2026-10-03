import { DecoderModel, decoderProgram } from '../src/model.ts'
import { Session } from 'affon:compute'
import { pack_token_windows } from '../src/data/index.ts'
import { createLookupTokenizer } from '@affon/tokenizers'

import {
  evaluateDecoderLM,
  trainDecoderLM,
} from '../src/index.ts'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const tokenizer = createLookupTokenizer({
  '<bos>': 0,
  '<eos>': 1,
  '<unk>': 2,
  hello: 3,
}, {
  specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
})

const tokenWindows = pack_token_windows([
  tokenizer.encode('hello', { addBos: true, addEos: true }),
  tokenizer.encode('hello', { addBos: true, addEos: true }),
], {
  seqLen: 2,
  stride: 1,
})

const model = DecoderModel(tokenizer.vocabSize, 16, {
  numLayers: 1,
  numHeads: 4,
  hiddenDim: 64,
  causal: true,
  positional: 'learned',
  maxSeqLen: 64,
})
const session = new Session()
const state = session.initialize(decoderProgram(model, 2, 2))
const runtime = { model, session, state }

const evalLoss = evaluateDecoderLM(runtime, tokenWindows, { batchSize: 2 })
assertType<IsExact<typeof evalLoss, number>>()

const training = trainDecoderLM(runtime, tokenWindows, {
  seqLen: 2,
  batchSize: 2,
  epochs: 1,
})
assertType<IsExact<typeof training.steps, number>>()
assertType<IsExact<(typeof training.history)[number]['trainLoss'], number>>()
assertType<IsExact<typeof training.checkpoints, import('../src/index.ts').DecoderLMCheckpoint[]>>()
