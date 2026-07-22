import { describe, expect, test } from 'std:test'

import {
  createHFTokenizerFromJSON,
  createLookupTokenizer,
  createSentencePieceTokenizer,
  tokenizerArtifact,
} from '../src/index.ts'

describe('@affon/tokenizers', () => {
  test('creates lookup tokenizers with special tokens', () => {
    const tokenizer = createLookupTokenizer({
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
      world: 4,
    }, {
      specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    })

    expect(tokenizer.family).toBe('lookup')
    expect(tokenizer.vocabSize).toBe(5)
    expect(tokenizer.encode('hello world', { addBos: true, addEos: true })).toEqual([0, 3, 4, 1])
    expect(tokenizer.decode([0, 3, 4, 1], { skipSpecialTokens: true })).toBe('hello world')
    expect(tokenizer.encode('missing')).toEqual([2])
  })

  test('wraps Hugging Face tokenizer JSON compatibility', () => {
    const tokenizer = createHFTokenizerFromJSON({
      model: {
        type: 'WordLevel',
        vocab: {
          '<bos>': 0,
          '<eos>': 1,
          '<unk>': 2,
          '<pad>': 5,
          hello: 3,
          world: 4,
        },
        unk_token: '<unk>',
      },
      pre_tokenizer: { type: 'Whitespace' },
      added_tokens: [
        { id: 0, content: '<bos>', special: true },
        { id: 1, content: '<eos>', special: true },
        { id: 2, content: '<unk>', special: true },
        { id: 5, content: '<pad>', special: true },
      ],
    })

    expect(tokenizer.family).toBe('hf-tokenizer-json')
    expect(tokenizer.vocabSize).toBe(6)
    expect(tokenizer.allSpecialTokenIds).toEqual([0, 1, 2, 5])
    expect(tokenizer.encode('hello world', { addBos: true, addEos: true })).toEqual([0, 3, 4, 1])
  })

  test('wraps SentencePiece compatibility', () => {
    const tokenizer = createSentencePieceTokenizer({
      pieces: [
        { piece: '<unk>', type: 'unknown' },
        { piece: '<s>', type: 'control' },
        { piece: '</s>', type: 'control' },
        { piece: '▁hello' },
        { piece: '▁world' },
      ],
      unkId: 0,
      bosId: 1,
      eosId: 2,
    })

    expect(tokenizer.family).toBe('sentencepiece')
    expect(tokenizer.vocabSize).toBe(5)
    expect(tokenizer.encode('hello world', { addBos: true, addEos: true })).toEqual([1, 3, 4, 2])
    expect(tokenizer.decode([1, 3, 4, 2], { skipSpecialTokens: true })).toBe('hello world')
  })

  test('describes tokenizer artifacts', () => {
    const artifact = tokenizerArtifact('hf-tokenizer-json', ['tokenizer.json'])

    expect(artifact.family).toBe('hf-tokenizer-json')
    expect(artifact.files).toEqual(['tokenizer.json'])
  })
})
