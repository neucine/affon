import { describe, test, expect } from 'std:test'
import dataset from 'affon:dataset'
import {
  createHFTokenizerFromFile,
  createHFTokenizerFromJSON,
  createSentencePieceTokenizer,
} from '../../../packages/@affon/tokenizers/src/index.ts'

describe('dataset tokenizer adapters', () => {
  test('hf tokenizer adapter satisfies the generic text tokenizer interface', () => {
    const tok = createHFTokenizerFromFile('test/e2e/dataset/fixtures/hf-wordlevel-tokenizer.json')
    expect(tok.encode('hello world')).toEqual([1, 2])
    expect(tok.encode('hello unknown')).toEqual([1, 0])
    expect(tok.decode([1, 2])).toBe('hello world')

    const ds = dataset.text.read('test/e2e/dataset/fixtures/lines-hf.txt').encode(tok)
    expect(ds.toArray()).toEqual([[1, 2], [3]])
  })

  test('hf BPE tokenizer adapter supports merged subwords', () => {
    const tok = createHFTokenizerFromFile('test/e2e/dataset/fixtures/hf-bpe-tokenizer.json')
    expect(tok.encode('hello world')).toEqual([9, 17])
    expect(tok.decode([9, 17])).toBe('hello world')
  })

  test('hf WordPiece tokenizer adapter supports continuing subwords', () => {
    const tok = createHFTokenizerFromJSON({
      model: {
        type: 'WordPiece',
        vocab: {
          '[UNK]': 0,
          hello: 1,
          play: 2,
          '##ing': 3,
        },
        unk_token: '[UNK]',
        continuing_subword_prefix: '##',
      },
      pre_tokenizer: { type: 'Whitespace' },
    })
    expect(tok.encode('hello playing')).toEqual([1, 2, 3])
    expect(tok.decode([1, 2, 3])).toBe('hello playing')
  })

  test('SentencePiece tokenizer adapter satisfies the generic text tokenizer interface', () => {
    const tok = createSentencePieceTokenizer({
      pieces: [
        { piece: '<unk>', type: 'unknown' },
        { piece: '<s>', type: 'control' },
        { piece: '</s>', type: 'control' },
        { piece: '▁hello' },
        { piece: '▁world' },
        { piece: '▁play' },
        { piece: 'ing' },
      ],
      unkId: 0,
      bosId: 1,
      eosId: 2,
    })

    expect(tok.encode('hello playing', { addBos: true, addEos: true })).toEqual([1, 3, 5, 6, 2])
    expect(tok.decode([1, 3, 5, 6, 2], { skipSpecialTokens: true })).toBe('hello playing')
    expect(tok.vocabSize).toBe(7)
    expect(tok.tokenId('▁play')).toBe(5)
  })
})
