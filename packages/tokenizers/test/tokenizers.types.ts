import {
  createHFTokenizerFromJSON,
  createLookupTokenizer,
  createSentencePieceTokenizer,
  tokenizerArtifact,
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
assertType<IsExact<typeof tokenizer.specialTokenIds.bos, number | undefined>>()
assertType<IsExact<typeof tokenizer.vocabSize, number>>()
const eosTokenId = tokenizer.tokenId('<eos>')
assertType<IsExact<typeof eosTokenId, number | undefined>>()
const encoded = tokenizer.encode('hello', { addBos: true })
assertType<IsExact<typeof encoded, number[]>>()
const decoded = tokenizer.decode(encoded, { skipSpecialTokens: true })
assertType<IsExact<typeof decoded, string>>()

const artifact = tokenizerArtifact('hf-tokenizer-json', ['tokenizer.json'])
assertType<IsExact<typeof artifact.family, 'lookup' | 'hf-tokenizer-json' | 'sentencepiece' | 'tiktoken'>>()

const hfTokenizer = createHFTokenizerFromJSON({
  model: {
    type: 'WordLevel',
    vocab: {
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      hello: 3,
    },
    unk_token: '<unk>',
  },
  pre_tokenizer: { type: 'Whitespace' },
})
const hfEncoded = hfTokenizer.encode('hello', { addBos: true })
assertType<IsExact<typeof hfEncoded, number[]>>()

const hfBpeTokenizer = createHFTokenizerFromJSON({
  model: {
    type: 'BPE',
    vocab: {
      '<bos>': 0,
      '<eos>': 1,
      '<unk>': 2,
      h: 3,
      '##i': 4,
      hi: 5,
    },
    unk_token: '<unk>',
    merges: [['h', '##i']],
    continuing_subword_prefix: '##',
  },
  pre_tokenizer: { type: 'Whitespace' },
})
const hfBpeEncoded = hfBpeTokenizer.encode('hi', { addBos: true })
assertType<IsExact<typeof hfBpeEncoded, number[]>>()

const hfByteLevelTokenizer = createHFTokenizerFromJSON({
  model: {
    type: 'BPE',
    vocab: {
      '<unk>': 0,
      hi: 1,
      'Ġt': 2,
      h: 3,
      e: 4,
      'Ġth': 5,
      'Ġthe': 6,
    },
    unk_token: '<unk>',
    merges: [['Ġt', 'h'], ['Ġth', 'e']],
  },
  pre_tokenizer: { type: 'ByteLevel' },
})
const hfByteLevelEncoded = hfByteLevelTokenizer.encode('hi the')
assertType<IsExact<typeof hfByteLevelEncoded, number[]>>()

const hfWordPieceTokenizer = createHFTokenizerFromJSON({
  model: {
    type: 'WordPiece',
    vocab: {
      '<unk>': 0,
      play: 1,
      '##ing': 2,
    },
    unk_token: '<unk>',
    continuing_subword_prefix: '##',
  },
  pre_tokenizer: { type: 'Whitespace' },
})
const hfWordPieceEncoded = hfWordPieceTokenizer.encode('playing')
assertType<IsExact<typeof hfWordPieceEncoded, number[]>>()

const sentencePieceTokenizer = createSentencePieceTokenizer({
  pieces: [
    { piece: '<unk>', type: 'unknown' },
    { piece: '▁hello' },
    { piece: 'world' },
  ],
  unkId: 0,
})
const sentencePieceEncoded = sentencePieceTokenizer.encode('hello world')
assertType<IsExact<typeof sentencePieceEncoded, number[]>>()
