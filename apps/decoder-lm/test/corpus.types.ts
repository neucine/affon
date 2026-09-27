import { createLookupTokenizer } from '../../../packages/@affon/tokenizers/src/index.ts'
import { createPackedTextCorpusFromConfig, loadTokenRows, pack_token_windows, saveTokenRows } from '../src/data/index.ts'
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

const packedFromConfig = createPackedTextCorpusFromConfig(tokenizer, {
  paths: ['train-a.txt', 'train-b.txt'],
  validationPaths: ['val.txt'],
  addBos: true,
  addEos: true,
  seqLen: 3,
  stride: 1,
  tokenCacheKey: 'train-v1',
  preprocess: {
    replacements: [{ from: '<unk>', to: '' }],
    normalizeWhitespace: true,
  },
})
assertType<IsExact<typeof packedFromConfig.validationRows, number[][]>>()
saveTokenRows('cache.json', packedFromConfig.trainRows, { key: 'train-v1' })
const loadedTokenRows = loadTokenRows('cache.json')
assertType<IsExact<typeof loadedTokenRows, number[][]>>()

const tokenWindows = pack_token_windows([
  [1, 2, 3, 4],
  [4, 5, 6, 7],
], {
  seqLen: 3,
  stride: 1,
})
assertType<IsExact<typeof tokenWindows, number[][]>>()
