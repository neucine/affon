import fs from 'std:fs'
import { text } from 'affon:dataset/text.ts'
import { describe, expect, test } from 'std:test'
import { createLookupTokenizer } from '../../../packages/@affon/tokenizers/src/index.ts'

import {
  createPackedTextCorpusFromConfig,
  loadTokenRows,
  saveTokenRows,
} from '../src/data/corpus.ts'

describe('decoder-lm corpus', () => {
  test('uses dataset.text for line and paragraph splitting', () => {
    expect(text.fromString('a\n\nb\n', { mode: 'line' }).toArray()).toEqual(['a', 'b'])
    expect(text.fromString('a\n\nb\n\nc', { mode: 'paragraph' }).toArray()).toEqual(['a', 'b', 'c'])
  })

  test('uses dataset.text directly for tokenized rows and windows', () => {
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

    const rows = text.rows(['hello world', 'hello there']).encode(tokenizer, {
      addBos: true,
      addEos: true,
    }).toArray()
    expect(rows).toEqual([
      [0, 3, 4, 1],
      [0, 3, 5, 1],
    ])

    const windows = text.encoded(rows).window({
      seqLen: 3,
      stride: 1,
      joinWithTokenId: tokenizer.specialTokenIds.eos,
    }).toArray()

    expect(windows.length > 0).toBeTruthy()
  })

  test('loads packed corpora from multiple files and token caches', () => {
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

    const prefix = `/tmp/affon-transformers-corpus-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const trainA = `${prefix}-train-a.txt`
    const trainB = `${prefix}-train-b.txt`
    const valFile = `${prefix}-val.txt`
    const cacheFile = `${prefix}-train-cache.json`
    fs.writeFileSync(trainA, 'hello world\n')
    fs.writeFileSync(trainB, 'hello there\n')
    fs.writeFileSync(valFile, 'hello world\n')

    const fromFiles = createPackedTextCorpusFromConfig(tokenizer, {
      paths: [trainA, trainB],
      validationPaths: [valFile],
      addBos: true,
      addEos: true,
      seqLen: 3,
      stride: 1,
    })
    saveTokenRows(cacheFile, fromFiles.trainRows)
    expect(loadTokenRows(cacheFile)).toEqual(fromFiles.trainRows)

    const fromCache = createPackedTextCorpusFromConfig(tokenizer, {
      tokenCachePath: cacheFile,
      validationPaths: [valFile],
      addBos: true,
      addEos: true,
      seqLen: 3,
      stride: 1,
    })
    expect(fromCache.trainRows).toEqual(fromFiles.trainRows)
    expect(fromCache.trainWindows).toEqual(fromFiles.trainWindows)
  })

  test('ignores token caches when the configured cache key changes', () => {
    const tokenizer = createLookupTokenizer({
      '<eos>': 0,
      hello: 1,
      world: 2,
      there: 3,
    }, {
      specialTokens: { eos: '<eos>' },
    })

    const prefix = `/tmp/affon-transformers-corpus-cache-key-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    const cacheFile = `${prefix}.cache.json`
    fs.writeFileSync(corpusPath, 'hello world\n')
    saveTokenRows(cacheFile, [[3, 0]], { key: 'old' })

    const packed = createPackedTextCorpusFromConfig(tokenizer, {
      path: corpusPath,
      tokenCachePath: cacheFile,
      tokenCacheKey: 'new',
      addEos: true,
      seqLen: 2,
      stride: 1,
    })

    expect(packed.trainRows).toEqual([[1, 2, 0]])
  })

  test('preprocesses text rows before tokenization', () => {
    const tokenizer = createLookupTokenizer({
      '<eos>': 0,
      hello: 1,
      world: 2,
    }, {
      specialTokens: { eos: '<eos>' },
    })

    const prefix = `/tmp/affon-transformers-corpus-preprocess-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, 'hello <unk> world\n')

    const packed = createPackedTextCorpusFromConfig(tokenizer, {
      path: corpusPath,
      addEos: true,
      seqLen: 2,
      stride: 1,
      preprocess: {
        replacements: [{ from: '<unk>', to: '' }],
        normalizeWhitespace: true,
      },
    })

    expect(packed.trainRows).toEqual([[1, 2, 0]])
  })

  test('uses dataset.text line reading for file corpora', () => {
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

    const prefix = `/tmp/affon-transformers-corpus-lines-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, ' hello world \n\nhello there\n')

    const packed = createPackedTextCorpusFromConfig(tokenizer, {
      path: corpusPath,
      addBos: true,
      addEos: true,
      seqLen: 3,
      stride: 1,
      split: {
        mode: 'line',
        trim: true,
        skipEmpty: true,
      },
    })

    expect(packed.trainRows).toEqual([
      [0, 3, 4, 1],
      [0, 3, 5, 1],
    ])
  })

  test('uses dataset.text paragraph reading for paragraph corpora', () => {
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

    const prefix = `/tmp/affon-transformers-corpus-paragraphs-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const corpusPath = `${prefix}.txt`
    fs.writeFileSync(corpusPath, ' hello world \n\nhello there\n')

    const packed = createPackedTextCorpusFromConfig(tokenizer, {
      path: corpusPath,
      addBos: true,
      addEos: true,
      seqLen: 3,
      stride: 1,
      split: {
        mode: 'paragraph',
        trim: true,
        skipEmpty: true,
      },
    })

    expect(packed.trainRows).toEqual([
      [0, 3, 4, 1],
      [0, 3, 5, 1],
    ])
  })
})
