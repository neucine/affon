import { describe, test, expect } from 'std:test'
import dataset, { read, DataLoader } from 'affon:dataset'
import { createHFTokenizerFromJSON, createLookupTokenizer } from 'tokenizers'
import { captureError } from '../../support/errors.ts'

describe('dataset contracts', () => {
  test('sample validates non-negative integer input', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')

    for (const n of [-1, 1.5, NaN]) {
      const err = captureError(() => ds.sample(n))
      expect(err).toBeInstanceOf(AffonError)
      expect(err.code).toBe('invalid_arg')
      expect(err.message).toContain('sample() requires a non-negative integer')
    }
  })

  test('DataLoader validates positive integer batch size', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv').select('sepal_length')

    for (const batchSize of [0, -1, 1.5]) {
      const err = captureError(() => new DataLoader(ds, { batchSize: batchSize as any }))
      expect(err).toBeInstanceOf(Error)
      expect(err.message).toContain('DataLoader.batchSize must be a positive integer')
    }
  })

  test('split validates ratio count', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv')
    const err = captureError(() => ds.split(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
    expect(err).toBeInstanceOf(AffonError)
    expect(err.code).toBe('invalid_arg')
    expect(err.message).toContain('split() supports at most 8 ratios')
  })

  test('numeric csv parse rejects invalid numeric cells', () => {
    const err = captureError(() => read('test/e2e/dataset/fixtures/invalid-numeric.csv').toTensor())
    expect(err).toBeInstanceOf(AffonError)
    expect(err.code).toBe('invalid_arg')
    expect(err.message).toContain('InvalidNumericValue')
  })

  test('tensor export dtype is limited to f32 or f64', () => {
    const ds = read('test/e2e/dataset/fixtures/iris.csv').select('sepal_length', 'sepal_width')

    for (const dtype of ['i64', 'bad-dtype']) {
      const err = captureError(() => ds.toTensor({ dtype: dtype as any }))
      expect(err).toBeInstanceOf(AffonError)
      expect(err.code).toBe('invalid_arg')
      expect(err.message).toContain("toTensor(): dtype must be 'f32' or 'f64'")
    }
  })

  test('text dataset validates positive integer batch size', () => {
    const ds = dataset.text.read('test/e2e/dataset/fixtures/lines.txt')

    for (const batchSize of [0, -1, 1.5]) {
      const err = captureError(() => new dataset.text.DataLoader(ds, { batchSize: batchSize as any }))
      expect(err).toBeInstanceOf(Error)
      expect(err.message).toContain('TextDataLoader.batchSize must be a positive integer')
    }
  })

  test('lookup tokenizer uses unk token when configured', () => {
    const tok = createLookupTokenizer(
      { '<unk>': 0, alpha: 1 },
      { specialTokens: { unk: '<unk>' } }
    )
    expect(tok.encode('alpha beta')).toEqual([1, 0])
  })

  test('padded text loader validates pad and length options', () => {
    const tok = createLookupTokenizer(
      { '<pad>': 0, '<unk>': 1, alpha: 2 },
      { specialTokens: { pad: '<pad>', unk: '<unk>' } }
    )
    const ds = dataset.text.read('test/e2e/dataset/fixtures/lines.txt').encode(tok)

    for (const opts of [
      { batchSize: 0, padId: 0 },
      { batchSize: 1, padId: -1 },
      { batchSize: 1, padId: 0, maxLength: 0 },
    ]) {
      const err = captureError(() => ds.paddedLoader(opts as any))
      expect(err).toBeInstanceOf(Error)
    }
  })

  test('tensorized text loader shares padded option validation', () => {
    const tok = createLookupTokenizer(
      { '<pad>': 0, '<unk>': 1, alpha: 2 },
      { specialTokens: { pad: '<pad>', unk: '<unk>' } }
    )
    const ds = dataset.text.read('test/e2e/dataset/fixtures/lines.txt').encode(tok)
    const err = captureError(() => ds.tensorLoader({ batchSize: 1, padId: -1 as any }))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('PaddedTextDataLoader.padId must be a non-negative integer')
  })

  test('readDelimited validates row width', () => {
    const err = captureError(() => dataset.text.readDelimited('test/e2e/dataset/fixtures/lines.txt', {
      delimiter: '\t',
      columns: ['text', 'label'],
    }))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('readDelimited() requires at least 2 delimited fields per row')
  })

  test('readDelimited requires columns', () => {
    const err = captureError(() => dataset.text.readDelimited('test/e2e/dataset/fixtures/labeled-lines.tsv', {
      delimiter: '\t',
      columns: [] as any,
    }))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('readDelimited() requires a non-empty columns array')
  })

  test('record tensor loader validates selected fields and numeric targets', () => {
    let err = captureError(() => dataset.text.readDelimited('test/e2e/dataset/fixtures/labeled-lines.tsv', {
      delimiter: '\t',
      columns: ['text', 'label'],
    }).paddedLoader({ batchSize: 1, padId: 0 }))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('paddedLoader() requires selected input or target fields')

    err = captureError(() => {
      const loader = dataset.text.readDelimited('test/e2e/dataset/fixtures/labeled-lines.tsv', {
      delimiter: '\t',
      columns: ['text', 'label'],
      }).input('text').tensorLoader({ batchSize: 1, padId: 0 })
      Array.from(loader)
    })
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('tensorLoader() can only tensorize numeric selected fields')
  })

  test('label encoder validates unknown labels and ids', () => {
    const labels = dataset.text.labelEncoder({ pos: 0 })
    let err = captureError(() => labels.encode('neg'))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('Unknown label: neg')

    err = captureError(() => labels.decode(1))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('Unknown label id: 1')

    err = captureError(() => labels.encodeMany(['pos', 'neg']))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('Unknown label: neg')
  })

  test('hf tokenizer adapter rejects unsupported model families', () => {
    const err = captureError(() => createHFTokenizerFromJSON({
      model: {
        type: 'Unigram',
        vocab: { hello: 0 },
      },
    }).encode('hello'))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('Unsupported HF tokenizer model type: Unigram')
  })
})
