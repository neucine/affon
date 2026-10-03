import { describe, test, expect, values } from 'std:test'
import { text } from 'affon:dataset/text.ts'
import { writeFileSync } from 'std:fs'
import { Session } from 'affon:compute'
import { createLookupTokenizer } from '../../../packages/@affon/tokenizers/src/index.ts'

describe('dataset text', () => {
  test('text namespace reads line-oriented corpora', () => {
    const ds = text.read('test/e2e/dataset/fixtures/lines.txt')
    expect(ds.length).toBe(4)
    expect(ds.toArray()).toEqual(['alpha', 'beta', 'gamma', 'delta'])
  })

  test('text dataset supports trim, filtering, mapping, split, and batching', () => {
    const ds = text.read('test/e2e/dataset/fixtures/lines-spaced.txt', { trim: true })
      .filter((line) => line !== 'skip')
      .map((line, index) => `${index}:${line}`)

    expect(ds.toArray()).toEqual(['0:alpha', '1:beta', '2:gamma', '3:delta'])

    const [train, testSplit] = ds.split(0.5)
    expect(train.toArray()).toEqual(['0:alpha', '1:beta'])
    expect(testSplit.toArray()).toEqual(['2:gamma', '3:delta'])

    const loader = ds.loader({ batchSize: 3 })
    const batches: string[][] = []
    for (const batch of loader) batches.push(batch)
    expect(loader.length).toBe(2)
    expect(batches).toEqual([
      ['0:alpha', '1:beta', '2:gamma'],
      ['3:delta'],
    ])
  })

  test('text lookup tokenizer encodes corpora and batches encoded rows', () => {
    const tok = createLookupTokenizer(
      { '<bos>': 0, '<eos>': 1, '<unk>': 2, alpha: 3, beta: 4, gamma: 5, delta: 6 },
      { specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' } }
    )
    const ds = text.read('test/e2e/dataset/fixtures/lines.txt').encode(tok, { addBos: true, addEos: true })

    expect(ds.toArray()).toEqual([
      [0, 3, 1],
      [0, 4, 1],
      [0, 5, 1],
      [0, 6, 1],
    ])

    expect(tok.decode([0, 3, 1], { skipSpecialTokens: true })).toBe('alpha')

    const loader = ds.loader({ batchSize: 2 })
    const batches: number[][][] = []
    for (const batch of loader) batches.push(batch)
    expect(loader.length).toBe(2)
    expect(batches).toEqual([
      [[0, 3, 1], [0, 4, 1]],
      [[0, 5, 1], [0, 6, 1]],
    ])
  })

  test('encoded text dataset exposes padded batches with attention masks', () => {
    const tok = createLookupTokenizer(
      { '<pad>': 0, '<unk>': 1, alpha: 2, beta: 3, gamma: 4, delta: 5 },
      { specialTokens: { pad: '<pad>', unk: '<unk>' } }
    )
    const ds = text.read('test/e2e/dataset/fixtures/phrases.txt').encode(tok)

    const loader = ds.paddedLoader({ batchSize: 2, padId: 0 })
    const batches: Array<{ inputIds: number[][]; attentionMask: number[][] }> = []
    for (const batch of loader) batches.push(batch)

    expect(loader.length).toBe(2)
    expect(batches).toEqual([
      {
        inputIds: [[2, 3], [4, 0]],
        attentionMask: [[1, 1], [1, 0]],
      },
      {
        inputIds: [[5]],
        attentionMask: [[1]],
      },
    ])
  })

  test('encoded text dataset exports tensorized padded batches', () => {
    const session = new Session({ device: 'cpu' })
    const tok = createLookupTokenizer(
      { '<pad>': 0, '<unk>': 1, alpha: 2, beta: 3, gamma: 4, delta: 5 },
      { specialTokens: { pad: '<pad>', unk: '<unk>' } }
    )
    const ds = text.read('test/e2e/dataset/fixtures/phrases.txt').encode(tok)

    const loader = ds.tensorLoader({ batchSize: 2, padId: 0, session })
    const batches: Array<{ inputIds: any; attentionMask: any }> = []
    for (const batch of loader) batches.push(batch)

    expect(loader.length).toBe(2)
    expect(batches[0].inputIds.shape).toEqual([2, 2])
    expect(batches[0].attentionMask.shape).toEqual([2, 2])
    expect(batches[0].inputIds.dtype).toBe('i64')
    expect(batches[0].attentionMask.dtype).toBe('i64')
    expect(values(batches[0].inputIds)).toEqual([[2, 3], [4, 0]])
    expect(values(batches[0].attentionMask)).toEqual([[1, 1], [1, 0]])
    session.dispose()
  })

  test('text datasets can read paragraph corpora', () => {
    const prefix = `/tmp/affon-dataset-paragraphs-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const path = `${prefix}.txt`
    writeFileSync(path, ' first block \n\nsecond line\nstill second\n\n third ')

    const ds = text.readParagraphs(path, {
      trim: true,
      skipEmpty: true,
    })

    expect(ds.toArray()).toEqual([
      'first block',
      'second line\nstill second',
      'third',
    ])
  })

  test('text datasets can start from in-memory rows', () => {
    const ds = text.rows(['alpha', '', 'beta']).filter((text) => text.length > 0)
    expect(ds.toArray()).toEqual(['alpha', 'beta'])
  })

  test('text datasets can start from raw strings with line or paragraph mode', () => {
    expect(text.fromString(' a \n\nb\n', {
      mode: 'line',
      trim: true,
      skipEmpty: true,
    }).toArray()).toEqual(['a', 'b'])

    expect(text.fromString(' a \n\nb\n\nc ', {
      mode: 'paragraph',
      trim: true,
      skipEmpty: true,
    }).toArray()).toEqual(['a', 'b', 'c'])
  })

  test('encoded text datasets support token windowing as a pipeline operation', () => {
    const tok = createLookupTokenizer(
      { '<bos>': 0, '<eos>': 1, hello: 2, world: 3, goodbye: 4 },
      { specialTokens: { bos: '<bos>', eos: '<eos>' } }
    )

    const windows = text.read('test/e2e/dataset/fixtures/lines-hf.txt')
      .encode(tok, { addBos: true, addEos: true })
      .window({ seqLen: 3, stride: 1 })
      .toArray()

    expect(windows).toEqual([
      [0, 2, 3, 1],
      [2, 3, 1, 0],
      [3, 1, 0, 4],
      [1, 0, 4, 1],
    ])

    const encoded = text.encoded([
      [0, 2, 3, 1],
      [0, 2, 4, 1],
    ])
    expect(encoded.window({ seqLen: 3 }).toArray()).toEqual([
      [0, 2, 3, 1],
      [1, 0, 2, 4],
    ])
  })

  test('encoded text datasets support shuffle before downstream preparation', () => {
    const encoded = text.encoded([
      [1],
      [2],
      [3],
    ])

    const shuffled = encoded.shuffle().toArray()
    expect(shuffled).toHaveLength(3)
    expect(shuffled.sort((a, b) => a[0] - b[0])).toEqual([[1], [2], [3]])
  })

  test('text tokenization supports single-sequence truncation', () => {
    const tok = createLookupTokenizer(
      { '<bos>': 0, '<eos>': 1, alpha: 2, beta: 3, gamma: 4, delta: 5 },
      { specialTokens: { bos: '<bos>', eos: '<eos>' } }
    )
    const ds = text.read('test/e2e/dataset/fixtures/phrases.txt')
      .encode(tok, { addBos: true, addEos: true, maxLength: 4 })

    expect(ds.toArray()[0]).toEqual([0, 2, 3, 1])
    expect(ds.toArray()[1]).toEqual([0, 4, 1])
    expect(ds.toArray()[2]).toEqual([0, 5, 1])
  })
})
