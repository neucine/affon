import { describe, test, expect, values } from 'std:test'
import dataset from 'affon:dataset'
import { createLookupTokenizer } from '../../../packages/tokenizers/src/index.ts'

describe('dataset text records', () => {
  test('text records support single-label classification through field transforms', () => {
    const tok = createLookupTokenizer(
      { '<pad>': 0, '<unk>': 1, good: 2, bad: 3, neutral: 4 },
      { specialTokens: { pad: '<pad>', unk: '<unk>' } }
    )
    const labels = dataset.text.labelEncoder({ pos: 0, neg: 1, neu: 2 })
    const ds = dataset.text.readDelimited('test/e2e/dataset/fixtures/labeled-lines.tsv', {
      delimiter: '\t',
      columns: ['text', 'label'],
    })
      .encode('text', tok, { into: 'inputIds' })
      .encodeLabels('label', labels, { into: 'labels' })
      .input('inputIds')
      .target('labels')

    expect(ds.toArray()).toEqual([
      { text: 'good', label: 'pos', inputIds: [2], labels: 0 },
      { text: 'bad', label: 'neg', inputIds: [3], labels: 1 },
      { text: 'neutral', label: 'neu', inputIds: [4], labels: 2 },
    ])

    const tensorBatches = [...ds.tensorLoader({ batchSize: 2, padId: 0 })]
    expect(values(tensorBatches[0].inputIds)).toEqual([[2], [3]])
    expect(values(tensorBatches[0].attentionMask)).toEqual([[1], [1]])
    expect(values(tensorBatches[0].labels)).toEqual([0, 1])
    expect(tensorBatches[0].labels.dtype).toBe('i64')
    expect(labels.decode(2)).toBe('neu')
  })

  test('text records support multi-label transforms and multi-hot targets', () => {
    const tok = createLookupTokenizer(
      { '<pad>': 0, '<unk>': 1, good: 2, bad: 3, neutral: 4 },
      { specialTokens: { pad: '<pad>', unk: '<unk>' } }
    )
    const labels = dataset.text.labelEncoder({ pos: 0, neg: 1, neu: 2, featured: 3 })
    const ds = dataset.text.readDelimited('test/e2e/dataset/fixtures/multilabel-lines.tsv', {
      delimiter: '\t',
      columns: ['text', 'labels'],
    })
      .splitLabels('labels')
      .encode('text', tok, { into: 'inputIds' })
      .toMultiHot('labels', labels, { into: 'labels' })
      .input('inputIds')
      .target('labels')

    expect(ds.toArray()).toEqual([
      { text: 'good', labels: [1, 0, 0, 1], inputIds: [2] },
      { text: 'bad', labels: [0, 1, 0, 0], inputIds: [3] },
      { text: 'neutral', labels: [0, 0, 1, 1], inputIds: [4] },
    ])

    const tensorBatches = [...ds.tensorLoader({ batchSize: 2, padId: 0 })]
    expect(values(tensorBatches[0].inputIds)).toEqual([[2], [3]])
    expect(values(tensorBatches[0].attentionMask)).toEqual([[1], [1]])
    expect(values(tensorBatches[0].labels)).toEqual([
      [1, 0, 0, 1],
      [0, 1, 0, 0],
    ])
    expect(labels.encodeMany(['featured', 'pos'])).toEqual([3, 0])
    expect(labels.decodeMany([2, 3])).toEqual(['neu', 'featured'])
  })

  test('text records support numeric target casting for regression', () => {
    const tok = createLookupTokenizer(
      { '<pad>': 0, '<unk>': 1, good: 2, bad: 3, neutral: 4 },
      { specialTokens: { pad: '<pad>', unk: '<unk>' } }
    )
    const ds = dataset.text.readDelimited('test/e2e/dataset/fixtures/scored-lines.tsv', {
      delimiter: '\t',
      columns: ['text', 'score'],
    })
      .encode('text', tok, { into: 'inputIds' })
      .cast('score', 'f32', { into: 'labels' })
      .input('inputIds')
      .target('labels')

    expect(ds.toArray()).toEqual([
      { text: 'good', score: '0.75', inputIds: [2], labels: 0.75 },
      { text: 'bad', score: '-0.5', inputIds: [3], labels: -0.5 },
      { text: 'neutral', score: '0.125', inputIds: [4], labels: 0.125 },
    ])

    const tensorBatches = [...ds.tensorLoader({ batchSize: 2, padId: 0 })]
    expect(values(tensorBatches[0].inputIds)).toEqual([[2], [3]])
    expect(values(tensorBatches[0].attentionMask)).toEqual([[1], [1]])
    expect(values(tensorBatches[0].labels)).toEqual([0.75, -0.5])
    expect(tensorBatches[0].labels.dtype).toBe('f32')
  })

  test('text records support pair encoding with segment ids', () => {
    const tok = createLookupTokenizer(
      { '<bos>': 0, '<eos>': 1, '<sep>': 2, '<unk>': 3, hello: 4, world: 5, good: 6, bye: 7 },
      { specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' } }
    )
    const ds = dataset.text.readDelimited('test/e2e/dataset/fixtures/text-pairs.tsv', {
      delimiter: '\t',
      columns: ['left', 'right'],
    })
      .encodePair({ left: 'left', right: 'right' }, tok, {
        into: 'inputIds',
        tokenTypesInto: 'tokenTypeIds',
        withTokenTypes: true,
        addBos: true,
        addEos: true,
        separatorText: '<sep>',
      })
      .input('inputIds', 'tokenTypeIds')

    expect(ds.toArray()).toEqual([
      { left: 'hello', right: 'world', inputIds: [0, 4, 2, 5, 1], tokenTypeIds: [0, 0, 0, 1, 1] },
      { left: 'good', right: 'bye', inputIds: [0, 6, 2, 7, 1], tokenTypeIds: [0, 0, 0, 1, 1] },
    ])

    const batches = [...ds.tensorLoader({ batchSize: 2, padId: 0 })]
    expect(values(batches[0].inputIds)).toEqual([
      [0, 4, 2, 5, 1],
      [0, 6, 2, 7, 1],
    ])
    expect(values(batches[0].tokenTypeIds)).toEqual([
      [0, 0, 0, 1, 1],
      [0, 0, 0, 1, 1],
    ])
    expect(values(batches[0].attentionMask)).toEqual([
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
    ])
  })

  test('text pair encoding supports explicit truncation policies', () => {
    const tok = createLookupTokenizer(
      { '<bos>': 0, '<eos>': 1, left: 2, a: 3, b: 4, right: 5, c: 6, d: 7 },
      { specialTokens: { bos: '<bos>', eos: '<eos>' } }
    )
    const pairs = dataset.text.readDelimited('test/e2e/dataset/fixtures/text-pairs.tsv', {
      delimiter: '\t',
      columns: ['left', 'right'],
    }).map(() => ({ left: 'left a b', right: 'right c d' }))

    expect(pairs.encodePair({ left: 'left', right: 'right' }, tok, { into: 'inputIds', maxLength: 5, truncation: 'only_first' }).toArray()[0].inputIds).toEqual([2, 3, 5, 6, 7])
    expect(pairs.encodePair({ left: 'left', right: 'right' }, tok, { into: 'inputIds', maxLength: 5, truncation: 'only_second' }).toArray()[0].inputIds).toEqual([2, 3, 4, 5, 6])
    expect(pairs.encodePair({ left: 'left', right: 'right' }, tok, {
      into: 'inputIds',
      tokenTypesInto: 'tokenTypeIds',
      withTokenTypes: true,
      addBos: true,
      addEos: true,
      maxLength: 6,
      truncation: 'longest_first',
    }).toArray()[0]).toEqual({
      left: 'left a b',
      right: 'right c d',
      inputIds: [0, 2, 3, 5, 6, 1],
      tokenTypeIds: [0, 0, 0, 1, 1, 1],
    })
  })

  test('text pair encoding supports explicit pair templates', () => {
    const tok = createLookupTokenizer(
      { '<bos>': 0, '<eos>': 1, '<sep>': 2, left: 3, right: 4 },
      { specialTokens: { bos: '<bos>', eos: '<eos>' } }
    )
    const pairs = dataset.text.readDelimited('test/e2e/dataset/fixtures/text-pairs.tsv', {
      delimiter: '\t',
      columns: ['left', 'right'],
    }).map(() => ({ left: 'left', right: 'right' }))

    expect(pairs.encodePair({ left: 'left', right: 'right' }, tok, {
      into: 'inputIds',
      tokenTypesInto: 'tokenTypeIds',
      withTokenTypes: true,
      separatorText: '<sep>',
      pairTemplate: ['bos', 'left', 'separator', 'right', 'separator', 'eos'],
    }).toArray()[0]).toEqual({
      left: 'left',
      right: 'right',
      inputIds: [0, 3, 2, 4, 2, 1],
      tokenTypeIds: [0, 0, 0, 1, 1, 1],
    })
  })

  test('text pair encoding supports pair template presets', () => {
    const tok = createLookupTokenizer(
      { '<bos>': 0, '<eos>': 1, '<sep>': 2, left: 3, right: 4 },
      { specialTokens: { bos: '<bos>', eos: '<eos>', sep: '<sep>' } }
    )
    const pairs = dataset.text.readDelimited('test/e2e/dataset/fixtures/text-pairs.tsv', {
      delimiter: '\t',
      columns: ['left', 'right'],
    }).map(() => ({ left: 'left', right: 'right' }))

    expect(pairs.encodePair({ left: 'left', right: 'right' }, tok, {
      into: 'inputIds',
      tokenTypesInto: 'tokenTypeIds',
      withTokenTypes: true,
      pairTemplatePreset: 'bert',
    }).toArray()[0]).toEqual({
      left: 'left',
      right: 'right',
      inputIds: [0, 3, 2, 4, 2, 1],
      tokenTypeIds: [0, 0, 0, 1, 1, 1],
    })
  })
})
