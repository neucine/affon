# Text Datasets

`affon:dataset` owns the text input pipeline surface for line corpora, record-oriented text datasets, tokenization, truncation, batching, and model-ready tensor export.

This layer is intentionally generic:
- raw text examples and text pairs
- single-label classification
- multi-label classification
- scalar regression
- tokenizer adapters behind one `TextTokenizer` interface

It does not own transformer model code, generation, or model-family-specific forward passes.

The current text API is usable now, but it should be read as an early slice of a broader dataset pipeline direction. Across tabular, text, and future image helpers, the intended shared shape is:
- read examples
- transform examples
- derive or select inputs
- derive or select targets
- batch
- tensorize

Text-specific operators such as tokenization, pair templating, and truncation stay domain-specific inside that shared pipeline grammar.

On the tabular side, the same shared grammar now surfaces through `.input(...)`, `.target(...)`, and `.tensorLoader(...)`, even though the underlying transforms remain column-oriented.

## Import Shape

```typescript
import dataset from 'affon:dataset'
```

Most text APIs live under `dataset.text.*`.

## Tokenizers

Every tokenizer family is adapted to the same public interface:

```typescript
interface TextTokenizer {
  encode(text: string, opts?): number[]
  decode(ids: readonly number[], opts?): string
  specialTokens?: { bos?: string; eos?: string; pad?: string; unk?: string; sep?: string }
  specialTokenIds?: { bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }
  vocabSize: number
  tokenId(token: string): number | undefined
  token(id: number): string | undefined
}
```

Tokenizer constructors live in the first-party tokenizer package:

```typescript
import { createLookupTokenizer } from '@affon/tokenizers'
```

Current package constructors:
- `createLookupTokenizer(...)`
- `createHFTokenizerFromJSON(...)`
- `createHFTokenizerFromFile(...)`
- `createSentencePieceTokenizer(...)`
- `createSentencePieceTokenizerFromFile(...)`

The HF adapter currently supports common `WordLevel`, `BPE`, and `WordPiece` JSON layouts.
The SentencePiece adapter currently accepts a simplified JSON model shape.

Boundary note:
- `affon:dataset` treats tokenizers as part of the text pipeline contract: encode examples, batch, pad, and tensorize.
- Tokenizer implementations and external format adapters live in `@affon/tokenizers`.
- Model-family tokenization recipes, prompt templates, and corpus-packing policy should live in packages or apps rather than in `affon:dataset`.

## Raw Text

```typescript
const lines = dataset.text.read('lines.txt', {
  trim: true,
  skipEmpty: true,
})

const paragraphs = dataset.text.readParagraphs('notes.txt', {
  trim: true,
  skipEmpty: true,
})

const rows = dataset.text.rows([
  'alpha',
  'beta',
])

const parsed = dataset.text.fromString('alpha\n\nbeta', {
  mode: 'paragraph',
  trim: true,
  skipEmpty: true,
})
```

`TextDataset` supports:
- `map(...)`
- `filter(...)`
- `shuffle()`
- `sample(n)`
- `split(...)`
- `loader(...)`
- `encode(tokenizer, opts?)`
- `records(field?)`

## Record Pipelines

Use `readDelimited(...)` when you want one example stream with field transforms:

```typescript
const records = dataset.text.readDelimited('labeled-lines.tsv', {
  delimiter: '\t',
  columns: ['text', 'label'],
})
```

`TextRecordDataset` supports:
- `map(...)`
- `filter(...)`
- `shuffle()`
- `sample(n)`
- `split(...)`
- `loader(...)`
- `encode(field, tokenizer, opts?)`
- `encodePair({ left, right }, tokenizer, opts?)`
- `splitLabels(field, opts?)`
- `encodeLabels(field, labelEncoder, opts?)`
- `toMultiHot(field, labelEncoder, opts?)`
- `cast(field, 'i64' | 'f32', opts?)`
- `input(...fields)`
- `target(...fields)`
- `paddedLoader(...)`
- `tensorLoader(...)`

## Tokenized Text

```typescript
import { createLookupTokenizer } from '@affon/tokenizers'

const tok = createLookupTokenizer(
  {
    '<pad>': 0,
    '<bos>': 1,
    '<eos>': 2,
    '<sep>': 3,
    '<unk>': 4,
    hello: 5,
    world: 6,
  },
  {
    specialTokens: {
      pad: '<pad>',
      bos: '<bos>',
      eos: '<eos>',
      sep: '<sep>',
      unk: '<unk>',
    },
  },
)

const encoded = dataset.text.read('lines.txt').encode(tok, {
  addBos: true,
  addEos: true,
  maxLength: 128,
})

const windows = encoded.window({
  seqLen: 128,
  stride: 64,
})
```

`TextEncodeOpts` supports:
- `addBos`
- `addEos`
- `maxLength`
- `truncation: 'longest_first' | 'only_first' | 'only_second'`

For single-sequence text, `maxLength` truncates the encoded sequence directly.

`EncodedTextDataset` supports:
- `map(...)`
- `filter(...)`
- `shuffle()`
- `sample(n)`
- `split(...)`
- `window({ seqLen, stride?, joinWithTokenId? })`
- `loader(...)`
- `paddedLoader(...)`
- `tensorLoader(...)`

If you already have token rows, you can enter the same pipeline directly:

```typescript
const encodedRows = dataset.text.encoded([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
])

const windows = encodedRows.window({ seqLen: 3 })
```

## Classification

```typescript
const labelEncoder = dataset.text.labelEncoder({
  pos: 0,
  neg: 1,
  neu: 2,
})

const classified = dataset.text
  .readDelimited('labeled-lines.tsv', {
    delimiter: '\t',
    columns: ['text', 'label'],
  })
  .encode('text', tok, { into: 'inputIds' })
  .encodeLabels('label', labelEncoder, { into: 'labels' })
  .input('inputIds')
  .target('labels')

for (const batch of classified.tensorLoader({ batchSize: 32, padId: 0 })) {
  // batch.inputIds: i64 tensor
  // batch.attentionMask: i64 tensor
  // batch.labels: i64 tensor
}
```

## Multi-Label Classification

```typescript
const tags = dataset.text.labelEncoder({
  pos: 0,
  neg: 1,
  neu: 2,
  featured: 3,
})

const multilabel = dataset.text
  .readDelimited('multilabel-lines.tsv', {
    delimiter: '\t',
    columns: ['text', 'labels'],
  })
  .splitLabels('labels')
  .encode('text', tok, { into: 'inputIds' })
  .toMultiHot('labels', tags, { into: 'labels' })
  .input('inputIds')
  .target('labels')

for (const batch of multilabel.tensorLoader({ batchSize: 32, padId: 0 })) {
  // batch.labels: multi-hot i64 tensor [batch, numClasses]
}
```

Useful `LabelEncoder` helpers:
- `size`
- `encodeMany(...)`
- `decodeMany(...)`

## Regression

```typescript
const scored = dataset.text
  .readDelimited('scored-lines.tsv', {
    delimiter: '\t',
    columns: ['text', 'score'],
  })
  .encode('text', tok, { into: 'inputIds' })
  .cast('score', 'f32', { into: 'labels' })
  .input('inputIds')
  .target('labels')

for (const batch of scored.tensorLoader({ batchSize: 32, padId: 0 })) {
  // batch.labels: f32 tensor
}
```

## Text Pairs

```typescript
const pairs = dataset.text
  .readDelimited('text-pairs.tsv', {
    delimiter: '\t',
    columns: ['left', 'right'],
  })
  .encodePair({ left: 'left', right: 'right' }, tok, {
    into: 'inputIds',
    tokenTypesInto: 'tokenTypeIds',
    withTokenTypes: true,
    pairTemplatePreset: 'bert',
    maxLength: 128,
    truncation: 'longest_first',
  })
  .input('inputIds', 'tokenTypeIds')

for (const batch of pairs.tensorLoader({ batchSize: 16, padId: 0 })) {
  // batch.inputIds
  // batch.attentionMask
  // batch.tokenTypeIds
}
```

Pair templating:
- `pairTemplatePreset: 'joined'`
- `pairTemplatePreset: 'bert'`
- `pairTemplate: Array<'bos' | 'left' | 'separator' | 'right' | 'eos'>`

`bert` expands to:
- `bos,left,separator,right,separator,eos`

If the tokenizer exposes `specialTokenIds.sep`, the preset can use it directly. Otherwise pass `separatorText`.

## Batch Shapes

Common tensorized outputs:

Single text:

```typescript
{
  inputIds: Tensor<number[], "i64">
  attentionMask: Tensor<number[], "i64">
}
```

Single-label classification:

```typescript
{
  inputIds: Tensor<number[], "i64">
  attentionMask: Tensor<number[], "i64">
  labels: Tensor<number[], "i64">
}
```

Multi-label classification:

```typescript
{
  inputIds: Tensor<number[], "i64">
  attentionMask: Tensor<number[], "i64">
  labels: Tensor<number[], "i64">
}
```

Regression:

```typescript
{
  inputIds: Tensor<number[], "i64">
  attentionMask: Tensor<number[], "i64">
  labels: Tensor<number[], "f32">
}
```

Text pairs:

```typescript
{
  inputIds: Tensor<number[], "i64">
  attentionMask: Tensor<number[], "i64">
  tokenTypeIds: Tensor<number[], "i64">
}
```

## Boundary

`dataset.text` should stay responsible for:
- reading examples
- tokenization
- truncation
- templating
- padding
- batching
- tensor export

It should not absorb:
- model forward logic
- generation helpers
- beam search
- decoder caches
- transformer-family training semantics that do not generalize beyond dataset preparation
