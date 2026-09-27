import fs from 'std:fs'
import { describe, expect, test } from 'std:test'
import { createHFTokenizerFromJSON } from '../../../packages/@affon/tokenizers/src/index.ts'
import { load_bert_processor } from '../../../packages/@affon/huggingface/src/index.ts'
import { resize_rgb } from '../../../packages/@affon/huggingface/src/processors/shared/resize-rgb.ts'

const bert = JSON.parse(fs.readFileSync('apps/hf-inference/tests/fixtures/bert-processor-reference.json'))
const resize = JSON.parse(fs.readFileSync('apps/hf-inference/tests/fixtures/resize-reference.json'))
describe('independent BERT processor references', () => {
  const tokenizer = createHFTokenizerFromJSON(bert.spec)
  for (const row of bert.raw) test(`raw ${JSON.stringify(row.text)}`, () => {
    expect(tokenizer.encode(row.text)).toEqual(row.ids)
  })
  for (let i = 0; i < bert.batches.length; i++) test(`template/padding batch ${i}`, () => {
    const processor = load_bert_processor('apps/hf-inference/tests/fixtures/processor-fixture')
    const row = bert.batches[i]
    expect(processor.encode_batch(row.texts, row.pairs ?? undefined)).toEqual(row.expected)
  })
})
describe('independent Pillow bilinear references', () => {
  for (let i = 0; i < resize.cases.length; i++) test(`resize ${i}`, () => {
    const row = resize.cases[i]
    expect(resize_rgb(row.input, row.height, row.width)).toEqual(row.expected)
  })
})
