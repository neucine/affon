import fs from 'std:fs'
import { describe, expect, test } from 'std:test'
import { createHFTokenizerFromJSON, createHFTokenizerFromFile } from '../src/index.ts'
import type { HFTokenizerJSON } from '../src/index.ts'

const reference = JSON.parse(fs.readFileSync('packages/@affon/tokenizers/test/hf-bytelevel-reference.json')) as {
  fixtures: { spec: HFTokenizerJSON; cases: { text: string; ids: number[]; decoded: string; skip_special: string }[] }[]
}
describe('HF ByteLevel independent reference parity', () => {
  for (const fixture of reference.fixtures) {
    const spec = fixture.spec
    const tokenizer = createHFTokenizerFromJSON(spec)
    for (const sample of fixture.cases) {
      test(`prefix=${spec.pre_tokenizer?.add_prefix_space} regex=${spec.pre_tokenizer?.use_regex} ${JSON.stringify(sample.text)}`, () => {
        expect(tokenizer.encode(sample.text)).toEqual(sample.ids)
        expect(tokenizer.decode(sample.ids, { skipSpecialTokens: false })).toBe(sample.decoded)
        expect(tokenizer.decode(sample.ids, { skipSpecialTokens: true })).toBe(sample.skip_special)
      })
    }
  }
  test('added vocabulary is exposed through both JSON and file entrypoints', () => {
    const spec = reference.fixtures[0].spec
    const path = `/tmp/affon-hf-added-${Date.now()}.json`
    fs.writeFileSync(path, JSON.stringify(spec))
    for (const tokenizer of [createHFTokenizerFromJSON(spec), createHFTokenizerFromFile(path)]) {
      const id = spec.added_tokens!.find(token => token.content === 'tag')!.id
      expect(tokenizer.vocabSize).toBe(id + 1)
      expect(tokenizer.tokenId('tag')).toBe(id)
      expect(tokenizer.token(id)).toBe('tag')
      expect(tokenizer.allSpecialTokenIds.includes(id)).toBe(false)
    }
  })
})
