import checkpoint from 'affon:checkpoint'
import { describe, expect, test, values } from 'std:test'

const fixture = (name: string) => `test/e2e/checkpoint/fixtures/${name}.safetensors`

describe('HF SafeTensors interoperability', () => {
  test('ignores standard string metadata while loading named tensors', () => {
    const state = checkpoint.load(fixture('hf-metadata')) as any
    expect(Object.keys(state)).toEqual(['weight'])
    expect(values(state.weight)).toEqual([0])
  })
  test('cleans up already loaded tensors when later metadata is invalid', () => {
    const path = fixture('partial-invalid')
    for (let i = 0; i < 3; i++) expect(() => checkpoint.load(path)).toThrow()
  })
  test('rejects malformed headers without native traps', () => {
    for (const name of ['invalid-root', 'invalid-metadata', 'negative-shape',
      'invalid-offsets', 'wrong-size', 'unsupported-dtype']) {
      expect(() => checkpoint.load(fixture(name))).toThrow()
    }
  })
})
