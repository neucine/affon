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
      'invalid-offsets', 'wrong-size', 'bf16-wrong-size', 'unsupported-dtype']) {
      expect(() => checkpoint.load(fixture(name))).toThrow()
    }
  })
})

test('BF16 storage widens exactly to f32, including subnormals and special values', () => {
  const state = checkpoint.load(fixture('bf16')) as any
  const v = values(state.weight)
  expect(state.weight.dtype).toBe('f32')
  expect(v.slice(0, 4)).toEqual([0, -0, 1, -2.5])
  expect(Object.is(v[1], -0)).toBe(true)
  expect(v[4]).toBe(2 ** -133)
  expect(v[5]).toBe((2 - 2 ** -7) * 2 ** 127)
  expect(v[6]).toBe(Infinity); expect(v[7]).toBe(-Infinity)
  expect(Number.isNaN(v[8])).toBe(true)
})

test('checkpoint metadata and selective reads preserve storage dtype and validate requested names', () => {
  expect(checkpoint.inspect(fixture('bf16'))).toEqual({weight:{dtype:'BF16',shape:[9]}})
  expect(Object.keys(checkpoint.load(fixture('bf16'),{names:[]}))).toEqual([])
  const state=checkpoint.load(fixture('hf-metadata'),{names:['weight']}) as any
  expect(values(state.weight)).toEqual([0])
  expect(()=>checkpoint.load(fixture('bf16'),{names:['absent']})).toThrow('Missing requested')
  expect(()=>checkpoint.load(fixture('bf16'),{names:['weight','weight']})).toThrow('Duplicate')
  expect(()=>checkpoint.load(fixture('bf16'),{names:[3] as any})).toThrow()
  expect(()=>checkpoint.inspect(fixture('bf16-wrong-size'))).toThrow()
  expect(()=>checkpoint.load(fixture('partial-invalid'),{names:[]})).toThrow()
})

test('BF16 streaming handles a full 64 KiB input chunk and a partial tail without count overflow', () => {
  const state=checkpoint.load(fixture('bf16-chunks')) as any
  const v=values(state.weight)
  expect(v.length).toBe(32769)
  expect(v.every((x:number,i:number)=>x===(i===32768?2:1))).toBe(true)
})
