import { test, expect } from 'std:test'
import { createHFTokenizerFromJSON } from '../src/index.ts'
test('Digits before ByteLevel prevents BPE merges across numeric characters', () => {
  const model = {type: 'BPE', vocab: {'1':0,'2':1,'3':2,'12':3,'123':4,'a':5}, merges:['1 2','12 3']}
  const tokenizer = createHFTokenizerFromJSON({model, pre_tokenizer:{type:'Sequence',pretokenizers:[{type:'Digits',individual_digits:true},{type:'ByteLevel',add_prefix_space:false,use_regex:true}]}})
  expect(tokenizer.encode('a123a')).toEqual([5,0,1,2,5])
  expect(tokenizer.decode([5,0,1,2,5])).toBe('a123a')
  expect(createHFTokenizerFromJSON({model,pre_tokenizer:{type:'ByteLevel'}}).encode('123')).toEqual([4])
  const unsupported = createHFTokenizerFromJSON({model,pre_tokenizer:{type:'Sequence',pretokenizers:[{type:'Digits',individual_digits:false},{type:'ByteLevel'}]}})
  expect(() => unsupported.encode('123')).toThrow()
})
