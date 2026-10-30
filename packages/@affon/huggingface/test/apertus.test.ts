import { expect, test } from 'std:test'
import fs from 'std:fs'
import { Session, Tensor } from 'affon:compute'
import { load_apertus, load_model } from '../src/index.ts'
import { execute_causal } from './execute-causal.ts'

const directory = 'packages/@affon/huggingface/test/fixtures/apertus'

test('Apertus native full-prefix logits and hidden states match Transformers', () => {
  const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))
  const model = load_model(directory, {task:'text-generation'})
  const result = execute_causal(model, reference.ids)
  const compare = (actual:unknown, expected:unknown) => {
    const a=(actual as number[]).flat(Infinity) as number[], b=(expected as number[]).flat(Infinity) as number[]
    expect(a.length).toBe(b.length)
    let largest=0
    for(let i=0;i<a.length;i++) largest=Math.max(largest,Math.abs(a[i]-b[i]))
    expect(largest<2e-4).toBe(true)
  }
  compare(result.logits,reference.logits)
  result.hidden_states.forEach((value,index)=>compare(value,reference.hidden_states[index]))
  expect(()=>load_apertus(directory).forward(65)).toThrow()
})

test('Apertus loader accepts published BF16 storage across all tensor groups', () => {
  const model=load_apertus('packages/@affon/huggingface/test/fixtures/apertus-bf16')
  expect(model.parameters.lm_head.shape).toEqual([32,16])
  expect(model.parameters.block_0_up.shape).toEqual([24,16])
  expect(model.parameters.block_0_key.shape).toEqual([8,16])
  const reference=JSON.parse(fs.readFileSync('packages/@affon/huggingface/test/fixtures/apertus-bf16/reference.json'))
  const source=model.forward(4)
  expect(source.inspect().parameters.every(value=>value.spec.dtype==='bf16')).toBe(true)
  const result=execute_causal(model,[1,7,9,4])
  const actual=(result.logits as number[]).flat(Infinity) as number[]
  const expected=(reference.logits as number[]).flat(Infinity) as number[]
  const maxError=Math.max(...actual.map((value,index)=>Math.abs(value-expected[index])))
  expect(Number.isFinite(maxError) && maxError<0.02).toBe(true)
})

test('Apertus fixed-capacity KV decode reuses one Program across positions', () => {
  for (const fixture of ['apertus','apertus-bf16']) {
    const model=load_apertus(`packages/@affon/huggingface/test/fixtures/${fixture}`)
    const session=new Session({device:'cpu'})
    const capacity=5, layers=model.config.num_hidden_layers
    const prefill=model.prefill(3,capacity)
    const state=session.initialize(prefill,{parameters:model.parameters})
    const ids=session.tensor([1,7,9],{dtype:'i64'})
    const mask=session.tensor([[[[0,1,1],[0,0,1],[0,0,0]]]],{dtype:'i64'})
    const prefilled=session.compile(prefill).run({ids,mask},state) as Tensor[]
    const decode=model.decode(capacity)
    expect(model.decode(capacity)).toBe(decode)
    const executable=session.compile(decode)
    const compare=(actual:Tensor, tokenIds:number[])=>{
      const got=(actual.to_array() as number[]).flat(Infinity) as number[]
      const expected=(execute_causal(model,tokenIds,tokenIds.length-1).logits as number[]).flat(Infinity) as number[]
      let maximum=0
      for(let i=0;i<got.length;i++) maximum=Math.max(maximum,Math.abs(got[i]-expected[i]))
      expect(maximum< (fixture==='apertus' ? 2e-4 : 0.02)).toBe(true)
    }
    const step=(token:number,position:number,past:Tensor[])=>{
      const stepIds=session.tensor([token],{dtype:'i64'})
      const stepPosition=session.tensor([position],{dtype:'i64'})
      const inputs:Record<string,Tensor>={ids:stepIds,position:stepPosition}
      for(let i=0;i<layers;i++) {
        inputs[`past_${i}_k`]=past[1+2*i]
        inputs[`past_${i}_v`]=past[2+2*i]
      }
      try { return executable.run(inputs,state) as Tensor[] }
      finally { stepIds.dispose();stepPosition.dispose() }
    }
    try {
      compare(prefilled[0],[1,7,9])
      const first=step(4,3,prefilled)
      try {
        compare(first[0],[1,7,9,4])
        const second=step(5,4,first)
        try { compare(second[0],[1,7,9,4,5]) }
        finally { second.forEach(value=>value.dispose()) }
      } finally { first.forEach(value=>value.dispose()) }
    } finally {
      prefilled.forEach(value=>value.dispose())
      ids.dispose();mask.dispose();state.dispose();session.dispose()
    }
    expect(()=>model.prefill(6,5)).toThrow()
    expect(()=>model.decode(model.config.max_position_embeddings+1)).toThrow()
  }
})

test('Apertus BF16 xIELU stays finite with published-scale coefficients', () => {
  const model=load_apertus('packages/@affon/huggingface/test/fixtures/apertus-bf16')
  const alphaP=Tensor.from([166],{dtype:'bf16'}), alphaN=Tensor.from([40.75],{dtype:'bf16'})
  model.parameters.block_0_alphaP=alphaP
  model.parameters.block_0_alphaN=alphaN
  try {
    const logits=(execute_causal(model,[1,7,9,4]).logits as number[]).flat(Infinity) as number[]
    expect(logits.every(Number.isFinite)).toBe(true)
  } finally { alphaP.dispose(); alphaN.dispose() }
})
