import { beforeAll, afterAll, test, expect } from 'std:test'
import fs from 'std:fs'
import { run, getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import { Session } from 'affon:compute'
import type { Device } from 'affon:compute'
import { load_model } from '../src/index.ts'
let directory=''
const device=(getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const session=new Session({device})
const tensor=(values:any,options:{dtype?:'f32'|'f64'|'i64';device?:Device}={})=>session.tensor(values,{dtype:options.dtype}) as any
const config={model_type:'an-unimplemented-architecture',id2label:{'0':'a','1':'b','2':'c'}}
function forward(model: ReturnType<typeof load_model>, values: unknown) {
  if (!('parameters' in model) || !('input_name' in model)) throw Error('Expected ONNX definition')
  const state=session.initialize(model.forward,{parameters:model.parameters})
  const input=tensor(values,{dtype:'f32',device})
  try {
    const result=session.compile(model.forward).run({[model.input_name]:input},state) as any
    const outputs=Array.isArray(result)?result:[result]
    const output=outputs[model.output_names.indexOf(model.output_name)]
    try {return output.to_array()}
    finally {for(const value of outputs)value.dispose()}
  } finally {input.dispose();state.dispose()}
}
beforeAll(async()=>{
  directory=(await run({cmd:'mktemp',args:['-d','/tmp/affon-hf-onnx.XXXXXX']})).stdout.trim()
  fs.writeFileSync(`${directory}/config.json`,JSON.stringify(config))
  checkpoint.save({bias:tensor([1,2,3],{dtype:'f32',device:'cpu'})},`${directory}/weights.safetensors`)
  fs.writeFileSync(`${directory}/graph.json`,JSON.stringify({format:'affon-onnx-static/v1',opset:17,
    inputs:{image:[1,3,1,1]},outputs:['scores'],constants:{bias:[3]},nodes:[
      {op:'Flatten',name:'flatten',inputs:['image'],output:'flat',shape:[1,3],attrs:{shape:[1,3]}},
      {op:'Add',name:'bias',inputs:['flat','bias'],output:'scores',shape:[1,3],attrs:{}}
    ]}))
})
afterAll(async()=>{if(directory)await run({cmd:'rm',args:['-rf',directory]});session.dispose()})
test('HF ONNX backend binds task inputs/outputs without architecture dispatch',()=>{
  const model=load_model(directory,{task:'image-classification',backend:'onnx',graph_dir:directory})
  expect(model.backend).toBe('onnx')
  expect(model.config.id2label).toEqual(config.id2label)
  expect(forward(model,[[[[4]],[[5]],[[6]]]])).toEqual([[5,7,9]])
  expect(()=>load_model(directory,{task:'image-classification'})).toThrow()
})
test('HF ONNX backend rejects invalid binding, unsupported task and mismatched labels',()=>{
  const options={task:'image-classification',backend:'onnx',graph_dir:directory} as const
  expect(()=>load_model(directory,{...options,input_name:'wrong'})).toThrow()
  expect(()=>load_model(directory,{...options,output_name:'wrong'})).toThrow()
  expect(()=>load_model(directory,{...options,task:'text-generation'} as any)).toThrow()
  fs.writeFileSync(`${directory}/config.json`,JSON.stringify({...config,id2label:{'0':'wrong'}}))
  try { expect(()=>load_model(directory,options)).toThrow() }
  finally { fs.writeFileSync(`${directory}/config.json`,JSON.stringify(config)) }
})

test('HF AST graph validates feature dimensions and binds audio logits',()=>{
  const old=fs.readFileSync(`${directory}/graph.json`)
  const graph=JSON.parse(old);graph.inputs.image=[1,1,3]
  const audioConfig={...config,model_type:'audio-spectrogram-transformer',max_length:1,num_mel_bins:3}
  fs.writeFileSync(`${directory}/config.json`,JSON.stringify(audioConfig))
  fs.writeFileSync(`${directory}/graph.json`,JSON.stringify(graph))
  const options={task:'audio-classification',backend:'onnx',graph_dir:directory} as const
  try {
    const model=load_model(directory,options)
    expect(forward(model,[[[4,5,6]]])).toEqual([[5,7,9]])
    expect(()=>load_model(directory,{...options,task:'image-classification'})).toThrow()
    fs.writeFileSync(`${directory}/config.json`,JSON.stringify({...audioConfig,max_length:2}))
    expect(()=>load_model(directory,options)).toThrow()
  } finally {
    fs.writeFileSync(`${directory}/config.json`,JSON.stringify(config))
    fs.writeFileSync(`${directory}/graph.json`,old)
  }
})
