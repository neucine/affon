import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import {getEnv} from 'std:process'
import type {Device, Tensor} from 'affon:compute'
import {load_whisper} from '../../../packages/@affon/huggingface/src/adapters/whisper.ts'
import {load_whisper_processor} from '../../../packages/@affon/huggingface/src/processors/whisper.ts'
import {WhisperProgramRuntime} from '../src/inference/program-runtime.ts'
const root=getEnv('AFFON_WHISPER_DIR')??'apps/hf-inference/artifacts/whisper',device=(getEnv('AFFON_DEVICE')??'cpu') as Device
const reference=checkpoint.load(`${root}/reference.safetensors`) as Record<string,Tensor>
const processor=load_whisper_processor(`${root}/source`,device)
const features=processor.process(new Float32Array(reference.waveform.to_array() as number[]),16000)
const actual=(features.to_array() as number[][][]).flat(2),expected=(reference.features.to_array() as number[][][]).flat(2)
let feature_error=0;for(let i=0;i<actual.length;i++)feature_error=Math.max(feature_error,Math.abs(actual[i]-expected[i]))
console.log(JSON.stringify({device,feature_error}))
if(feature_error>3e-5)throw Error('Whisper feature parity failed')
const model=load_whisper(`${root}/source`,{task:'automatic-speech-recognition',backend:'onnx',graph_dir:root})
const runtime=new WhisperProgramRuntime(model,device)
const result=runtime.transcribe(features,(step,logits)=>{
 const ref=reference[`logits_${step}`];if(!ref)throw Error('Unexpected decoder step')
 const truth=ref.to_array() as number[]
 let max=0;for(let i=0;i<logits.length;i++)max=Math.max(max,Math.abs(logits[i]-truth[i]))
 const passed=logits.every((v,i)=>Math.abs(v-truth[i])<=1e-4+1e-4*Math.abs(truth[i]))
 console.log(JSON.stringify({step,logit_error:max,passed}))
 if(!passed)throw Error('Whisper logits mismatch')
})
console.log(JSON.stringify(result))

const expectedReport=JSON.parse(fs.readFileSync(`${root}/whisper.json`))
if(JSON.stringify(result.tokens)!==JSON.stringify(expectedReport.ids)||result.text!==expectedReport.hf_text)throw Error('Whisper generation parity failed')
runtime.dispose();features.dispose()
