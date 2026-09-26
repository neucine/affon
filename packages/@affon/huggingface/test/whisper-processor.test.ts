import {test,expect,beforeAll,afterAll} from 'std:test'
import fs from 'std:fs'
import {run} from 'std:process'
import {load_processor} from '../src/processor.ts'
let directory=''
beforeAll(async()=>{
 directory=(await run({cmd:'mktemp',args:['-d','/tmp/whisper-processor.XXXXXX']})).stdout.trim()
 fs.writeFileSync(`${directory}/config.json`,JSON.stringify({model_type:'whisper'}))
 fs.writeFileSync(`${directory}/preprocessor_config.json`,JSON.stringify({feature_extractor_type:'WhisperFeatureExtractor',feature_size:80,sampling_rate:16000,n_fft:400,hop_length:160,chunk_length:30}))
})
afterAll(async()=>{if(directory)await run({cmd:'rm',args:['-rf',directory]})})
test('Whisper centered STFT and Slaney features match HF, including padded tail',()=>{
 const p=load_processor(directory,{task:'automatic-speech-recognition'})
 const cases=JSON.parse(fs.readFileSync('packages/@affon/huggingface/test/fixtures/whisper-processor.json'))
 for(const c of cases){
  const wave=Float32Array.from({length:c.length},(_,i)=>((i*73)%251-125)/256)
  const features=p.process(wave,16000)
  expect(features.shape).toEqual([1,80,3000])
  const values=(features.to_array() as number[][][])[0]
  for(let b=0;b<80;b++)for(let i=0;i<c.indices.length;i++)expect(Math.abs(values[b][c.indices[i]]-c.values[b][i])<3e-5).toBe(true)
 }
 expect(()=>p.process(new Float32Array(1),16000)).toThrow()
})
