import { test, expect } from 'std:test'
import fs from 'std:fs'
import { run } from 'std:process'
import checkpoint from 'affon:checkpoint'
import type { Tensor } from 'affon:compute'
import { open_checkpoint } from '../src/adapters/checkpoint.ts'
import { load_llama } from '../src/adapters/llama.ts'
import { snapshot_download } from '../src/index.ts'
import { parse_checkpoint_index } from '../src/hub/checkpoint-index.ts'
import { generate_causal } from './execute-causal.ts'
const fixture = 'packages/@affon/huggingface/test/fixtures/llama'
test('indexed shards match a single-file model and enforce tensor ownership', async () => {
  const temp = (await run({cmd:'mktemp',args:['-d','/tmp/affon-shard-test.XXXXXX']})).stdout.trim()
  try {
    const weights = checkpoint.load(`${fixture}/model.safetensors`) as Record<string, Tensor>
    const entries = Object.entries(weights), names = ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']
    const map: Record<string,string> = Object.create(null)
    for (let i=0;i<2;i++) {
      const part = entries.filter((_,j)=>j%2===i)
      checkpoint.save(Object.fromEntries(part),`${temp}/${names[i]}`)
      for (const [name] of part) map[name]=names[i]
    }
    fs.writeFileSync(`${temp}/config.json`,fs.readFileSync(`${fixture}/config.json`))
    const writeIndex = () => fs.writeFileSync(`${temp}/model.safetensors.index.json`,JSON.stringify({weight_map:map}))
    writeIndex()
    expect(generate_causal(load_llama(temp),[1,7,12],3)).toEqual(generate_causal(load_llama(fixture),[1,7,12],3))
    map['absent.weight']=names[0];writeIndex()
    expect(()=>open_checkpoint(temp)).toThrow('Missing indexed')
    delete map['absent.weight'];map[entries[0][0]]=names[1];writeIndex()
    expect(()=>open_checkpoint(temp)).toThrow('ownership mismatch')
    map[entries[0][0]]=names[0];writeIndex()
    checkpoint.save(weights,`${temp}/${names[1]}`)
    expect(()=>open_checkpoint(temp)).toThrow()
  } finally {await run({cmd:'rm',args:['-rf',temp]})}
})
test('shard indexes reject paths, URLs, empty maps and invalid tensor entries', () => {
  for (const file of ['../escape.safetensors','sub/file.safetensors','/tmp/file.safetensors','https://host/file.safetensors','x.bin','x.safetensors\0'])
    expect(()=>parse_checkpoint_index({weight_map:{w:file}})).toThrow()
  for (const value of [null,{}, {weight_map:[]}, {weight_map:{}}, {weight_map:{w:42}}]) expect(()=>parse_checkpoint_index(value)).toThrow()
})
test('offline indexed snapshots verify every shard and reject missing shard manifests', async () => {
  const root=(await run({cmd:'mktemp',args:['-d','/tmp/affon-shard-cache.XXXXXX']})).stdout.trim()
  const revision='0123456789abcdef0123456789abcdef01234567'
  const dir=`${root}/models--owner--model/${revision}`, options={revision,cache_dir:root,local_files_only:true}
  try {
    await run({cmd:'mkdir',args:['-p',dir]})
    fs.writeFileSync(`${dir}/config.json`,'{}')
    fs.writeFileSync(`${dir}/model.safetensors.index.json`,JSON.stringify({weight_map:{a:'part-a.safetensors',b:'part-b.safetensors'}}))
    fs.writeFileSync(`${dir}/part-a.safetensors`,'test a');fs.writeFileSync(`${dir}/part-b.safetensors`,'test b')
    const files=[]
    for (const name of ['config.json','model.safetensors.index.json','part-a.safetensors','part-b.safetensors'])
      files.push({name,size:fs.statSync(`${dir}/${name}`).size,sha256:(await run({cmd:'shasum',args:['-a','256',`${dir}/${name}`]})).stdout.split(/\s/)[0]})
    const write=()=>fs.writeFileSync(`${dir}/affon-snapshot.json`,JSON.stringify({format:'affon-hf-snapshot/v1',model_id:'owner/model',revision,files}))
    write();expect(await snapshot_download('owner/model',options)).toBe(dir)
    files.pop();write()
    let message='';try {await snapshot_download('owner/model',options)} catch(e){message=String(e)}
    expect(message.includes('missing weights')).toBe(true)
  } finally {await run({cmd:'rm',args:['-rf',root]})}
})
