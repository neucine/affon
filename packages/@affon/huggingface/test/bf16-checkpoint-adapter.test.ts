import {expect,test} from 'std:test'
import {run} from 'std:process'
import checkpoint from 'affon:checkpoint'
import {Session} from 'affon:compute'
import {prepare_encoder_checkpoint} from '../src/adapters/encoder-checkpoint.ts'

test('shared BERT/ViT/Llama linear mapper preserves BF16 dtype and transposition', async () => {
  const directory=(await run({cmd:'mktemp',args:['-d','/tmp/affon-bf16-adapter.XXXXXX']})).stdout.trim()
  const session=new Session({device:'cpu'})
  const weight=session.tensor([[1,2,3],[4,5,6]],{dtype:'bf16'})
  try {
    checkpoint.save({'linear.weight':weight},`${directory}/model.safetensors`)
    const adapter=prepare_encoder_checkpoint(directory)
    adapter.require_weight('linear.weight',[2,3],true)
    adapter.finish()
    expect(adapter.weights['linear.weight'].dtype).toBe('bf16')
    expect(adapter.weights['linear.weight'].shape).toEqual([3,2])
    expect(adapter.weights['linear.weight'].to_array()).toEqual([[1,4],[2,5],[3,6]])
  } finally {
    weight.dispose();session.dispose();await run({cmd:'rm',args:['-rf',directory]})
  }
})
